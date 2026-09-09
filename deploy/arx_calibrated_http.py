"""Sessioned ARX calibrated-v3 and feedback-v4 HTTP API.

The wire action is deliberately *not* a robot command. Arm columns are
absolute joint targets, while gripper columns retain the checkpoint's
calibrated-feedback or VR-intent semantics. Only the reviewed ARX calibrated
client may convert them into X5 command coordinates.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
import logging
from pathlib import Path
import threading
import time
import uuid
from typing import Any

from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse
import numpy as np
from pydantic import BaseModel

from deploy.arx_lift2s_http import _read_jpeg
from tau0_vla.adapters.arx_lift2s.calibrated import (
    PROTOCOL as V3_PROTOCOL,
    VERSION as V3_VERSION,
    calibrate_joint,
    contract as calibrated_contract,
)
from tau0_vla.adapters.arx_lift2s.feedback import (
    BODY as FEEDBACK_BODY,
    PROTOCOL as V4_PROTOCOL,
    VERSION as V4_VERSION,
    contract as feedback_contract,
)
from tau0_vla.adapters.arx_lift2s.deploy_io import build_native_action_perm
from tau0_vla.adapters.arx_lift2s.layout import ARX_LIFT2S_JOINT_NAMES
from tau0_vla.data import action_slices


CLIENT_ADAPTER_VERSION = "arx-calibrated-client-v1"
CAMERA_NAMES = ("head", "left_wrist", "right_wrist")
ACTION_DIM = 14
ACTION_HORIZON = 30
FPS = 30
SUPPORTED_EXPERIMENTS = ("joint-feedback", "joint-vr")
logger = logging.getLogger(__name__)


class SessionRequest(BaseModel):
    protocol_version: str
    calibration_version: str
    client_adapter_version: str
    experiment: str
    task_instruction: str
    client_name: str
    robot_id: str
    calibration_id: str
    open_baselines: dict[str, float]


@dataclass
class _Session:
    session_id: str
    task_instruction: str
    experiment: str
    client_name: str
    robot_id: str
    calibration_id: str
    open_baselines: dict[str, float]
    last_request_id: int = 0


def validate_model_contract(spec) -> dict[str, Any]:
    saved = getattr(spec, "deployment_contract", None)
    if isinstance(saved, dict) and saved == feedback_contract():
        if (
            spec.unified_registry_key != FEEDBACK_BODY
            or bool(spec.unified_has_eef)
            or spec.action_semantics != saved["action_semantics"]
        ):
            raise ValueError("feedback checkpoint encoding and deployment contract disagree")
    elif not isinstance(saved, dict) or saved != calibrated_contract(saved.get("experiment")):
        raise ValueError("checkpoint is missing a supported ARX deployment contract")
    mode = saved["experiment"]
    if mode not in SUPPORTED_EXPERIMENTS:
        raise ValueError(f"calibrated ARX HTTP supports {SUPPORTED_EXPERIMENTS}, got {mode!r}")
    if saved["protocol_version"] == V3_PROTOCOL:
        if (
            spec.unified_registry_key != "arx_calibrated_joint_v1"
            or bool(spec.unified_has_eef)
            or spec.action_semantics != saved["action_semantics"]
        ):
            raise ValueError("calibrated checkpoint encoding and deployment contract disagree")
    if tuple(getattr(spec, "cam_keys", ())) != CAMERA_NAMES:
        raise ValueError(f"calibrated ARX checkpoint requires cameras {CAMERA_NAMES}")
    if int(getattr(spec, "action_chunk_size", 0)) != ACTION_HORIZON:
        raise ValueError("calibrated ARX checkpoint requires a 30-step action horizon")
    return saved


def _finite_vector(value: Any, name: str) -> np.ndarray:
    try:
        array = np.asarray(value, dtype=np.float32)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be numeric") from error
    if array.shape != (ACTION_DIM,) or not np.isfinite(array).all():
        raise ValueError(f"{name} must be a finite {ACTION_DIM}-vector")
    return array


def _validate_open_baselines(value: Any) -> dict[str, float]:
    if not isinstance(value, dict) or set(value) != {"left", "right"}:
        raise ValueError("open_baselines requires exactly left and right")
    result = {side: float(value[side]) for side in ("left", "right")}
    if not np.isfinite(list(result.values())).all():
        raise ValueError("open_baselines must be finite")
    return result


def adapt_request(raw: dict[str, Any], images: dict[str, np.ndarray], spec) -> dict[str, Any]:
    """Build the policy payload from raw robot feedback and fixed session data."""
    saved = validate_model_contract(spec)
    if (
        raw.get("protocol_version") != saved["protocol_version"]
        or raw.get("calibration_version") != saved["contract_version"]
    ):
        raise ValueError("explicit matching protocol and calibration version required")
    if raw.get("experiment") != saved["experiment"]:
        raise ValueError("request experiment does not match checkpoint")
    if any(key in raw for key in ("state", "observation_state", "calibrated_state")):
        raise ValueError("send raw feedback and open baselines; pre-calibrated state is not accepted")
    joint = _finite_vector(raw.get("raw_joint_feedback"), "raw_joint_feedback")
    eef = None
    if saved["protocol_version"] == V3_PROTOCOL:
        eef = _finite_vector(raw.get("raw_eef_feedback"), "raw_eef_feedback")
    opened = _validate_open_baselines(raw.get("open_baselines"))
    instruction = raw.get("task_instruction")
    if not isinstance(instruction, str) or not instruction.strip():
        raise ValueError("task_instruction must not be empty")
    for key in ("request_id", "sample_monotonic_ns"):
        if isinstance(raw.get(key), bool) or not isinstance(raw.get(key), int) or raw[key] <= 0:
            raise ValueError(f"{key} must be a positive integer")
    if set(images) != set(CAMERA_NAMES):
        raise ValueError("three camera images required")
    state = calibrate_joint(joint, [opened["left"], opened["right"]])
    return {
        "state": state,
        "images": images,
        "prompt": instruction.strip(),
        "meta": {
            "request_id": raw["request_id"],
            "sample_monotonic_ns": raw["sample_monotonic_ns"],
            **({"raw_eef_feedback": eef} if eef is not None else {}),
        },
    }


def _contract_payload(saved, *, model_id: str, checkpoint_sha256: str | None, record_dir: Path):
    return {
        **saved,
        "protocol_version": saved["protocol_version"],
        "calibration_version": saved["contract_version"],
        "required_client_adapter_version": CLIENT_ADAPTER_VERSION,
        "robot": "ARX LIFT2s",
        "fps": FPS,
        "camera_names": list(CAMERA_NAMES),
        "state_dim": ACTION_DIM,
        "action_dim": ACTION_DIM,
        "action_horizon": ACTION_HORIZON,
        "action_dt": 1.0 / FPS,
        "joint_names": list(ARX_LIFT2S_JOINT_NAMES),
        "wire_action_field": "calibrated_action_chunk",
        "wire_action_is_robot_command": False,
        "model_id": model_id,
        "checkpoint_sha256": checkpoint_sha256,
        "record_dir": str(record_dir),
        "request_fields": [
            "protocol_version",
            "raw_joint_feedback",
            "open_baselines",
            "calibration_version",
            "experiment",
            "request_id",
            "sample_monotonic_ns",
            "task_instruction",
            *(["raw_eef_feedback"] if saved["protocol_version"] == V3_PROTOCOL else []),
        ],
    }


def build_calibrated_app(
    policy,
    *,
    model_id: str | None = None,
    checkpoint_sha256: str | None = None,
    record_dir: str | Path | None = None,
    allowed_client_ips: tuple[str, ...] = (),
) -> FastAPI:
    saved = validate_model_contract(policy.data_spec)
    if bool(getattr(policy, "rtc_enabled", False)):
        raise ValueError("ARX calibrated/feedback experiments require RTC disabled")
    model_id = model_id or policy.data_spec.finch_config_name
    record_root = Path(record_dir or "outputs/arx_calibrated_inference")
    model_record_dir = record_root / str(model_id).replace("/", "_")
    model_record_dir.mkdir(parents=True, exist_ok=True)
    native_perm = build_native_action_perm(action_slices(policy.data_spec))

    api_prefix = "/arx/v3" if saved["protocol_version"] == V3_PROTOCOL else "/arx/v4"
    app = FastAPI(title=f"ARX {saved['protocol_version']}")
    state_lock = threading.Lock()
    inference_lock = threading.Lock()
    record_lock = threading.Lock()
    record_state_lock = threading.Lock()
    record_error: str | None = None
    active: _Session | None = None

    def write_record(
        *,
        target: Path,
        request_json: str,
        response_json: str,
        joint: np.ndarray,
        eef: np.ndarray,
        open_baselines: np.ndarray,
        calibrated_state: np.ndarray,
        actions: np.ndarray,
        received: str,
        images: dict[str, np.ndarray],
    ) -> None:
        nonlocal record_error
        temporary = target.with_suffix(".tmp")
        try:
            with record_lock:
                with temporary.open("wb") as stream:
                    np.savez_compressed(
                        stream,
                        request_json=request_json,
                        response_json=response_json,
                        raw_joint_feedback=joint,
                        raw_eef_feedback=eef,
                        open_baselines=open_baselines,
                        calibrated_native_state=calibrated_state,
                        calibrated_action_chunk=actions,
                        request_received_utc=received,
                        **{f"image_{name}": image for name, image in images.items()},
                    )
                temporary.replace(target)
        except Exception as error:
            with record_state_lock:
                record_error = f"{type(error).__name__}: {error}"
            logger.exception("ARX calibrated background recording failed: %s", target)

    if allowed_client_ips:
        allowed = frozenset(allowed_client_ips)

        @app.middleware("http")
        async def restrict_client(request, call_next):
            if request.client is None or request.client.host not in allowed:
                return JSONResponse(status_code=403, content={"detail": "client IP not allowed"})
            return await call_next(request)

    contract_payload = _contract_payload(
        saved,
        model_id=model_id,
        checkpoint_sha256=checkpoint_sha256,
        record_dir=model_record_dir,
    )

    @app.get("/health")
    async def health():
        with record_state_lock:
            current_record_error = record_error
        return {
            "status": "ok",
            "ready": current_record_error is None,
            "route": policy.data_spec.finch_config_name,
            "model_id": model_id,
            "checkpoint_sha256": checkpoint_sha256,
            "protocol_version": saved["protocol_version"],
            "experiment": saved["experiment"],
            "required_client_adapter_version": CLIENT_ADAPTER_VERSION,
            "recording_mode": "background-serialized",
            "recording_error": current_record_error,
        }

    @app.get(f"{api_prefix}/policy-contract")
    async def policy_contract():
        return contract_payload

    @app.post(f"{api_prefix}/sessions")
    async def create_session(request: SessionRequest):
        nonlocal active
        if request.protocol_version != saved["protocol_version"]:
            raise HTTPException(status_code=409, detail="protocol version mismatch")
        if request.calibration_version != saved["contract_version"]:
            raise HTTPException(status_code=409, detail="calibration version mismatch")
        if request.client_adapter_version != CLIENT_ADAPTER_VERSION:
            raise HTTPException(status_code=409, detail="client adapter version mismatch")
        if request.experiment != saved["experiment"]:
            raise HTTPException(status_code=409, detail="experiment does not match checkpoint")
        instruction = request.task_instruction.strip()
        if not instruction:
            raise HTTPException(status_code=422, detail="task_instruction must not be empty")
        if not request.client_name.strip() or not request.robot_id.strip() or not request.calibration_id.strip():
            raise HTTPException(
                status_code=422,
                detail="client_name, robot_id and calibration_id must not be empty",
            )
        try:
            opened = _validate_open_baselines(request.open_baselines)
        except (TypeError, ValueError) as error:
            raise HTTPException(status_code=422, detail=str(error)) from error
        session = _Session(
            session_id=uuid.uuid4().hex,
            task_instruction=instruction,
            experiment=request.experiment,
            client_name=request.client_name.strip(),
            robot_id=request.robot_id.strip(),
            calibration_id=request.calibration_id.strip(),
            open_baselines=opened,
        )
        with state_lock:
            active = session
        logger.info(
            "ARX calibrated session created: session=%s client=%s robot=%s calibration=%s",
            session.session_id,
            session.client_name,
            session.robot_id,
            session.calibration_id,
        )
        return {
            "protocol_version": saved["protocol_version"],
            "session_id": session.session_id,
            "model_id": model_id,
            "experiment": session.experiment,
            "calibration_id": session.calibration_id,
        }

    @app.post(f"{api_prefix}/sessions/{{session_id}}/action-chunks")
    async def action_chunk(
        session_id: str,
        background_tasks: BackgroundTasks,
        metadata: str = Form(...),
        head: UploadFile = File(...),
        left_wrist: UploadFile = File(...),
        right_wrist: UploadFile = File(...),
    ):
        request_started = time.monotonic_ns()
        with record_state_lock:
            current_record_error = record_error
        if current_record_error is not None:
            raise HTTPException(
                status_code=503,
                detail=f"request recording is unavailable: {current_record_error}",
            )
        try:
            request = json.loads(metadata)
        except json.JSONDecodeError as error:
            raise HTTPException(status_code=422, detail="metadata is not valid JSON") from error
        if not isinstance(request, dict):
            raise HTTPException(status_code=422, detail="metadata must be a JSON object")
        if request.get("protocol_version") != saved["protocol_version"]:
            raise HTTPException(status_code=409, detail="protocol version mismatch")
        try:
            for key in ("request_id", "sample_monotonic_ns"):
                if type(request.get(key)) is not int or request[key] < 1:
                    raise ValueError(f"{key} must be a positive integer")
            request_id = request["request_id"]
            sample_monotonic_ns = request["sample_monotonic_ns"]
            joint = _finite_vector(request.get("raw_joint_feedback"), "raw_joint_feedback")
            eef = (
                _finite_vector(request.get("raw_eef_feedback"), "raw_eef_feedback")
                if saved["protocol_version"] == V3_PROTOCOL
                else np.asarray([], dtype=np.float32)
            )
        except (KeyError, TypeError, ValueError) as error:
            raise HTTPException(status_code=422, detail=str(error)) from error
        if request_id < 1 or sample_monotonic_ns < 1:
            raise HTTPException(
                status_code=422,
                detail="request_id and sample_monotonic_ns must be positive",
            )
        with state_lock:
            session = active
            if session is None or session.session_id != session_id:
                raise HTTPException(status_code=409, detail="inactive session")
            expected = session.last_request_id + 1
            if request_id != expected:
                raise HTTPException(
                    status_code=409,
                    detail=f"request_id {request_id} does not follow {session.last_request_id}",
                )

        images = {
            "head": await _read_jpeg(head, "head"),
            "left_wrist": await _read_jpeg(left_wrist, "left_wrist"),
            "right_wrist": await _read_jpeg(right_wrist, "right_wrist"),
        }
        raw = {
            "protocol_version": saved["protocol_version"],
            "calibration_version": saved["contract_version"],
            "experiment": session.experiment,
            "request_id": request_id,
            "sample_monotonic_ns": sample_monotonic_ns,
            "task_instruction": session.task_instruction,
            "raw_joint_feedback": joint,
            "open_baselines": session.open_baselines,
        }
        if saved["protocol_version"] == V3_PROTOCOL:
            raw["raw_eef_feedback"] = eef
        try:
            payload = adapt_request(raw, images, policy.data_spec)
        except (KeyError, TypeError, ValueError) as error:
            raise HTTPException(status_code=422, detail=str(error)) from error

        received = datetime.now(timezone.utc).isoformat()
        preprocess_ms = (time.monotonic_ns() - request_started) / 1e6
        started = time.monotonic_ns()
        with inference_lock:
            # Upload reads above may yield to another request/session. Recheck
            # before using the model, so concurrent duplicates cannot both run.
            with state_lock:
                if active is not session or request_id != session.last_request_id + 1:
                    raise HTTPException(status_code=409, detail="session or request order changed during upload")
            result = policy.infer(payload)
        inference_ms = (time.monotonic_ns() - started) / 1e6
        actions = np.asarray(result["actions"], dtype=np.float32)
        if actions.ndim != 2 or actions.shape[0] != ACTION_HORIZON:
            raise HTTPException(status_code=500, detail=f"policy returned invalid action shape {actions.shape}")
        actions = actions[:, native_perm]
        if actions.shape != (ACTION_HORIZON, ACTION_DIM) or not np.isfinite(actions).all():
            raise HTTPException(status_code=500, detail="policy returned invalid calibrated action chunk")
        with state_lock:
            if active is not session:
                raise HTTPException(status_code=409, detail="session changed during inference")
            session.last_request_id = request_id

        response = {
            "protocol_version": saved["protocol_version"],
            "calibration_version": saved["contract_version"],
            "required_client_adapter_version": CLIENT_ADAPTER_VERSION,
            "session_id": session_id,
            "request_id": request_id,
            "sample_monotonic_ns": sample_monotonic_ns,
            "request_received_utc": received,
            "model_id": model_id,
            "experiment": saved["experiment"],
            "control_mode": saved["control_mode"],
            "gripper_semantics": saved["gripper_action"],
            "component_source_offsets": saved["component_source_offsets"],
            "offset_unit": saved["offset_unit"],
            "action_dt": 1.0 / FPS,
            "calibrated_action_chunk": actions.tolist(),
            "wire_action_is_robot_command": False,
            "inference_ms": inference_ms,
            "preprocess_ms": preprocess_ms,
            "recording_mode": "background-serialized",
        }
        for key in ("source_fps", "temporal_stride"):
            if key in saved:
                response[key] = saved[key]
        model_record_dir.mkdir(parents=True, exist_ok=True)
        target = model_record_dir / f"request-{request_id}-session-{session_id}.npz"
        background_tasks.add_task(
            write_record,
            target=target,
            request_json=json.dumps(
                {
                    **request,
                    "open_baselines": session.open_baselines,
                    "task_instruction": session.task_instruction,
                    "experiment": session.experiment,
                    "calibration_id": session.calibration_id,
                },
                separators=(",", ":"),
            ),
            response_json=json.dumps(response, separators=(",", ":")),
            joint=joint,
            eef=eef,
            open_baselines=np.asarray(
                [session.open_baselines["left"], session.open_baselines["right"]],
                dtype=np.float32,
            ),
            calibrated_state=np.asarray(payload["state"], dtype=np.float32),
            actions=actions,
            received=received,
            images=images,
        )
        logger.info(
            "ARX calibrated action chunk: session=%s request=%d sample=%d inference_ms=%.1f shape=%s",
            session_id,
            request_id,
            sample_monotonic_ns,
            inference_ms,
            tuple(actions.shape),
        )
        return response

    return app


__all__ = [
    "ACTION_DIM",
    "ACTION_HORIZON",
    "CAMERA_NAMES",
    "CLIENT_ADAPTER_VERSION",
    "FPS",
    "SUPPORTED_EXPERIMENTS",
    "SessionRequest",
    "adapt_request",
    "build_calibrated_app",
    "validate_model_contract",
]
