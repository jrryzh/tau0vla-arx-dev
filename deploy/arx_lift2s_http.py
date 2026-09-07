"""Versioned ARX LIFT2s HTTP control contract for Tau0VLA serving."""

from __future__ import annotations

import json
import logging
import threading
import time
import uuid
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from pydantic import BaseModel

from tau0_vla.adapters.arx_lift2s.deploy_io import CAMERA_NAMES, decode_jpeg
from tau0_vla.adapters.arx_lift2s.layout import ARX_LIFT2S_JOINT_NAMES


PROTOCOL_VERSION = "arx_lift2s_http_v1"
RTC_PROTOCOL_VERSION = "arx_lift2s_http_v2"
FPS = 30
ACTION_DIM = 14
ACTION_HORIZON = 30
ACTION_SEMANTICS = "state_t_plus_1"
MAX_JPEG_BYTES = 8 * 1024 * 1024
logger = logging.getLogger(__name__)


class SessionRequest(BaseModel):
    protocol_version: str
    task_instruction: str
    client_name: str = "arx1"


@dataclass
class _Session:
    session_id: str
    task_instruction: str
    protocol_version: str
    last_request_id: int = 0


def _validate_data_spec(data_spec) -> None:
    expected = {
        "robot_name": "arx_lift2s_unified",
        "unified_registry_key": "arx_lift2s_14",
        "action_chunk_size": ACTION_HORIZON,
        "action_semantics": ACTION_SEMANTICS,
    }
    for key, value in expected.items():
        actual = getattr(data_spec, key, None)
        if actual != value:
            raise ValueError(f"ARX HTTP contract requires {key}={value!r}, got {actual!r}")
    if tuple(getattr(data_spec, "cam_keys", ())) != CAMERA_NAMES:
        raise ValueError(f"ARX HTTP contract requires cameras {CAMERA_NAMES!r}")
    if bool(getattr(data_spec, "unified_has_eef", True)):
        raise ValueError("ARX HTTP contract only supports joint-control checkpoints")


def _parse_metadata(raw: str) -> dict[str, Any]:
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as error:
        raise HTTPException(status_code=422, detail="metadata is not valid JSON") from error
    if not isinstance(value, dict):
        raise HTTPException(status_code=422, detail="metadata must be a JSON object")
    return value


async def _read_jpeg(upload: UploadFile, camera: str) -> np.ndarray:
    if upload.content_type not in (None, "image/jpeg"):
        raise HTTPException(status_code=415, detail=f"{camera} must use image/jpeg")
    data = await upload.read(MAX_JPEG_BYTES + 1)
    if not data or len(data) > MAX_JPEG_BYTES:
        raise HTTPException(status_code=413, detail=f"{camera} JPEG is empty or too large")
    try:
        return decode_jpeg(data)
    except (OSError, ValueError) as error:
        raise HTTPException(status_code=422, detail=f"invalid {camera} JPEG: {error}") from error


def build_router(
    *,
    policy,
    native_action: Callable[[np.ndarray], np.ndarray],
    model_id: str,
    checkpoint_sha256: str | None,
) -> APIRouter:
    """Build the ARX-only API while sharing the already-loaded policy."""
    _validate_data_spec(policy.data_spec)
    router = APIRouter()
    v1 = APIRouter(prefix="/api/v1/arx-lift2s")
    v2 = APIRouter(prefix="/api/v2/arx-lift2s")
    lock = threading.Lock()
    active: _Session | None = None

    rtc_enabled = bool(getattr(policy, "rtc_enabled", False))
    rtc_max_delay = int(getattr(policy, "rtc_max_delay", 0) or 0)

    def _contract(protocol_version: str) -> dict[str, Any]:
        result = {
            "protocol_version": protocol_version,
            "robot": "ARX LIFT2s",
            "fps": FPS,
            "camera_names": list(CAMERA_NAMES),
            "state_dim": ACTION_DIM,
            "action_dim": ACTION_DIM,
            "action_horizon": ACTION_HORIZON,
            "action_dt": 1.0 / FPS,
            "action_semantics": ACTION_SEMANTICS,
            "joint_names": list(ARX_LIFT2S_JOINT_NAMES),
            "model_id": model_id,
            "checkpoint_sha256": checkpoint_sha256,
        }
        if protocol_version == RTC_PROTOCOL_VERSION:
            result.update(
                rtc_enabled=rtc_enabled,
                rtc_max_delay=rtc_max_delay,
                rtc_delay_unit="action_steps",
            )
        return result

    @v1.get("/policy-contract")
    async def policy_contract():
        return _contract(PROTOCOL_VERSION)

    @v2.get("/policy-contract")
    async def rtc_policy_contract():
        return _contract(RTC_PROTOCOL_VERSION)

    async def _create_session(request: SessionRequest, protocol_version: str):
        nonlocal active
        if request.protocol_version != protocol_version:
            raise HTTPException(status_code=409, detail="protocol version mismatch")
        instruction = request.task_instruction.strip()
        if not instruction:
            raise HTTPException(status_code=422, detail="task_instruction must not be empty")
        session = _Session(
            session_id=uuid.uuid4().hex,
            task_instruction=instruction,
            protocol_version=protocol_version,
        )
        with lock:
            active = session
        logger.info("ARX session created: session=%s client=%s", session.session_id, request.client_name)
        return {
            "session_id": session.session_id,
            "protocol_version": protocol_version,
            "model_id": model_id,
        }

    @v1.post("/sessions")
    async def create_session(request: SessionRequest):
        return await _create_session(request, PROTOCOL_VERSION)

    @v2.post("/sessions")
    async def create_rtc_session(request: SessionRequest):
        return await _create_session(request, RTC_PROTOCOL_VERSION)

    async def _action_chunk(
        session_id: str,
        metadata: str,
        head: UploadFile,
        left_wrist: UploadFile,
        right_wrist: UploadFile,
        protocol_version: str,
    ):
        request = _parse_metadata(metadata)
        if request.get("protocol_version") != protocol_version:
            raise HTTPException(status_code=409, detail="protocol version mismatch")
        try:
            request_id = int(request["request_id"])
            sample_monotonic_ns = int(request["sample_monotonic_ns"])
            state = np.asarray(request["observation_state"], dtype=np.float32)
        except (KeyError, TypeError, ValueError) as error:
            raise HTTPException(status_code=422, detail="invalid request metadata") from error
        if request_id < 1 or sample_monotonic_ns < 1:
            raise HTTPException(status_code=422, detail="request_id and sample_monotonic_ns must be positive")
        if state.shape != (ACTION_DIM,) or not np.isfinite(state).all():
            raise HTTPException(status_code=422, detail="observation_state must be a finite 14-vector")

        rtc_delay = 0
        action_prefix = np.empty((0, ACTION_DIM), dtype=np.float32)
        if protocol_version == RTC_PROTOCOL_VERSION:
            if not rtc_enabled:
                raise HTTPException(status_code=409, detail="checkpoint does not support training-time RTC")
            raw_delay = request.get("rtc_delay")
            if isinstance(raw_delay, bool) or not isinstance(raw_delay, int):
                raise HTTPException(status_code=422, detail="rtc_delay must be an integer action-step count")
            rtc_delay = int(raw_delay)
            if rtc_delay < 0:
                raise HTTPException(status_code=422, detail="rtc_delay must be non-negative")
            if rtc_delay > rtc_max_delay:
                raise HTTPException(
                    status_code=422,
                    detail=f"rtc_delay exceeds checkpoint rtc_max_delay={rtc_max_delay}",
                )
            try:
                action_prefix = np.asarray(request.get("action_prefix"), dtype=np.float32)
            except (TypeError, ValueError) as error:
                raise HTTPException(status_code=422, detail="action_prefix must be numeric") from error
            if rtc_delay == 0 and action_prefix.size == 0:
                action_prefix = action_prefix.reshape(0, ACTION_DIM)
            if action_prefix.shape != (rtc_delay, ACTION_DIM):
                raise HTTPException(
                    status_code=422,
                    detail=f"action_prefix must have shape [{rtc_delay}, {ACTION_DIM}]",
                )
            if not np.isfinite(action_prefix).all():
                raise HTTPException(status_code=422, detail="action_prefix must contain only finite values")
        with lock:
            session = active
            if session is None or session.session_id != session_id:
                raise HTTPException(status_code=409, detail="inactive session")
            if session.protocol_version != protocol_version:
                raise HTTPException(status_code=409, detail="session protocol version mismatch")
            expected_request_id = session.last_request_id + 1
            if request_id != expected_request_id:
                raise HTTPException(
                    status_code=409,
                    detail=f"request_id {request_id} does not follow {session.last_request_id}",
                )

        images = {
            "head": await _read_jpeg(head, "head"),
            "left_wrist": await _read_jpeg(left_wrist, "left_wrist"),
            "right_wrist": await _read_jpeg(right_wrist, "right_wrist"),
        }
        started = time.monotonic()
        policy_result = policy.infer(
            {
                "prompt": session.task_instruction,
                "images": images,
                "state": state,
                "meta": {
                    "session_id": session_id,
                    "request_id": request_id,
                    "sample_monotonic_ns": sample_monotonic_ns,
                    **(
                        {"rtc_delay": rtc_delay, "action_prefix": action_prefix}
                        if protocol_version == RTC_PROTOCOL_VERSION
                        else {}
                    ),
                },
            }
        )
        actions = policy_result["actions"]
        inference_ms = (time.monotonic() - started) * 1000.0
        actions = np.asarray(native_action(actions), dtype=np.float32)
        if actions.shape != (ACTION_HORIZON, ACTION_DIM):
            raise HTTPException(status_code=500, detail=f"policy returned invalid action shape {actions.shape}")
        if not np.isfinite(actions).all():
            raise HTTPException(status_code=500, detail="policy returned NaN or Inf")
        if protocol_version == RTC_PROTOCOL_VERSION and rtc_delay:
            # Do this after the canonical→wire permutation.  It guarantees the
            # response prefix is byte-for-byte the finite float32 command prefix
            # supplied by the client, independent of normalize/decode roundoff.
            actions[:rtc_delay] = action_prefix
        with lock:
            if active is not session:
                raise HTTPException(status_code=409, detail="session changed during inference")
            session.last_request_id = request_id
        logger.info(
            "ARX action chunk: session=%s request=%d sample=%d inference_ms=%.1f "
            "rtc_delay=%d prefix_error=%.6g shape=%s",
            session_id,
            request_id,
            sample_monotonic_ns,
            inference_ms,
            rtc_delay,
            float(policy_result.get("rtc_prefix_max_abs_error", 0.0)),
            tuple(actions.shape),
        )
        response = {
            "protocol_version": protocol_version,
            "session_id": session_id,
            "request_id": request_id,
            "sample_monotonic_ns": sample_monotonic_ns,
            "actions": actions.tolist(),
            "action_dt": 1.0 / FPS,
            "action_semantics": ACTION_SEMANTICS,
            "inference_ms": inference_ms,
            "model_id": model_id,
        }
        if protocol_version == RTC_PROTOCOL_VERSION:
            response.update(
                rtc_delay=rtc_delay,
                rtc_prefix_max_abs_error=float(policy_result.get("rtc_prefix_max_abs_error", 0.0)),
            )
        return response

    @v1.post("/sessions/{session_id}/action-chunks")
    async def action_chunk(
        session_id: str,
        metadata: str = Form(...),
        head: UploadFile = File(...),
        left_wrist: UploadFile = File(...),
        right_wrist: UploadFile = File(...),
    ):
        return await _action_chunk(
            session_id, metadata, head, left_wrist, right_wrist, PROTOCOL_VERSION
        )

    @v2.post("/sessions/{session_id}/action-chunks")
    async def rtc_action_chunk(
        session_id: str,
        metadata: str = Form(...),
        head: UploadFile = File(...),
        left_wrist: UploadFile = File(...),
        right_wrist: UploadFile = File(...),
    ):
        return await _action_chunk(
            session_id, metadata, head, left_wrist, right_wrist, RTC_PROTOCOL_VERSION
        )

    router.include_router(v1)
    router.include_router(v2)
    return router


__all__ = [
    "ACTION_DIM",
    "ACTION_HORIZON",
    "ACTION_SEMANTICS",
    "CAMERA_NAMES",
    "FPS",
    "PROTOCOL_VERSION",
    "RTC_PROTOCOL_VERSION",
    "SessionRequest",
    "build_router",
]
