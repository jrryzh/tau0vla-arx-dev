"""ARX v3 server-only calibrated protocol; no robot driver mapping is implied."""
from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from threading import Lock
import time
import uuid

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
import numpy as np

from tau0_vla.adapters.arx_lift2s.calibrated import (
    PROTOCOL, VERSION, POSE_INDICES, calibrate_joint, contract,
)
from tau0_vla.data import action_slices
from deploy.arx_lift2s_http import _read_jpeg


def validate_model_contract(spec):
    saved = getattr(spec, "deployment_contract", None)
    if not isinstance(saved, dict) or saved != contract(saved.get("experiment")):
        raise ValueError("checkpoint is missing the calibrated deployment contract")
    mode = saved["experiment"]
    expected_key = "arx_calibrated_eef_v1" if mode == "eef-vr" else "arx_calibrated_joint_v1"
    if spec.unified_registry_key != expected_key or bool(spec.unified_has_eef) != (mode == "eef-vr") or spec.action_semantics != saved["action_semantics"]:
        raise ValueError("checkpoint encoding and deployment contract disagree")
    return saved


def adapt_request(raw, images, spec):
    saved = validate_model_contract(spec)
    if raw.get("protocol_version") != PROTOCOL or raw.get("calibration_version") != VERSION:
        raise ValueError("explicit v3 protocol and calibration version required")
    if raw.get("experiment") != saved["experiment"]:
        raise ValueError("request experiment does not match checkpoint")
    if any(key in raw for key in ("state", "observation_state", "calibrated_state")):
        raise ValueError("send raw feedback and open baselines; pre-calibrated state is not accepted")
    joint = np.asarray(raw["raw_joint_feedback"], dtype=np.float64)
    eef = np.asarray(raw["raw_eef_feedback"], dtype=np.float64)
    if joint.shape != (14,) or eef.shape != (14,) or not np.isfinite(eef).all():
        raise ValueError("raw joint and EEF feedback must each be finite 14-vectors")
    opened = raw["open_baselines"]
    if set(opened) != {"left", "right"}:
        raise ValueError("open_baselines requires left and right")
    state = calibrate_joint(joint, [opened["left"], opened["right"]])
    if saved["control_mode"] == "eef":
        state = np.concatenate([state, eef[POSE_INDICES]]).astype(np.float32)
    instruction = raw["task_instruction"]
    if not isinstance(instruction, str) or not instruction.strip():
        raise ValueError("task_instruction must not be empty")
    for key in ("request_id", "sample_monotonic_ns"):
        if isinstance(raw.get(key), bool) or not isinstance(raw.get(key), int) or raw[key] <= 0:
            raise ValueError(f"{key} must be a positive integer")
    if set(images) != {"head", "left_wrist", "right_wrist"}:
        raise ValueError("three camera images required")
    return {"state": state, "images": images, "prompt": instruction.strip(), "meta": {
        "request_id": raw["request_id"], "sample_monotonic_ns": raw["sample_monotonic_ns"]}}


def infer_and_record(policy, raw, images, *, model_id, record_dir):
    payload = adapt_request(raw, images, policy.data_spec)
    saved = validate_model_contract(policy.data_spec)
    received = datetime.now(timezone.utc).isoformat()
    started = time.monotonic_ns()
    result = policy.infer(payload)
    actions = np.asarray(result["actions"], dtype=np.float32)
    slices = action_slices(policy.data_spec)
    width = sum(dim for _, _, dim in slices)
    if actions.shape != (30, width) or not np.isfinite(actions).all():
        raise ValueError("policy returned invalid action chunk")
    split = {name: actions[:, offset:offset+dim].tolist() for name, offset, dim in slices}
    response = {"protocol_version": PROTOCOL, "model_id": model_id, "request_id": raw["request_id"],
        "sample_monotonic_ns": raw["sample_monotonic_ns"], "request_received_utc": received,
        "experiment": saved["experiment"], "control_mode": saved["control_mode"], "gripper_semantics": saved["gripper_action"],
        "calibration_version": VERSION, "action_dt": 1/30, "pose_convention": saved["pose_convention"],
        "actions": split, "inference_ms": (time.monotonic_ns()-started)/1e6, "robot_client_adapted": False}
    directory = Path(record_dir)
    directory.mkdir(parents=True, exist_ok=True)
    identifier = uuid.uuid4().hex
    target = directory / f"request-{raw['request_id']}-{identifier}.npz"
    temporary = target.with_suffix(".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, request_json=json.dumps(raw), response_json=json.dumps(response),
            raw_joint_feedback=np.asarray(raw["raw_joint_feedback"]), raw_eef_feedback=np.asarray(raw["raw_eef_feedback"]),
            open_baselines=np.array([raw["open_baselines"][s] for s in ("left", "right")]),
            calibrated_native_state=payload["state"], actions=actions, **{f"image_{k}": v for k, v in images.items()})
    temporary.replace(target)
    return response


def build_calibrated_app(policy, *, model_id=None, checkpoint_sha256=None, record_dir=None, allowed_client_ips=()):
    saved = validate_model_contract(policy.data_spec)
    if getattr(policy, "rtc_enabled", False):
        raise ValueError("calibrated v1 experiment requires RTC disabled")
    model_id = model_id or policy.data_spec.finch_config_name
    record_dir = Path(record_dir or "outputs/arx_calibrated_inference") / str(model_id).replace("/", "_")
    app = FastAPI(title="ARX calibrated v3 — server protocol")
    lock = Lock()
    if allowed_client_ips:
        from fastapi import Request
        from fastapi.responses import JSONResponse

        @app.middleware("http")
        async def restrict_client(request: Request, call_next):
            if request.client is None or request.client.host not in allowed_client_ips:
                return JSONResponse(status_code=403, content={"detail": "client IP not allowed"})
            return await call_next(request)

    @app.get("/arx/v3/policy-contract")
    async def policy_contract():
        return {**saved, "model_id": model_id, "checkpoint_sha256": checkpoint_sha256,
            "request_fields": ["raw_joint_feedback", "raw_eef_feedback", "open_baselines", "calibration_version", "experiment", "request_id", "sample_monotonic_ns", "task_instruction"],
            "record_dir": str(record_dir)}

    @app.post("/arx/v3/action-chunks")
    async def action_chunk(metadata: str = Form(...), head: UploadFile = File(...), left_wrist: UploadFile = File(...), right_wrist: UploadFile = File(...)):
        try:
            raw = json.loads(metadata)
            if not isinstance(raw, dict):
                raise ValueError("metadata must be an object")
            images = {name: await _read_jpeg(upload, name) for name, upload in (("head", head), ("left_wrist", left_wrist), ("right_wrist", right_wrist))}
            adapt_request(raw, images, policy.data_spec)
        except (ValueError, TypeError, KeyError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        with lock:
            return infer_and_record(policy, raw, images, model_id=model_id, record_dir=record_dir)

    return app
