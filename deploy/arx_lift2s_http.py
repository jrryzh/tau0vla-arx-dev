"""Versioned ARX LIFT2s HTTP control contract for Tau0VLA serving."""

from __future__ import annotations

import json
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
FPS = 30
ACTION_DIM = 14
ACTION_HORIZON = 30
ACTION_SEMANTICS = "state_t_plus_1"
MAX_JPEG_BYTES = 8 * 1024 * 1024


class SessionRequest(BaseModel):
    protocol_version: str
    task_instruction: str
    client_name: str = "arx1"


@dataclass
class _Session:
    session_id: str
    task_instruction: str
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
    router = APIRouter(prefix="/api/v1/arx-lift2s")
    lock = threading.Lock()
    active: _Session | None = None

    @router.get("/policy-contract")
    async def policy_contract():
        return {
            "protocol_version": PROTOCOL_VERSION,
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

    @router.post("/sessions")
    async def create_session(request: SessionRequest):
        nonlocal active
        if request.protocol_version != PROTOCOL_VERSION:
            raise HTTPException(status_code=409, detail="protocol version mismatch")
        instruction = request.task_instruction.strip()
        if not instruction:
            raise HTTPException(status_code=422, detail="task_instruction must not be empty")
        session = _Session(session_id=uuid.uuid4().hex, task_instruction=instruction)
        with lock:
            active = session
        return {
            "session_id": session.session_id,
            "protocol_version": PROTOCOL_VERSION,
            "model_id": model_id,
        }

    @router.post("/sessions/{session_id}/action-chunks")
    async def action_chunk(
        session_id: str,
        metadata: str = Form(...),
        head: UploadFile = File(...),
        left_wrist: UploadFile = File(...),
        right_wrist: UploadFile = File(...),
    ):
        request = _parse_metadata(metadata)
        if request.get("protocol_version") != PROTOCOL_VERSION:
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
        with lock:
            session = active
            if session is None or session.session_id != session_id:
                raise HTTPException(status_code=409, detail="inactive session")
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
        actions = policy.infer(
            {
                "prompt": session.task_instruction,
                "images": images,
                "state": state,
                "meta": {
                    "session_id": session_id,
                    "request_id": request_id,
                    "sample_monotonic_ns": sample_monotonic_ns,
                },
            }
        )["actions"]
        inference_ms = (time.monotonic() - started) * 1000.0
        actions = np.asarray(native_action(actions), dtype=np.float32)
        if actions.shape != (ACTION_HORIZON, ACTION_DIM):
            raise HTTPException(status_code=500, detail=f"policy returned invalid action shape {actions.shape}")
        if not np.isfinite(actions).all():
            raise HTTPException(status_code=500, detail="policy returned NaN or Inf")
        with lock:
            if active is not session:
                raise HTTPException(status_code=409, detail="session changed during inference")
            session.last_request_id = request_id
        return {
            "protocol_version": PROTOCOL_VERSION,
            "session_id": session_id,
            "request_id": request_id,
            "sample_monotonic_ns": sample_monotonic_ns,
            "actions": actions.tolist(),
            "action_dt": 1.0 / FPS,
            "action_semantics": ACTION_SEMANTICS,
            "inference_ms": inference_ms,
            "model_id": model_id,
        }

    return router


__all__ = [
    "ACTION_DIM",
    "ACTION_HORIZON",
    "ACTION_SEMANTICS",
    "CAMERA_NAMES",
    "FPS",
    "PROTOCOL_VERSION",
    "SessionRequest",
    "build_router",
]
