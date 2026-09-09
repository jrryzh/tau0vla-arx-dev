#!/usr/bin/env python3
"""No-ROS, no-motion synthetic HTTP contract/latency probe for feedback-v4.

Creates a session (invalidating an existing session). Run only while rollout is
stopped. Synthetic images and joint values test serving, not physical performance.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from io import BytesIO
import json
from pathlib import Path
import sys
import time

import numpy as np
from PIL import Image
import requests


PROTOCOL = "arx-feedback-v4"
CALIBRATION = "arx-feedback-open-v1"
CLIENT = "arx-calibrated-client-v1"
CAMERAS = ("head", "left_wrist", "right_wrist")
OFFSETS = {"state": 0, "arm_action": 1, "gripper_action": 1}


def check_fields(payload, expected, label):
    for key, value in expected.items():
        if payload.get(key) != value:
            raise RuntimeError(f"{label}: {key}={payload.get(key)!r}, expected {value!r}")


def validate_contract(health, contract, args):
    identity = {"model_id": args.model_id, "checkpoint_sha256": args.checkpoint_sha256}
    check_fields(health, {**identity, "ready": True, "route": args.route,
                         "protocol_version": PROTOCOL, "experiment": "joint-feedback"}, "health")
    check_fields(contract, {
        **identity,
        "protocol_version": PROTOCOL,
        "contract_version": CALIBRATION,
        "calibration_version": CALIBRATION,
        "required_client_adapter_version": CLIENT,
        "experiment": "joint-feedback",
        "control_mode": "joint",
        "action_semantics": "feedback_open_joint_feedback",
        "gripper_action": "calibrated_feedback_position",
        "component_source_offsets": OFFSETS,
        "offset_unit": "uploaded_30fps_frame",
        "camera_names": list(CAMERAS),
        "fps": 30,
        "state_dim": 14,
        "action_dim": 14,
        "action_horizon": 30,
        "wire_action_field": "calibrated_action_chunk",
        "wire_action_is_robot_command": False,
    }, "contract")
    fields = contract.get("request_fields", [])
    if "raw_joint_feedback" not in fields or "raw_eef_feedback" in fields:
        raise RuntimeError("v4 must require joint feedback without EEF feedback")


def synthetic_images(width, height):
    """Deterministic, distinct per-camera patterns; no hardware is accessed."""
    y, x = np.indices((height, width))
    files = {}
    for index, name in enumerate(CAMERAS):
        pixels = np.stack(((x + 47 * index) % 256, (y + 79 * index) % 256,
                           ((x // 16 + y // 16) * 31 + 101 * index) % 256), axis=-1).astype(np.uint8)
        stream = BytesIO()
        Image.fromarray(pixels).save(stream, format="JPEG", quality=90)
        files[name] = (f"synthetic-{name}.jpg", stream.getvalue(), "image/jpeg")
    return files


def json_response(response):
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, dict):
        raise RuntimeError("HTTP response must be a JSON object")
    return payload


def validate_action(payload, metadata, session_id, model_id):
    check_fields(payload, {
        "protocol_version": PROTOCOL,
        "calibration_version": CALIBRATION,
        "required_client_adapter_version": CLIENT,
        "session_id": session_id,
        "request_id": metadata["request_id"],
        "sample_monotonic_ns": metadata["sample_monotonic_ns"],
        "model_id": model_id,
        "experiment": "joint-feedback",
        "gripper_semantics": "calibrated_feedback_position",
        "component_source_offsets": OFFSETS,
        "offset_unit": "uploaded_30fps_frame",
        "wire_action_is_robot_command": False,
        "recording_mode": "background-serialized",
    }, "action")
    actions = np.asarray(payload.get("calibrated_action_chunk"), dtype=np.float32)
    if actions.shape != (30, 14) or not np.isfinite(actions).all():
        raise RuntimeError(f"invalid action chunk: {actions.shape}; require finite 30x14")


def stats(values):
    return {"mean": float(np.mean(values)), "p50": float(np.percentile(values, 50)),
            "p95": float(np.percentile(values, 95)), "p99": float(np.percentile(values, 99)),
            "max": float(np.max(values))}


def verify_recordings(record_dir, session_id, metadata_rows, timeout=30.0):
    paths = [Path(record_dir) / f"request-{row['request_id']}-session-{session_id}.npz" for row in metadata_rows]
    deadline = time.monotonic() + timeout
    # Recording completes after the response; wait for the atomic .tmp -> .npz rename.
    while not all(path.is_file() for path in paths):
        if time.monotonic() >= deadline:
            raise RuntimeError(f"recordings missing after {timeout}s: {[str(p) for p in paths if not p.is_file()]}")
        time.sleep(0.1)
    for path, metadata in zip(paths, metadata_rows):
        with np.load(path, allow_pickle=False) as record:
            request = json.loads(str(record["request_json"]))
            response = json.loads(str(record["response_json"]))
            check_fields(request, metadata, str(path))
            check_fields(response, {"session_id": session_id, "request_id": metadata["request_id"],
                                   "sample_monotonic_ns": metadata["sample_monotonic_ns"]}, str(path))
            if record["raw_eef_feedback"].size != 0:
                raise RuntimeError(f"v4 synthetic request unexpectedly recorded EEF: {path}")
            if record["calibrated_action_chunk"].shape != (30, 14):
                raise RuntimeError(f"recorded action shape mismatch: {path}")
            np.testing.assert_allclose(record["raw_joint_feedback"], metadata["raw_joint_feedback"])
            for name in CAMERAS:
                if record[f"image_{name}"].ndim != 3:
                    raise RuntimeError(f"invalid recorded image {name}: {path}")
    return [str(path) for path in paths]


def run(args):
    if args.requests < 1 or args.warmup < 0 or args.width < 1 or args.height < 1:
        raise ValueError("requests and image dimensions must be positive; warmup must be nonnegative")
    base = args.server_url.rstrip("/")
    timeout = (5, args.timeout)
    with requests.Session() as session:
        session.trust_env = False  # Direct Ethernet must not inherit HTTP(S)_PROXY.
        health = json_response(session.get(f"{base}/health", timeout=timeout))
        contract = json_response(session.get(f"{base}/arx/v4/policy-contract", timeout=timeout))
        validate_contract(health, contract, args)
        opened = {"left": args.left_open_baseline, "right": args.right_open_baseline}
        joint = np.asarray([
            -.0002, .9447, .8597, -.5755, .0006, -.0013, opened["left"],
            -.0002, .9466, .8604, -.5724, -.0002, -.0006, opened["right"],
        ], dtype=np.float32)
        if not np.isfinite(joint).all():
            raise ValueError("synthetic open baselines must be finite")
        created = json_response(session.post(f"{base}/arx/v4/sessions", json={
            "protocol_version": PROTOCOL, "calibration_version": CALIBRATION,
            "client_adapter_version": CLIENT, "experiment": "joint-feedback",
            "task_instruction": args.task_instruction, "client_name": "synthetic-no-motion-probe",
            "robot_id": "synthetic-no-hardware", "calibration_id": "synthetic-not-robot-calibration",
            "open_baselines": opened,
        }, timeout=timeout))
        check_fields(created, {"protocol_version": PROTOCOL, "model_id": args.model_id,
                               "experiment": "joint-feedback"}, "session")
        session_id = created["session_id"]
        endpoint = f"{base}/arx/v4/sessions/{session_id}/action-chunks"
        files = synthetic_images(args.width, args.height)
        rows, metadata_rows = [], []
        for index in range(args.warmup + args.requests):
            metadata = {"protocol_version": PROTOCOL, "request_id": index + 1,
                        "sample_monotonic_ns": time.monotonic_ns(), "raw_joint_feedback": joint.tolist()}
            started = time.perf_counter()
            response = session.post(endpoint, data={"metadata": json.dumps(metadata)}, files=files, timeout=timeout)
            elapsed = (time.perf_counter() - started) * 1000
            payload = json_response(response)
            validate_action(payload, metadata, session_id, args.model_id)
            metadata_rows.append(metadata)
            rows.append({"request_id": index + 1, "warmup": index < args.warmup,
                         "sample_monotonic_ns": metadata["sample_monotonic_ns"], "rtt_ms": elapsed,
                         "inference_ms": float(payload["inference_ms"]),
                         "preprocess_ms": float(payload["preprocess_ms"])})
            print(f"synthetic {index + 1}/{args.warmup + args.requests}: RTT={elapsed:.1f} ms", file=sys.stderr)
        rejection_status = {}
        for name, request_id in (("duplicate", len(rows)), ("skipped", len(rows) + 2)):
            invalid = {**metadata_rows[-1], "request_id": request_id}
            rejected = session.post(endpoint, data={"metadata": json.dumps(invalid)}, files=files, timeout=timeout)
            if rejected.status_code != 409:
                raise RuntimeError(f"{name} request was not rejected with 409: {rejected.status_code}")
            rejection_status[name] = rejected.status_code
        end_health = json_response(session.get(f"{base}/health", timeout=timeout))
        validate_contract(end_health, contract, args)
    timed = [row for row in rows if not row["warmup"]]
    record_dir = contract["record_dir"]
    record_paths = [str(Path(record_dir) / f"request-{row['request_id']}-session-{session_id}.npz") for row in rows]
    if args.verify_records:
        record_paths = verify_recordings(record_dir, session_id, metadata_rows)
    return {
        "passed": True, "test_kind": "synthetic-no-motion-http", "physical_performance_validated": False,
        "created_at": datetime.now(timezone.utc).isoformat(), "server_url": base,
        "model_id": args.model_id, "checkpoint_sha256": args.checkpoint_sha256, "route": args.route,
        "protocol_version": PROTOCOL, "session_id": session_id, "task_instruction": args.task_instruction,
        "synthetic_open_baselines": opened, "image_size": [args.width, args.height],
        "warmup_requests": args.warmup, "timed_requests": args.requests,
        "rtt_ms": stats([row["rtt_ms"] for row in timed]),
        "inference_ms": stats([row["inference_ms"] for row in timed]),
        "timed_rtt_over_500ms": sum(row["rtt_ms"] > 500 for row in timed),
        "order_rejections": rejection_status,
        "recordings_verified_locally": args.verify_records, "record_paths": record_paths,
        "requests": rows,
    }


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server-url", default="http://192.168.50.2:8001")
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--route", default="arx-lift2s-0908-all-joint-feedback-ft")
    parser.add_argument("--task-instruction", required=True)
    parser.add_argument("--left-open-baseline", type=float, default=-3.39)
    parser.add_argument("--right-open-baseline", type=float, default=-3.39)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--requests", type=int, default=30)
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--verify-records", action="store_true", help="verify NPZ files locally on the model server")
    parser.add_argument("--report", type=Path)
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    try:
        report = run(arguments)
    except Exception as error:
        report = {"passed": False, "test_kind": "synthetic-no-motion-http",
                  "physical_performance_validated": False, "error": f"{type(error).__name__}: {error}"}
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if arguments.report:
        arguments.report.parent.mkdir(parents=True, exist_ok=True)
        arguments.report.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    sys.exit(0 if report["passed"] else 1)
