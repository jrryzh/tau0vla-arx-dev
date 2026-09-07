#!/usr/bin/env python3
"""Benchmark a loaded calibrated-v3 server with safe synthetic observations."""
from __future__ import annotations

import argparse
from io import BytesIO
import json
import time

import numpy as np
from PIL import Image
import requests


PROTOCOL = "arx-calibrated-v3"
CALIBRATION = "arx-open-baseline-v1"
CLIENT = "arx-calibrated-client-v1"


def run(args) -> dict:
    base = args.server_url.rstrip("/")
    session = requests.Session()
    health = session.get(f"{base}/health", timeout=(1, 5)).json()
    contract = session.get(f"{base}/arx/v3/policy-contract", timeout=(1, 5)).json()
    if health.get("ready") is not True or health.get("protocol_version") != PROTOCOL:
        raise RuntimeError(f"server is not calibrated-v3 ready: {health}")
    if contract.get("experiment") != args.experiment:
        raise RuntimeError(f"checkpoint experiment mismatch: {contract.get('experiment')}")
    opened = {"left": args.left_open_baseline, "right": args.right_open_baseline}
    created = session.post(
        f"{base}/arx/v3/sessions",
        json={
            "protocol_version": PROTOCOL,
            "calibration_version": CALIBRATION,
            "client_adapter_version": CLIENT,
            "experiment": args.experiment,
            "task_instruction": args.task_instruction,
            "client_name": "calibrated-benchmark",
            "robot_id": "synthetic",
            "calibration_id": "synthetic-benchmark",
            "open_baselines": opened,
        },
        timeout=(1, 5),
    )
    created.raise_for_status()
    session_id = created.json()["session_id"]
    image = BytesIO()
    Image.fromarray(np.zeros((224, 224, 3), dtype=np.uint8)).save(image, format="JPEG")
    jpeg = image.getvalue()
    files = {name: (f"{name}.jpg", jpeg, "image/jpeg") for name in ("head", "left_wrist", "right_wrist")}
    joint = np.zeros(14, dtype=np.float32)
    joint[[6, 13]] = [opened["left"], opened["right"]]
    latencies = []
    inference = []
    total = args.warmup + args.requests
    for index in range(total):
        request_id = index + 1
        metadata = {
            "protocol_version": PROTOCOL,
            "request_id": request_id,
            "sample_monotonic_ns": time.monotonic_ns(),
            "raw_joint_feedback": joint.tolist(),
            "raw_eef_feedback": np.zeros(14, dtype=np.float32).tolist(),
        }
        started = time.monotonic()
        response = session.post(
            f"{base}/arx/v3/sessions/{session_id}/action-chunks",
            data={"metadata": json.dumps(metadata, separators=(",", ":"))},
            files=files,
            timeout=(1, args.timeout),
        )
        elapsed = (time.monotonic() - started) * 1000
        response.raise_for_status()
        payload = response.json()
        action = np.asarray(payload.get("calibrated_action_chunk"), dtype=np.float32)
        if action.shape != (30, 14) or not np.isfinite(action).all():
            raise RuntimeError(f"invalid calibrated action chunk: {action.shape}")
        if payload.get("session_id") != session_id or payload.get("request_id") != request_id:
            raise RuntimeError("session/request identity mismatch")
        if index >= args.warmup:
            latencies.append(elapsed)
            inference.append(float(payload["inference_ms"]))
        print(
            f"Benchmark {index+1}/{total}: request={request_id}, RTT={elapsed:.1f} ms, "
            f"inference={float(payload['inference_ms']):.1f} ms"
        )
    summary = {
        "model_id": health["model_id"],
        "experiment": args.experiment,
        "requests": args.requests,
        "rtt_mean_ms": float(np.mean(latencies)),
        "rtt_p99_ms": float(np.percentile(latencies, 99)),
        "inference_mean_ms": float(np.mean(inference)),
        "inference_p99_ms": float(np.percentile(inference, 99)),
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return summary


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server-url", default="http://127.0.0.1:8001")
    parser.add_argument("--experiment", choices=("joint-feedback", "joint-vr"), required=True)
    parser.add_argument("--task-instruction", required=True)
    parser.add_argument("--left-open-baseline", type=float, default=-3.34)
    parser.add_argument("--right-open-baseline", type=float, default=-3.34)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--requests", type=int, default=30)
    parser.add_argument("--timeout", type=float, default=5.0)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
