#!/usr/bin/env python3
"""Fail closed unless a qzcli dry-run is the intended 2x8 H200 launch."""

from __future__ import annotations

import argparse
import json
import sys


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--compute-group", required=True)
    parser.add_argument("--spec", required=True)
    parser.add_argument("--repo", required=True)
    args = parser.parse_args()

    text = sys.stdin.read()
    start = text.find("{")
    if start < 0:
        raise SystemExit("qzcli dry-run did not contain a JSON payload")
    payload = json.loads(text[start:])
    configs = payload.get("framework_config") or []
    if len(configs) != 1:
        raise SystemExit("qzcli payload must contain exactly one framework config")
    config = configs[0]
    price = config.get("resource_spec_price") or {}

    expected = {
        "framework": (payload.get("framework"), "pytorch"),
        "logic_compute_group_id": (payload.get("logic_compute_group_id"), args.compute_group),
        "gpu_count": (config.get("gpu_count"), 8),
        "instance_count": (config.get("instance_count"), 2),
        "shm_gi": (config.get("shm_gi"), 1200),
        "gpu_type": (price.get("gpu_type"), "NVIDIA_H200_SXM_141G"),
        "quota_id": (price.get("quota_id"), args.spec),
    }
    wrong = {name: pair for name, pair in expected.items() if pair[0] != pair[1]}
    if wrong:
        raise SystemExit(f"qzcli dry-run payload mismatch: {wrong}")
    if args.repo not in str(payload.get("command", "")):
        raise SystemExit("qzcli dry-run command does not reference the expected repository")

    print("[QZCLI] dry-run payload validated", file=sys.stderr)


if __name__ == "__main__":
    main()
