#!/usr/bin/env python3
"""Fail closed unless a qzcli dry-run matches the requested H200 topology."""

from __future__ import annotations

import argparse
import json
import sys


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--compute-group", required=True)
    parser.add_argument("--spec", required=True)
    parser.add_argument("--repo", required=True)
    parser.add_argument("--instances", type=int, required=True)
    parser.add_argument("--gpus-per-node", type=int, default=8)
    parser.add_argument("--shm-gi", type=int, default=1200)
    parser.add_argument("--world-size", type=int, required=True)
    parser.add_argument("--global-batch", type=int, required=True)
    parser.add_argument("--per-device-batch", type=int, required=True)
    parser.add_argument("--gradient-accumulation", type=int, required=True)
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
        "logic_compute_group_id": (
            payload.get("logic_compute_group_id"),
            args.compute_group,
        ),
        "gpu_count": (config.get("gpu_count"), args.gpus_per_node),
        "instance_count": (config.get("instance_count"), args.instances),
        "shm_gi": (config.get("shm_gi"), args.shm_gi),
        "gpu_type": (price.get("gpu_type"), "NVIDIA_H200_SXM_141G"),
        "quota_id": (price.get("quota_id"), args.spec),
    }
    wrong = {name: pair for name, pair in expected.items() if pair[0] != pair[1]}
    if wrong:
        raise SystemExit(f"qzcli dry-run payload mismatch: {wrong}")
    command = str(payload.get("command", ""))
    if args.repo not in command:
        raise SystemExit(
            "qzcli dry-run command does not reference the expected repository"
        )
    calculated_world_size = args.instances * args.gpus_per_node
    calculated_global_batch = (
        calculated_world_size * args.per_device_batch * args.gradient_accumulation
    )
    if args.world_size != calculated_world_size:
        raise SystemExit(
            f"world size mismatch: requested {args.world_size}, topology gives {calculated_world_size}"
        )
    if args.global_batch != calculated_global_batch:
        raise SystemExit(
            f"global batch mismatch: requested {args.global_batch}, topology gives {calculated_global_batch}"
        )
    command_contract = (
        f"EXPECTED_NNODES={args.instances}",
        f"EXPECTED_GPUS_PER_NODE={args.gpus_per_node}",
        f"REQUIRE_WORLD_SIZE={args.world_size}",
        f"REQUIRE_GLOBAL_BATCH={args.global_batch}",
        f"--per_device_train_batch_size {args.per_device_batch}",
        f"--gradient_accumulation_steps {args.gradient_accumulation}",
    )
    missing_contract = [value for value in command_contract if value not in command]
    if missing_contract:
        raise SystemExit(
            f"qzcli command is missing launch contract values: {missing_contract}"
        )

    print("[QZCLI] dry-run payload validated", file=sys.stderr)


if __name__ == "__main__":
    main()
