#!/usr/bin/env python3
"""Validate H200 smoke hardware metadata and durable training evidence."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


TRAINABLE_GROUPS = {"vision_tower", "vision_projector", "llm", "vla_dit"}


def _load_last_json(path: Path) -> dict:
    text = path.read_text(errors="replace")
    decoder = json.JSONDecoder()
    objects = []
    for match in re.finditer(r"\{", text):
        try:
            obj, _ = decoder.raw_decode(text[match.start() :])
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            objects.append(obj)
    job_objects = [obj for obj in objects if "job_id" in obj and "status" in obj]
    if job_objects:
        return job_objects[-1]
    if not objects:
        raise ValueError(f"no JSON object found in {path}")
    return objects[-1]


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--status-json", type=Path, required=True)
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--compute-group", required=True)
    parser.add_argument("--instances", type=int, required=True)
    parser.add_argument("--gpus-per-node", type=int, required=True)
    parser.add_argument("--shm-gi", type=int, required=True)
    parser.add_argument("--world-size", type=int, required=True)
    parser.add_argument("--global-batch", type=int, required=True)
    parser.add_argument("--worker-log", action="append", type=Path, required=True)
    parser.add_argument("--training-log", action="append", type=Path, required=True)
    args = parser.parse_args()

    status = _load_last_json(args.status_json)
    _require(status.get("job_id") == args.job_id, "status job_id mismatch")
    _require(status.get("status") == "job_succeeded", "smoke job did not succeed")
    _require(
        status.get("logic_compute_group_id") == args.compute_group,
        "compute group mismatch",
    )
    configs = status.get("framework_config") or []
    _require(len(configs) == 1, "expected one framework configuration")
    config = configs[0]
    _require(config.get("instance_count") == args.instances, "instance_count mismatch")
    _require(config.get("gpu_count") == args.gpus_per_node, "GPU count mismatch")
    _require(config.get("shm_gi") == args.shm_gi, "shared-memory size mismatch")
    gpu_info = (config.get("instance_spec_price_info") or {}).get("gpu_info") or {}
    gpu_name = " ".join(
        str(gpu_info.get(key, ""))
        for key in ("gpu_product_simple", "gpu_type", "gpu_type_display")
    ).upper()
    _require("H200" in gpu_name, "job metadata does not prove H200 GPUs")
    command = str(status.get("command", ""))
    _require("python scripts/check_h200_worker.py" in command, "H200 preflight was not gated")
    _require(
        f"EXPECTED_NNODES={args.instances}" in command
        and f"EXPECTED_GPUS_PER_NODE={args.gpus_per_node}" in command,
        "H200 preflight topology mismatch",
    )

    _require(len(args.worker_log) == args.instances, "worker log count mismatch")
    peaks = []
    worker_texts = []
    for path in args.worker_log:
        text = path.read_text(errors="replace")
        worker_texts.append(text)
        values = [int(value) for value in re.findall(r"\[H200_PEAK_MEMORY\] peak_mib=(\d+)", text)]
        _require(bool(values), f"missing peak-memory evidence in {path}")
        peaks.append(max(values))

    _require(len(args.training_log) >= args.instances, "durable training log count mismatch")
    training_text = "\n".join(path.read_text(errors="replace") for path in args.training_log)
    evidence = training_text + "\n" + "\n".join(worker_texts)
    _require(
        f"Distributed contract verified: world_size={args.world_size}" in evidence
        and f"global_batch={args.global_batch}" in evidence,
        "distributed/global-batch contract is missing",
    )
    groups = set(re.findall(r"Trainable group verified: ([a-z_]+)=", evidence))
    _require(TRAINABLE_GROUPS <= groups, "not all four model groups were trainable")
    _require("'loss':" in evidence, "loss metric is missing")
    _require("'grad_norm':" in evidence, "gradient norm is missing")
    _require("'train_runtime':" in evidence, "completed training summary is missing")
    _require("'global_step': 20" in evidence, "smoke did not complete step 20")
    failure = re.search(
        r"out of memory|\bnan\b|nccl[^\n]*(?:timeout|error)|traceback",
        evidence,
        flags=re.IGNORECASE,
    )
    _require(failure is None, f"failure signature found: {failure.group(0) if failure else ''}")

    print(
        "smoke_validation=ok "
        f"job_id={args.job_id} instances={args.instances} "
        f"world_size={args.world_size} global_batch={args.global_batch} "
        f"peak_mib={max(peaks)}"
    )


if __name__ == "__main__":
    main()
