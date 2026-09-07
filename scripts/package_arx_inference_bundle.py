#!/usr/bin/env python3
"""Create a self-contained, inference-only Tau0VLA checkpoint bundle."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import subprocess


INFERENCE_FILES = (
    "chat_template.jinja",
    "config.json",
    "model.safetensors",
    "policy_manifest.json",
    "processor_config.json",
    "resolved_config_full.yaml",
    "run_spec.json",
    "tokenizer.json",
    "tokenizer_config.json",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_head(root: Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def package(args) -> Path:
    checkpoint = args.checkpoint.resolve()
    run_root = args.run_root.resolve()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing bundle: {output}")
    missing = [name for name in INFERENCE_FILES if not (checkpoint / name).is_file()]
    if missing:
        raise FileNotFoundError(f"checkpoint is missing inference files: {missing}")
    manifest = json.loads((checkpoint / "policy_manifest.json").read_text(encoding="utf-8"))
    routes = manifest.get("routes") or []
    if len(routes) != 1:
        raise ValueError(f"ARX deployment requires exactly one route, got {routes}")
    spec_root = run_root / "finch_data_spec"
    if not spec_root.is_dir():
        raise FileNotFoundError(f"missing run data spec: {spec_root}")
    spec_files = list(spec_root.glob("*/spec.json"))
    matched = []
    for path in spec_files:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("finch_config_name") == routes[0]:
            matched.append(path.parent)
    if len(matched) != 1:
        raise ValueError(f"route {routes[0]!r} resolved to {len(matched)} data specs")
    spec_dir = matched[0]
    for name in ("spec.json", "norm_stats.json", "components.json", "field_descriptions.json"):
        if not (spec_dir / name).is_file():
            raise FileNotFoundError(f"data spec is missing {name}: {spec_dir}")

    output.mkdir(parents=True)
    for name in INFERENCE_FILES:
        shutil.copy2(checkpoint / name, output / name)
    shutil.copytree(spec_dir, output / "finch_data_spec" / spec_dir.name)

    spec = json.loads((spec_dir / "spec.json").read_text(encoding="utf-8"))
    run_spec = json.loads((checkpoint / "run_spec.json").read_text(encoding="utf-8"))
    deployment_contract = spec.get("deployment_contract")
    if deployment_contract is not None and deployment_contract.get("experiment") not in (
        "joint-feedback",
        "joint-vr",
    ):
        raise ValueError("inference bundle supports calibrated joint-feedback/joint-vr only")
    deployment = {
        "schema_version": 2,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_checkpoint": str(checkpoint),
        "source_run": str(run_root),
        "source_checkpoint_name": checkpoint.name,
        "training_git_commit": run_spec.get("git_hash"),
        "training_git_dirty": bool(run_spec.get("git_dirty")),
        "service_compatibility_commit": _git_head(args.repo_root.resolve()),
        "model_id": args.model_id,
        "task_instruction": args.task_instruction,
        "route": spec.get("finch_config_name"),
        "robot_name": spec.get("robot_name"),
        "deployment_contract": deployment_contract,
        "model_sha256": _sha256(output / "model.safetensors"),
        "run_spec_sha256": _sha256(output / "run_spec.json"),
        "deployment_kind": "inference-only",
        "excluded_training_state": [
            "global_step*/",
            "optimizer.pt",
            "scheduler.pt",
            "rng_state_*.pth",
            "training_args.bin",
            "data_state_rank*.pt",
        ],
    }
    (output / "deployment_manifest.json").write_text(
        json.dumps(deployment, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    checksum_paths = sorted(
        path for path in output.rglob("*") if path.is_file() and path.name != "SHA256SUMS"
    )
    (output / "SHA256SUMS").write_text(
        "".join(f"{_sha256(path)}  {path.relative_to(output).as_posix()}\n" for path in checksum_paths),
        encoding="utf-8",
    )
    return output


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--task-instruction", required=True)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    return parser.parse_args()


if __name__ == "__main__":
    target = package(parse_args())
    print(target)
