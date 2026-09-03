#!/usr/bin/env python3
"""Check that a distributed checkpoint contains every resumable component."""

from __future__ import annotations

import argparse
from pathlib import Path


def _matches(root: Path, patterns: tuple[str, ...]) -> list[Path]:
    return sorted({path for pattern in patterns for path in root.rglob(pattern)})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--world-size", type=int, default=16)
    args = parser.parse_args()
    root = args.checkpoint.resolve()
    if not root.is_dir():
        raise SystemExit(f"checkpoint directory does not exist: {root}")

    required_files = ("trainer_state.json", "scheduler.pt", "run_spec.json")
    missing = [name for name in required_files if not (root / name).exists()]
    if missing:
        raise SystemExit(f"checkpoint missing required files: {missing}")

    model = _matches(root, ("model.safetensors", "pytorch_model.bin", "*model_states.pt"))
    optimizer = _matches(root, ("optimizer.pt", "*optim_states.pt"))
    data_states = _matches(root, ("data_state_rank*.pt",))
    if not model:
        raise SystemExit("checkpoint contains no model state")
    if not optimizer:
        raise SystemExit("checkpoint contains no optimizer state")
    if len(data_states) != args.world_size:
        raise SystemExit(f"expected {args.world_size} rank data states, found {len(data_states)}")

    print(f"checkpoint={root}")
    print(f"model_state={model[0].relative_to(root)}")
    print(f"optimizer_state={optimizer[0].relative_to(root)}")
    print("scheduler_state=scheduler.pt")
    print(f"data_states={len(data_states)}")
    print("run_spec=run_spec.json")
    print("checkpoint_validation=ok")


if __name__ == "__main__":
    main()
