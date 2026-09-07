#!/usr/bin/env python3
"""Select a complete checkpoint belonging to this exact ARX training run."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

import yaml


def select_checkpoint(run_dir: Path, config_path: Path, world_size: int) -> Path | None:
    run_dir = run_dir.resolve()
    expected = yaml.safe_load(config_path.read_text())
    candidates = sorted(
        (p for p in run_dir.glob("checkpoint-*") if re.fullmatch(r"checkpoint-\d+", p.name)),
        key=lambda p: int(p.name.split("-")[-1]),
        reverse=True,
    )
    for checkpoint in candidates:
        # A symlink to another run must never turn into an implicit transfer run.
        if checkpoint.resolve().parent != run_dir:
            raise ValueError(f"checkpoint leaves this run directory: {checkpoint}")
        result = subprocess.run(
            [sys.executable, str(Path(__file__).with_name("validate_checkpoint.py")),
             str(checkpoint), "--world-size", str(world_size), "--deployment"],
            capture_output=True, text=True,
        )
        if result.returncode:
            print(f"Skipping incomplete {checkpoint.name}: {result.stderr.strip()}", file=sys.stderr)
            continue
        spec = json.loads((checkpoint / "run_spec.json").read_text())
        for section in ("model_args", "data_args", "training_args"):
            for key, value in expected[section].items():
                if key == "report_to" and value == "none":
                    value = []
                if key == "output_dir":
                    if Path(spec[section][key]).resolve() != run_dir:
                        raise ValueError(f"{checkpoint.name}: output directory does not match")
                elif spec.get(section, {}).get(key) != value:
                    raise ValueError(f"{checkpoint.name}: incompatible {section}.{key}")
        return checkpoint
    if candidates:
        raise ValueError("Existing checkpoints are all incomplete; refusing a silent restart from base")
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--world-size", type=int, default=16)
    args = parser.parse_args()
    try:
        checkpoint = select_checkpoint(args.run_dir, args.config, args.world_size)
    except (ValueError, OSError, KeyError) as exc:
        raise SystemExit(str(exc)) from exc
    if checkpoint:
        print(checkpoint)


if __name__ == "__main__":
    main()
