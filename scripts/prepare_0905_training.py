#!/usr/bin/env python3
"""Prepare one approved 0905 split, retaining source provenance and validation."""
from __future__ import annotations

import argparse
import fcntl
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import time

REPO = Path(__file__).resolve().parents[1]
PROFILES = (
    "0905-datasets-first50", "0905-datasets-last50", "0905-datasets-all100",
    "0905-zyp-gyt-first50", "0905-zyp-gyt-all100",
)
TASK = "Pick up the tool and place it into the tray."


def source_snapshot(root: Path) -> list[tuple[str, int, int]] | None:
    if not root.is_dir():
        return None
    files = list(root.iterdir())
    if any(p.name.startswith(".") for p in files):
        return None
    episodes = [p for p in files if p.name.startswith("episode_") and p.suffix == ".hdf5"]
    if len(episodes) != 100:
        return None
    return sorted((p.name, p.stat().st_size, p.stat().st_mtime_ns) for p in episodes)


def select_numbers(profile: str, available: list[int]) -> list[int]:
    available = sorted(available)
    if "datasets" in profile:
        expected = [i for i in range(101) if i != 78]
        if available[:100] != expected:
            raise ValueError("datasets-0905 no longer matches the approved 100-episode selection")
        available = expected
    elif len(available) != 100:
        raise ValueError("zyp-gyt source must contain exactly 100 episodes")
    if profile.endswith("first50"):
        return available[:50]
    if profile.endswith("last50"):
        return available[50:]
    return available


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("profile", choices=PROFILES)
    args = parser.parse_args()
    profile = args.profile
    evidence = REPO / "outputs/0905_preparation" / profile
    evidence.mkdir(parents=True, exist_ok=True)
    with (evidence / "preparation.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        (evidence / "preparation.pid").write_text(str(os.getpid()) + "\n")
        if "datasets" in profile:
            source = REPO / "data/datasets-0905/pickplace_right_to_bowl"
        else:
            source = REPO / "data/pickplace_zyp_gyt_0905"
            previous = None
            while True:
                snapshot = source_snapshot(source)
                if snapshot is not None and snapshot == previous:
                    break
                previous = snapshot
                incoming = REPO / "data/.pickplace_zyp_gyt_0905.rsync-incoming"
                print(f"Waiting for final directory and 100 stable episodes; incoming={len(list(incoming.glob('episode_*.hdf5')))}", flush=True)
                time.sleep(60)
        numbers = select_numbers(profile, [int(p.stem.split("_")[-1]) for p in source.glob("episode_*.hdf5")])
        spec = importlib.util.spec_from_file_location("converter", REPO / "tools/convert_official_hdf5_to_lerobot.py")
        converter = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(converter)
        paths = [source / f"episode_{n}.hdf5" for n in numbers]
        manifest = converter.validate_selection(paths, 60, 30, TASK, "state_t_plus_1", requested_start=min(numbers), requested_end=max(numbers))
        (evidence / "source_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        dataset = REPO / "data/0905_training" / profile / "lerobot_v3_30fps_state_t_plus_1"
        dataset.parent.mkdir(parents=True, exist_ok=True)
        config_name = "arx_lift2s_" + profile.replace("-", "_")
        config = REPO / "configs" / config_name
        python = REPO / ".venv/bin/python"
        env = dict(os.environ, PYTHONPATH="src:.", OMP_NUM_THREADS="4")

        def run(stage: str, command: list) -> None:
            print(f"{profile}: {stage} started", flush=True)
            with (evidence / f"{stage}.log").open("w") as handle:
                subprocess.run([str(x) for x in command], cwd=REPO, env=env, stdout=handle, stderr=subprocess.STDOUT, check=True)
            print(f"{profile}: {stage} passed", flush=True)

        if not (dataset / "meta/arx.json").exists():
            if dataset.exists():
                raise RuntimeError(f"Incomplete conversion exists: {dataset}; inspect before retry")
            run("conversion", [python, "tools/convert_official_hdf5_to_lerobot.py", "--input", source,
                "--start", min(numbers), "--end", max(numbers), "--allow-missing-episodes",
                "--source-fps", 60, "--fps", 30, "--task", TASK, "--action-mode", "state_t_plus_1",
                "--output", dataset, "--repo-id", config_name])
        actual = json.loads((dataset / "meta/arx.json").read_text())
        for key in ("selected_source_episode_numbers", "total_frames", "episodes"):
            if actual[key] != manifest[key]:
                raise ValueError(f"Converted provenance changed: {key}")
        run("validation", [python, "scripts/validate_arx_lerobot_conversion.py", dataset, "--source", source,
            "--expected-episodes", len(numbers), "--expected-frames", manifest["total_frames"],
            "--expected-task", TASK, "--expected-missing", *manifest["missing_source_episode_numbers"]])
        for entry in manifest["episodes"]:
            if converter.sha256(source / entry["file"]) != entry["sha256"]:
                raise ValueError(f"Source changed while converting: {entry['file']}")
        run("statistics", [python, "scripts/norm_stats/compute_unified_ft_stats.py", "--body", "arx_lift2s_unified",
            "--repos", dataset, "--action-horizon", 30, "--positive-labels", "--negative-labels",
            "--partials-dir", evidence / "stats"])
        run("statistics_merge", [python, "scripts/norm_stats/merge_stats.py", "--partials", evidence / "stats",
            "--out", config / "norm_stats.json"])
        import numpy as np
        partial = np.load(next((evidence / "stats").glob("*.npz")))
        anchors = sum(max(0, e["output_frames"] - 29) for e in manifest["episodes"])
        assert int(partial["n_frames"]) == anchors
        batches = ((anchors + 15) // 16) // 8
        report = {"profile": profile, "source": str(source), "dataset": str(dataset), "episodes": len(numbers),
            "selected_numbers": numbers, "frames": manifest["total_frames"], "anchors": anchors,
            "expected_batches_per_rank": batches, "expected_vla_epoch": 10000 / batches,
            "max_steps": 10000, "world_size": 16, "global_batch": 128,
            "source_manifest": str(evidence / "source_manifest.json"), "validation": "ok"}
        temp = evidence / "ready.json.tmp"
        temp.write_text(json.dumps(report, indent=2) + "\n")
        temp.replace(evidence / "ready.json")
        print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
