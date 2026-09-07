#!/usr/bin/env python3
"""Check the actual training anchor stream and distributed batch lengths."""
import argparse
import importlib
import json
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO))
from prepare_0905_training import PROFILES, TASK


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("profile", choices=PROFILES)
    args = parser.parse_args()
    evidence = REPO / "outputs/0905_preparation" / args.profile
    ready = json.loads((evidence / "ready.json").read_text())
    name = "arx_lift2s_" + args.profile.replace("-", "_")
    importlib.import_module(f"configs.{name}.data")
    import numpy as np
    from torch.utils.data import BatchSampler, DistributedSampler
    from tau0_vla.data import FinchDataLoader
    loader = FinchDataLoader.from_config_name(name + "_ft")
    dataset = loader.dataset
    assert len(dataset) == ready["anchors"]
    batches = [len(BatchSampler(DistributedSampler(dataset, num_replicas=16, rank=rank, seed=42), batch_size=8, drop_last=True)) for rank in range(16)]
    assert batches == [ready["expected_batches_per_rank"]] * 16
    manifest = json.loads((evidence / "source_manifest.json").read_text())
    start = 0
    checked = 0
    for episode in manifest["episodes"]:
        count = max(0, episode["output_frames"] - 29)
        if count:
            for index in {start, start + count - 1}:
                sample = dataset[index]
                assert sample["state"].shape == (40,)
                assert sample["action"].shape == (30, 40)
                assert sample["state_mask"].sum() == sample["action_mask"].sum() == 14
                assert np.isfinite(sample["state"]).all() and np.isfinite(sample["action"]).all()
                assert set(sample["images"]) == {"head", "left_wrist", "right_wrist"}
                assert TASK in sample["prompt"]
                checked += 1
        start += count
    report = {"samples": len(dataset), "batches_per_rank": batches, "vla_epoch_at_10000": 10000 / batches[0], "episode_boundary_samples_checked": checked, "validation": "ok"}
    (evidence / "dataloader_verification.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
