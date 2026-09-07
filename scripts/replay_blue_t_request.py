#!/usr/bin/env python3
"""Replay a v3 NPZ request offline through a checkpoint, recording its result."""
import argparse
import json
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO / "src")]
import numpy as np
from deploy.arx_calibrated_http import infer_and_record
from deploy.policy import Tau0VLAPolicy


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True, type=Path)
    p.add_argument("--request", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()
    with np.load(args.request, allow_pickle=False) as recording:
        raw = json.loads(str(recording["request_json"]))
        images = {k: recording[f"image_{k}"] for k in ("head", "left_wrist", "right_wrist")}
    policy = Tau0VLAPolicy.from_checkpoint(args.model, device=args.device)
    result = infer_and_record(policy, raw, images, model_id=str(args.model.resolve()), record_dir=args.out)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
