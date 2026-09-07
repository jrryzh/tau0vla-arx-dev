#!/usr/bin/env python3
"""Fit each experiment's own valid-window statistics, then gate on DataLoader parity."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time
from prepare_blue_t_training import REPO, EVIDENCE, MODES, TASKS, profile, config_name, dataset_path, save, sha256
from tau0_vla.adapters.arx_lift2s.calibrated import contract


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("name",choices=TASKS)
    args=parser.parse_args()
    name=args.name
    python=REPO / ".venv/bin/python"
    env=dict(os.environ,PYTHONPATH="src:.",OMP_NUM_THREADS="1",OPENBLAS_NUM_THREADS="1")
    for mode in MODES:
        evidence=EVIDENCE / profile(name,mode)
        while not (evidence / "converted.json").exists():
            time.sleep(30)
        if (evidence / "ready.json").exists():
            continue
        def run(stage, command):
            print(name,mode,stage,"started",flush=True)
            with (evidence / (stage+".log")).open("w") as stream:
                subprocess.run([str(x) for x in command],cwd=REPO,env=env,stdout=stream,stderr=subprocess.STDOUT,check=True)
        if mode == MODES[0]:
            run("video_random_access",[python,"scripts/optimize_blue_t_videos.py",name])
        body="arx_calibrated_eef_v1" if mode=="eef-vr" else "arx_calibrated_joint_v1"
        run("statistics",[python,"scripts/norm_stats/compute_unified_ft_stats.py","--body",body,"--repos",dataset_path(name,mode),
            "--action-horizon",30,"--positive-labels","--negative-labels","--partials-dir",evidence / "stats"])
        stats_path=REPO / "configs" / config_name(name,mode) / "norm_stats.json"
        run("statistics_merge",[python,"scripts/norm_stats/merge_stats.py","--partials",evidence / "stats","--out",stats_path])
        stats=json.loads(stats_path.read_text())
        stats["calibrated_provenance"]={"dataset":name,"contract":contract(mode),"source_manifest_sha256":sha256(evidence / "source_manifest.json")}
        save(stats_path,stats)
        run("dataloader_verification",[python,"scripts/verify_blue_t_dataloader.py",name,mode])
        print(name,mode,"ready",flush=True)


if __name__=="__main__":
    main()
