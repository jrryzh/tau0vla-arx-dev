#!/usr/bin/env python3
"""Keep six independent H200 controllers, recovery and reporting running to 10k."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import time

from prepare_blue_t_training import REPO,EVIDENCE,TASKS,MODES,profile,save


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--credentials",type=Path,required=True)
    p.add_argument("--manage-resources",action="store_true")
    p.add_argument("--mixed",action="store_true")
    args=p.parse_args()
    evidence_root=REPO / "outputs/0907_bluet_mixed_preparation" if args.mixed else EVIDENCE
    names=("BlueT",) if args.mixed else tuple(TASKS)
    campaign_flags=["--mixed"] if args.mixed else []
    evidence_root.mkdir(parents=True,exist_ok=True)
    with (evidence_root / "campaign.lock").open("a") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        (evidence_root / "campaign.pid").write_text(str(os.getpid())+"\n")
        children={}; last_restart={}; resource_time=0; report_time=0
        while True:
            statuses={}
            for name in names:
                for mode in MODES:
                    slug=profile(name,mode)
                    state=REPO / "outputs" / ("qzcli_arx_h200_"+slug.replace("-","_"))
                    state.mkdir(parents=True,exist_ok=True)
                    if (state / "completed_job_id").exists():
                        statuses[slug]="completed_10000"; continue
                    if not (evidence_root / slug / "ready.json").exists():
                        statuses[slug]="preparing"; continue
                    child=children.get(slug)
                    if child is not None and child.poll() is None:
                        statuses[slug]="controller_running"; continue
                    if (state / "pending_submission.json").exists():
                        statuses[slug]="submission_requires_reconciliation"; continue
                    if child is not None and child.returncode and not (state / "formal_job_id").exists():
                        statuses[slug]="smoke_requires_review"; continue
                    if time.time()-last_restart.get(slug,0)<120:
                        statuses[slug]="recovery_backoff"; continue
                    pidfile=state / "controller.pid"
                    if pidfile.exists():
                        cmdline=Path('/proc') / pidfile.read_text().strip() / 'cmdline'
                        if cmdline.exists() and b'qzcli_arx_h200.sh' in cmdline.read_bytes() and slug.encode() in cmdline.read_bytes():
                            statuses[slug]="external_controller_running"; continue
                    with (state / "controller.log").open("a") as log:
                        children[slug]=subprocess.Popen(["bash","scripts/qzcli_arx_h200.sh","auto","--profile",slug,"--credentials",str(args.credentials)],cwd=REPO,stdout=log,stderr=subprocess.STDOUT)
                    last_restart[slug]=time.time()
                    statuses[slug]="controller_started"
            save(evidence_root / "campaign_status.json",statuses)
            if time.time()-report_time>=60:
                with (evidence_root / "monitor.log").open("a") as log:
                    subprocess.run([str(REPO / '.venv/bin/python'),"scripts/report_blue_t_campaign.py","--credentials",str(args.credentials),*campaign_flags],cwd=REPO,stdout=log,stderr=subprocess.STDOUT)
                report_time=time.time()
            if args.manage_resources and time.time()-resource_time>=120:
                runtime=Path('/opt/qzcli-runtime-0.3.0-py313/bin/python')
                if not runtime.exists(): runtime=Path('/opt/qzcli-runtime/bin/python')
                if not runtime.exists():
                    runtime=Path('/inspire/qb-ilm2/project/robot-learning-system/zhangjinyu-253108120325/miniconda3/bin/python3.13')
                for action in ("release-one","replenish","verify"):
                    with (evidence_root / "resources.log").open("a") as log:
                        result=subprocess.run([str(runtime),"scripts/manage_blue_t_resources.py",action,"--credentials",str(args.credentials),*campaign_flags],cwd=REPO,stdout=log,stderr=subprocess.STDOUT)
                    if result.returncode: break
                resource_time=time.time()
            print(json.dumps(statuses),flush=True)
            if all(s=="completed_10000" for s in statuses.values()):
                return
            time.sleep(30)


if __name__=="__main__":
    main()
