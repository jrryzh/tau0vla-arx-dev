#!/usr/bin/env python3
"""Start each verified 0905 profile once and keep its controller attached."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import time

from prepare_0905_training import PROFILES, REPO

RUNS = tuple((p + suffix, p) for p in PROFILES for suffix in ("", "-20k"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--credentials", required=True, type=Path)
    args = parser.parse_args()
    evidence = REPO / "outputs/0905_preparation"
    with (evidence / "campaign.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        (evidence / "campaign.pid").write_text(str(os.getpid()) + "\n")
        children = {}
        failed = set()
        while True:
            for profile, dataset_profile in RUNS:
                if profile in children or profile in failed:
                    continue
                prepared = evidence / dataset_profile
                if not (prepared / "ready.json").exists():
                    continue
                state = REPO / "outputs" / ("qzcli_arx_h200_" + profile.replace("-", "_"))
                state.mkdir(exist_ok=True)
                if (state / "controller.pid").exists():
                    pid = (state / "controller.pid").read_text().strip()
                    cmdline = Path(f"/proc/{pid}/cmdline")
                    if cmdline.exists() and b"qzcli_arx_h200.sh" in cmdline.read_bytes() and profile.encode() in cmdline.read_bytes():
                        print(f"{profile}: controller already running; leaving it attached to its owner", flush=True)
                        failed.add(profile)
                        continue
                if not (prepared / "dataloader_verification.json").exists():
                    with (prepared / "dataloader_verification.log").open("w") as log:
                        result = subprocess.run([str(REPO / ".venv/bin/python"), "scripts/verify_0905_dataloader.py", dataset_profile], cwd=REPO, stdout=log, stderr=subprocess.STDOUT)
                    if result.returncode:
                        print(f"{profile}: DataLoader verification failed; see log", flush=True)
                        failed.add(profile)
                        continue
                log = (state / "controller.log").open("a")
                children[profile] = subprocess.Popen(["bash", "scripts/qzcli_arx_h200.sh", "auto", "--profile", profile, "--credentials", str(args.credentials)], cwd=REPO, stdout=log, stderr=subprocess.STDOUT)
                log.close()
                print(f"{profile}: launched controller pid={children[profile].pid}", flush=True)
            status = {p: ("preflight_failed_or_external_controller" if p in failed else "preparing" if p not in children else "controller_running" if children[p].poll() is None else f"controller_exit_{children[p].returncode}") for p, _ in RUNS}
            (evidence / "campaign_status.json").write_text(json.dumps(status, indent=2) + "\n")
            if len(children) + len(failed) == len(RUNS) and all(p.poll() is not None for p in children.values()):
                print(json.dumps(status), flush=True)
                return
            time.sleep(30)


if __name__ == "__main__":
    main()
