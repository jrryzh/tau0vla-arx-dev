#!/usr/bin/env python3
"""Audited, serialized release and one-for-one replacement of approved fill jobs.

Run with qzcli's Python environment. No mutation is made by the audit command.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import json
from pathlib import Path
import re
import shlex
import subprocess
import time

from qzcli_session import ensure_session

REPO = Path(__file__).resolve().parents[1]
STATE = REPO / "outputs/0905_preparation/resources"
WORKSPACE = "ws-21bd7e9f-5f97-4ffa-831e-966a436c7818"
GROUP = "lcg-d8eb9030-2233-47f7-b8cb-988c3e7c0ec9"
OWNER = "user-eae5205f-ce2f-4a21-b4c4-6ceab9da1a65"
TERMINAL = {"job_succeeded", "job_failed", "job_stopped"}
EXISTING = {
    "qwen35_vla_fill_16g_20260904_r4_35": "job-eef0366e-d204-4e4e-bd68-3c8de60dd52b",
    "qwen35_vla_fill_16g_20260904_r4_36": "job-d950031c-218f-4f1e-a765-0db434c4f7bd",
}


class AuthenticatedAPI:
    """Retry expired-cookie reads once; mutations are never retried."""
    def __init__(self, api, credentials):
        self.api = api
        self.credentials = credentials

    def __getattr__(self, name):
        def call(*args, **kwargs):
            from qzcli.api import QzAPIError
            from qzcli.config import get_cookie
            args = list(args)
            args[1] = get_cookie()["cookie"]
            try:
                return getattr(self.api, name)(*args, **kwargs)
            except QzAPIError as exc:
                if not name.startswith(("list_", "get_")) or getattr(exc, "code", None) != 401:
                    raise
                ensure_session(self.credentials, force=True)
                args[1] = get_cookie()["cookie"]
                return getattr(self.api, name)(*args, **kwargs)
        return call


def approved_fill(job: dict) -> bool:
    return (job.get("created_by", {}).get("id") == OWNER
        and job.get("logic_compute_group_id") == GROUP
        and job.get("name", "").startswith("qwen35_vla_fill_16g_"))


def choose_release(jobs: list[dict], idle: int) -> dict | None:
    queued = [j for j in jobs if j.get("logic_compute_group_id") == GROUP and j.get("status") == "job_queuing"]
    ours = [j for j in queued if j["name"].startswith("arx-0905-") and j.get("node_count", 0) < 2]
    if not ours or idle >= 2:
        return None
    first = min(ours, key=lambda j: (-int(j.get("priority", 0)), int(j["created_at"])))
    ahead = [j for j in queued if not j["name"].startswith("arx-0905-")
        and (-int(j.get("priority", 0)), int(j["created_at"])) <= (-int(first.get("priority", 0)), int(first["created_at"]))]
    fills = [j for j in ahead if approved_fill(j)]
    if fills:
        return min(fills, key=lambda j: int(j["created_at"]))
    if ahead:
        return None
    running = [j for j in jobs if approved_fill(j) and j.get("status") == "job_running"
        and j.get("node_count") == 2 and "0905_replacement" not in j["name"]
        and j.get("job_id") not in EXISTING.values()]
    return min(running, key=lambda j: int(j["created_at"])) if running else None


def save(path: Path, value) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def all_jobs(api, cookie) -> list[dict]:
    rows = []
    page = 1
    while True:
        result = api.list_jobs_with_cookie(WORKSPACE, cookie, page_num=page, page_size=100)
        rows.extend(result["jobs"])
        if len(rows) >= result["total"]:
            return rows
        page += 1
        time.sleep(1)


def notebook_queue(api, cookie) -> list[dict]:
    rows = []
    page = 1
    while True:
        result = api.list_notebooks_with_cookie(WORKSPACE, cookie, page=page, page_size=100)
        items = result["list"]
        rows.extend(items)
        if len(rows) >= result["total"]:
            break
        if not items:
            raise RuntimeError("Notebook listing ended before its reported total")
        page += 1
        time.sleep(1)
    return [j for j in rows if j.get("logic_compute_group_id") == GROUP and str(j.get("status", "")).lower() in {"pending", "queuing", "notebook_pending"}]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["audit", "release-one", "replenish", "verify"])
    parser.add_argument("--credentials", required=True, type=Path)
    args = parser.parse_args()
    STATE.mkdir(parents=True, exist_ok=True)
    with (STATE / "coordinator.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        ensure_session(args.credentials)
        from qzcli.api import get_api
        from qzcli.config import get_cookie
        api = AuthenticatedAPI(get_api(), args.credentials)
        cookie = get_cookie()["cookie"]
        rows = all_jobs(api, cookie)
        notebooks = notebook_queue(api, cookie)
        result = subprocess.run(["qzcli", "avail", "--workspace", WORKSPACE, "--group", GROUP, "--nodes", "2", "--export"], capture_output=True, text=True)
        free = re.findall(r"(\d+) 空节点", result.stdout)
        if not free:
            raise RuntimeError("Cannot establish idle nodes; refusing resource mutations")
        idle = int(free[-1])
        snapshot = {"time": datetime.now(timezone.utc).isoformat(), "idle": idle, "jobs": rows, "pending_notebooks": notebooks}
        save(STATE / "latest_audit.json", snapshot)
        print("idle_nodes=", idle, "queued_vla=", [(j["name"], j.get("node_count")) for j in rows if j.get("logic_compute_group_id") == GROUP and j["status"] == "job_queuing"], "pending_notebooks=", len(notebooks), flush=True)
        ledger_path = STATE / "ledger.json"
        ledger = json.loads(ledger_path.read_text()) if ledger_path.exists() else {"stops": [], "replacements": {}}
        if args.action == "audit":
            return
        if args.action == "release-one":
            if notebooks:
                print("Pending development jobs require reassessment; no stop performed")
                return
            if any(r["result"] != "stop_accepted" for r in ledger["stops"]):
                raise RuntimeError("An earlier stop needs reconciliation before another mutation")
            candidate = choose_release(rows, idle)
            if candidate is None:
                print("No justified filler stop")
                return
            submitted_profiles = sum(
                (directory / "smoke_job_id").exists() or (directory / "formal_job_id").exists()
                for directory in (REPO / "outputs").glob("qzcli_arx_h200_0905*") if directory.is_dir()
            )
            initial_idle = ledger.setdefault("initial_idle", idle)
            release_limit = max(0, (submitted_profiles * 2 - initial_idle + 1) // 2)
            running_stops = sum(r["before_status"] == "job_running" for r in ledger["stops"])
            if candidate["status"] == "job_running" and running_stops >= release_limit:
                print("Enough nodes have already been released for submitted profiles; wait for smoke/formal resource turnover")
                return
            if any(by_stop.get("result") == "stop_accepted" and
                   next((j["status"] for j in rows if j["job_id"] == by_stop["job_id"]), "unknown") not in TERMINAL
                   for by_stop in ledger["stops"]):
                print("An earlier filler is still releasing resources; no additional stop")
                return
            detail = api.get_job_detail_with_cookie(candidate["job_id"], cookie)
            assert approved_fill(detail) and detail["status"] == candidate["status"]
            assert "bash scripts/run_qwen35_capacity_hold.sh 16 " in detail["command"]
            config = detail["framework_config"][0]
            assert config["gpu_count"] == 8 and config["instance_count"] == 2
            assert detail["job_id"] not in {r["job_id"] for r in ledger["stops"]}
            record = {"job_id": detail["job_id"], "name": detail["name"], "before_status": detail["status"],
                "idle_before": idle, "time": snapshot["time"], "result": "requested"}
            save(STATE / f"{detail['job_id']}_before.json", detail)
            ledger["stops"].append(record)
            save(ledger_path, ledger)
            accepted = api.stop_job_with_cookie(detail["job_id"], cookie)
            record["result"] = "stop_accepted" if accepted else "stop_uncertain"
            save(ledger_path, ledger)
            print(json.dumps(record), flush=True)
            return
        historical = json.loads((REPO / "outputs/0904_pickandplace_preparation/queued_fill_cancellation_results.json").read_text())
        historical += [json.loads((REPO / f"outputs/0904_pickandplace_preparation/stop_r4_{n}_approved.json").read_text()) for n in ("01", "03")]
        historical.append({"job_id": "job-83decba4-e3d8-444d-868d-a077323895b5", "name": "qwen35_vla_fill_16g_20260904_r4_02"})
        assert len(historical) == 36
        obligations = {r["job_id"]: r for r in historical + ledger["stops"]}
        by_id = {j["job_id"]: j for j in rows}
        for old in historical:
            candidates = [j for j in rows if approved_fill(j) and j["status"] not in TERMINAL
                and (j["job_id"] == EXISTING.get(old["name"])
                     or j["name"] == old["name"] + "_copy"
                     or j["name"].startswith(old["name"] + "_copy_"))]
            if candidates and old["job_id"] not in ledger["replacements"]:
                jid = min(candidates, key=lambda j: int(j["created_at"]))["job_id"]
                detail = api.get_job_detail_with_cookie(jid, cookie)
                assert approved_fill(detail) and detail["status"] not in TERMINAL
                assert old["name"] + "/RELEASE" in detail["command"]
                ledger["replacements"][old["job_id"]] = {"job_id": jid, "name": detail["name"], "preexisting": True}
        save(ledger_path, ledger)
        if args.action == "verify":
            missing = set(obligations) - set(ledger["replacements"])
            unhealthy = [r for r in ledger["replacements"].values() if by_id.get(r["job_id"], {}).get("status") not in {"job_running", "job_queuing"}]
            assert not (set(ledger["replacements"]) - set(obligations)), "Unrequested replacement obligation"
            replacements = list(ledger["replacements"].values())
            assert len({r["job_id"] for r in replacements}) == len(replacements), "A replacement was credited twice"
            formal_ids = [(p / "formal_job_id").read_text().strip() for p in (REPO / "outputs").glob("qzcli_arx_h200_0905*") if (p / "formal_job_id").exists()]
            assert len(formal_ids) == 10, "Ten formal jobs must precede replacement acceptance"
            last_formal_created = max(int(by_id[jid]["created_at"]) for jid in formal_ids)
            for replacement in replacements:
                detail = by_id[replacement["job_id"]]
                assert approved_fill(detail) and detail["workspace_id"] == WORKSPACE
                assert detail["gpu_count"] == 16 and len(detail["framework_config"]) == 1
                config = detail["framework_config"][0]
                assert config["gpu_count"] == 8 and config["instance_count"] == 2
                assert "bash scripts/run_qwen35_capacity_hold.sh 16 " in detail["command"]
                if not replacement.get("preexisting"):
                    assert detail["priority_name"] == "9"
                    assert int(detail["created_at"]) > last_formal_created
                    assert str(REPO / "outputs/0905_capacity_holds" / replacement["name"] / "RELEASE") in detail["command"]
            verification = {"time": snapshot["time"], "obligations": len(obligations),
                "replacements": len(replacements), "preexisting": sum(r.get("preexisting", False) for r in replacements),
                "missing": sorted(missing), "unhealthy": unhealthy, "gpu_per_replacement": 16,
                "unique_mapping_verified": True, "new_replacements_after_all_ten_formals": True,
                "new_replacement_priority": 9, "validation": "ok" if not missing and not unhealthy else "pending"}
            save(STATE / "verification.json", verification)
            print(json.dumps(verification))
            if missing or unhealthy:
                raise SystemExit(1)
            return
        profiles = ("0905_datasets_first50", "0905_datasets_last50", "0905_datasets_all100", "0905_zyp_gyt_first50", "0905_zyp_gyt_all100")
        for profile in [p + suffix for p in profiles for suffix in ("", "_20k")]:
            jid = (REPO / f"outputs/qzcli_arx_h200_{profile}/formal_job_id").read_text().strip()
            assert by_id[jid]["status"] in {"job_running", "job_queuing", "job_succeeded"}
        template = json.loads((REPO / "outputs/0905_preparation/job-eef0366e-d204-4e4e-bd68-3c8de60dd52b_detail.json").read_text())
        config = template["framework_config"][0]
        for old_id, old in obligations.items():
            if old_id in ledger["replacements"]:
                continue
            ensure_session(args.credentials)
            name = "qwen35_vla_fill_16g_0905_replacement_" + old_id.removeprefix("job-").replace("-", "")[:20]
            assert sum(other.removeprefix("job-").replace("-", "")[:20] == old_id.removeprefix("job-").replace("-", "")[:20] for other in obligations) == 1
            matches = [j for j in rows if j["name"] == name]
            if len(matches) > 1:
                raise RuntimeError("Duplicate replacement name requires reconciliation")
            if matches:
                jid = matches[0]["job_id"]
            else:
                hold = REPO / "outputs/0905_capacity_holds" / name
                hold.mkdir(parents=True, exist_ok=True)
                assert not (hold / "RELEASE").exists()
                command = "cd /inspire/qb-ilm/project/robot-learning-system/public/weidafeng/codes/AgibotVLA && bash scripts/run_qwen35_capacity_hold.sh 16 " + shlex.quote(str(hold / "RELEASE")) + " " + shlex.quote(str(hold / "heartbeat"))
                cmd = ["qzcli", "create", "--name", name, "--command", command, "--workspace", WORKSPACE,
                    "--compute-group", GROUP, "--spec", "e9411c6b-91b6-4973-b843-633a91998beb",
                    "--gpu-type", "NVIDIA_H200_SXM_141G", "--cpu", str(config["cpu"]), "--memory", str(config["mem_gi"]),
                    "--gpus", "8", "--instances", "2", "--shm", "1200", "--image", config["image"],
                    "--image-type", config["image_type"], "--framework", "pytorch", "--priority", "9"]
                dry = subprocess.run([*cmd, "--dry-run"], capture_output=True, text=True, check=True)
                (STATE / f"{name}_dry.log").write_text(dry.stdout)
                payload = json.loads(dry.stdout[dry.stdout.index("{"):])
                assert payload["name"] == name and payload["command"] == command
                assert payload["workspace_id"] == WORKSPACE and payload["logic_compute_group_id"] == GROUP
                assert payload["task_priority"] == 9
                assert len(payload["framework_config"]) == 1
                submitted = payload["framework_config"][0]
                assert submitted["instance_count"] == 2 and submitted["gpu_count"] == 8
                assert submitted["cpu"] == config["cpu"] and submitted["mem_gi"] == config["mem_gi"]
                assert submitted["image"] == config["image"] and submitted["shm_gi"] == 1200
                result = subprocess.run([*cmd, "--json"], capture_output=True, text=True)
                (STATE / f"{name}_submit.log").write_text(result.stdout + result.stderr)
                if result.returncode:
                    raise RuntimeError("Replacement submission uncertain; rerun only after full queue reconciliation")
                objects = [json.loads(line) for line in result.stdout.splitlines() if line.startswith("{")]
                jid = objects[-1]["job_id"]
                time.sleep(3)
            detail = api.get_job_detail_with_cookie(jid, cookie)
            assert approved_fill(detail) and detail.get("priority_name") == "9", "Platform did not confirm replacement priority 9"
            ledger["replacements"][old_id] = {"job_id": jid, "name": name, "preexisting": False}
            save(ledger_path, ledger)
            print("replacement", old["name"], "->", jid, flush=True)
            time.sleep(3)
        print("replacement_count=", len(ledger["replacements"]), "obligations=", len(obligations), flush=True)


if __name__ == "__main__":
    main()
