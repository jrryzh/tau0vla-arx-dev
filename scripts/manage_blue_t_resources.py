#!/usr/bin/env python3
"""Audited 0907-only replacement ledger for the previously approved 16-H200 fillers."""
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

from manage_0905_resources import AuthenticatedAPI, WORKSPACE, GROUP, TERMINAL, approved_fill, all_jobs, notebook_queue, save
from qzcli_session import ensure_session

REPO = Path(__file__).resolve().parents[1]
STATE = REPO / "outputs/0907_blue_t_preparation/resources"
PROFILES = tuple(f"0907-{name}-{mode}" for name in ("blue", "t") for mode in ("joint-vr", "joint-feedback", "eef-vr"))
MIXED_PROFILES = tuple(f"0907-bluet-{mode}" for mode in ("joint-vr", "joint-feedback", "eef-vr"))


def owns_job(job, profiles):
    return any(job["name"].startswith("arx-"+profile+"-") for profile in profiles)


def choose(jobs, idle, profiles=PROFILES):
    queued = [j for j in jobs if j.get("logic_compute_group_id") == GROUP and j["status"] == "job_queuing"]
    ours = [j for j in queued if owns_job(j,profiles)]
    if not ours or idle >= 2:
        return None
    order = lambda j: (-int(j.get("priority", 0)), int(j["created_at"]))
    first = min(ours, key=order)
    ahead = [j for j in queued if not owns_job(j,profiles) and order(j) <= order(first)]
    fills = [j for j in ahead if approved_fill(j)]
    if fills:
        return min(fills, key=order)
    if ahead:
        return None
    running = [j for j in jobs if approved_fill(j) and j["status"] == "job_running" and j.get("node_count") == 2
               and "0907_replacement" not in j["name"] and "0907_mixed_replacement" not in j["name"]]
    return min(running, key=lambda j: int(j["created_at"])) if running else None


def reassessed_formal_deficit(jobs, formal_ids, idle, stops, now, profiles=PROFILES):
    """Recheck actual allocation after the original released capacity was scheduled."""
    by_id={j["job_id"]:j for j in jobs}
    if len(formal_ids)!=len(profiles) or len(set(formal_ids))!=len(profiles):
        return None
    formal=[by_id.get(j,{}) for j in formal_ids]
    if any(j.get("status") not in {"job_running","job_queuing"} for j in formal):
        return None
    if any(owns_job(j,profiles) and j["job_id"] not in formal_ids and j["status"] not in TERMINAL for j in jobs):
        return None
    queued=[j for j in formal if j["status"]=="job_queuing"]
    if not queued or any(now-int(j["created_at"])/1000<120 for j in queued):
        return None
    if any(now-datetime.fromisoformat(s["time"]).timestamp()<120 for s in stops):
        return None
    allocated=sum(int(j.get("node_count",0)) for j in formal)
    deficit=2*len(profiles)-allocated-idle
    return {"allocated_formal_nodes":allocated,"idle_nodes":idle,"missing_nodes":deficit,
            "queued_formal_ids":[j["job_id"] for j in queued]} if deficit>=2 else None


def main():
    global STATE
    p = argparse.ArgumentParser()
    p.add_argument("action", choices=("audit", "release-one", "replenish", "verify"))
    p.add_argument("--credentials", required=True, type=Path)
    p.add_argument("--reassess-deficit", action="store_true", help="Manually recheck a persistent formal allocation deficit beyond initial release accounting")
    p.add_argument("--mixed",action="store_true",help="Manage only the three Blue+T mixed profiles and their separate ledger")
    p.add_argument("--wait-lock-seconds",type=float,default=0,help="Wait up to 60 seconds for another authorized resource operation")
    args = p.parse_args()
    if not 0<=args.wait_lock_seconds<=60:p.error("--wait-lock-seconds must be in [0,60]")
    profiles=MIXED_PROFILES if args.mixed else PROFILES
    if args.mixed:STATE=REPO/"outputs/0907_bluet_mixed_preparation/resources"
    replacement_tag="0907_mixed_replacement" if args.mixed else "0907_replacement"
    STATE.mkdir(parents=True, exist_ok=True)
    with (REPO / "outputs/.arx_0907_capacity_coordinator.lock").open("a") as lock:
        deadline=time.monotonic()+args.wait_lock_seconds
        while True:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if time.monotonic()>=deadline:
                    print("Another authorized resource operation is in progress",flush=True)
                    return
                time.sleep(min(1,deadline-time.monotonic()))
        ensure_session(args.credentials)
        from qzcli.api import get_api
        from qzcli.config import get_cookie
        api = AuthenticatedAPI(get_api(), args.credentials)
        cookie = get_cookie()["cookie"]
        jobs = all_jobs(api, cookie)
        notebooks = notebook_queue(api, cookie)
        result = subprocess.run(["qzcli", "avail", "--workspace", WORKSPACE, "--group", GROUP, "--nodes", "2", "--export"], capture_output=True, text=True)
        matched = re.findall(r"(\d+) 空节点", result.stdout)
        if not matched:
            raise RuntimeError("idle capacity query failed; refusing mutations")
        idle = int(matched[-1])
        now = datetime.now(timezone.utc).isoformat()
        snapshot = {"time": now, "idle": idle, "jobs": jobs, "pending_notebooks": notebooks}
        save(STATE / "latest_audit.json", snapshot)
        ledger_path = STATE / "ledger.json"
        ledger = json.loads(ledger_path.read_text()) if ledger_path.exists() else {"stops": [], "replacements": {}, "initial_idle": idle}
        save(ledger_path, ledger)
        print(json.dumps({"time":now,"idle_nodes":idle,"queued_jobs":sum(j.get("logic_compute_group_id")==GROUP and j["status"]=="job_queuing" for j in jobs),"pending_notebooks":len(notebooks)}),flush=True)
        if args.action == "audit":
            return
        by_id = {j["job_id"]: j for j in jobs}
        if any(s["result"] != "stop_accepted" for s in ledger["stops"]):
            raise RuntimeError("uncertain previous stop; reconcile before another mutation")
        if args.action == "release-one":
            if notebooks or any(by_id.get(s["job_id"],{}).get("status") not in TERMINAL for s in ledger["stops"]):
                print("Waiting for pending notebooks or an earlier resource release",flush=True)
                return
            candidate = choose(jobs,idle,profiles)
            if candidate is None:
                print("No justified approved filler stop",flush=True)
                return
            submitted = sum(any((REPO / "outputs" / ("qzcli_arx_h200_"+profile.replace("-","_")) / k).exists() for k in ("smoke_job_id","formal_job_id")) for profile in profiles)
            limit = max(0,(submitted*2-ledger["initial_idle"]+1)//2)
            reassessment=None
            if candidate["status"] == "job_running" and sum(s["before_status"]=="job_running" for s in ledger["stops"]) >= limit:
                paths=[REPO / "outputs" / ("qzcli_arx_h200_"+profile.replace("-","_")) / "formal_job_id" for profile in profiles]
                formal_ids=[p.read_text().strip() for p in paths if p.exists()]
                if args.reassess_deficit:
                    reassessment=reassessed_formal_deficit(jobs,formal_ids,idle,ledger["stops"],time.time(),profiles)
                if reassessment is None:
                    print("Initial released capacity accounted for; persistent formal deficit requires an explicit reassessment",flush=True)
                    return
            detail = api.get_job_detail_with_cookie(candidate["job_id"],cookie)
            assert approved_fill(detail) and detail["status"] == candidate["status"]
            assert "bash scripts/run_qwen35_capacity_hold.sh 16 " in detail["command"]
            fc = detail["framework_config"]
            assert len(fc)==1 and fc[0]["gpu_count"]==8 and fc[0]["instance_count"]==2
            assert detail["job_id"] not in {s["job_id"] for s in ledger["stops"]}
            save(STATE / (detail["job_id"]+"_before.json"),detail)
            record = {"job_id":detail["job_id"],"name":detail["name"],"before_status":detail["status"],"time":now,"idle_before":idle,"gpus":16,"result":"requested"}
            if reassessment is not None: record["formal_deficit_reassessment"]=reassessment
            ledger["stops"].append(record); save(ledger_path,ledger)
            (STATE/"verification.json").unlink(missing_ok=True)
            accepted = api.stop_job_with_cookie(detail["job_id"],cookie)
            record["result"] = "stop_accepted" if accepted else "stop_uncertain"
            save(ledger_path,ledger)
            print(json.dumps(record),flush=True)
            return
        formal_ids=[]
        for profile in profiles:
            path=REPO / "outputs" / ("qzcli_arx_h200_"+profile.replace("-","_")) / "formal_job_id"
            if not path.exists() or by_id.get(path.read_text().strip(),{}).get("status") not in {"job_running","job_queuing","job_succeeded"}:
                print(f"Replenishment waits for all {len(profiles)} formal jobs to be accepted",flush=True)
                return
            formal_ids.append(path.read_text().strip())
        last_formal = max(int(by_id[j]["created_at"]) for j in formal_ids)
        if args.action == "replenish":
            for old in ledger["stops"]:
                old_id=old["job_id"]
                if old_id in ledger["replacements"]:
                    continue
                template=json.loads((STATE / (old_id+"_before.json")).read_text())
                fc=template["framework_config"][0]
                name="qwen35_vla_fill_16g_"+replacement_tag+"_"+old_id.removeprefix("job-").replace("-", "")
                matches=[j for j in jobs if j["name"]==name]
                if len(matches)>1:
                    raise RuntimeError("duplicate replacement requires reconciliation")
                if matches:
                    jid=matches[0]["job_id"]
                else:
                    hold=REPO / "outputs/0907_capacity_holds" / name
                    hold.mkdir(parents=True,exist_ok=True)
                    assert not (hold / "RELEASE").exists()
                    command="cd /inspire/qb-ilm/project/robot-learning-system/public/weidafeng/codes/AgibotVLA && bash scripts/run_qwen35_capacity_hold.sh 16 "+shlex.quote(str(hold / "RELEASE"))+" "+shlex.quote(str(hold / "heartbeat"))
                    cmd=["qzcli","create","--name",name,"--command",command,"--workspace",WORKSPACE,"--compute-group",GROUP,
                        "--spec","e9411c6b-91b6-4973-b843-633a91998beb","--gpu-type","NVIDIA_H200_SXM_141G","--cpu",str(fc["cpu"]),"--memory",str(fc["mem_gi"]),
                        "--gpus","8","--instances","2","--shm","1200","--image",fc["image"],"--image-type",fc["image_type"],"--framework","pytorch","--priority","9"]
                    dry=subprocess.run([*cmd,"--dry-run"],capture_output=True,text=True,check=True)
                    (STATE / (name+"_dry.log")).write_text(dry.stdout)
                    pending=STATE / (name+"_pending.json")
                    if pending.exists():
                        raise RuntimeError("uncertain replacement submission; reconcile exact name")
                    save(pending,{"name":name,"old_job_id":old_id,"requested_at":now})
                    actual=subprocess.run([*cmd,"--json"],capture_output=True,text=True,check=True)
                    (STATE / (name+"_submit.log")).write_text(actual.stdout)
                    objects=[json.loads(line) for line in actual.stdout.splitlines() if line.lstrip().startswith('{')]
                    jid=objects[-1]["job_id"]
                    pending.unlink()
                ledger["replacements"][old_id]={"job_id":jid,"name":name,"gpus":16}
                save(ledger_path,ledger)
                print("replacement",jid,flush=True)
        else:
            assert len({v["job_id"] for v in ledger["replacements"].values()})==len(ledger["replacements"])
            assert set(ledger["replacements"])=={s["job_id"] for s in ledger["stops"]}
            for replacement in ledger["replacements"].values():
                detail=api.get_job_detail_with_cookie(replacement["job_id"],cookie)
                assert approved_fill(detail) and detail["status"] in {"job_running","job_queuing"}
                assert int(detail["created_at"]) >= last_formal
                assert detail["framework_config"][0]["gpu_count"]==8 and detail["framework_config"][0]["instance_count"]==2
            save(STATE / "verification.json",{"validation":"ok","time":now,"stops":len(ledger["stops"]),"replacements":len(ledger["replacements"]),"gpus":16*len(ledger["stops"]),"after_all_formal_submissions":True,"formal_profiles":list(profiles),**({"after_all_six_formal_submissions":True} if not args.mixed else {})})


if __name__ == "__main__":
    main()
