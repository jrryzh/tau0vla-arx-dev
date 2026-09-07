#!/usr/bin/env python3
"""Persist six-run curves and the strict checkpoint-500/two-growth-check acceptance."""
import argparse
import csv
import fcntl
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import re
import subprocess
import sys
import time

from prepare_blue_t_training import REPO, EVIDENCE, TASKS, MODES, profile, config_name, save
from report_0905_campaign import metrics_from_log


def advance_acceptance(acceptance, *, step, now, job_id, healthy, reset=False, metrics=None):
    """Require two distinct later observations after validation or a recovery."""
    if not acceptance["checkpoint_validated"]:
        acceptance["accepted"] = False
        return
    if reset or acceptance.get("growth_job_id", job_id) != job_id:
        acceptance.update(growth_checks=[], baseline_step=step, validated_at=now)
    acceptance["growth_job_id"] = job_id
    checks = acceptance["growth_checks"]
    last = checks[-1] if checks else {"time":acceptance["validated_at"],"step":acceptance["baseline_step"]}
    if healthy and now-last["time"] >= 30 and step > last["step"]:
        checks.append({"time":now,"step":step,"job_id":job_id,**(metrics or {})})
    acceptance["accepted"] = len(checks) >= 2 and healthy


def report(credentials=None, *, mixed=False):
    evidence_root=REPO / "outputs/0907_bluet_mixed_preparation" if mixed else EVIDENCE
    names=("BlueT",) if mixed else tuple(TASKS)
    accepted_key="all_three_accepted" if mixed else "all_six_accepted"
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import yaml
    now=time.time()
    reports=[]
    fig,axes=plt.subplots(len(names),2,figsize=(13,4*len(names)),constrained_layout=True,squeeze=False)
    for row_index,name in enumerate(names):
        for mode in MODES:
            slug=profile(name,mode)
            cname=config_name(name,mode)
            run_name=cname+"_h200_formal"
            run=REPO / "outputs" / run_name / run_name
            state=REPO / "outputs" / ("qzcli_arx_h200_"+slug.replace("-","_"))
            evidence=evidence_root / slug
            row={"profile":slug,"run_directory":str(run),"config":str(REPO / "configs" / cname / "train_h200.yaml"),"accepted":False}
            if (evidence / "ready.json").exists():
                row["data"]=json.loads((evidence / "ready.json").read_text())
            for kind in ("smoke","formal"):
                id_path=state / (kind+"_job_id")
                if id_path.exists():
                    row[kind+"_job_id"]=id_path.read_text().strip()
            if row.get("smoke_job_id"):
                validation=state / (row["smoke_job_id"]+"_validation.log")
                if validation.exists():
                    row["smoke_validation"]=validation.read_text().strip()
            replay=evidence / "checkpoint_500_inference_validation.json"
            if replay.exists():
                row["checkpoint_inference"]=json.loads(replay.read_text())
            status="not_submitted"
            if credentials and row.get("formal_job_id"):
                command=[str(REPO / ".venv/bin/python"),"scripts/qzcli_session.py","--credentials",str(credentials),"--","status",row["formal_job_id"],"--json"]
                try:
                    result=subprocess.run(command,cwd=REPO,capture_output=True,text=True,timeout=120)
                except subprocess.TimeoutExpired:
                    result=subprocess.CompletedProcess(command,124,stdout="",stderr="Scheduler status query timed out")
                    row["scheduler_query_error"]="timeout; training logs are still reported"
                from validate_h200_smoke import _load_last_json
                status_path=state / "report_status.json"
                status_path.write_text(result.stdout)
                try:
                    status=_load_last_json(status_path)["status"] if result.returncode==0 else "unknown"
                except (ValueError,KeyError):
                    status="unknown"
            row["formal_status"]=status
            logs=sorted((run / "log").glob("training_log_nodeIdx000_*.txt"))
            metrics=[]
            for path in logs:
                metrics.extend(metrics_from_log(path))
            for metric in metrics:
                for key in ("loss","grad_norm","vla_loss","learning_rate","vla_epoch","epoch"):
                    if key in metric:
                        metric[key] = float(metric[key])
            if metrics:
                latest=metrics[-1]
                row["metrics"]=latest
                steps=[m["global_step"] for m in metrics]
                axes[row_index,0].plot(steps,[m["loss"] for m in metrics],label=mode)
                axes[row_index,1].plot(steps,[m["grad_norm"] for m in metrics],label=mode)
                with (evidence / "training_metrics.csv").open("w",newline="") as stream:
                    writer=csv.DictWriter(stream,fieldnames=sorted(set().union(*(m.keys() for m in metrics))))
                    writer.writeheader();writer.writerows(metrics)
                current_logs=[]
                for node in ("000","001"):
                    paths=sorted((run / "log").glob(f"training_log_nodeIdx{node}_*.txt"))
                    if paths: current_logs.append(paths[-1].read_text(errors="replace"))
                anomalies=[m for m in metrics_from_log(logs[-1]) if not all(math.isfinite(float(m[k])) for k in ("loss","grad_norm"))]
                errors=sum(len(re.findall(r"Traceback \(most recent call last\)|CUDA out of memory|NCCL error|ChildFailedError",text)) for text in current_logs)
                row["current_log_errors"]=errors
                row["nonfinite_metrics"]=len(anomalies)
                if (run / "run_spec.json").exists():
                    actual=json.loads((run / "run_spec.json").read_text())
                    expected=yaml.safe_load(Path(row["config"]).read_text())
                    for section in ("model_args","data_args","training_args"):
                        for key,value in expected[section].items():
                            if key=="output_dir": value=str(run)
                            if key=="report_to" and value=="none": value=[]
                            if actual[section].get(key)!=value:
                                raise ValueError(f"{slug}: training config mismatch {section}.{key}")
                acceptance_path=evidence / "acceptance.json"
                acceptance=json.loads(acceptance_path.read_text()) if acceptance_path.exists() else {"growth_checks":[],"checkpoint_validated":False}
                checkpoint=run / "checkpoint-500"
                if not acceptance["checkpoint_validated"] and int(latest["global_step"])>=500:
                    check=subprocess.run([sys.executable,"scripts/validate_checkpoint.py",str(checkpoint),"--world-size","16","--expected-step","500","--deployment"],cwd=REPO,capture_output=True,text=True)
                    (evidence / "checkpoint_500_validation.log").write_text(check.stdout+check.stderr)
                    if check.returncode==0:
                        specs=list((checkpoint / "finch_data_spec").glob("*/spec.json"))
                        assert len(specs)==1
                        saved=json.loads(specs[0].read_text())
                        from tau0_vla.adapters.arx_lift2s.calibrated import contract
                        assert saved["deployment_contract"]==contract(mode)
                        assert json.loads(specs[0].with_name("norm_stats.json").read_text())==json.loads((REPO / "configs" / cname / "norm_stats.json").read_text())
                        acceptance.update(checkpoint_validated=True,checkpoint=str(checkpoint),validated_at=now,baseline_step=int(latest["global_step"]),formal_job_id=row.get("formal_job_id"))
                healthy=status in {"job_running","job_succeeded"} and not(errors or anomalies)
                advance_acceptance(acceptance,step=int(latest["global_step"]),now=now,job_id=row.get("formal_job_id"),healthy=healthy,
                    reset=bool(errors or anomalies or status in {"job_failed","job_stopped"}),
                    metrics={key:latest.get(key) for key in ("loss","grad_norm","vla_epoch")})
                save(acceptance_path,acceptance)
                row["accepted"]=acceptance["accepted"]
                row["acceptance"]=acceptance
            reports.append(row)
        for col,label in enumerate(("Training loss","Gradient norm")):
            axes[row_index,col].set(title=f"{name}: {label}",xlabel="Optimization step",ylabel=label)
            if axes[row_index,col].lines: axes[row_index,col].legend()
    evidence_root.mkdir(parents=True,exist_ok=True)
    fig.savefig(evidence_root / "training_comparison.png",dpi=150)
    fig.savefig(evidence_root / "training_comparison.pdf")
    plt.close(fig)
    summary={"time":datetime.now(timezone.utc).isoformat(),accepted_key:all(r["accepted"] for r in reports),"all_accepted":all(r["accepted"] for r in reports),"runs":reports}
    save(evidence_root / "campaign_report.json",summary)
    lines=["# Blue + T mixed calibrated v1" if mixed else "# Blue / T calibrated v1", "", "Each experiment targets 10,000 steps. This handoff requires complete checkpoint-500 and two later checks showing continued step growth without new training errors. Robot client EEF publishing and VR-to-driver gripper mapping remain unimplemented.", "",
        "| Profile | Step | Loss | vla_epoch | State | Formal job |", "|---|---:|---:|---:|---|---|"]
    for row in reports:
        m=row.get("metrics",{})
        lines.append(f"| {row['profile']} | {m.get('global_step','—')} | {m.get('loss','—')} | {m.get('vla_epoch','—')} | {'accepted' if row['accepted'] else row['formal_status']} | {row.get('formal_job_id','—')} |")
    lines += ["", "| Profile | Configuration | Model / checkpoint | Smoke |", "|---|---|---|---|"]
    for row in reports:
        checkpoint=Path(row["run_directory"]) / "checkpoint-500"
        lines.append(f"| {row['profile']} | [train_h200.yaml]({row['config']}) | [checkpoint-500]({checkpoint}) | {row.get('smoke_validation','pending')} |")
    lines += ["", "[Comparison curves](training_comparison.png) · [Machine-readable report](campaign_report.json) · [Per-side calibration audit](calibration_audit.json)", "",
        "[Calibration summary](calibration_summary.json) · [Calibration plot](calibration_baselines.png) · [Resource replacement ledger](resources/ledger.json) · [Replacement verification](resources/verification.json) · [Base inference probes](base_inference_validation.json)", "",
        "[Smoke results](smoke_summary.json) · [Trained checkpoint-500 replays](checkpoint_500_inference_summary.json) · [Regression test log](full_tests_final.log)", "",
        "Loss curves use each experiment's own labels and normalization. They do not establish robot success rates or rank control modes.", "",
        "Raw source L is named T only in derived datasets/configs/models. Source HDF5 files are read-only; each episode manifest records its original path and SHA256. Calibration is q minus the side-specific full-open baseline, without stroke rescaling or clipping negative noise.", "",
        "Each dataset/profile directory holds conversion_validation.json, source_manifest.json, norm_stats partials, ready.json, saved deployment contract, and an offline_request.npz. The three modes share identical source anchors and hard-linked lossless videos.", ""]
    if mixed:
        lines=[line.replace(" · [Base inference probes](base_inference_validation.json)","") for line in lines]
        lines += ["", "Mixed training samples uniformly from 58,756 valid windows: Blue 34,986 and T 23,770. Both task instructions are preserved; each mode has newly fitted mixed normalization statistics.", ""]
    (evidence_root / "report.md").write_text("\n".join(lines))
    print(json.dumps({accepted_key:summary[accepted_key],"runs":[{"profile":r["profile"],"step":r.get("metrics",{}).get("global_step"),"status":r["formal_status"],"accepted":r["accepted"]} for r in reports]}),flush=True)
    return summary


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--credentials",type=Path)
    p.add_argument("--mixed",action="store_true")
    args=p.parse_args()
    evidence_root=REPO / "outputs/0907_bluet_mixed_preparation" if args.mixed else EVIDENCE
    evidence_root.mkdir(parents=True,exist_ok=True)
    with (evidence_root / "report.lock").open("a") as lock:
        try:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:
            print("Another report update is in progress",flush=True)
        else:
            report(args.credentials,mixed=args.mixed)
