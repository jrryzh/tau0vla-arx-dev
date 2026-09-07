#!/usr/bin/env python3
"""Write reviewable campaign progress and strict healthy-training acceptance."""
import ast
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import re
import subprocess
import time

import yaml

from prepare_0905_training import PROFILES, REPO


def metrics_from_log(path):
    metrics = []
    for line in path.read_text(errors="replace").replace("\r", "\n").splitlines():
        if "'global_step'" in line:
            try:
                value = ast.literal_eval(line[line.index("{"):line.rindex("}") + 1])
            except (ValueError, SyntaxError):
                continue
            if "loss" not in value or "grad_norm" not in value:
                continue
            metrics.append(value)
    return metrics


def verify_config(run, profile):
    config_name = "arx_lift2s_" + profile.replace("-", "_")
    config = yaml.safe_load((REPO / "configs" / config_name / "train_h200.yaml").read_text())
    actual = json.loads((run / "run_spec.json").read_text())
    for section in ("model_args", "data_args", "training_args"):
        for key, expected in config[section].items():
            if (section, key) == ("training_args", "output_dir"):
                expected = str(run)
            if (section, key) == ("training_args", "report_to") and expected == "none":
                expected = []
            assert actual[section][key] == expected, (profile, section, key, expected, actual[section].get(key))
    assert not actual["training_args"].get("resume_from_checkpoint") or str(run) in actual["training_args"]["resume_from_checkpoint"]
    dataset_profile = profile.removesuffix("-20k")
    expected_root = str(REPO / "data/0905_training" / dataset_profile / "lerobot_v3_30fps_state_t_plus_1")
    specs = list((run / "finch_data_spec").glob("*/spec.json"))
    assert len(specs) == 1
    spec = json.loads(specs[0].read_text())
    assert set(spec["finch_config_names_by_root"]) == {expected_root}
    assert spec["state_dim"] == spec["action_dim"] == 40 and spec["action_chunk_size"] == 30
    assert spec["action_semantics"] == "state_t_plus_1" and spec["action_offset_frames"] == 1
    assert set(spec["cam_keys"]) == {"head", "left_wrist", "right_wrist"}
    saved_stats = json.loads(specs[0].with_name("norm_stats.json").read_text())
    source_stats = REPO / "configs" / ("arx_lift2s_" + dataset_profile.replace("-", "_")) / "norm_stats.json"
    assert saved_stats == json.loads(source_stats.read_text()), f"Normalization differs from the approved dataset: {profile}"


def main():
    evidence = REPO / "outputs/0905_preparation"
    audit_path = evidence / "resources/latest_audit.json"
    audit = json.loads(audit_path.read_text()) if audit_path.exists() else {}
    jobs = {j["job_id"]: j for j in audit.get("jobs", [])}
    audit_fresh = audit_path.exists() and time.time() - audit_path.stat().st_mtime < 120
    rows = []
    for dataset in PROFILES:
        for suffix, target in (("", 10000), ("-20k", 20000)):
            profile = dataset + suffix
            slug = profile.replace("-", "_")
            name = "arx_lift2s_" + slug + "_h200_formal"
            run = REPO / "outputs" / name / name
            state = REPO / "outputs" / ("qzcli_arx_h200_" + slug)
            row = {"profile": profile, "target": target, "run_directory": str(run), "healthy": False,
                "curve": str(state / "dashboard/training_curves.png")}
            for kind in ("smoke", "formal"):
                path = state / f"{kind}_job_id"
                if path.exists():
                    row[kind + "_job_id"] = path.read_text().strip()
            logs = sorted((run / "log").glob("training_log_nodeIdx000_*.txt"))
            if logs:
                metrics = metrics_from_log(logs[-1])
                if metrics:
                    row["latest"] = metrics[-1]
                    row["formal_status"] = jobs.get(row.get("formal_job_id"), {}).get("status", "not_refreshed")
                    passed = sorted({int(m["global_step"]) for m in metrics if int(m["global_step"]) > 500})
                    finite = all(math.isfinite(float(m[k])) for m in metrics for k in ("loss", "grad_norm"))
                    fresh = time.time() - logs[-1].stat().st_mtime < 180
                    successful = row["formal_status"] == "job_succeeded" and int(metrics[-1]["global_step"]) == target
                    if finite and len(passed) >= 2 and audit_fresh and (successful or (row["formal_status"] == "job_running" and fresh)):
                        verify_config(run, profile)
                        marker = state / "healthy_acceptance.json"
                        if not marker.exists() or successful:
                            step = target if successful else 500
                            result = subprocess.run([str(REPO / ".venv/bin/python"), "scripts/validate_checkpoint.py",
                                str(run / f"checkpoint-{step}"), "--world-size", "16", "--expected-step", str(step), "--deployment"],
                                cwd=REPO, capture_output=True, text=True)
                            (state / f"checkpoint_{step}_deployment_validation.log").write_text(result.stdout + result.stderr)
                            if result.returncode == 0:
                                marker.write_text(json.dumps({"profile": profile, "target": target, "validated_step": step,
                                    "job_id": row["formal_job_id"], "observed_later_steps": passed[:2],
                                    "time": datetime.now(timezone.utc).isoformat()}, indent=2) + "\n")
                        if marker.exists():
                            acceptance = json.loads(marker.read_text())
                            row["healthy"] = acceptance["profile"] == profile and acceptance["target"] == target
            rows.append(row)
    report = {"updated_at": datetime.now(timezone.utc).isoformat(), "audit_fresh": audit_fresh,
        "healthy_count": sum(r["healthy"] for r in rows), "total_runs": 10, "runs": rows}
    (evidence / "training_status.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    lines = ["# 0905：五组数据 × 10k／20k 对照训练", "", f"更新时间：{report['updated_at']}", "",
        "每个任务独立从 tau-0-vla-base 初始化，16 张 H200、全局 batch 128；对应 10k／20k 使用相同数据与统计。", "",
        f"已通过正常训练验收：{report['healthy_count']} / 10。", "",
        "| Profile | 进度 | Loss | vla_epoch | 状态 |", "|---|---:|---:|---:|---|"]
    for r in rows:
        m = r.get("latest", {})
        lines.append(f"| {r['profile']} | {m.get('global_step', 0)} / {r['target']} | {m.get('loss', '—')} | {m.get('vla_epoch', '—')} | {'正常，已验收' if r['healthy'] else r.get('formal_status', '准备中')} |")
    lines += ["", "## 数据验收", "", "30 FPS，三路相机，14D 原生状态／动作映射到 unified 40D；action(t)=state(t+1)，action horizon 30。", "",
        "| 数据组 | 轨迹数 | 帧数 | 有效起点 | 每 rank 的 DataLoader batches | 10k／20k 预计 vla_epoch |", "|---|---:|---:|---:|---:|---:|"]
    for profile in PROFILES:
        ready = evidence / profile / "ready.json"
        if ready.exists():
            data = json.loads(ready.read_text())
            batches = data["expected_batches_per_rank"]
            lines.append(f"| {profile} | {data['episodes']} | {data['frames']} | {data['anchors']} | {batches} | {10000/batches:.4f}／{20000/batches:.4f} |")
    lines += ["", "两路损坏的 zyp-gyt-all100 腕部视频已从源 HDF5 重建；修复记录、原文件哈希及逐帧／轨迹边界验证见 "
        f"[视频修复记录]({evidence / '0905-zyp-gyt-all100/video_repair'})。整组通过完整校验后才会提交训练。"]
    ledger_path = evidence / "resources/ledger.json"
    if ledger_path.exists():
        ledger = json.loads(ledger_path.read_text())
        fulfilled = list(ledger["replacements"].values())
        preexisting = sum(r.get("preexisting", False) for r in fulfilled)
        active = sum(jobs.get(r["job_id"], {}).get("status") in {"job_running", "job_queuing"} for r in fulfilled)
        lines += ["", "## 资源账目", "", f"历史需补 36 个，本次实际释放 {len(ledger['stops'])} 个；已记录补位 {len(fulfilled)} 个，其中已有 {preexisting} 个、本次新提交 {len(fulfilled)-preexisting} 个。最近队列审计确认 {active} 个补位处于运行或排队状态。", "",
            "新补位均在十个正式任务提交后提交，优先级 9；训练优先级 10。", "",
            f"[逐项停止／补位账本]({ledger_path}) · [队列审计]({audit_path})"]
    for r in rows:
        lines += ["", f"## {r['profile']}", "", f"正式任务：`{r.get('formal_job_id', '尚未提交')}`；smoke：`{r.get('smoke_job_id', '尚未提交')}`。", "",
            f"[模型目录]({r['run_directory']})", "", f"[Loss 曲线]({r['curve']})"]
    (evidence / "training_status.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({"healthy_count": report["healthy_count"], "runs": [{"profile": r["profile"], "step": r.get("latest", {}).get("global_step", 0), "healthy": r["healthy"]} for r in rows]}), flush=True)


if __name__ == "__main__":
    main()
