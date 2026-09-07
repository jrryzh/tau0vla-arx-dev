#!/usr/bin/env python3
"""Read-only endpoint and reset audit; emits diagnostics, never calibration for serving."""
import argparse
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np


def runs(mask, minimum=5):
    edges = np.diff(np.r_[False, mask, False].astype(int))
    return [(int(a), int(b)) for a, b in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)) if b-a >= minimum]


def endpoint(q, command, closed):
    extreme = q.max() if closed else q.min()
    # Require near-extreme feedback, small adjacent steps and the endpoint input.
    stable = np.r_[False, abs(np.diff(q)) <= .001]
    mask = (abs(q-extreme) <= .002) & stable & np.isclose(command, 0 if closed else 5, atol=1e-5)
    intervals = runs(mask)
    if not intervals:
        raise ValueError("No stable endpoint plateau of at least five frames")
    samples = np.concatenate([q[a:b] for a, b in intervals])
    return {"median": float(np.median(samples)), "min": float(samples.min()), "max": float(samples.max()),
            "frames": int(len(samples)), "runs": [{"start": a, "stop_exclusive": b,
            "median": float(np.median(q[a:b])), "span": float(np.ptp(q[a:b]))} for a, b in intervals]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("data/gripper校准"))
    parser.add_argument("--out", type=Path, default=Path("outputs/0906_gripper_calibration"))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    result = {"source": str(args.source.resolve()), "endpoint_method": {
        "near_extreme_tolerance": .002, "max_adjacent_step": .001, "minimum_contiguous_frames": 5,
        "input_endpoint": {"open": 5, "closed": 0},
        "orientation_evidence": "Wrist images inspected: lower feedback is open, higher feedback is empty closed.",
        "note": "Frame rate is metadata only; no timestamp or reset marker is available. Input topic is not verified motor command."}, "episodes": []}
    arrays = []
    for path in sorted(args.source.glob("episode_*.hdf5"), key=lambda p: int(p.stem.split("_")[-1])):
        before = path.stat()
        with path.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        with h5py.File(path, "r") as f:
            q = f["observations/qpos"][()]
            a = f["action"][()]
            cmd = f["action_poscmd"][()]
            assert q.shape == a.shape == cmd.shape and q.shape[1] == 14
            assert all(np.isfinite(x).all() for x in (q, a, cmd))
            attrs = {k: v.item() if isinstance(v, np.generic) else v for k, v in f.attrs.items()}
        after = path.stat()
        assert (before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns), "Source changed during read"
        row = {"file": path.name, "sha256": digest, "bytes": before.st_size, "frames": len(q), "attrs": attrs, "sides": {}}
        for side, i in (("left", 6), ("right", 13)):
            x, c = q[:, i], cmd[:, i]
            opened, closed = endpoint(x, c, False), endpoint(x, c, True)
            row["sides"][side] = {"raw_min": float(x.min()), "raw_max": float(x.max()),
                "raw_span": float(np.ptp(x)), "open": opened, "closed": closed,
                "stroke": closed["median"]-opened["median"],
                "source_action_threshold_mismatches": int(np.count_nonzero(a[:, i] != np.where(x > -2.1, 0, x))),
                "poscmd_min": float(c.min()), "poscmd_max": float(c.max())}
        result["episodes"].append(row)
        arrays.append((q, cmd))
    assert result["episodes"], "No source files"
    result["comparison"] = {}
    for side in ("left", "right"):
        values = [r["sides"][side] for r in result["episodes"]]
        o = np.array([v["open"]["median"] for v in values])
        c = np.array([v["closed"]["median"] for v in values])
        d = c-o
        result["comparison"][side] = {"open_shift_from_episode0": (o-o[0]).tolist(),
            "closed_shift_from_episode0": (c-c[0]).tolist(), "stroke_median": float(np.median(d)),
            "stroke_min": float(d.min()), "stroke_max": float(d.max()), "stroke_spread": float(np.ptp(d)),
            "stroke_spread_percent": float(np.ptp(d)/np.median(d)*100), "closed_endpoint_spread": float(np.ptp(c)),
            "open_endpoint_spread": float(np.ptp(o)),
            "closed_only_with_episode0_stroke_open_residual": (d-d[0]).tolist()}
    (args.out/"audit.json").write_text(json.dumps(result, ensure_ascii=False, indent=2)+"\n")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(len(arrays), 2, figsize=(14, 3*len(arrays)), squeeze=False, constrained_layout=True)
    for n, (row, (q, cmd)) in enumerate(zip(result["episodes"], arrays)):
        for col, (side, i) in enumerate((("left", 6), ("right", 13))):
            ax = axes[n, col]
            ax.plot(q[:, i], label="feedback", color="C0")
            for label, color in (("open", "C2"), ("closed", "C3")):
                value = row["sides"][side][label]
                ax.axhline(value["median"], color=color, linestyle=":", label=f"{label} plateau")
                for run in value["runs"]:
                    ax.plot(np.arange(run["start"], run["stop_exclusive"]), q[run["start"]:run["stop_exclusive"], i], color=color, linewidth=3)
            twin = ax.twinx()
            twin.plot(cmd[:, i], "--", color="grey", alpha=.5, label="recorded PosCmd")
            twin.set(ylim=(-.2, 5.2), ylabel="recorded PosCmd gripper")
            ax.set(title=f'{row["file"]} / {side}', xlabel="source frame", ylabel="raw feedback")
            ax.legend(loc="center right", fontsize=8)
    fig.savefig(args.out/"feedback_and_input.png", dpi=160)
    fig.savefig(args.out/"feedback_and_input.pdf")
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
    for ax, side in zip(axes, ("left", "right")):
        for label in ("open", "closed"):
            raw = np.array([r["sides"][side][label]["median"] for r in result["episodes"]])
            ax.plot(np.arange(len(raw)), raw-raw[0], "o-", label=label+" endpoint shift")
        ax.set(title=f"{side}: endpoint shifts relative to episode 0", xlabel="episode", ylabel="raw position shift", xticks=np.arange(len(arrays)))
        ax.legend()
    fig.savefig(args.out/"endpoint_shifts.png", dpi=160)
    fig.savefig(args.out/"endpoint_shifts.pdf")
    plt.close(fig)

    lines = ["# 新采集夹爪校准数据审计", "", f'覆盖 {len(arrays)} 条记录、{sum(r["frames"] for r in result["episodes"])} 帧。原始数据只读，未改变训练或部署。', "",
        "低反馈端为打开，高反馈端为空夹闭合；已抽查腕部视频。端点使用靠近极值且已稳定的连续片段中位数，排除过渡帧；详见 audit.json 的筛选参数及帧范围。", "",
        "| 文件 | 帧数 | 侧 | 打开反馈 | 闭合反馈 | 行程 |", "|---|---:|---|---:|---:|---:|"]
    for row in result["episodes"]:
        for side, v in row["sides"].items():
            lines.append(f'| {row["file"]} | {row["frames"]} | {side} | {v["open"]["median"]:.6f} | {v["closed"]["median"]:.6f} | {v["stroke"]:.6f} |')
    lines += ["", "## 跨记录比较", ""]
    for side, v in result["comparison"].items():
        lines.append(f'- {side}: 闭合端点最大差 {v["closed_endpoint_spread"]:.6f}，打开端点最大差 {v["open_endpoint_spread"]:.6f}；行程中位数 {v["stroke_median"]:.6f}，行程最大差 {v["stroke_spread"]:.6f}（{v["stroke_spread_percent"]:.4f}%）。')
        lines.append(f'  以 episode 0 行程固定、仅更新闭合点时，打开端点残差：{v["closed_only_with_episode0_stroke_open_residual"]}。此为本批端点一致性检查，不是独立部署验证。')
    lines += ["", "## 解释与使用边界", "",
        "- 本批记录之间存在小幅反馈基线变化，开闭端点基本同幅平移，支持本台机器、当前设置下的固定行程加会话零点校准。未复现历史数据中约 6 或更大的数值区间切换。不能据此排除其他 reset 类型、断电或驱动版本产生大幅变化。",
        "- 左右行程独立，不能共享一个数值。每条内部两次闭合的稳定程度不同，过短闭合不能直接当作可靠端点；实际校准需等待反馈稳定并采集一段平台。",
        "- 新增 action_poscmd 来自 /ARX_VR_L_filtered 和 /ARX_VR_R_filtered。视频及平台对应关系为该输入 5=打开、0=闭合；这不证明电机底层 joint_pos 控制接口也接受 0/5。",
        "- 原 action 仍逐帧满足反馈 > -2.1 则写 0、否则复制反馈。反馈与输入命令有不同数值语义，必须分别规范化。",
        "- 状态规范化可用 u=(q-open)/(closed-open)，0=打开、1=闭合。固定每侧有符号行程 D 后，闭合校准给出 c，打开端点为 c-D。控制输出须按实际部署接口的命令契约映射。",
        "- 历史数据只有在能够定位同一标定会话的已知空夹开或闭端点时，才能结合经确认适用的行程推导映射；本批参数不能直接套用于所有历史机器和记录段。",
        "- HDF5 无 reset 事件、重启类型或逐帧时间戳；因果判断依赖用户对采集流程的说明，横轴使用源帧号。",
        "", "## 产物", "", "- [数值、平台帧范围及源文件 SHA256](audit.json)",
        "- [反馈与控制输入曲线](feedback_and_input.png)", "- [开闭端点平移对比](endpoint_shifts.png)",
        "", "复现：`.venv/bin/python scripts/audit_arx_gripper_calibration.py`", ""]
    (args.out/"report.md").write_text("\n".join(lines))
    print(json.dumps({"episodes": len(arrays), "frames": sum(r["frames"] for r in result["episodes"]), "comparison": result["comparison"]}, indent=2))


if __name__ == "__main__":
    main()
