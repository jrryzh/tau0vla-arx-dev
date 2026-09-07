#!/usr/bin/env python3
"""Audit the 0905 gripper contract without modifying data or training artifacts."""
import argparse
import csv
import importlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import h5py
import numpy as np
import pyarrow.dataset as pads

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO / "src")]
GRIP = [6, 13]
ARMS = [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12]
SOURCES = {
    "datasets": REPO / "data/datasets-0905/pickplace_right_to_bowl",
    "zyp-gyt": REPO / "data/pickplace_zyp_gyt_0905",
}


def quantiles(a):
    return np.quantile(a, [0, .01, .1, .5, .9, .99, 1], axis=0).tolist()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    report = {"quantile_levels": [0, .01, .1, .5, .9, .99, 1], "sources": {}, "conversions": {}, "roundtrips": {}}
    cache, rows = {}, []
    for name, source in SOURCES.items():
        manifest = json.loads((REPO / f"outputs/0905_preparation/0905-{name}-all100/source_manifest.json").read_text())
        for ep in manifest["episodes"]:
            n = ep["source_episode_number"]
            with h5py.File(source / ep["file"], "r") as f:
                q, a = f["observations/qpos"][()], f["action"][()]
                assert np.isfinite(q).all() and np.isfinite(a).all()
            cache[name, n] = q, a
            row = {"source": name, "episode": n, "frames": len(q),
                   "arm_action_not_same_frame_state_values": int(np.count_nonzero(a[:, ARMS] != q[:, ARMS])),
                   "threshold_rule_mismatched_values": int(np.count_nonzero(a[:, GRIP] != np.where(q[:, GRIP] > -2.1, 0, q[:, GRIP])))}
            for side, idx in zip(("left", "right"), GRIP):
                v, command = q[:, idx], a[:, idx]
                row.update({f"{side}_q_min": float(v.min()), f"{side}_q_max": float(v.max()),
                    f"{side}_q_median": float(np.median(v)), f"{side}_q_initial": float(v[0]),
                    f"{side}_source_action_zero_fraction": float(np.mean(command == 0)),
                    f"{side}_state_action_mae": float(np.mean(np.abs(command - v))),
                    f"{side}_max_step": float(np.abs(np.diff(v)).max()),
                    f"{side}_span_after_2s": float(np.ptp(v[120:])),
                    f"{side}_threshold_crossings": int(np.count_nonzero(np.diff((v > -2.1).astype(int))))})
            rows.append(row)
        selected = [r for r in rows if r["source"] == name]
        q = np.concatenate([cache[name, e["source_episode_number"]][0] for e in manifest["episodes"]])
        a = np.concatenate([cache[name, e["source_episode_number"]][1] for e in manifest["episodes"]])
        zeros = a[:, 6] == 0
        report["sources"][name] = {
            "source": str(source), "episodes": len(selected), "frames": len(q),
            "state_gripper_quantiles": quantiles(q[:, GRIP]), "source_action_gripper_quantiles": quantiles(a[:, GRIP]),
            "arm_action_not_same_frame_state_values": sum(r["arm_action_not_same_frame_state_values"] for r in selected),
            "threshold_rule_mismatched_values": sum(r["threshold_rule_mismatched_values"] for r in selected),
            "left_feedback_when_source_action_zero_quantiles": quantiles(q[zeros, 6]),
            "left_feedback_below_minus6_episodes": [r["episode"] for r in selected if r["left_q_min"] < -6],
            "right_episode_span_quantiles": quantiles(np.array([r["right_q_max"] - r["right_q_min"] for r in selected])),
            "right_span_after_2s_quantiles": quantiles(np.array([r["right_span_after_2s"] for r in selected])),
        }
        nominal = [r for r in selected if r["left_q_min"] > -6]
        report["sources"][name]["nominal_baseline_episode_count"] = len(nominal)
        report["sources"][name]["never_reaches_minus0p2_episodes"] = [r["episode"] for r in nominal if r["left_q_max"] < -.2]

    from tau0_vla.data.data_spec import build_unified_action_prefix_encoder, build_unified_state_encoder
    from tau0_vla.adapters.arx_lift2s.deploy_io import restore_native_action
    profiles = json.loads((REPO / "outputs/0905_preparation/campaign_scope.json").read_text())["datasets"]
    for profile in profiles:
        ready = json.loads((REPO / f"outputs/0905_preparation/{profile}/ready.json").read_text())
        name = "datasets" if "datasets" in profile else "zyp-gyt"
        expected_q, expected_a = [], []
        for n in ready["selected_numbers"]:
            q, _ = cache[name, n]
            expected_q.append(q[::2][:-1])
            expected_a.append(q[::2][1:])
        q, a = np.concatenate(expected_q), np.concatenate(expected_a)
        table = pads.dataset(str(Path(ready["dataset"]) / "data"), format="parquet").to_table(columns=["index", "observation.state", "action"]).sort_by([("index", "ascending")])
        actual_q = np.stack(table["observation.state"].to_numpy())
        actual_a = np.stack(table["action"].to_numpy())
        np.testing.assert_array_equal(actual_q, q)
        np.testing.assert_array_equal(actual_a, a)
        report["conversions"][profile] = {"frames": len(q), "all_14d_state_values_exact": True,
            "all_14d_action_values_equal_qpos_source_t_plus_2": True}
        config_name = "arx_lift2s_" + profile.replace("-", "_")
        module = importlib.import_module(f"configs.{config_name}.data")
        config = getattr(module, config_name + "_ft")()
        assembler = config._build_component_assembler(field_descriptions={}, disable_component_normalization=False)
        for suffix in ("", "_20k"):
            run_name = config_name + suffix + "_h200_formal"
            run = REPO / "outputs" / run_name / run_name
            spec_path = next((run / "finch_data_spec").glob("*/spec.json"))
            stats_path = spec_path.with_name("norm_stats.json")
            assert json.loads(stats_path.read_text()) == json.loads((REPO / "configs" / config_name / "norm_stats.json").read_text())
            spec = SimpleNamespace(**json.loads(spec_path.read_text()), norm_stats_path=str(stats_path), artifacts_dir=str(spec_path.parent))
            encode_state = build_unified_state_encoder(spec)
            encode_action = build_unified_action_prefix_encoder(spec)
            errors = []
            for n in ready["selected_numbers"]:
                source_q, _ = cache[name, n]
                last_t = 2 * (len(source_q[::2]) - 31)
                for t in (0, (last_t // 4) * 2, last_t):
                    state = source_q[t]
                    target = source_q[t+2:t+62:2]
                    assert target.shape == (30, 14)
                    trained = assembler({"prompt": "audit", "images": {}, "_state_raw": state, "_action_raw": target})
                    deployed = encode_state(state)
                    np.testing.assert_allclose(trained["state"], deployed["state"], atol=1e-5)
                    np.testing.assert_allclose(trained["action"], encode_action(state, target), atol=1e-5)
                    restored = restore_native_action(trained["action"], spec, state_abs=deployed["state_abs"])
                    error = float(np.max(np.abs(restored-target)))
                    assert error < 1e-5, (run_name, n, t, error)
                    errors.append(error)
            report["roundtrips"][run_name] = {"windows_checked": len(errors), "horizon": 30,
                "training_serving_state_and_action_encoding_match": True, "native_restoration_max_abs_error": max(errors),
                "saved_stats_match_own_config": True}
    report["official_reference"] = {"repo": "https://github.com/ARXroboticsX/ROS2_LIFT_Play",
        "commit": "befc6712f461cf965781f671cf9db948da73778b", "collection": "act/collect.py:162-187",
        "inference_gate": "act/inference.py:246-250,376-386", "machine_version_verified": False}
    report["limits"] = [
        "No robot rollout trace or machine-specific collector/driver was available for this audit.",
        "No source timestamps or calibration metadata are present in the audited manifests.",
        "Both model groups share threshold and baseline issues; these do not alone establish the cause of the performance difference.",
        "Raw HDF5, converted data, normalization, model weights and deployment runtime are never modified by this script."]
    (args.out / "audit.json").write_text(json.dumps(report, indent=2) + "\n")
    with (args.out / "episodes.csv").open("w") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 3, figsize=(16, 8), constrained_layout=True)
    for row, name in enumerate(SOURCES):
        subset = [r for r in rows if r["source"] == name]
        ns = [r["episode"] for r in subset]
        ax = axes[row, 0]
        ax.plot(ns, [r["left_q_min"] for r in subset], label="left episode min")
        ax.plot(ns, [r["left_q_max"] for r in subset], label="left episode max")
        ax.set(title=f"{name}: active left gripper ranges", xlabel="source episode", ylabel="raw position"); ax.legend()
        ax = axes[row, 1]
        ax.plot(ns, [r["right_q_median"] for r in subset], '.-', label="right median")
        ax.set(title=f"{name}: right gripper baselines", xlabel="source episode", ylabel="raw position"); ax.legend()
        n = 2 if name == "datasets" else 24
        q, a = cache[name, n]; idx = np.arange(0, len(q), 2)[:-1]
        ax = axes[row, 2]
        ax.plot(idx/60, q[idx,6], label="observation qpos")
        ax.plot(idx/60, a[idx,6], label="source action (thresholded)", alpha=.75)
        ax.plot(idx/60, q[idx+2,6], '--', label="current training target", alpha=.75)
        ax.axhline(-2.1, color="grey", linestyle=":", label="collector threshold")
        ax.set(title=f"{name}: episode {n}, left", xlabel="nominal seconds (no timestamps)", ylabel="raw position"); ax.legend(fontsize=8)
    fig.savefig(args.out / "gripper_gap.png", dpi=160)
    fig.savefig(args.out / "gripper_gap.pdf")
    print(json.dumps({"source_frames": {k:v["frames"] for k,v in report["sources"].items()},
        "converted_groups_verified": len(report["conversions"]), "model_roundtrips_verified": len(report["roundtrips"]),
        "output": str(args.out)}, indent=2))


if __name__ == "__main__":
    main()
