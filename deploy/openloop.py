"""Open-loop policy evaluation CLI.

Per index: ``policy.infer(payload)`` → ``align_actions_for_evaluation``
→ compare against native-space GT from ``repack_action_target``. Inference
goes through the *deploy* encoder (``encode_payload`` inside ``infer``),
so train/deploy drift shows up in the numbers. Old ckpts without
``policy_manifest.json`` are auto-retrofit from ``run_spec.json``.

Usage::

    python deploy/openloop.py --ckpt <ckpt> --out-dir <dir> --max-inferences 100
    python deploy/openloop.py --ckpt <ckpt> --no-plot            # metrics only
    python deploy/openloop.py --ckpt <ckpt> --route <leaf> --config <other>
"""

from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path

_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
del _sys, _Path

import argparse
import importlib
import logging
from pathlib import Path
from typing import Any, Dict

import numpy as np

from deploy._bootstrap import (
    discover_checkpoint_config_modules,
    ensure_configs_registered,
    ensure_policy_manifest,
)
from tau0_vla.data import (
    FinchDataLoader,
    action_slices,
    align_actions_for_evaluation,
    load_checkpoint_spec,
    unified_native_gt_from_raw,
)
from tau0_vla.data.config import get_config
from tau0_vla.data.source import _load_field_descriptions

logger = logging.getLogger(__name__)


# ─── Dataset unwrap + raw → canonical obs / native GT ───────────────────
# ``canonical_obs_from_raw`` + ``native_gt_from_raw`` aren't strictly
# openloop-specific — any offline tool with a raw lerobot dict in hand can
# reuse them — but they only make sense paired with the dataset iteration
# below, so keeping them here avoids a fifth file. A replay bridge that
# needs the same behavior can import them directly.

def _unwrap_to_lerobot_dataset(dataset: Any) -> Any:
    """Descend ``MappedDataset._dataset`` to the raw lerobot leaf."""
    current = dataset
    for _ in range(8):
        inner = getattr(current, "_dataset", None)
        if inner is None:
            return current
        current = inner
    raise AttributeError(f"Dataset chain too deep from {type(dataset).__name__}")


def canonical_obs_from_raw(raw: Dict[str, Any], config) -> Dict[str, Any]:
    """raw lerobot dict → ``{prompt, images, state, meta}``. This is the
    shape a real robot SDK would send to ``policy.infer`` — pre-wrap
    prompt, pre-resize images, pre-augment raw state. ``encode_payload``
    inside ``policy.infer`` applies prompt_template / image resize / state
    pipeline exactly once."""
    repacked = config._repack_raw_sample(raw)

    def _as_numpy(value: Any) -> np.ndarray:
        """Normalize LeRobot tensor/array values for the safe NPZ wire format."""
        if hasattr(value, "detach"):
            value = value.detach()
        if hasattr(value, "cpu"):
            value = value.cpu()
        if hasattr(value, "numpy"):
            value = value.numpy()
        return np.asarray(value)

    def _image_as_hwc(image: Any) -> np.ndarray:
        array = _as_numpy(image)
        if array.ndim == 3 and array.shape[0] in (1, 3) and array.shape[-1] not in (1, 3):
            array = np.moveaxis(array, 0, -1)
        return array

    return {
        "prompt": repacked["prompt"],
        "images": {key: _image_as_hwc(image) for key, image in repacked["images"].items()},
        "state":  _as_numpy(repacked["_state_raw"]).astype(np.float32, copy=False).reshape(-1),
        "meta":   {},
    }


def native_gt_from_raw(
    raw: Dict[str, Any], config, field_descriptions, *, native_action_dim: int,
) -> np.ndarray:
    """raw lerobot dict → native-space action chunk, zero-pad stripped to
    ``native_action_dim``. Native dims are pre-Quat2Rot6D /
    pre-RelativeToState — e.g. 7 per EefPose arm, not the 9 of rot6d
    model-space."""
    gt = np.asarray(
        config.repack_action_target(raw, field_descriptions=field_descriptions),
        dtype=np.float32,
    )
    return gt[..., :native_action_dim]


def _episode_index(raw: Dict[str, Any]) -> int:
    """Peek raw ``episode_index`` — used only for plot boundary marks."""
    ep = raw.get("episode_index")
    if ep is None:
        return -1
    try:
        return int(ep) if np.ndim(ep) == 0 else int(np.asarray(ep).item())
    except Exception:
        return -1


# ─── Quat hemisphere handling ───────────────────────────────────────────
# ``align_actions_for_evaluation`` only sign-aligns the first 7 × n_EefPose
# dims, missing dual-arm's second block. We catch the rest in per-step
# alignment + unwrap quat signs along the stitched trajectory.

def _quat_dim_ranges(slices) -> list[tuple[int, int]]:
    ranges: list[tuple[int, int]] = []
    for name, offset, native_dim in slices:
        if name != "eef_pose":
            continue
        if native_dim % 7 != 0:
            raise ValueError(f"eef_pose native dim {native_dim} not a multiple of 7")
        for b in range(native_dim // 7):
            ranges.append((offset + b * 7 + 3, 4))
    return ranges


def _align_pred_quat_signs_to_gt(
    gt: np.ndarray, pred: np.ndarray, quat_ranges: list[tuple[int, int]],
) -> np.ndarray:
    """Per-step: flip ``pred``'s quat when opposite-hemisphere from ``gt``."""
    pred = np.asarray(pred, dtype=np.float32).copy()
    gt = np.asarray(gt, dtype=np.float32)
    for q_start, q_len in quat_ranges:
        gt_q = gt[..., q_start:q_start + q_len]
        pred_q = pred[..., q_start:q_start + q_len]
        dot = np.sum(gt_q * pred_q, axis=-1, keepdims=True)
        pred[..., q_start:q_start + q_len] = np.where(dot < 0, -pred_q, pred_q)
    return pred


def _unwrap_quat_signs_along_axis(
    traj: np.ndarray, quat_ranges: list[tuple[int, int]],
) -> np.ndarray:
    """Time-series continuity: walk ``traj`` forward, flip frames that are
    opposite-hemisphere from the previous. Removes spurious jumps at
    chunk boundaries in the stitched trajectory plot."""
    out = np.asarray(traj, dtype=np.float32).copy()
    if out.ndim == 1 or out.shape[0] < 2:
        return out
    for q_start, q_len in quat_ranges:
        for t in range(1, out.shape[0]):
            q_prev = out[t - 1, q_start:q_start + q_len]
            q_curr = out[t, q_start:q_start + q_len]
            if np.isnan(q_prev).any() or np.isnan(q_curr).any():
                continue
            if float(np.dot(q_prev, q_curr)) < 0.0:
                out[t, q_start:q_start + q_len] = -q_curr
    return out


# ─── Labels + plotting ──────────────────────────────────────────────────

def _action_dim_labels(slices: list[tuple[str, int, int]]) -> list[str]:
    """One label per native-dim slot, driven by the data pipeline's
    ``action_slices``. ``len(result) == sum(native_dim for _, _, native_dim in slices)``
    by construction.

    EefPose is 7 dims per arm (xyz + quat xyzw); Gripper is 1 dim per arm.
    Other components (ArmJoint / Waist / ChassisVelocity / unknowns) get
    ``<name>_<i>`` per dim.
    """
    labels: list[str] = []
    for name, _offset, native_dim in slices:
        if name == "eef_pose":
            if native_dim % 7 != 0:
                raise ValueError(f"eef_pose native_dim {native_dim} not a multiple of 7")
            arms = native_dim // 7
            arm_tags = ["L", "R"][:arms] if arms <= 2 else [f"A{i}" for i in range(arms)]
            for tag in arm_tags:
                labels.extend(f"{tag}_{sub}" for sub in ("x", "y", "z", "qx", "qy", "qz", "qw"))
        elif name == "gripper":
            if native_dim == 2:
                labels.extend(["L_grip", "R_grip"])
            else:
                labels.extend(f"grip_{i}" for i in range(native_dim))
        else:
            labels.extend(f"{name}_{i}" for i in range(native_dim))
    return labels


def _lazy_matplotlib_pyplot():
    # Deferred: ~300 ms saved on --no-plot runs. Agg pinned before pyplot
    # import so headless runs don't try to open a display.
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def _plot_summary(
    *, per_sample_errors: np.ndarray, labels: list[str], out_path: Path,
) -> None:
    mse_per_dim = np.mean(per_sample_errors ** 2, axis=(0, 1))
    dim = mse_per_dim.shape[0]
    # labels length is >= dim (full native), take the prefix that matches
    # the possibly-truncated eval_action_dim.
    labels = labels[:dim]

    plt = _lazy_matplotlib_pyplot()
    fig, ax = plt.subplots(figsize=(max(8.0, dim * 0.3), 3.5))
    ax.bar(range(dim), mse_per_dim, color="tab:red")
    ax.set_xticks(range(dim))
    ax.set_xticklabels(labels, rotation=75, fontsize=7, ha="right")
    ax.set_ylabel("mean MSE")
    ax.set_title(f"per-dim open-loop MSE (n={per_sample_errors.shape[0]} samples × "
                 f"{per_sample_errors.shape[1]} horizon steps)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)


def _plot_trajectory(
    *,
    gt_all: np.ndarray,
    pred_all: np.ndarray,
    anchor_offsets: np.ndarray,
    episode_boundaries: list[int],
    quat_ranges: list[tuple[int, int]],
    labels: list[str],
    out_path: Path,
) -> None:
    """Stitch sequential chunks into one long GT/pred trace per dim."""
    gt_all = np.asarray(gt_all, dtype=np.float32)
    pred_all = np.asarray(pred_all, dtype=np.float32)
    N, H, D = gt_all.shape
    labels = labels[:D]

    anchor_offsets = np.asarray(anchor_offsets, dtype=np.int64)
    total_len = int(anchor_offsets[-1] + H)

    gt_line = np.full((total_len, D), np.nan, dtype=np.float32)
    pred_line = np.full((total_len, D), np.nan, dtype=np.float32)
    for k in range(N):
        gt_line[anchor_offsets[k]: anchor_offsets[k] + H] = gt_all[k]
        pred_line[anchor_offsets[k]: anchor_offsets[k] + H] = pred_all[k]

    # Cross-chunk quat sign unwrap: each infer call can land on the
    # opposite hemisphere; unwrap GT first, then align pred to it.
    if quat_ranges:
        gt_line = _unwrap_quat_signs_along_axis(gt_line, quat_ranges)
        pred_line = _align_pred_quat_signs_to_gt(gt_line, pred_line, quat_ranges)

    cols = min(4, D)
    rows = (D + cols - 1) // cols
    plt = _lazy_matplotlib_pyplot()
    fig, axes = plt.subplots(rows, cols, figsize=(4.0 * cols, 2.2 * rows), squeeze=False)
    t_full = np.arange(total_len)

    for d in range(D):
        ax = axes[d // cols][d % cols]
        ax.plot(t_full, gt_line[:, d], color="tab:blue", linewidth=1.4, label="gt", zorder=3)
        ax.plot(t_full, pred_line[:, d], color="tab:orange",
                linewidth=1.0, linestyle="--", label=f"pred (H={H})", zorder=2)
        for k in range(1, N):
            ax.axvline(int(anchor_offsets[k]), color="tab:orange",
                       linewidth=0.3, alpha=0.25, zorder=1)
        for b in episode_boundaries:
            ax.axvline(b, color="gray", linewidth=0.7, linestyle=":", alpha=0.8, zorder=4)
        ax.set_title(labels[d], fontsize=9)
        ax.tick_params(labelsize=7)
        if d == 0:
            ax.legend(fontsize=7, loc="best")
    for d in range(D, rows * cols):
        axes[d // cols][d % cols].axis("off")
    fig.suptitle(
        f"open-loop trajectory — {N} anchors × H={H} (total {total_len} frames)",
        fontsize=10,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def _print_per_dim_breakdown(
    errors: np.ndarray, gt_all: np.ndarray, pred_all: np.ndarray, labels: list[str],
) -> None:
    """rmse/std: <0.3 captures most signal; ~1.0 ≈ mean-predictor; >1 worse."""
    per_dim_mse = np.mean(errors ** 2, axis=(0, 1))
    per_dim_std = np.sqrt(np.var(gt_all, axis=(0, 1)))
    gt_range = gt_all.max(axis=(0, 1)) - gt_all.min(axis=(0, 1))
    pred_range = pred_all.max(axis=(0, 1)) - pred_all.min(axis=(0, 1))

    print(f"\n  per-dim breakdown (errors vs GT magnitude):")
    print(f"  {'dim':>4}  {'label':<12}  {'gt0[:]':>10}  {'pred0[:]':>10}  "
          f"{'gt_range':>10}  {'pred_range':>10}  {'rmse':>10}  {'rmse/std':>8}")
    for d in range(per_dim_mse.shape[0]):
        rmse = float(np.sqrt(per_dim_mse[d]))
        lbl = labels[d]
        print(f"  {d:>4}  {lbl:<12}  {float(gt_all[0, 0, d]):>10.4f}  {float(pred_all[0, 0, d]):>10.4f}  "
              f"{float(gt_range[d]):>10.4f}  {float(pred_range[d]):>10.4f}  "
              f"{rmse:>10.4f}  {rmse / (float(per_dim_std[d]) + 1e-9):>8.3f}")


# ─── Eval loop ──────────────────────────────────────────────────────────

def run_eval(
    policy,
    *,
    start_idx: int,
    max_inferences: int,
    eval_action_dim: int | None,
    config_override: str | None,
    out_dir: Path | None,
) -> None:
    """Shared eval loop for local + server-backed openloop. ``policy`` needs
    ``.data_spec`` + ``.infer(payload) -> {"actions": ndarray[chunk, dim]}``."""
    # Eval dataset: default to the route's leaf (policy.data_spec), or an
    # explicit --config override. NEVER spec.finch_config_name — that's the
    # DataMix top-level, would mix every leaf's samples and hand the
    # route-specific policy wrong-shape raws.
    dataset_config_name = config_override or policy.data_spec.finch_config_name
    if config_override:
        logger.info("Dataset swap: --config %s (policy: %s)",
                    config_override, policy.data_spec.finch_config_name)
    dataset = FinchDataLoader.from_config_name(
        dataset_config_name, batch_size=1, num_workers=0, shuffle=False
    ).dataset
    raw_dataset = _unwrap_to_lerobot_dataset(dataset)
    config = get_config(policy.data_spec.finch_config_name)
    # GT comes from the dataset's current raw samples → scan the repo.
    # (policy uses the ckpt-frozen copy for its own output shape.)
    field_descriptions = _load_field_descriptions(
        str(Path(config._primary_repo_id()).resolve())
    )

    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)

    slices = action_slices(policy.data_spec)
    native_action_dim = sum(d for _, _, d in slices)
    quat_ranges = _quat_dim_ranges(slices)
    chunk_size = int(policy.data_spec.action_chunk_size)
    stride = chunk_size
    if quat_ranges:
        logger.info("quaternion slots in native action: %s", quat_ranges)

    n = int(max_inferences)
    anchor_offsets = np.arange(n) * stride
    if start_idx + int(anchor_offsets[-1]) >= len(dataset):
        max_n = (len(dataset) - 1 - start_idx) // stride + 1
        n = min(n, max_n)
        anchor_offsets = np.arange(n) * stride

    logger.info(
        "Running %d inferences from idx=%d, chunk_size=%d, frames [0, %d)%s",
        n, start_idx, chunk_size,
        int(anchor_offsets[-1]) + chunk_size,
        f"; plots → {out_dir}" if out_dir is not None else " (no plots)",
    )
    labels = _action_dim_labels(slices)

    collected: list[np.ndarray] = []
    gt_collected: list[np.ndarray] = []
    pred_collected: list[np.ndarray] = []
    episode_idx_collected: list[int] = []
    for step in range(n):
        idx = start_idx + int(anchor_offsets[step])
        raw = raw_dataset[idx]
        if not isinstance(raw, dict):
            raise TypeError(f"Expected lerobot dict, got {type(raw).__name__}")

        obs = canonical_obs_from_raw(raw, config)
        # GT must match pred's layout. Unified-40D routes have no declared
        # components, so ``repack_action_target`` (native_gt_from_raw) would order
        # slots differently and omit grippers — build GT via the unified scatter +
        # the same native gather ``restore_action`` uses for pred instead.
        if policy.data_spec.unified_registry_key is not None:
            gt = unified_native_gt_from_raw(raw, policy.data_spec)
        else:
            gt = native_gt_from_raw(raw, config, field_descriptions,
                                    native_action_dim=native_action_dim)
        pred = policy.infer(obs)["actions"]
        aligned = align_actions_for_evaluation(
            gt, pred, action_components=policy.data_spec.action_components,
        )
        if quat_ranges:
            aligned = _align_pred_quat_signs_to_gt(gt, aligned, quat_ranges)
        if eval_action_dim is not None:
            gt = gt[..., : eval_action_dim]
            aligned = aligned[..., : eval_action_dim]

        collected.append(aligned - gt)
        gt_collected.append(np.asarray(gt, dtype=np.float32))
        pred_collected.append(np.asarray(aligned, dtype=np.float32))
        episode_idx_collected.append(_episode_index(raw))
        if (step + 1) % 10 == 0 or step == n - 1:
            logger.info("  [%d/%d] idx=%d", step + 1, n, idx)

    if not collected:
        logger.warning("No samples evaluated.")
        return

    errors = np.stack(collected, axis=0)
    gt_all = np.stack(gt_collected, axis=0)
    pred_all = np.stack(pred_collected, axis=0)

    if out_dir is not None:
        _plot_summary(per_sample_errors=errors, labels=labels,
                      out_path=out_dir / "summary_mse.png")
        ep_ids = np.asarray(episode_idx_collected)
        episode_boundaries = [
            int(anchor_offsets[i]) for i in range(1, len(ep_ids)) if ep_ids[i] != ep_ids[i - 1]
        ]
        _plot_trajectory(
            gt_all=gt_all, pred_all=pred_all,
            anchor_offsets=anchor_offsets[:n],
            episode_boundaries=episode_boundaries,
            quat_ranges=quat_ranges, labels=labels,
            out_path=out_dir / "trajectory.png",
        )

    header = f"\n=== openloop results (n={n}"
    if out_dir is not None:
        header += f", plots → {out_dir}"
    header += ") ==="
    print(header)
    _print_per_dim_breakdown(errors, gt_all, pred_all, labels)
    print(f"  mean MSE {float(np.mean(errors ** 2)):.6f}")
    print(f"  mean  L1 {float(np.mean(np.abs(errors))):.6f}")
    if out_dir is not None:
        print(f"  per-dim summary: {out_dir / 'summary_mse.png'}")
        print(f"  trajectory:      {out_dir / 'trajectory.png'}")


# ─── CLI ────────────────────────────────────────────────────────────────

def main() -> None:
    args = _parse_args()
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

    ensure_configs_registered()
    ensure_policy_manifest(args.ckpt, verbose=True)
    discover_checkpoint_config_modules(args.ckpt, verbose=True)

    spec = load_checkpoint_spec(args.ckpt, route=args.route)
    if spec.finch_config_name is None:
        raise ValueError(
            f"{args.ckpt}/policy_manifest.json has no finch_config_name; "
            "openloop needs the training data contract to replay GT."
        )
    policy_cls = getattr(importlib.import_module(args.policy_module), args.policy_class)
    policy = policy_cls.from_checkpoint(args.ckpt, route=spec.route, device=args.device)

    run_eval(
        policy,
        start_idx=args.start_idx,
        max_inferences=args.max_inferences,
        eval_action_dim=args.eval_action_dim,
        config_override=args.config,
        out_dir=Path(args.out_dir) if args.out_dir is not None else None,
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ckpt", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=None, help="Required unless --no-plot.")
    parser.add_argument("--no-plot", action="store_true")
    parser.add_argument("--route", default=None, help="Leaf config for multi-config DataMix ckpts.")
    parser.add_argument("--config", default=None, help="Override eval dataset config.")
    parser.add_argument("--policy-module", default="deploy.policy")
    parser.add_argument("--policy-class", default="Tau0VLAPolicy")
    parser.add_argument("--max-inferences", type=int, default=20)
    parser.add_argument("--start-idx", type=int, default=0)
    parser.add_argument("--device", default=None)
    parser.add_argument("--eval-action-dim", type=int, default=None,
                        help="Truncate pred/GT to first N dims before metrics.")
    args = parser.parse_args()
    if not args.no_plot and args.out_dir is None:
        parser.error("--out-dir is required unless --no-plot is set")
    return args


if __name__ == "__main__":
    main()
