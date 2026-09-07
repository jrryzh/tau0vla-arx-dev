"""Versioned Blue/T contracts. Raw HDF5 feedback is never an action command."""
from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import ClassVar

import numpy as np

from .layout import ARX_LIFT2S_JOINT_NAMES, ArxLift2sUnified

VERSION = "arx-open-baseline-v1"
PROTOCOL = "arx-calibrated-v3"
MODES = ("joint-vr", "joint-feedback", "eef-vr")
POSE_INDICES = [*range(6), *range(7, 13)]
GRIP_INDICES = [6, 13]
POSE_NAMES = [f"{side}_{axis}" for side in ("left", "right") for axis in ("x", "y", "z", "roll", "pitch", "yaw")]
POSE_CONVENTION = {"position_unit": "meter", "angle_unit": "radian", "rpy_convention": "XYZ extrinsic; R=Rz(yaw)@Ry(pitch)@Rx(roll)", "model_rotation": "rot6d", "returned_rotation": "quaternion_xyzw", "action_transform": "relative_to_current_eef_state"}


def estimate_open(q, vr) -> dict:
    """Use each contiguous open command plateau's final ten feedback frames."""
    q, vr = np.asarray(q, dtype=np.float64), np.asarray(vr, dtype=np.float64)
    if q.ndim != 1 or vr.shape != q.shape or not np.isfinite([q, vr]).all():
        raise ValueError("feedback/VR must be finite aligned vectors")
    edges = np.diff(np.r_[False, vr >= 4.99, False].astype(int))
    accepted = []
    for start, stop in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
        if stop - start < 30:
            continue
        tail = q[stop-10:stop]
        spread = float(np.percentile(tail, 90) - np.percentile(tail, 10))
        shift = float(abs(np.median(tail[:5]) - np.median(tail[5:])))
        if spread <= .003 and shift <= .0015:
            accepted.append({"start": int(start), "stop_exclusive": int(stop), "feedback_start": int(stop-10), "median": float(np.median(tail)), "p90_minus_p10": spread, "half_median_difference": shift})
    if not accepted:
        raise ValueError("no qualifying full-open plateau")
    values = [row["median"] for row in accepted]
    if np.ptp(values) > .02:
        raise ValueError(f"full-open plateau baseline drift exceeds 0.02: {np.ptp(values):.6g}")
    return {"algorithm": VERSION, "baseline": float(np.percentile(values, 10)), "plateau_spread": float(np.ptp(values)), "platforms": accepted}


def calibrate_joint(raw_joint, open_baselines):
    raw = np.asarray(raw_joint, dtype=np.float64)
    baselines = np.asarray(open_baselines, dtype=np.float64)
    if raw.shape[-1:] != (14,) or baselines.shape != (2,) or not np.isfinite(raw).all() or not np.isfinite(baselines).all():
        raise ValueError("expected finite joint feedback [...,14] and two open baselines")
    result = raw.copy()
    result[..., GRIP_INDICES] -= baselines
    return result.astype(np.float32)


def labels(qpos, eef, poscmd, baselines, mode):
    if mode not in MODES:
        raise ValueError(mode)
    qpos, eef, poscmd = (np.asarray(x) for x in (qpos, eef, poscmd))
    if qpos.ndim != 2 or qpos.shape[1] != 14 or not (qpos.shape == eef.shape == poscmd.shape) or not all(np.isfinite(x).all() for x in (qpos, eef, poscmd)):
        raise ValueError("source qpos/eef/poscmd must be finite aligned [frames,14]")
    if np.any((poscmd[:, GRIP_INDICES] < 0) | (poscmd[:, GRIP_INDICES] > 5)):
        raise ValueError("VR gripper command outside [0,5]")
    indices = np.arange(0, len(qpos), 2)[:-1]
    state = calibrate_joint(qpos[indices], baselines)
    action = calibrate_joint(qpos[indices+2], baselines)
    if mode != "joint-feedback":
        action[:, GRIP_INDICES] = 1 - poscmd[indices][:, GRIP_INDICES] / 5
    if mode == "eef-vr":
        state = np.concatenate([state, eef[indices][:, POSE_INDICES]], axis=-1).astype(np.float32)
        action = np.concatenate([action, poscmd[indices][:, POSE_INDICES]], axis=-1).astype(np.float32)
    return indices, state, action


def contract(mode):
    if mode not in MODES:
        raise ValueError(mode)
    return {"contract_version": VERSION, "protocol_version": PROTOCOL, "control_mode": mode.split("-")[0], "experiment": mode,
            "action_semantics": "calibrated_" + mode.replace("-", "_"), "action_offset_frames": None,
            "component_source_offsets": {"state": 0, "arm_action": 0 if mode == "eef-vr" else 2, "gripper_action": 2 if mode == "joint-feedback" else 0},
            "offset_unit": "source_frame", "fps": 30, "source_fps": 60, "temporal_stride": 2, "horizon": 30,
            "state_gripper": "raw_feedback_minus_episode_side_open_baseline",
            "gripper_action": "calibrated_feedback_position" if mode == "joint-feedback" else "vr_closure_intent_0_open_1_closed",
            "arm_action_source": "action_poscmd" if mode == "eef-vr" else "observations/qpos",
            "eef_state_source": "observations/eef", "pose_convention": POSE_CONVENTION,
            "time_alignment": "collection_index; no inferred latency compensation", "robot_client_adapted": False}


def validate_calibrated_contract(root, mode):
    root = Path(root)
    sidecar = json.loads((root / "meta/arx.json").read_text())
    info = json.loads((root / "meta/info.json").read_text())
    for key, expected in contract(mode).items():
        if sidecar.get(key) != expected:
            raise ValueError(f"calibrated ARX contract mismatch: {key}")
    dim = 26 if mode == "eef-vr" else 14
    names = list(ARX_LIFT2S_JOINT_NAMES) + (POSE_NAMES if dim == 26 else [])
    for key in ("observation.state", "action"):
        if info["features"][key]["shape"] != [dim] or info["features"][key]["names"] != names:
            raise ValueError(f"calibrated ARX layout mismatch: {key}")
    if info["fps"] != 30 or not sidecar.get("validation_passed"):
        raise ValueError("calibrated data must pass complete conversion validation")
    for camera in ("head", "left_wrist", "right_wrist"):
        if f"observation.images.{camera}" not in info["features"]:
            raise ValueError(f"missing camera {camera}")
    return sidecar


def inline_eef(sample):
    updated = dict(sample)
    for role in ("state", "action"):
        raw = np.asarray(updated[f"_{role}_raw"], dtype=np.float32)
        if raw.shape[-1] != 26 or not np.isfinite(raw).all():
            raise ValueError("calibrated EEF native vector requires finite 14 joint/gripper + 12 pose")
        updated[f"_eef_{role}_raw"] = raw[..., 14:26]
    return updated


@dataclasses.dataclass(frozen=True)
class ArxCalibratedJoint(ArxLift2sUnified):
    robot_name: ClassVar[str] = "arx_calibrated_joint_v1"
    _unified_registry_key: ClassVar[str] = "arx_calibrated_joint_v1"


@dataclasses.dataclass(frozen=True)
class ArxCalibratedEEF(ArxLift2sUnified):
    robot_name: ClassVar[str] = "arx_calibrated_eef_v1"
    _unified_registry_key: ClassVar[str] = "arx_calibrated_eef_v1"

    def _eef_provider(self):
        return inline_eef


ARX_CALIBRATED_UNIFIED_CLASSES = {c.robot_name: c for c in (ArxCalibratedJoint, ArxCalibratedEEF)}
