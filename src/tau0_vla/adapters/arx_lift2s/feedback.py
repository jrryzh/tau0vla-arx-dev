"""Feedback-only open-platform calibration, distinct from VR-certified v1."""
from __future__ import annotations

import dataclasses
import hashlib
import json
from pathlib import Path
from typing import ClassVar

import numpy as np

from .calibrated import calibrate_joint
from .layout import ARX_LIFT2S_JOINT_NAMES, ArxLift2sUnified


VERSION = "arx-feedback-open-v1"
PROTOCOL = "arx-feedback-v4"
BODY = "arx_feedback_joint_v1"
PARAMETERS = {
    "window_frames": 15,
    "half_frames": 7,
    "max_p90_p10": 0.003,
    "max_half_median_difference": 0.0015,
    "low_percentile": 1,
    "low_band": 0.05,
    "baseline_percentile": 10,
    "max_platform_drift": 0.02,
}


def contract() -> dict:
    return {
        "contract_version": VERSION,
        "protocol_version": PROTOCOL,
        "control_mode": "joint",
        "experiment": "joint-feedback",
        "action_semantics": "feedback_open_joint_feedback",
        "action_offset_frames": None,
        "component_source_offsets": {"state": 0, "arm_action": 1, "gripper_action": 1},
        "offset_unit": "uploaded_30fps_frame",
        "fps": 30,
        "horizon": 30,
        "state_gripper": "raw_feedback_minus_episode_side_open_baseline",
        "gripper_action": "calibrated_feedback_position",
        "arm_action_source": "uploaded_next_frame_feedback",
        "calibration_evidence": (
            "feedback stable low platforms; full-open direction confirmed by operator; "
            "no VR evidence"
        ),
        "time_alignment": "uploaded collection index; no extra shift or latency compensation",
        "robot_client_adapted": False,
    }


def estimate_open(values) -> dict:
    samples = np.asarray(values, dtype=np.float64)
    if samples.ndim != 1 or len(samples) < 15 or not np.isfinite(samples).all():
        raise ValueError("finite feedback sequence of at least 15 frames required")
    windows = np.lib.stride_tricks.sliding_window_view(samples, 15)
    medians = np.median(windows, axis=1)
    spread = np.percentile(windows, 90, axis=1) - np.percentile(windows, 10, axis=1)
    shift = np.abs(
        np.median(windows[:, :7], axis=1) - np.median(windows[:, -7:], axis=1)
    )
    cutoff = float(np.percentile(samples, 1) + 0.05)
    indices = np.flatnonzero((spread <= 0.003) & (shift <= 0.0015) & (medians <= cutoff))
    result = {
        "algorithm": VERSION,
        "parameters": PARAMETERS,
        "low_band_upper": cutoff,
        "platforms": [
            {
                "start": int(index),
                "stop_exclusive": int(index + 15),
                "median": float(medians[index]),
                "p90_minus_p10": float(spread[index]),
                "half_median_difference": float(shift[index]),
            }
            for index in indices
        ],
    }
    if not len(indices):
        return {**result, "accepted": False, "reason": "no_platform"}
    drift = float(np.ptp(medians[indices]))
    return {
        **result,
        "accepted": drift <= 0.02,
        "reason": None if drift <= 0.02 else "platform_drift",
        "baseline": float(np.percentile(medians[indices], 10)),
        "platform_spread": drift,
    }


@dataclasses.dataclass(frozen=True)
class ArxFeedbackJoint(ArxLift2sUnified):
    robot_name: ClassVar[str] = BODY
    _unified_registry_key: ClassVar[str] = BODY


ARX_FEEDBACK_UNIFIED_CLASSES = {BODY: ArxFeedbackJoint}


def validate_contract(root):
    root = Path(root)
    metadata = json.loads((root / "meta/arx.json").read_text())
    info = json.loads((root / "meta/info.json").read_text())
    if any(metadata.get(key) != value for key, value in contract().items()) or not metadata.get(
        "validation_passed"
    ):
        raise ValueError("feedback calibration contract not validated")
    if info["fps"] != 30:
        raise ValueError("30fps required")
    for key in ("observation.state", "action"):
        if info["features"][key]["shape"] != [14] or info["features"][key]["names"] != list(
            ARX_LIFT2S_JOINT_NAMES
        ):
            raise ValueError("feedback layout mismatch")
    return metadata


def make_config(config_file, name):
    from tau0_vla.data import FrameFilter
    from tau0_vla.data.modalities import ArmJoint, Gripper, Image, Prompt
    from tau0_vla.data.modalities.image import ColorJitter, ResizeWithPad

    directory = Path(config_file).resolve().parent
    root = directory.parents[1] / "data/0908_feedback_v1" / name
    validate_contract(root)
    stats = directory / "norm_stats.json"
    provenance = json.loads(stats.read_text())["calibrated_provenance"]
    expected = {
        "dataset": name,
        "contract": contract(),
        "source_manifest_sha256": hashlib.sha256((root / "meta/arx.json").read_bytes()).hexdigest(),
    }
    if provenance != expected:
        raise ValueError("statistics provenance mismatch")
    transforms = [
        ColorJitter(prob=0.33, brightness=0.3, contrast=0.4, saturation=0.5, hue=0.03),
        ResizeWithPad(224, 224),
    ]
    return ArxFeedbackJoint(
        repo_id=str(root),
        images=[Image(camera, transforms=transforms) for camera in ("head", "left_wrist", "right_wrist")],
        prompt=Prompt(
            template=(
                "You are controlling a robot.\n"
                "Robot type: ARX LIFT2s\n"
                "Control mode: joint\n"
                "Whole-body control: disabled\n"
                "Task: {instruction}"
            )
        ),
        filter_by_segments=False,
        frame_filter=FrameFilter(positive=(), negative=()),
        state=[ArmJoint(normalize="none"), Gripper(normalize="none")],
        action=[ArmJoint(normalize="none"), Gripper(normalize="none")],
        action_horizon=30,
        action_semantics=contract()["action_semantics"],
        action_offset_frames=None,
        deployment_contract=contract(),
        state_padding_dim=40,
        action_padding_dim=40,
        source_kwargs={"video_backend": "pyav"},
        norm_stats_path=str(stats),
        return_all_norm_forms=True,
    )


__all__ = [
    "ARX_FEEDBACK_UNIFIED_CLASSES",
    "BODY",
    "PARAMETERS",
    "PROTOCOL",
    "VERSION",
    "ArxFeedbackJoint",
    "contract",
    "estimate_open",
    "make_config",
    "validate_contract",
]
