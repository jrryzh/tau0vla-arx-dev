"""Native 14D layout for the dual-arm ARX LIFT2s."""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import ClassVar

from tau0_vla.data.robots.base import RobotConfig
from tau0_vla.data.robots.unified import _UnifiedMixin

ARX_LIFT2S_JOINT_NAMES = tuple(
    [f"left_j{i}" for i in range(6)]
    + ["left_gripper"]
    + [f"right_j{i}" for i in range(6)]
    + ["right_gripper"]
)
EXPECTED_ACTION_SEMANTICS = "state_t_plus_1"
EXPECTED_ACTION_OFFSET_FRAMES = 1
EXPECTED_FPS = 30


def validate_dataset_contract(repo_id: str | Path) -> dict:
    """Fail closed unless a local LeRobot repo has the exact ARX contract."""
    root = Path(repo_id)
    info_path = root / "meta" / "info.json"
    sidecar_path = root / "meta" / "arx.json"
    if not info_path.is_file() or not sidecar_path.is_file():
        raise ValueError(f"ARX dataset requires {info_path} and {sidecar_path}")

    info = json.loads(info_path.read_text(encoding="utf-8"))
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    if int(info.get("fps", -1)) != EXPECTED_FPS or int(sidecar.get("fps", -1)) != EXPECTED_FPS:
        raise ValueError("ARX dataset must be 30 FPS in both info.json and arx.json")
    if sidecar.get("action_semantics") != EXPECTED_ACTION_SEMANTICS:
        raise ValueError(
            "ARX training refuses action_semantics="
            f"{sidecar.get('action_semantics')!r}; expected {EXPECTED_ACTION_SEMANTICS!r}"
        )
    if int(sidecar.get("action_offset_frames", -1)) != EXPECTED_ACTION_OFFSET_FRAMES:
        raise ValueError("ARX state(t+1) data requires action_offset_frames=1")
    if tuple(sidecar.get("joint_names") or ()) != ARX_LIFT2S_JOINT_NAMES:
        raise ValueError("ARX joint_names do not match the fixed native 14D order")

    features = info.get("features") or {}
    for key in ("observation.state", "action"):
        feature = features.get(key) or {}
        if list(feature.get("shape") or ()) != [14]:
            raise ValueError(f"{key} must have shape [14]")
        if tuple(feature.get("names") or ()) != ARX_LIFT2S_JOINT_NAMES:
            raise ValueError(f"{key} names do not match the fixed native order")
    expected_cameras = {
        "observation.images.head",
        "observation.images.left_wrist",
        "observation.images.right_wrist",
    }
    missing = expected_cameras - set(features)
    if missing:
        raise ValueError(f"ARX dataset is missing camera features: {sorted(missing)}")
    return sidecar


@dataclasses.dataclass(frozen=True)
class ArxLift2s(RobotConfig):
    robot_name: ClassVar[str] = "arx_lift2s"
    repack: ClassVar[dict[str, object]] = {
        "prompt": "task",
        "images": {
            "head": "observation.images.head",
            "left_wrist": "observation.images.left_wrist",
            "right_wrist": "observation.images.right_wrist",
        },
        "state": {
            "raw": "observation.state",
            "semantic": {
                "arm_joint": "state/joint/position",
                "gripper": [
                    "state/left_effector/position",
                    "state/right_effector/position",
                ],
            },
        },
        "action": {
            "raw": "action",
            "semantic": {
                "arm_joint": "action/joint/position",
                "gripper": [
                    "action/left_effector/position",
                    "action/right_effector/position",
                ],
            },
        },
    }

    @classmethod
    def native_dim_labels(cls, *, is_eef: bool, native_dim: int) -> list[str]:
        if is_eef:
            raise NotImplementedError("ARX LIFT2s adapter is joint-only")
        return list(ARX_LIFT2S_JOINT_NAMES[:native_dim])


@dataclasses.dataclass(frozen=True)
class ArxLift2sUnified(_UnifiedMixin, ArxLift2s):
    robot_name: ClassVar[str] = "arx_lift2s_unified"
    _unified_registry_key: ClassVar[str] = "arx_lift2s_14"


ARX_LIFT2S_UNIFIED_CLASSES: dict[str, type] = {
    "arx_lift2s_unified": ArxLift2sUnified,
}
