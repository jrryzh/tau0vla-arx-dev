"""Live deployment I/O for the dual-arm ARX LIFT2s."""

from __future__ import annotations

import json
from io import BytesIO
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from PIL import Image

from tau0_vla.adapters.arx_lift2s.layout import ARX_LIFT2S_JOINT_NAMES
from tau0_vla.data import action_slices, restore_action


CAMERA_NAMES = ("head", "left_wrist", "right_wrist")
UNIFIED_SIDE_TO_CANONICAL = (
    ("left_arm", "right_arm", "arm_joint"),
    ("left_gripper", "right_gripper", "gripper"),
)
_EXPECTED_STATE_FIELDS = {
    "state/joint/position": [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12],
    "state/left_effector/position": [6],
    "state/right_effector/position": [13],
}


def build_native_action_perm(slices: Sequence[tuple[str, int, int]]) -> list[int]:
    """Map restored unified slots to the fixed ARX 14D native order."""
    by_name = {name: (int(offset), int(dim)) for name, offset, dim in slices}
    expected = {
        "left_arm": 6,
        "left_gripper": 1,
        "right_arm": 6,
        "right_gripper": 1,
    }
    if set(by_name) != set(expected):
        raise ValueError(f"unexpected ARX restored action slices: {sorted(by_name)}")
    for name, width in expected.items():
        if by_name[name][1] != width:
            raise ValueError(f"ARX slice {name!r} must have width {width}")
    ordered = ("left_arm", "left_gripper", "right_arm", "right_gripper")
    return [by_name[name][0] + index for name in ordered for index in range(by_name[name][1])]


def restore_native_action(action_inferred, data_spec, *, state_abs) -> np.ndarray:
    """Restore normalized unified actions to the fixed native ARX 14D order."""
    canonical = restore_action(action_inferred, data_spec, state=state_abs)
    perm = build_native_action_perm(action_slices(data_spec))
    return np.asarray(canonical, dtype=np.float32)[..., perm]


def load_state_field_descriptions(artifacts_dir: str | Path) -> dict[str, Any]:
    path = Path(artifacts_dir) / "field_descriptions.json"
    return json.loads(path.read_text(encoding="utf-8"))["state"]


def state_dim_from_field_descriptions(state_fd: Mapping[str, Any]) -> int:
    """Validate the saved raw ARX state contract and return its native width."""
    if set(state_fd) != set(_EXPECTED_STATE_FIELDS):
        raise ValueError(f"unexpected ARX state fields: {sorted(state_fd)}")
    for key, expected_indices in _EXPECTED_STATE_FIELDS.items():
        indices = list(state_fd[key].get("indices") or ())
        if indices != expected_indices:
            raise ValueError(f"ARX state field {key!r} indices {indices} != {expected_indices}")
    return len(ARX_LIFT2S_JOINT_NAMES)


def decode_jpeg(value: Any) -> np.ndarray:
    """Decode compressed JPEG bytes to an RGB uint8 HWC array."""
    if isinstance(value, np.ndarray) and value.ndim == 3:
        array = np.asarray(value)
        if array.shape[-1] != 3 or array.dtype != np.uint8:
            raise ValueError("ARX image arrays must be uint8 HWC RGB")
        return array
    if isinstance(value, np.ndarray):
        value = np.asarray(value, dtype=np.uint8).reshape(-1).tobytes()
    elif isinstance(value, (bytearray, memoryview)):
        value = bytes(value)
    if not isinstance(value, bytes) or not value:
        raise ValueError("ARX camera payload must contain JPEG bytes")
    with Image.open(BytesIO(value)) as image:
        image.load()
        rgb = image.convert("RGB")
        if rgb.width > 4096 or rgb.height > 4096:
            raise ValueError("ARX camera image dimensions exceed 4096x4096")
        return np.asarray(rgb, dtype=np.uint8)


def _payload_value(raw: Mapping[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in raw:
            return raw[key]
    raise KeyError(keys[0])


def build_payload_adapter(*, cam_keys: Sequence[str], state_fd: Mapping[str, Any], state_dim: int):
    """Build an ARX SDK/NPZ payload adapter for the legacy flat endpoint."""
    validated_dim = state_dim_from_field_descriptions(state_fd)
    if state_dim != validated_dim:
        raise ValueError(f"ARX state width {state_dim} != {validated_dim}")
    if tuple(cam_keys) != CAMERA_NAMES:
        raise ValueError(f"ARX cameras {tuple(cam_keys)!r} != {CAMERA_NAMES!r}")

    def adapt(raw: Mapping[str, Any]) -> dict[str, Any]:
        state = np.asarray(_payload_value(raw, "state", "observation.state"), dtype=np.float32).reshape(-1)
        if state.shape != (state_dim,) or not np.isfinite(state).all():
            raise ValueError(f"ARX state must be a finite {state_dim}-vector")
        image_map = raw.get("images")
        images = {}
        for camera in CAMERA_NAMES:
            if isinstance(image_map, Mapping) and camera in image_map:
                value = image_map[camera]
            else:
                value = _payload_value(raw, camera, f"observation.images.{camera}")
            images[camera] = decode_jpeg(value)
        prompt = str(_payload_value(raw, "prompt", "task", "instruction")).strip()
        if not prompt:
            raise ValueError("ARX prompt must not be empty")
        return {"prompt": prompt, "images": images, "state": state, "meta": dict(raw.get("meta") or {})}

    return adapt


def canonicalize_action_dict(split: dict[str, Any]) -> dict[str, Any]:
    """Merge per-side unified actions into canonical arm/gripper keys."""
    out = dict(split)
    for left_key, right_key, canonical in UNIFIED_SIDE_TO_CANONICAL:
        left = out.pop(left_key, None)
        right = out.pop(right_key, None)
        if left is None and right is None:
            continue
        if left is None or right is None:
            raise ValueError(f"ARX action response has only one of {left_key!r}/{right_key!r}")
        out[canonical] = np.concatenate(
            [np.asarray(left, dtype=np.float32), np.asarray(right, dtype=np.float32)], axis=-1
        ).tolist()
    return out


def build_sdk_action_perm(data_spec, slices) -> list[int]:
    if getattr(data_spec, "unified_registry_key", None) != "arx_lift2s_14":
        raise ValueError("ARX live serving requires unified registry 'arx_lift2s_14'")
    return build_native_action_perm(slices)


def apply_sdk_action_perm(actions, sdk_action_perm: Sequence[int] | None) -> np.ndarray:
    arr = np.asarray(actions, dtype=np.float32)
    if sdk_action_perm is not None:
        arr = arr[..., list(sdk_action_perm)]
    return arr


__all__ = [
    "CAMERA_NAMES",
    "UNIFIED_SIDE_TO_CANONICAL",
    "apply_sdk_action_perm",
    "build_native_action_perm",
    "build_payload_adapter",
    "build_sdk_action_perm",
    "canonicalize_action_dict",
    "decode_jpeg",
    "load_state_field_descriptions",
    "restore_native_action",
    "state_dim_from_field_descriptions",
]
