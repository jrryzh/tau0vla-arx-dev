"""Action restoration for the ARX LIFT2s state-as-action route."""

from __future__ import annotations

from typing import Sequence

import numpy as np

from tau0_vla.data import action_slices, restore_action


def build_native_action_perm(slices: Sequence[tuple[str, int, int]]) -> list[int]:
    """Map canonical restored slots to L-arm, L-grip, R-arm, R-grip order."""
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
    return [
        by_name[name][0] + index
        for name in ordered
        for index in range(by_name[name][1])
    ]


def restore_native_action(action_inferred, data_spec, *, state_abs) -> np.ndarray:
    """Restore normalized unified actions to the fixed native ARX 14D order."""
    canonical = restore_action(action_inferred, data_spec, state=state_abs)
    perm = build_native_action_perm(action_slices(data_spec))
    return np.asarray(canonical, dtype=np.float32)[..., perm]


__all__ = ["build_native_action_perm", "restore_native_action"]
