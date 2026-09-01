from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from tau0_vla.adapters.arx_lift2s import (
    ARX_LIFT2S_JOINT_NAMES,
    ArxLift2sUnified,
    validate_dataset_contract,
)
from tau0_vla.adapters.arx_lift2s.deploy_io import restore_native_action
from tau0_vla.data.modalities import ArmJoint, Gripper


def _config() -> ArxLift2sUnified:
    return ArxLift2sUnified(
        repo_id="unused",
        state=[ArmJoint(normalize="none"), Gripper(normalize="none")],
        action=[ArmJoint(normalize="none"), Gripper(normalize="none")],
        action_horizon=2,
        state_padding_dim=40,
        action_padding_dim=40,
    )


class ArxLayoutTest(unittest.TestCase):
    def test_native_14d_scatter_relative_action_and_masks(self):
        state = np.arange(14, dtype=np.float32)
        action = np.stack([state + 10, state + 20])
        assembler = _config()._build_component_assembler(
            field_descriptions={}, disable_component_normalization=True
        )
        out = assembler(
            {"prompt": "task", "images": {}, "_state_raw": state, "_action_raw": action}
        )

        np.testing.assert_array_equal(out["state"][24:30], state[0:6])
        np.testing.assert_array_equal(out["state"][32:38], state[7:13])
        self.assertEqual(out["state"][18], state[6])
        self.assertEqual(out["state"][19], state[13])
        np.testing.assert_array_equal(out["action"][:, 24:30], [[10] * 6, [20] * 6])
        np.testing.assert_array_equal(out["action"][:, 32:38], [[10] * 6, [20] * 6])
        np.testing.assert_array_equal(out["action"][:, 18], action[:, 6])
        np.testing.assert_array_equal(out["action"][:, 19], action[:, 13])
        active = [18, 19, *range(24, 30), *range(32, 38)]
        np.testing.assert_array_equal(np.flatnonzero(out["state_mask"]), active)
        np.testing.assert_array_equal(np.flatnonzero(out["action_mask"]), active)

    def test_restore_native_action_round_trip(self):
        state_native = np.arange(14, dtype=np.float32)
        target_native = state_native + np.linspace(0.1, 1.4, 14, dtype=np.float32)
        assembler = _config()._build_component_assembler(
            field_descriptions={}, disable_component_normalization=True
        )
        out = assembler(
            {
                "prompt": "task",
                "images": {},
                "_state_raw": state_native,
                "_action_raw": target_native[None],
            }
        )
        stats = {
            "mean": [0.0] * 40,
            "std": [1.0] * 40,
            "q01": [-1.0] * 40,
            "q99": [1.0] * 40,
        }
        with tempfile.TemporaryDirectory() as directory:
            stats_path = Path(directory) / "norm_stats.json"
            stats_path.write_text(
                json.dumps(
                    {
                        "format_version": 2,
                        "norm_stats": {"state": stats, "action": stats},
                        "per_embodiment": {
                            "arx_lift2s_14": {"state": stats, "action": stats}
                        },
                        "config_summary": {},
                    }
                )
            )
            spec = SimpleNamespace(
                unified_registry_key="arx_lift2s_14",
                unified_has_eef=False,
                norm_stats_path=str(stats_path),
            )
            restored = restore_native_action(out["action"], spec, state_abs=out["state"])
        np.testing.assert_allclose(restored[0], target_native, atol=1e-6)


def _valid_metadata(root: Path) -> None:
    features = {
        key: {"shape": [14], "names": list(ARX_LIFT2S_JOINT_NAMES)}
        for key in ("observation.state", "action")
    }
    for camera in ("head", "left_wrist", "right_wrist"):
        features[f"observation.images.{camera}"] = {"dtype": "video"}
    meta = root / "meta"
    meta.mkdir()
    (meta / "info.json").write_text(json.dumps({"fps": 30, "features": features}))
    (meta / "arx.json").write_text(
        json.dumps(
            {
                "fps": 30,
                "action_semantics": "state_t_plus_1",
                "action_offset_frames": 1,
                "joint_names": list(ARX_LIFT2S_JOINT_NAMES),
            }
        )
    )


class ArxContractTest(unittest.TestCase):
    def test_dataset_contract_accepts_exact_schema(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _valid_metadata(root)
            validate_dataset_contract(root)

    def test_dataset_contract_rejects_wrong_semantics(self):
        cases = [
            ("action_semantics", "official_current_qpos_with_gripper_threshold"),
            ("action_offset_frames", 0),
            ("fps", 60),
        ]
        for field, value in cases:
            with self.subTest(field=field), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                _valid_metadata(root)
                path = root / "meta" / "arx.json"
                payload = json.loads(path.read_text())
                payload[field] = value
                path.write_text(json.dumps(payload))
                with self.assertRaises(ValueError):
                    validate_dataset_contract(root)


if __name__ == "__main__":
    unittest.main()
