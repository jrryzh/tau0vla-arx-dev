from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path
from unittest import mock


REPO = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "convert_official_hdf5_to_lerobot",
    REPO / "tools/convert_official_hdf5_to_lerobot.py",
)
assert SPEC is not None and SPEC.loader is not None
CONVERTER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CONVERTER)


class EpisodeSelectionTest(unittest.TestCase):
    def test_strict_mode_still_rejects_a_missing_episode(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "episode_1.hdf5").touch()
            (root / "episode_3.hdf5").touch()
            with self.assertRaises(FileNotFoundError):
                CONVERTER.selected_paths(root, 1, 3)

    def test_sparse_mode_selects_present_files_in_numeric_order(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ("episode_11.hdf5", "episode_2.hdf5", "episode_1.hdf5"):
                (root / name).touch()
            paths = CONVERTER.selected_paths(root, 1, 11, allow_missing=True)
            self.assertEqual(
                [path.name for path in paths],
                ["episode_1.hdf5", "episode_2.hdf5", "episode_11.hdf5"],
            )

    def test_sparse_mode_rejects_malformed_episode_filename(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "episode_1.hdf5").touch()
            (root / "episode_bad.hdf5").touch()
            with self.assertRaisesRegex(ValueError, "episode_<integer>"):
                CONVERTER.selected_paths(root, 1, 3, allow_missing=True)

    def test_manifest_records_requested_selected_and_missing_source_numbers(self):
        paths = [Path("episode_1.hdf5"), Path("episode_3.hdf5")]

        def inspect(path: Path) -> dict:
            return {
                "file": path.name,
                "source_episode_number": CONVERTER.episode_number(path),
                "frames": 10,
                "source_action_semantics": "",
                "image_shapes": {
                    camera: [480, 640, 3] for camera in CONVERTER.CAMERA_NAMES
                },
            }

        with mock.patch.object(CONVERTER, "inspect_episode", side_effect=inspect):
            manifest = CONVERTER.validate_selection(
                paths,
                60,
                30,
                "Pick up the tool and place it into the tray.",
                "state_t_plus_1",
                requested_start=1,
                requested_end=3,
            )
        self.assertEqual(manifest["selected_source_episode_numbers"], [1, 3])
        self.assertEqual(manifest["missing_source_episode_numbers"], [2])
        self.assertEqual(manifest["requested_episode_range"], {"start": 1, "end": 3})


if __name__ == "__main__":
    unittest.main()
