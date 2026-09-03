from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]


class H200ResourceSelectionTest(unittest.TestCase):
    def _run(self, home: Path, *args: str) -> subprocess.CompletedProcess[str]:
        env = os.environ.copy()
        env["HOME"] = str(home)
        return subprocess.run(
            [sys.executable, str(REPO / "scripts/qzcli_resource_select.py"), *args],
            text=True,
            capture_output=True,
            env=env,
            check=False,
        )

    def test_lists_only_h200_groups_and_selects_group_scoped_8_gpu_spec(self):
        with tempfile.TemporaryDirectory() as directory:
            home = Path(directory)
            cache = home / ".qzcli/resources.json"
            cache.parent.mkdir()
            cache.write_text(
                json.dumps(
                    {
                        "ws-b": {
                            "compute_groups": {
                                "group-h200-b": {"gpu_type_display": "NVIDIA H200 141GB"},
                                "group-a100": {"gpu_type": "NVIDIA_A100_SXM_80G"},
                            },
                            "specs": {
                                "spec-wrong-group": {
                                    "name": "8x H200",
                                    "gpu_count": 8,
                                    "cpu_count": 128,
                                    "memory_gb": 1024,
                                    "logic_compute_group_ids": ["some-other-group"],
                                },
                                "spec-h200-8": {
                                    "gpu_type": "NVIDIA_H200_SXM_141G",
                                    "gpu_count": 8,
                                    "cpu_count": 192,
                                    "memory_gb": 1800,
                                    "logic_compute_group_ids": ["group-h200-b"],
                                },
                                "spec-h200-4": {
                                    "gpu_type": "NVIDIA_H200_SXM_141G",
                                    "gpu_count": 4,
                                    "cpu_count": 96,
                                    "memory_gb": 900,
                                    "logic_compute_group_ids": ["group-h200-b"],
                                },
                            },
                        },
                        "ws-a": {
                            "compute_groups": {
                                "group-h200-a": {"name": "H200 production"},
                            },
                            "specs": {},
                        },
                    }
                )
            )

            groups = self._run(home, "groups")
            self.assertEqual(groups.returncode, 0, groups.stderr)
            self.assertEqual(
                groups.stdout.splitlines(),
                ["ws-a\tgroup-h200-a", "ws-b\tgroup-h200-b"],
            )

            spec = self._run(home, "spec", "ws-b", "group-h200-b")
            self.assertEqual(spec.returncode, 0, spec.stderr)
            self.assertEqual(spec.stdout.splitlines(), ["spec-h200-8", "192", "1800"])


class H200CheckpointValidationTest(unittest.TestCase):
    def _run(self, checkpoint: Path, world_size: int = 2) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [
                sys.executable,
                str(REPO / "scripts/validate_checkpoint.py"),
                str(checkpoint),
                "--world-size",
                str(world_size),
            ],
            text=True,
            capture_output=True,
            check=False,
        )

    def test_accepts_deepspeed_checkpoint_with_all_resume_artifacts(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "checkpoint-500"
            shard = checkpoint / "global_step500"
            shard.mkdir(parents=True)
            for name in ("trainer_state.json", "scheduler.pt", "run_spec.json"):
                (checkpoint / name).touch()
            (shard / "mp_rank_00_model_states.pt").touch()
            (shard / "zero_pp_rank_0_mp_rank_00_optim_states.pt").touch()
            for rank in range(2):
                (checkpoint / f"data_state_rank{rank}.pt").touch()

            result = self._run(checkpoint)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("checkpoint_validation=ok", result.stdout)

    def test_rejects_checkpoint_until_every_rank_data_state_exists(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "checkpoint-500"
            checkpoint.mkdir()
            for name in (
                "trainer_state.json",
                "scheduler.pt",
                "run_spec.json",
                "model.safetensors",
                "optimizer.pt",
                "data_state_rank0.pt",
            ):
                (checkpoint / name).touch()

            result = self._run(checkpoint)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("expected 2 rank data states, found 1", result.stderr)


class QzcliPayloadValidationTest(unittest.TestCase):
    def _run(self, payload: dict) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [
                sys.executable,
                str(REPO / "scripts/validate_qzcli_payload.py"),
                "--compute-group",
                "group-h200",
                "--spec",
                "spec-h200-8",
                "--repo",
                str(REPO),
            ],
            input="Dry run\n" + json.dumps(payload),
            text=True,
            capture_output=True,
            check=False,
        )

    def _payload(self) -> dict:
        return {
            "framework": "pytorch",
            "logic_compute_group_id": "group-h200",
            "command": f"cd {REPO} && bash scripts/train.sh",
            "framework_config": [
                {
                    "gpu_count": 8,
                    "instance_count": 2,
                    "shm_gi": 1200,
                    "resource_spec_price": {
                        "gpu_type": "NVIDIA_H200_SXM_141G",
                        "quota_id": "spec-h200-8",
                    },
                }
            ],
        }

    def test_accepts_expected_two_node_h200_payload(self):
        result = self._run(self._payload())
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("dry-run payload validated", result.stderr)

    def test_rejects_payload_with_wrong_instance_count(self):
        payload = self._payload()
        payload["framework_config"][0]["instance_count"] = 1
        result = self._run(payload)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("instance_count", result.stderr)


if __name__ == "__main__":
    unittest.main()
