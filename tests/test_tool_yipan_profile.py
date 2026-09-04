from __future__ import annotations

import unittest
from pathlib import Path

import yaml


REPO = Path(__file__).resolve().parents[1]
CONFIG_DIR = REPO / "configs/arx_lift2s_pickplace_tool_yipan"


class ToolYipanProfileTest(unittest.TestCase):
    def test_formal_config_is_full_parameter_global_batch_128_primary(self):
        config = yaml.safe_load((CONFIG_DIR / "train_h200.yaml").read_text())
        model = config["model_args"]
        training = config["training_args"]
        for key in ("tune_mm_vision", "tune_mm_mlp", "tune_mm_llm", "tune_vla_dit"):
            self.assertIs(model[key], True)
        self.assertEqual(
            config["data_args"]["config_name"], "arx_lift2s_pickplace_tool_yipan_ft"
        )
        self.assertIs(config["data_args"]["filter_dataset_by_stats"], False)
        self.assertEqual(training["per_device_train_batch_size"], 16)
        self.assertEqual(training["gradient_accumulation_steps"], 1)
        self.assertEqual(training["max_steps"], 10_000)
        self.assertEqual(training["save_steps"], 500)
        self.assertEqual(training["save_total_limit"], 20)
        self.assertEqual(training["deepspeed"], "scripts/deepspeed/zero1.json")

    def test_launcher_pins_vla_partition_and_only_approved_tool_attempts(self):
        launcher = (REPO / "scripts/qzcli_arx_h200.sh").read_text()
        self.assertIn("TARGET_GROUP=lcg-d8eb9030-2233-47f7-b8cb-988c3e7c0ec9", launcher)
        self.assertIn('SMOKE_ATTEMPTS=("1 16 1" "2 8 1")', launcher)
        self.assertIn("less than 900 GiB free", launcher)
        self.assertIn("CUDA OOM", launcher)
        self.assertIn("SKIP_FINAL_SAVE", launcher)


if __name__ == "__main__":
    unittest.main()
