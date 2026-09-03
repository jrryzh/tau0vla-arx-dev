from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from tau0_vla.utils.utils import load_config_from_yaml


class YamlOverrideTest(unittest.TestCase):
    def test_report_to_none_stays_transformers_sentinel(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "train.yaml"
            path.write_text(
                "experiment:\n"
                "  run_name: test\n"
                "model_args:\n"
                "  model_name_or_path: /tmp/model\n"
                "data_args: {}\n"
                "training_args:\n"
                "  output_dir: /tmp/output\n"
            )
            config = load_config_from_yaml(path, overrides={"report_to": "none"})
        self.assertEqual(config["training_args"]["report_to"], "none")


if __name__ == "__main__":
    unittest.main()
