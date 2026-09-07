from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path


spec = importlib.util.spec_from_file_location("training_plot", Path(__file__).resolve().parents[1] / "scripts/plot_training_live.py")
plot = importlib.util.module_from_spec(spec)
spec.loader.exec_module(plot)


class TrainingPlotTest(unittest.TestCase):
    def parse(self, text):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "training.txt"
            path.write_text(text)
            return plot.parse_logs([path])

    def test_6k_budget_survives_larger_dataset_progress_bars(self):
        rows, total = self.parse(
            "[INFO] CLI overrides: --max_steps 6000\n"
            "Loading: 9383/9383\n5999/6000\n"
            "{'loss': 0.1, 'global_step': 6000, 'vla_epoch': 89.552238}\n"
        )
        self.assertEqual(total, 6000)
        self.assertEqual(rows[0]["step"], 6000)
        self.assertEqual(rows[0]["vla_epoch"], 89.552238)

    def test_global_step_metrics_without_progress_bar(self):
        rows, total = self.parse("--max_steps 6000\n{'loss': 0.2, 'global_step': 510}\n")
        self.assertEqual((rows[0]["step"], total), (510, 6000))

    def test_legacy_logs_keep_progress_fallback(self):
        rows, total = self.parse("100/10000\n{'loss': 0.3}\n")
        self.assertEqual((rows[0]["step"], total), (100, 10000))
