from __future__ import annotations

import importlib.util
import json
import os
import shlex
import subprocess
import tempfile
import unittest
from pathlib import Path

import yaml


REPO = Path(__file__).resolve().parents[1]
LAUNCHER = (REPO / "scripts/qzcli_arx_h200.sh").read_text()
module_spec = importlib.util.spec_from_file_location(
    "arx_resume", REPO / "scripts/select_arx_resume_checkpoint.py"
)
resume = importlib.util.module_from_spec(module_spec)
module_spec.loader.exec_module(resume)


class PickAndPlaceLaunchTest(unittest.TestCase):
    def command(self, profile, kind, checkpoint=""):
        # Evaluate the real profile declarations and command generator, without
        # login, filesystem mutation, resource selection or submission.
        setup = LAUNCHER[LAUNCHER.index("PYTHON_BIN=\"$REPO/"):LAUNCHER.index('FORMAL_DIR="$FORMAL_ROOT/')]
        generator = LAUNCHER[LAUNCHER.index("remote_command() {"):LAUNCHER.index("run_smoke() {")]
        script = setup + generator + '\nRESUME_CHECKPOINT=$TEST_RESUME\nremote_command "$TEST_KIND" 2 8 1\n'
        env = dict(os.environ, REPO=str(REPO), PROFILE=profile, TEST_KIND=kind, TEST_RESUME=checkpoint)
        return subprocess.check_output(["bash", "-eu", "-c", script], env=env, text=True)

    def test_formal_and_smoke_commands_for_both_datasets(self):
        commands = []
        for number in ("01", "02"):
            config_path = REPO / f"configs/arx_lift2s_pickandplace_{number}/train_h200.yaml"
            config = yaml.safe_load(config_path.read_text())
            self.assertFalse(config["model_args"]["training_time_rtc"])
            self.assertEqual(config["data_args"]["config_name"], f"arx_lift2s_pickandplace_{number}_ft")
            self.assertEqual(config["training_args"]["max_steps"], 6000)
            for kind, steps, save in (("formal", "6000", "steps"), ("smoke", "20", "no")):
                command = self.command(f"pickandplace-{number}", kind)
                commands.append(command)
                tokens = shlex.split(command)
                self.assertIn(str(config_path), tokens)
                self.assertEqual(tokens[tokens.index("--max_steps") + 1], steps)
                self.assertEqual(tokens[tokens.index("--save_strategy") + 1], save)
                self.assertIn("AUTO_RESUME=0", tokens)
                self.assertIn("REQUIRE_WORLD_SIZE=16", tokens)
                self.assertIn("REQUIRE_GLOBAL_BATCH=128", tokens)
                self.assertIn("REQUIRE_ALL_TRAINABLE=1", tokens)
                self.assertEqual(tokens[tokens.index("--per_device_train_batch_size") + 1], "8")
                self.assertEqual(tokens[tokens.index("--gradient_accumulation_steps") + 1], "1")
                self.assertNotIn("--resume_from_checkpoint", tokens)
        self.assertEqual(len(set(commands)), 4)

    def test_old_profiles_retain_10k_and_auto_resume(self):
        for profile in ("pickplace", "tool-yipan"):
            command = self.command(profile, "formal")
            self.assertIn("--max_steps 10000", command)
            self.assertIn("AUTO_RESUME=1", command)

    def test_explicit_resume_is_only_added_to_formal(self):
        checkpoint = "/some run/checkpoint-500"
        formal = shlex.split(self.command("pickandplace-01", "formal", checkpoint))
        self.assertEqual(formal[formal.index("--resume_from_checkpoint") + 1], checkpoint)
        self.assertNotIn("--resume_from_checkpoint", self.command("pickandplace-01", "smoke", checkpoint))


class ResumeSelectionTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.run = Path(self.temp.name) / "run"
        self.run.mkdir()
        self.config = Path(self.temp.name) / "train.yaml"
        self.expected = {
            "model_args": {"training_time_rtc": False},
            "data_args": {"config_name": "arx_lift2s_pickandplace_01_ft"},
            "training_args": {"output_dir": "outputs/run", "max_steps": 6000, "report_to": "none"},
        }
        self.config.write_text(yaml.safe_dump(self.expected))

    def checkpoint(self, step):
        path = self.run / f"checkpoint-{step}"
        path.mkdir()
        spec = json.loads(json.dumps(self.expected))
        spec["training_args"].update(output_dir=str(self.run), report_to=[])
        (path / "run_spec.json").write_text(json.dumps(spec))
        (path / "trainer_state.json").write_text(json.dumps({"global_step": step}))
        for name in ("model.safetensors", "scheduler.pt", "optimizer.pt", "data_state_rank0.pt",
                     "config.json", "processor_config.json", "tokenizer.json", "policy_manifest.json",
                     "resolved_config_full.yaml"):
            (path / name).write_text("{}")
        contract = path / "finch_data_spec" / "data"
        contract.mkdir(parents=True)
        for name in ("spec.json", "norm_stats.json", "components.json", "field_descriptions.json"):
            (contract / name).write_text("{}")
        return path

    def test_fresh_run_returns_no_checkpoint(self):
        self.assertIsNone(resume.select_checkpoint(self.run, self.config, 1))

    def test_skips_partial_save_and_selects_previous_complete_checkpoint(self):
        complete = self.checkpoint(500)
        partial = self.checkpoint(1000)
        (partial / "data_state_rank0.pt").unlink()
        self.assertEqual(resume.select_checkpoint(self.run, self.config, 1), complete)

    def test_refuses_to_restart_over_only_incomplete_checkpoint(self):
        partial = self.checkpoint(500)
        (partial / "model.safetensors").write_text("")
        with self.assertRaisesRegex(ValueError, "all incomplete"):
            resume.select_checkpoint(self.run, self.config, 1)

    def test_rejects_other_dataset_or_schedule(self):
        checkpoint = self.checkpoint(500)
        spec = json.loads((checkpoint / "run_spec.json").read_text())
        spec["data_args"]["config_name"] = "arx_lift2s_pickandplace_02_ft"
        (checkpoint / "run_spec.json").write_text(json.dumps(spec))
        with self.assertRaisesRegex(ValueError, "data_args.config_name"):
            resume.select_checkpoint(self.run, self.config, 1)
        spec["data_args"] = self.expected["data_args"]
        spec["training_args"]["max_steps"] = 10000
        (checkpoint / "run_spec.json").write_text(json.dumps(spec))
        with self.assertRaisesRegex(ValueError, "training_args.max_steps"):
            resume.select_checkpoint(self.run, self.config, 1)

    def test_rejects_checkpoint_symlink_to_other_run(self):
        other = Path(self.temp.name) / "other-checkpoint"
        other.mkdir()
        (self.run / "checkpoint-500").symlink_to(other, target_is_directory=True)
        with self.assertRaisesRegex(ValueError, "leaves this run"):
            resume.select_checkpoint(self.run, self.config, 1)


class FormalMonitoringTest(unittest.TestCase):
    def monitor(self, final_exit, target=6000, accepted=False):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        root = Path(temp.name)
        state = root / "state"
        state.mkdir()
        (state / "formal_job_id").write_text("existing-job")
        if accepted:
            (state / "checkpoint_500_accepted").write_text(f"run:{target}\n")
        marker = root / "marker"
        marker.write_text("profile=pickandplace-01\ninstances=2\nworld_size=16\nbatch=8\naccumulation=1\n")
        run = root / "run"
        (run / "log").mkdir(parents=True)
        (run / "log/training_log_nodeIdx000_test.txt").write_text("{'global_step': 510}")
        fake_python = root / "python"
        fake_python.write_text(
            '#!/usr/bin/env bash\n'
            f'if [[ "$*" == *checkpoint-{target}* ]]; then\n'
            '  echo final_validation_called\n'
            f'  exit {final_exit}\n'
            'fi\necho intermediate_validation_called\n'
        )
        fake_python.chmod(0o755)
        functions = LAUNCHER[LAUNCHER.index("run_formal() {"):LAUNCHER.rindex('case "$MODE" in')]
        stubs = '''
qzcli() { echo "{'global_step': 510}"; }
log_has_step_after_500() { return 0; }
job_status() {
    count=$(cat "$STATE_DIR/status_calls" 2>/dev/null || echo 0)
    count=$((count + 1))
    echo "$count" > "$STATE_DIR/status_calls"
    if [ "$count" -ge 3 ]; then echo succeeded; else echo running; fi
}
wait_for_resources() { echo unexpected_submission >&2; exit 9; }
'''
        env = dict(os.environ, REPO=str(REPO), PYTHON_BIN=str(fake_python),
                   PROFILE="pickandplace-01", MARKER=str(marker), STATE_DIR=str(state),
                   FORMAL_ROOT=str(root), FORMAL_DIR=str(run), GPUS_PER_NODE="8",
                   GLOBAL_BATCH="128", MONITOR_TO_COMPLETION="1", FORMAL_STEPS=str(target), FORMAL_RUN="run",
                   FREE_KIB_REQUIRED="0", POLL_SECONDS="0")
        result = subprocess.run(["bash", "-eu", "-o", "pipefail", "-c", functions + stubs + "\nrun_formal"],
                                env=env, text=True, capture_output=True, timeout=10)
        return result, state

    def test_20k_restart_retains_acceptance_after_old_checkpoint_rotation(self):
        result, state = self.monitor(0, target=20000, accepted=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("final_validation_called", result.stdout)
        self.assertFalse((state / "checkpoint_500_validation.log").exists())
        self.assertTrue((state / "completed_job_id").exists())

    def test_continues_beyond_500_until_success_and_final_validation(self):
        result, state = self.monitor(0)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("status=running checkpoint_500_accepted=1", result.stdout)
        self.assertIn("final_validation_called", result.stdout)
        self.assertEqual((state / "completed_job_id").read_text().strip(), "existing-job")

    def test_failed_final_validation_never_marks_job_complete(self):
        result, state = self.monitor(1)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("final_validation_called", result.stdout)
        self.assertFalse((state / "completed_job_id").exists())


class PinnedQueueTest(unittest.TestCase):
    def select(self, query_output, monitor):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            scripts = root / "scripts"
            scripts.mkdir()
            (scripts / "qzcli_resource_select.py").write_text('print("spec-h200\\n180\\n1800")\n')
            function = LAUNCHER[LAUNCHER.index("select_resources() {"):LAUNCHER.index("wait_for_resources() {")]
            env = dict(os.environ, REPO=str(root), TARGET_WORKSPACE="workspace", TARGET_GROUP="vla-group",
                       MONITOR_TO_COMPLETION=str(monitor), QUERY_OUTPUT=query_output)
            return subprocess.run(
                ["bash", "-eu", "-c", function + '\nqzcli() { echo "$QUERY_OUTPUT"; return 1; }\nselect_resources 2'],
                env=env, text=True, capture_output=True,
            )

    def test_new_pinned_profile_can_queue_after_valid_shortage_query(self):
        result = self.select("[项目] vla分区: 1 空节点 [NVIDIA]", 1)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("pinned scheduler queue", result.stdout)

    def test_auth_failure_never_becomes_resource_selection(self):
        self.assertNotEqual(self.select("Cookie expired", 1).returncode, 0)

    def test_existing_profiles_still_wait_for_idle_nodes(self):
        self.assertNotEqual(self.select("[项目] vla分区: 1 空节点 [NVIDIA]", 0).returncode, 0)


if __name__ == "__main__":
    unittest.main()
