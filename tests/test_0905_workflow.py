from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile
import types
import unittest
from unittest import mock

import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
from prepare_0905_training import PROFILES, select_numbers, source_snapshot
from manage_0905_resources import GROUP, OWNER, choose_release, notebook_queue, AuthenticatedAPI
import qzcli_session
from report_0905_campaign import metrics_from_log


class SplitTest(unittest.TestCase):
    def test_final_training_summary_does_not_replace_last_loss_metric(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "train.log"
            path.write_text("{'global_step': 10000, 'loss': '0.001', 'grad_norm': '0.02'}\r\n{'global_step': 10000, 'train_loss': '0.01', 'train_runtime': 5000}\n")
            self.assertEqual(len(metrics_from_log(path)), 1)
            self.assertEqual(metrics_from_log(path)[0]["loss"], "0.001")

    def test_sparse_split_is_disjoint_and_exactly_100(self):
        available = list(reversed([n for n in range(102) if n != 78]))
        first, last, full = [select_numbers(p, available) for p in PROFILES[:3]]
        self.assertEqual(first, list(range(50)))
        self.assertEqual(last, list(range(50, 78)) + list(range(79, 101)))
        self.assertEqual(full, first + last)
        self.assertEqual(len(set(full)), 100)
        self.assertNotIn(101, full)

    def test_changed_or_incomplete_source_is_rejected(self):
        with self.assertRaises(ValueError):
            select_numbers(PROFILES[0], list(range(100)))
        with self.assertRaises(ValueError):
            select_numbers(PROFILES[3], list(range(99)))

    def test_transfer_gate_rejects_missing_partial_and_changing_files(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "final"
            self.assertIsNone(source_snapshot(root))
            root.mkdir()
            for n in range(99):
                (root / f"episode_{n}.hdf5").write_bytes(b"a")
            self.assertIsNone(source_snapshot(root))
            (root / "episode_99.hdf5").write_bytes(b"a")
            snapshot = source_snapshot(root)
            self.assertIsNotNone(snapshot)
            partial = root / ".episode_99.hdf5.part"
            partial.touch()
            self.assertIsNone(source_snapshot(root))
            partial.unlink()
            (root / "episode_99.hdf5").write_bytes(b"changed")
            self.assertNotEqual(snapshot, source_snapshot(root))


class ProfileTest(unittest.TestCase):
    def test_smoke_restart_tracks_existing_job_without_resubmission(self):
        launcher = (REPO / "scripts/qzcli_arx_h200.sh").read_text()
        function = launcher[launcher.index("run_smoke() {"):launcher.index("log_has_step_after_500() {")]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "smoke_job_id").write_text("existing-smoke")
            stubs = '\njob_status() { echo job_running; }\nwait_for_smoke() { echo "$1" > "$STATE_DIR/reused"; }\nwait_for_resources() { exit 91; }\n'
            result = subprocess.run(["bash", "-eu", "-c", function + stubs + "run_smoke"],
                env=dict(os.environ, MONITOR_TO_COMPLETION="1", STATE_DIR=directory, MARKER=str(root / "marker"),
                    PROFILE=PROFILES[0], TARGET_WORKSPACE="workspace", TARGET_GROUP="group"), capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual((root / "reused").read_text().strip(), "existing-smoke")
            self.assertIn("world_size=16", (root / "marker").read_text())

    def test_uncertain_submission_fence_prevents_duplicate_create(self):
        launcher = (REPO / "scripts/qzcli_arx_h200.sh").read_text()
        function = launcher[launcher.index("submit_job() {"):launcher.index("job_status() {")]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "pending_submission.json").write_text('{"name":"unknown-result"}')
            result = subprocess.run(["bash", "-eu", "-c", function + '\nqzcli() { touch "$STATE_DIR/unexpected"; }\nsubmit_job name command 2 8 1 10000'],
                env=dict(os.environ, MONITOR_TO_COMPLETION="1", STATE_DIR=directory), capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertFalse((root / "unexpected").exists())

    def test_ten_profiles_have_independent_10k_and_20k_commands(self):
        launcher = (REPO / "scripts/qzcli_arx_h200.sh").read_text()
        setup = launcher[launcher.index('PYTHON_BIN="$REPO/'):launcher.index('FORMAL_DIR="$FORMAL_ROOT/')]
        generator = launcher[launcher.index("remote_command() {"):launcher.index("run_smoke() {")]
        paths = set()
        original = yaml.safe_load((REPO / "configs/arx_lift2s_pickandplace_01/train_h200.yaml").read_text())
        for profile in [p + suffix for p in PROFILES for suffix in ("", "-20k")]:
            name = "arx_lift2s_" + profile.replace("-", "_")
            path = REPO / "configs" / name / "train_h200.yaml"
            config = yaml.safe_load(path.read_text())
            self.assertEqual(config["model_args"], original["model_args"])
            max_steps = 20000 if profile.endswith("-20k") else 10000
            self.assertEqual(config["training_args"]["max_steps"], max_steps)
            if profile.endswith("-20k"):
                baseline = yaml.safe_load((REPO / "configs" / name.removesuffix("_20k") / "train_h200.yaml").read_text())
                self.assertEqual(config["data_args"], baseline["data_args"])
                self.assertEqual(config["model_args"], baseline["model_args"])
            for k, v in original["training_args"].items():
                if k not in {"max_steps", "output_dir"}:
                    self.assertEqual(config["training_args"][k], v)
            for kind, steps in (("smoke", "20"), ("formal", str(max_steps))):
                command = subprocess.check_output(["bash", "-eu", "-c", setup + generator + f'\nremote_command {kind} 2 8 1'],
                    env=dict(os.environ, REPO=str(REPO), PROFILE=profile), text=True)
                tokens = shlex.split(command)
                self.assertIn(str(path), tokens)
                self.assertEqual(tokens[tokens.index("--max_steps") + 1], steps)
                self.assertIn("REQUIRE_GLOBAL_BATCH=128", tokens)
                self.assertIn("REQUIRE_WORLD_SIZE=16", tokens)
                self.assertIn("AUTO_RESUME=0", tokens)
                if kind == "formal":
                    paths.add(tokens[tokens.index("--output_dir") + 1])
        self.assertEqual(len(paths), 10)


class ReleaseTest(unittest.TestCase):
    def test_notebook_audit_reads_every_page_and_rejects_unknown_schema(self):
        api = mock.Mock()
        api.list_notebooks_with_cookie.side_effect = [
            {"list": [{"status": "RUNNING"}], "total": 2},
            {"list": [{"status": "PENDING", "logic_compute_group_id": GROUP}], "total": 2},
        ]
        with mock.patch("manage_0905_resources.time.sleep"):
            self.assertEqual(len(notebook_queue(api, "test")), 1)
        api.list_notebooks_with_cookie.side_effect = [{"items": [], "total": 2}]
        with self.assertRaises(KeyError):
            notebook_queue(api, "test")

    def job(self, name, *, status="job_running", nodes=2, created=1, owner=OWNER):
        return {"name": name, "status": status, "node_count": nodes, "created_at": str(created),
            "priority": 35, "created_by": {"id": owner}, "logic_compute_group_id": GROUP}

    def test_only_one_fill_when_an_unallocated_training_is_waiting(self):
        pending = self.job("arx-0905-datasets-first50", status="job_queuing", nodes=0, created=10)
        fillers = [self.job(f"qwen35_vla_fill_16g_{n}", created=n) for n in (1, 2, 3)]
        self.assertIsNone(choose_release(fillers, 0))
        self.assertIsNone(choose_release([pending, *fillers], 2))
        self.assertEqual(choose_release([pending, *fillers], 1), fillers[0])
        pending["node_count"] = 2
        self.assertIsNone(choose_release([pending, *fillers], 0))

    def test_other_jobs_and_owners_are_protected(self):
        pending = self.job("arx-0905-test", status="job_queuing", nodes=0, created=10)
        other_fill = self.job("qwen35_vla_fill_16g_other", owner="another-user")
        real_training = self.job("qwen35_vla_flex_16g_real")
        self.assertIsNone(choose_release([pending, other_fill, real_training], 0))
        ahead = self.job("someone-training", status="job_queuing", nodes=0)
        fill = self.job("qwen35_vla_fill_16g_ours")
        self.assertIsNone(choose_release([pending, ahead, fill], 0))

    def test_only_blocking_queued_fill_is_cancelled(self):
        pending = self.job("arx-0905-test", status="job_queuing", nodes=0, created=10)
        before = self.job("qwen35_vla_fill_16g_before", status="job_queuing", nodes=0)
        after = self.job("qwen35_vla_fill_16g_after", status="job_queuing", nodes=0, created=20)
        self.assertEqual(choose_release([pending, before, after], 0), before)
        self.assertIsNone(choose_release([pending, after], 0))


class SessionTest(unittest.TestCase):
    def test_direct_api_refreshes_expired_reads_but_never_retries_stop(self):
        class Expired(Exception):
            code = 401
        api = mock.Mock()
        api.list_jobs_with_cookie.side_effect = [Expired(), {"jobs": []}]
        api.stop_job_with_cookie.side_effect = Expired()
        modules = {"qzcli.api": types.SimpleNamespace(QzAPIError=Expired),
            "qzcli.config": types.SimpleNamespace(get_cookie=lambda: {"cookie": "fresh"})}
        with mock.patch.dict(sys.modules, modules), mock.patch("manage_0905_resources.ensure_session") as refresh:
            wrapper = AuthenticatedAPI(api, Path("credentials"))
            self.assertEqual(wrapper.list_jobs_with_cookie("workspace", "old"), {"jobs": []})
            refresh.assert_called_once_with(Path("credentials"), force=True)
            with self.assertRaises(Expired):
                wrapper.stop_job_with_cookie("job", "old")
            self.assertEqual(api.stop_job_with_cookie.call_count, 1)
            self.assertEqual(refresh.call_count, 1)

    def test_cached_refresh_is_shared_without_printing_password(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            credential = root / "credential"
            credential.write_text("test-user\nprivate-value\n")
            with mock.patch.object(qzcli_session.Path, "home", return_value=root), mock.patch.object(qzcli_session.subprocess, "run") as run:
                run.return_value.returncode = 0
                qzcli_session.ensure_session(credential)
                qzcli_session.ensure_session(credential)
                self.assertEqual(run.call_count, 1)
                self.assertNotIn("private-value", str(run.call_args.args))

    def test_uncertain_create_is_never_automatically_retried(self):
        with mock.patch.object(sys, "argv", ["session", "--credentials", "/some/path", "--", "create"]), mock.patch.object(qzcli_session, "ensure_session"), mock.patch.object(qzcli_session.subprocess, "run") as run:
            run.return_value = subprocess.CompletedProcess([], 1, "", "401")
            self.assertEqual(qzcli_session.main(), 1)
            self.assertEqual(run.call_count, 1)


if __name__ == "__main__":
    unittest.main()
