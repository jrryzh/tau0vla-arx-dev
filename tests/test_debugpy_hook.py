"""Regression tests for the opt-in debugpy listener."""

import os
import sys
import types
import unittest
from unittest import mock

# ``transformers`` is only used inside ``train()``. Keep this focused unit test
# importable in lightweight environments where the full training stack is not
# installed.
sys.modules.setdefault("transformers", types.ModuleType("transformers"))
logging_stub = types.ModuleType("tau0_vla.utils.logging")
logging_stub.setup_py_logging = mock.Mock()
sys.modules.setdefault("tau0_vla.utils.logging", logging_stub)

from tau0_vla.trainer.train import _maybe_wait_for_debugger


class DebugpyHookTest(unittest.TestCase):
    def run_hook(self, **environment: str):
        debugpy = types.ModuleType("debugpy")
        debugpy.listen = mock.Mock()
        debugpy.wait_for_client = mock.Mock()

        with (
            mock.patch.dict(os.environ, environment, clear=True),
            mock.patch.dict(sys.modules, {"debugpy": debugpy}),
        ):
            _maybe_wait_for_debugger()

        return debugpy

    def test_defaults_to_loopback_and_waits_for_selected_rank(self):
        debugpy = self.run_hook(VLA_DEBUGPY_RANKS="0")

        debugpy.listen.assert_called_once_with(("127.0.0.1", 5678))
        debugpy.wait_for_client.assert_called_once_with()

    def test_empty_host_also_defaults_to_loopback(self):
        debugpy = self.run_hook(VLA_DEBUGPY_RANKS="0", VLA_DEBUGPY_HOST="")

        debugpy.listen.assert_called_once_with(("127.0.0.1", 5678))
        debugpy.wait_for_client.assert_called_once_with()

    def test_explicit_host_and_rank_offset_are_honored(self):
        debugpy = self.run_hook(
            VLA_DEBUGPY_RANKS="0, 2",
            VLA_DEBUGPY_HOST="192.0.2.10",
            VLA_DEBUGPY_PORT="6000",
            RANK="2",
        )

        debugpy.listen.assert_called_once_with(("192.0.2.10", 6002))
        debugpy.wait_for_client.assert_called_once_with()

    def test_unselected_rank_does_not_listen_or_wait(self):
        debugpy = self.run_hook(VLA_DEBUGPY_RANKS="0,2", RANK="1")

        debugpy.listen.assert_not_called()
        debugpy.wait_for_client.assert_not_called()


if __name__ == "__main__":
    unittest.main()
