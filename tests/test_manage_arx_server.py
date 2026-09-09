"""Deployment failures must not switch to an unready/stale model."""
import importlib.util
from pathlib import Path
import json

import pytest


SPEC = importlib.util.spec_from_file_location("manage_arx_server", Path(__file__).parents[1] / "scripts/manage_arx_server.py")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def config():
    return {"model_id": "new", "checkpoint_sha256": "abc", "route": "feedback",
            "protocol_version": "arx-feedback-v4", "model_dir": "/bundle",
            "deployment_contract": {"protocol_version": "arx-feedback-v4", "fps": 30}}


def test_verify_live_checks_readiness_identity_and_contract(monkeypatch):
    expected = config()
    health = {**expected, "ready": True}
    contract = {**expected["deployment_contract"], "model_id": "new", "checkpoint_sha256": "abc"}
    monkeypatch.setattr(MODULE, "_get_json", lambda *args: contract)
    assert MODULE._verify_live("http://test", expected, health) == health
    with pytest.raises(RuntimeError, match="not ready"):
        MODULE._verify_live("http://test", expected, {**health, "ready": False})
    with pytest.raises(RuntimeError, match="route"):
        MODULE._verify_live("http://test", expected, {**health, "route": "other"})
    contract["fps"] = 15
    with pytest.raises(RuntimeError, match="contract mismatch"):
        MODULE._verify_live("http://test", expected, health)


def test_failed_promotion_without_verified_old_service_clears_stale_state(tmp_path, monkeypatch):
    monkeypatch.setattr(MODULE, "STATE_DIR", tmp_path)
    monkeypatch.setattr(MODULE, "_manifest", lambda bundle: config())
    monkeypatch.setattr(MODULE, "_verify_live", lambda *args: {})
    monkeypatch.setattr(MODULE, "_live_config", lambda: None)
    monkeypatch.setattr(MODULE.time, "sleep", lambda _: None)
    (tmp_path / "previous.json").write_text(json.dumps({"model_id": "stale"}))
    (tmp_path / "active.json").write_text(json.dumps({"model_id": "stale"}))
    started, stopped = [], []
    monkeypatch.setattr(MODULE, "_start", lambda manifest, **kw: started.append(manifest["model_id"]))
    monkeypatch.setattr(MODULE, "_stop", stopped.append)
    def timeout(*args, **kwargs):
        assert kwargs["timeout"] == 123
        raise TimeoutError("not loaded")
    monkeypatch.setattr(MODULE, "_wait_health", timeout)
    with pytest.raises(TimeoutError):
        MODULE.promote(Path("/bundle"), startup_timeout=123)
    assert started == ["new"]
    assert stopped[-1] == MODULE.PRODUCTION_SESSION
    assert not (tmp_path / "previous.json").exists()
    assert not (tmp_path / "active.json").exists()


def test_legacy_unverified_rollback_state_never_stops_live_service(tmp_path, monkeypatch):
    monkeypatch.setattr(MODULE, "STATE_DIR", tmp_path)
    (tmp_path / "previous.json").write_text(json.dumps(config()))
    monkeypatch.setattr(MODULE, "_stop", lambda _: pytest.fail("must validate before stopping"))
    with pytest.raises(ValueError, match="not captured"):
        MODULE.rollback()
