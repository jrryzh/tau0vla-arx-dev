from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "package_arx_inference_bundle", ROOT / "scripts/package_arx_inference_bundle.py"
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_package_is_self_contained_and_excludes_training_state(tmp_path):
    run = tmp_path / "run"
    checkpoint = run / "checkpoint-10000"
    checkpoint.mkdir(parents=True)
    route = "arx-lift2s-0907-blue-joint-feedback-ft"
    for name in MODULE.INFERENCE_FILES:
        (checkpoint / name).write_text(name, encoding="utf-8")
    (checkpoint / "policy_manifest.json").write_text(
        json.dumps({"routes": [route]}), encoding="utf-8"
    )
    (checkpoint / "run_spec.json").write_text(
        json.dumps({"git_hash": "training", "git_dirty": True}), encoding="utf-8"
    )
    (checkpoint / "optimizer.pt").write_text("excluded", encoding="utf-8")
    spec_dir = run / "finch_data_spec" / route
    spec_dir.mkdir(parents=True)
    deployment_contract = {"experiment": "joint-feedback"}
    for name in ("norm_stats.json", "components.json", "field_descriptions.json"):
        (spec_dir / name).write_text("{}", encoding="utf-8")
    (spec_dir / "spec.json").write_text(
        json.dumps({
            "finch_config_name": route,
            "robot_name": "arx_calibrated_joint_v1",
            "deployment_contract": deployment_contract,
        }),
        encoding="utf-8",
    )
    output = tmp_path / "bundle"
    result = MODULE.package(SimpleNamespace(
        checkpoint=checkpoint,
        run_root=run,
        output=output,
        model_id="blue-feedback-10000",
        task_instruction="pick blue",
        repo_root=ROOT,
        link_model=True,
    ))
    assert result == output
    assert (output / "finch_data_spec" / route / "spec.json").is_file()
    assert not (output / "optimizer.pt").exists()
    manifest = json.loads((output / "deployment_manifest.json").read_text())
    assert manifest["route"] == route
    assert manifest["training_git_dirty"] is True
    assert manifest["deployment_kind"] == "inference-only"
    assert manifest["model_storage"] == "hardlink"
    assert (output / "model.safetensors").stat().st_ino == (checkpoint / "model.safetensors").stat().st_ino
    assert "optimizer.pt" in manifest["excluded_training_state"]
    with pytest.raises(FileExistsError):
        MODULE.package(SimpleNamespace(
            checkpoint=checkpoint,
            run_root=run,
            output=output,
            model_id="blue-feedback-10000",
            task_instruction="pick blue",
            repo_root=ROOT,
            link_model=True,
        ))
