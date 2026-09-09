import importlib.util
from pathlib import Path

import numpy as np
import pytest


SPEC = importlib.util.spec_from_file_location("probe_arx_feedback", Path(__file__).parents[1] / "scripts/probe_arx_feedback.py")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_camera_fixture_is_deterministic_and_distinct():
    first = MODULE.synthetic_images(640, 480)
    assert first == MODULE.synthetic_images(640, 480)
    assert len({value[1] for value in first.values()}) == 3


def test_action_validation_rejects_wrong_shape_identity_and_nonfinite():
    metadata = {"request_id": 1, "sample_monotonic_ns": 123}
    payload = {"protocol_version": MODULE.PROTOCOL, "calibration_version": MODULE.CALIBRATION,
               "required_client_adapter_version": MODULE.CLIENT, "session_id": "session",
               **metadata, "model_id": "model", "experiment": "joint-feedback",
               "gripper_semantics": "calibrated_feedback_position", "component_source_offsets": MODULE.OFFSETS,
               "offset_unit": "uploaded_30fps_frame", "wire_action_is_robot_command": False,
               "recording_mode": "background-serialized", "calibrated_action_chunk": np.zeros((30, 14)).tolist()}
    MODULE.validate_action(payload, metadata, "session", "model")
    with pytest.raises(RuntimeError, match="request_id"):
        MODULE.validate_action({**payload, "request_id": 2}, metadata, "session", "model")
    for invalid in (np.zeros((29, 14)), np.full((30, 14), np.nan)):
        with pytest.raises(RuntimeError, match="invalid action"):
            MODULE.validate_action({**payload, "calibrated_action_chunk": invalid}, metadata, "session", "model")
