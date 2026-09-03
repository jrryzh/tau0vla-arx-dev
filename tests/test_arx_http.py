from __future__ import annotations

import json
import unittest
from io import BytesIO
from types import SimpleNamespace

import numpy as np
from fastapi import FastAPI
from fastapi.testclient import TestClient
from PIL import Image

from deploy.arx_lift2s_http import PROTOCOL_VERSION, build_router


def _jpeg(value: int) -> bytes:
    stream = BytesIO()
    Image.fromarray(np.full((12, 16, 3), value, dtype=np.uint8), mode="RGB").save(stream, format="JPEG")
    return stream.getvalue()


class _Policy:
    def __init__(self):
        self.data_spec = SimpleNamespace(
            robot_name="arx_lift2s_unified",
            unified_registry_key="arx_lift2s_14",
            action_chunk_size=30,
            action_semantics="state_t_plus_1",
            cam_keys=("head", "left_wrist", "right_wrist"),
            unified_has_eef=False,
        )
        self.last_payload = None

    def infer(self, payload):
        self.last_payload = payload
        state = np.asarray(payload["state"], dtype=np.float32)
        return {"actions": np.repeat(state[None], 30, axis=0)}


class ArxHttpTest(unittest.TestCase):
    def setUp(self):
        self.policy = _Policy()
        app = FastAPI()
        app.include_router(
            build_router(
                policy=self.policy,
                native_action=lambda value: value,
                model_id="test-model",
                checkpoint_sha256="abc123",
            )
        )
        self.client = TestClient(app)

    def test_contract_session_and_ordered_action_chunk(self):
        contract = self.client.get("/api/v1/arx-lift2s/policy-contract")
        self.assertEqual(contract.status_code, 200)
        self.assertEqual(contract.json()["action_horizon"], 30)
        session = self.client.post(
            "/api/v1/arx-lift2s/sessions",
            json={"protocol_version": PROTOCOL_VERSION, "task_instruction": "pick"},
        )
        self.assertEqual(session.status_code, 200)
        session_id = session.json()["session_id"]
        metadata = {
            "protocol_version": PROTOCOL_VERSION,
            "request_id": 1,
            "sample_monotonic_ns": 123,
            "observation_state": np.arange(14, dtype=np.float32).tolist(),
        }
        files = {name: (f"{name}.jpg", _jpeg(index), "image/jpeg") for index, name in enumerate(
            ("head", "left_wrist", "right_wrist"), start=1
        )}
        response = self.client.post(
            f"/api/v1/arx-lift2s/sessions/{session_id}/action-chunks",
            data={"metadata": json.dumps(metadata)},
            files=files,
        )
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(np.asarray(response.json()["actions"]).shape, (30, 14))
        self.assertEqual(self.policy.last_payload["images"]["head"].shape, (12, 16, 3))

        duplicate = self.client.post(
            f"/api/v1/arx-lift2s/sessions/{session_id}/action-chunks",
            data={"metadata": json.dumps(metadata)},
            files={name: (f"{name}.jpg", _jpeg(1), "image/jpeg") for name in (
                "head", "left_wrist", "right_wrist"
            )},
        )
        self.assertEqual(duplicate.status_code, 409)

    def test_rejects_invalid_state_and_old_session(self):
        first = self.client.post(
            "/api/v1/arx-lift2s/sessions",
            json={"protocol_version": PROTOCOL_VERSION, "task_instruction": "pick"},
        ).json()["session_id"]
        self.client.post(
            "/api/v1/arx-lift2s/sessions",
            json={"protocol_version": PROTOCOL_VERSION, "task_instruction": "pick again"},
        )
        metadata = {
            "protocol_version": PROTOCOL_VERSION,
            "request_id": 1,
            "sample_monotonic_ns": 123,
            "observation_state": [0.0] * 13,
        }
        response = self.client.post(
            f"/api/v1/arx-lift2s/sessions/{first}/action-chunks",
            data={"metadata": json.dumps(metadata)},
            files={name: (f"{name}.jpg", _jpeg(1), "image/jpeg") for name in (
                "head", "left_wrist", "right_wrist"
            )},
        )
        self.assertIn(response.status_code, (409, 422))


if __name__ == "__main__":
    unittest.main()
