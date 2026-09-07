from __future__ import annotations

import json
import unittest
from io import BytesIO
from types import SimpleNamespace

import httpx
import numpy as np
from fastapi import FastAPI
from PIL import Image

from deploy.arx_lift2s_http import PROTOCOL_VERSION, RTC_PROTOCOL_VERSION, build_router


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
        self.rtc_enabled = True
        self.rtc_max_delay = 8

    def infer(self, payload):
        self.last_payload = payload
        state = np.asarray(payload["state"], dtype=np.float32)
        return {"actions": np.repeat(state[None], 30, axis=0)}


class ArxHttpTest(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
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
        self.client = httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test")

    async def asyncTearDown(self):
        await self.client.aclose()

    async def test_contract_session_and_ordered_action_chunk(self):
        contract = await self.client.get("/api/v1/arx-lift2s/policy-contract")
        self.assertEqual(contract.status_code, 200)
        self.assertEqual(contract.json()["action_horizon"], 30)
        session = await self.client.post(
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
        response = await self.client.post(
            f"/api/v1/arx-lift2s/sessions/{session_id}/action-chunks",
            data={"metadata": json.dumps(metadata)},
            files=files,
        )
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(np.asarray(response.json()["actions"]).shape, (30, 14))
        self.assertEqual(self.policy.last_payload["images"]["head"].shape, (12, 16, 3))

        duplicate = await self.client.post(
            f"/api/v1/arx-lift2s/sessions/{session_id}/action-chunks",
            data={"metadata": json.dumps(metadata)},
            files={name: (f"{name}.jpg", _jpeg(1), "image/jpeg") for name in (
                "head", "left_wrist", "right_wrist"
            )},
        )
        self.assertEqual(duplicate.status_code, 409)

    async def test_rejects_invalid_state_and_old_session(self):
        first = (await self.client.post(
            "/api/v1/arx-lift2s/sessions",
            json={"protocol_version": PROTOCOL_VERSION, "task_instruction": "pick"},
        )).json()["session_id"]
        await self.client.post(
            "/api/v1/arx-lift2s/sessions",
            json={"protocol_version": PROTOCOL_VERSION, "task_instruction": "pick again"},
        )
        metadata = {
            "protocol_version": PROTOCOL_VERSION,
            "request_id": 1,
            "sample_monotonic_ns": 123,
            "observation_state": [0.0] * 13,
        }
        response = await self.client.post(
            f"/api/v1/arx-lift2s/sessions/{first}/action-chunks",
            data={"metadata": json.dumps(metadata)},
            files={name: (f"{name}.jpg", _jpeg(1), "image/jpeg") for name in (
                "head", "left_wrist", "right_wrist"
            )},
        )
        self.assertIn(response.status_code, (409, 422))

    async def test_v2_contract_exact_prefix_and_validation(self):
        contract = await self.client.get("/api/v2/arx-lift2s/policy-contract")
        self.assertEqual(contract.status_code, 200)
        self.assertTrue(contract.json()["rtc_enabled"])
        self.assertEqual(contract.json()["rtc_max_delay"], 8)
        session = await self.client.post(
            "/api/v2/arx-lift2s/sessions",
            json={"protocol_version": RTC_PROTOCOL_VERSION, "task_instruction": "pick"},
        )
        session_id = session.json()["session_id"]
        prefix = np.arange(42, dtype=np.float32).reshape(3, 14) / 10
        metadata = {
            "protocol_version": RTC_PROTOCOL_VERSION,
            "request_id": 1,
            "sample_monotonic_ns": 123,
            "observation_state": [0.0] * 14,
            "rtc_delay": 3,
            "action_prefix": prefix.tolist(),
        }

        def files():
            return {
                name: (f"{name}.jpg", _jpeg(1), "image/jpeg")
                for name in ("head", "left_wrist", "right_wrist")
            }

        response = await self.client.post(
            f"/api/v2/arx-lift2s/sessions/{session_id}/action-chunks",
            data={"metadata": json.dumps(metadata)},
            files=files(),
        )
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.json()["rtc_delay"], 3)
        np.testing.assert_array_equal(np.asarray(response.json()["actions"], dtype=np.float32)[:3], prefix)
        self.assertEqual(self.policy.last_payload["meta"]["rtc_delay"], 3)

        metadata.update(request_id=2, rtc_delay=2)  # prefix length deliberately remains three
        invalid = await self.client.post(
            f"/api/v2/arx-lift2s/sessions/{session_id}/action-chunks",
            data={"metadata": json.dumps(metadata)},
            files=files(),
        )
        self.assertEqual(invalid.status_code, 422)

        metadata.update(rtc_delay=9, action_prefix=[[0.0] * 14 for _ in range(9)])
        too_large = await self.client.post(
            f"/api/v2/arx-lift2s/sessions/{session_id}/action-chunks",
            data={"metadata": json.dumps(metadata)},
            files=files(),
        )
        self.assertEqual(too_large.status_code, 422)

    async def test_v2_rejects_non_rtc_checkpoint(self):
        policy = _Policy()
        policy.rtc_enabled = False
        policy.rtc_max_delay = 0
        app = FastAPI()
        app.include_router(
            build_router(
                policy=policy,
                native_action=lambda value: value,
                model_id="legacy-model",
                checkpoint_sha256=None,
            )
        )
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://legacy"
        ) as client:
            contract = (await client.get("/api/v2/arx-lift2s/policy-contract")).json()
            self.assertFalse(contract["rtc_enabled"])
            session = await client.post(
                "/api/v2/arx-lift2s/sessions",
                json={"protocol_version": RTC_PROTOCOL_VERSION, "task_instruction": "pick"},
            )
            metadata = {
                "protocol_version": RTC_PROTOCOL_VERSION,
                "request_id": 1,
                "sample_monotonic_ns": 1,
                "observation_state": [0.0] * 14,
                "rtc_delay": 0,
                "action_prefix": [],
            }
            response = await client.post(
                f"/api/v2/arx-lift2s/sessions/{session.json()['session_id']}/action-chunks",
                data={"metadata": json.dumps(metadata)},
                files={
                    name: (f"{name}.jpg", _jpeg(1), "image/jpeg")
                    for name in ("head", "left_wrist", "right_wrist")
                },
            )
            self.assertEqual(response.status_code, 409)


if __name__ == "__main__":
    unittest.main()
