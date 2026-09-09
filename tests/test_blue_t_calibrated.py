import json
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from tau0_vla.adapters.arx_lift2s.calibrated import (
    VERSION, PROTOCOL, POSE_INDICES, MODES, ArxCalibratedJoint, ArxCalibratedEEF,
    calibrate_joint, contract, estimate_open, labels,
)
from tau0_vla.adapters.arx_lift2s.feedback import (
    BODY as FEEDBACK_BODY,
    PROTOCOL as FEEDBACK_PROTOCOL,
    VERSION as FEEDBACK_VERSION,
    ArxFeedbackJoint,
    contract as feedback_contract,
)
from tau0_vla.data.data_spec import build_unified_state_encoder, encode_unified_action_prefix, restore_action, action_slices
from deploy.arx_calibrated_http import adapt_request, build_calibrated_app


def test_open_baseline_removes_episode_offset_without_closed_endpoint_fitting():
    vr = np.r_[np.full(40, 5.), np.zeros(50), np.full(40, 5.)]
    travel = np.r_[np.zeros(40), np.full(50, .37), np.zeros(40)]
    results = []
    for offset in (-5.8, .123, 8.2):
        report = estimate_open(travel+offset, vr)
        assert report["baseline"] == pytest.approx(offset)
        q = np.zeros((len(vr), 14))
        q[:, [6, 13]] = (travel+offset)[:, None]
        results.append(calibrate_joint(q, [report["baseline"]]*2))
    for result in results:
        np.testing.assert_allclose(result[:, 6], travel, atol=1e-7)
        assert result[50, 6] == pytest.approx(.37)  # object-held position stays .37, never stretched to 1
    q = np.zeros(14); q[6] = -2.0001; q[13] = -2
    assert calibrate_joint(q, [-2, -2])[6] < 0


def test_missing_unstable_and_shifted_platforms_rejected():
    with pytest.raises(ValueError, match="no qualifying"):
        estimate_open(np.zeros(100), np.zeros(100))
    with pytest.raises(ValueError, match="no qualifying"):
        estimate_open(np.arange(100)*.01, np.full(100, 5))
    q = np.r_[np.zeros(40), np.ones(40), np.full(40, .021)]
    with pytest.raises(ValueError, match="drift"):
        estimate_open(q, np.r_[np.full(40, 5), np.zeros(40), np.full(40, 5)])


@pytest.mark.parametrize("mode", MODES)
def test_component_source_indices_and_vr_direction(mode):
    q = np.arange(11*14).reshape(11, 14)/100
    eef = q + 5
    cmd = q + 10
    cmd[:, 6] = 5; cmd[:, 13] = 0
    indices, state, action = labels(q, eef, cmd, [-2, 3], mode)
    np.testing.assert_array_equal(indices, [0, 2, 4, 6, 8])
    np.testing.assert_allclose(state[:, 6], q[indices, 6]+2)
    np.testing.assert_allclose(action[:, :6], q[indices+2, :6])
    if mode == "joint-feedback":
        np.testing.assert_allclose(action[:, 6], q[indices+2, 6]+2)
    else:
        np.testing.assert_array_equal(action[:, 6], 0)
        np.testing.assert_array_equal(action[:, 13], 1)
    if mode == "eef-vr":
        np.testing.assert_allclose(state[:, 14:], eef[indices][:, POSE_INDICES])
        np.testing.assert_allclose(action[:, 14:], cmd[indices][:, POSE_INDICES])


def spec_for(tmp_path, mode):
    cls = ArxCalibratedEEF if mode == "eef-vr" else ArxCalibratedJoint
    stats = {"mean": [0.]*40, "std": [1.]*40, "q01": [-1.]*40, "q99": [1.]*40}
    path = tmp_path / (mode+".json")
    path.write_text(json.dumps({"format_version": 2, "norm_stats": {"state": stats, "action": stats},
        "per_embodiment": {cls._unified_registry_key: {"state": stats, "action": stats}}}))
    return SimpleNamespace(robot_cls=cls, robot_name=cls.robot_name, unified_registry_key=cls._unified_registry_key,
        unified_has_eef=mode == "eef-vr", config_modules=(), finch_config_name=mode+str(tmp_path), artifacts_dir=str(tmp_path),
        norm_stats_path=str(path), action_dim=40, action_chunk_size=30,
        cam_keys=("head", "left_wrist", "right_wrist"),
        deployment_contract=contract(mode), action_semantics=contract(mode)["action_semantics"])


@pytest.mark.parametrize("mode", ("joint-feedback", "joint-vr"))
def test_server_training_encoding_and_rotation_roundtrip(tmp_path, mode):
    spec = spec_for(tmp_path, mode)
    rng = np.random.default_rng(3)
    q = rng.uniform(-.5, .5, (64,14))
    eef = rng.uniform(-.5, .5, (64,14))
    cmd = rng.uniform(-.5, .5, (64,14)); cmd[:, [6,13]] = [5, 0]
    _, states, actions = labels(q,eef,cmd,[.12, -.13],mode)
    raw = {"protocol_version": PROTOCOL, "calibration_version": VERSION, "experiment": mode,
        "raw_joint_feedback": q[0].tolist(), "raw_eef_feedback": eef[0].tolist(),
        "open_baselines": {"left": .12, "right": -.13}, "task_instruction": "pick", "request_id": 1, "sample_monotonic_ns": 1}
    images = {key: np.zeros((8,8,3), np.uint8) for key in ("head", "left_wrist", "right_wrist")}
    payload = adapt_request(raw, images, spec)
    np.testing.assert_array_equal(payload["state"], states[0])
    assembler = spec.robot_cls(repo_id="unused")._build_component_assembler(field_descriptions={}, disable_component_normalization=True)
    train = assembler({"_state_raw": states[0], "_action_raw": actions[:30], "prompt": "", "images": {}})
    encoded = build_unified_state_encoder(spec)(payload["state"])
    active = list(range(20)) if mode == "eef-vr" else [18,19,*range(24,30),*range(32,38)]
    np.testing.assert_array_equal(np.flatnonzero(encoded["state_mask"]), active)
    np.testing.assert_array_equal(encoded["state"], train["state"])
    np.testing.assert_array_equal(encoded["action_mask"], train["action_mask"])
    model_actions = encode_unified_action_prefix(states[0], actions[:30], spec)
    np.testing.assert_allclose(model_actions, train["action"], atol=1e-6)
    restored = restore_action(model_actions, spec, state=encoded["state_abs"])
    split = {name: restored[:, offset:offset+dim] for name,offset,dim in action_slices(spec)}
    np.testing.assert_allclose(split["left_gripper"][:,0], actions[:30,6], atol=1e-6)
    np.testing.assert_allclose(split["right_gripper"][:,0], actions[:30,13], atol=1e-6)
    if mode == "eef-vr":
        for side,start in (("left",14),("right",20)):
            np.testing.assert_allclose(split[side+"_eef"][:,:3], actions[:30,start:start+3], atol=1e-6)
            actual = Rotation.from_quat(split[side+"_eef"][:,3:]).as_matrix()
            expected = Rotation.from_euler("xyz", actions[:30,start+3:start+6]).as_matrix()
            np.testing.assert_allclose(actual, expected, atol=1e-5)
    else:
        np.testing.assert_allclose(split["left_arm"], actions[:30,:6], atol=1e-6)
        np.testing.assert_allclose(split["right_arm"], actions[:30,7:13], atol=1e-6)
    with pytest.raises(ValueError, match="pre-calibrated"):
        adapt_request({**raw, "state": states[0]}, images, spec)
    import asyncio
    import httpx
    policy = SimpleNamespace(data_spec=spec, rtc_enabled=False, infer=lambda p: {"actions": restored})
    async def check():
        from io import BytesIO
        from PIL import Image
        jpeg = BytesIO()
        Image.fromarray(images["head"]).save(jpeg, format="JPEG")
        files = {key: (key+".jpg", jpeg.getvalue(), "image/jpeg") for key in images}
        transport = httpx.ASGITransport(app=build_calibrated_app(
            policy, model_id="calibrated-test", checkpoint_sha256="abc", record_dir=tmp_path
        ))
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            health = await client.get('/health')
            assert health.status_code == 200
            assert health.json()["protocol_version"] == PROTOCOL
            assert health.json()["recording_mode"] == "background-serialized"
            assert health.json()["recording_error"] is None
            assert (await client.get('/arx/v3/policy-contract')).status_code == 200
            assert (await client.post('/act')).status_code == 404
            assert (await client.post('/arx/v1/sessions')).status_code == 404
            session_payload = {
                "protocol_version": PROTOCOL,
                "calibration_version": VERSION,
                "client_adapter_version": "arx-calibrated-client-v1",
                "experiment": mode,
                "task_instruction": "pick",
                "client_name": "test",
                "robot_id": "ark-2",
                "calibration_id": "calibration",
                "open_baselines": {"left": .12, "right": -.13},
            }
            session = await client.post('/arx/v3/sessions', json=session_payload)
            assert session.status_code == 200, session.text
            session_id = session.json()["session_id"]
            request = {
                "protocol_version": PROTOCOL,
                "request_id": 1,
                "sample_monotonic_ns": 1,
                "raw_joint_feedback": q[0].tolist(),
                "raw_eef_feedback": eef[0].tolist(),
            }
            rejected = await client.post(
                f'/arx/v3/sessions/{session_id}/action-chunks',
                data={"metadata": json.dumps({**request, "protocol_version": "arx-v1"})},
                files=files,
            )
            assert rejected.status_code == 409
            response = await client.post(
                f'/arx/v3/sessions/{session_id}/action-chunks',
                data={"metadata": json.dumps(request)},
                files=files,
            )
            assert response.status_code == 200, response.text
            assert response.json()["gripper_semantics"] == contract(mode)["gripper_action"]
            assert response.json()["control_mode"] == mode.split('-')[0]
            assert response.json()["wire_action_is_robot_command"] is False
            assert response.json()["recording_mode"] == "background-serialized"
            assert response.json()["preprocess_ms"] >= 0
            assert np.asarray(response.json()["calibrated_action_chunk"]).shape == (30, 14)
            duplicate = await client.post(
                f'/arx/v3/sessions/{session_id}/action-chunks',
                data={"metadata": json.dumps(request)},
                files=files,
            )
            assert duplicate.status_code == 409
        recordings = list(tmp_path.rglob('request-*.npz'))
        assert len(recordings) == 1
        with np.load(recordings[0], allow_pickle=False) as recording:
            np.testing.assert_array_equal(recording['calibrated_native_state'], states[0])
            assert recording['calibrated_action_chunk'].shape == (30, 14)
            saved_request = json.loads(str(recording['request_json']))
            assert saved_request["calibration_id"] == "calibration"
            assert json.loads(str(recording['response_json']))['request_received_utc']
    asyncio.run(check())


def test_new_calibrated_session_invalidates_previous_session(tmp_path):
    import asyncio
    import httpx

    spec = spec_for(tmp_path, "joint-feedback")
    policy = SimpleNamespace(
        data_spec=spec,
        rtc_enabled=False,
        infer=lambda payload: {"actions": np.zeros((30, 14), dtype=np.float32)},
    )
    app = build_calibrated_app(policy, record_dir=tmp_path)
    payload = {
        "protocol_version": PROTOCOL,
        "calibration_version": VERSION,
        "client_adapter_version": "wrong-client",
        "experiment": "joint-feedback",
        "task_instruction": "pick",
        "client_name": "test",
        "robot_id": "ark-2",
        "calibration_id": "one",
        "open_baselines": {"left": -3.3, "right": -3.3},
    }

    async def check():
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            rejected = await client.post('/arx/v3/sessions', json=payload)
            assert rejected.status_code == 409
            payload["client_adapter_version"] = "arx-calibrated-client-v1"
            first = (await client.post('/arx/v3/sessions', json=payload)).json()["session_id"]
            second = (await client.post(
                '/arx/v3/sessions', json={**payload, "calibration_id": "two"}
            )).json()["session_id"]
            assert first != second

    asyncio.run(check())


def test_feedback_v4_sessioned_http_omits_eef_and_preserves_recording(tmp_path):
    import asyncio
    from io import BytesIO

    import httpx
    from PIL import Image

    spec = spec_for(tmp_path, "joint-feedback")
    spec.robot_cls = ArxFeedbackJoint
    spec.robot_name = FEEDBACK_BODY
    spec.unified_registry_key = FEEDBACK_BODY
    spec.finch_config_name = "arx-lift2s-0908-all-joint-feedback-ft"
    spec.deployment_contract = feedback_contract()
    spec.action_semantics = feedback_contract()["action_semantics"]
    policy = SimpleNamespace(
        data_spec=spec,
        rtc_enabled=False,
        infer=lambda payload: {"actions": np.zeros((30, 14), dtype=np.float32)},
    )
    app = build_calibrated_app(
        policy,
        model_id="feedback-v4-test",
        checkpoint_sha256="v4sha",
        record_dir=tmp_path,
    )
    image = BytesIO()
    Image.fromarray(np.zeros((8, 8, 3), dtype=np.uint8)).save(image, format="JPEG")
    files = {
        name: (f"{name}.jpg", image.getvalue(), "image/jpeg")
        for name in ("head", "left_wrist", "right_wrist")
    }

    async def check():
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            health = await client.get("/health")
            assert health.json()["protocol_version"] == FEEDBACK_PROTOCOL
            assert health.json()["recording_mode"] == "background-serialized"
            contract_response = await client.get("/arx/v4/policy-contract")
            assert contract_response.status_code == 200
            assert contract_response.json()["calibration_version"] == FEEDBACK_VERSION
            assert "raw_eef_feedback" not in contract_response.json()["request_fields"]
            assert (await client.get("/arx/v3/policy-contract")).status_code == 404
            opened = await client.post(
                "/arx/v4/sessions",
                json={
                    "protocol_version": FEEDBACK_PROTOCOL,
                    "calibration_version": FEEDBACK_VERSION,
                    "client_adapter_version": "arx-calibrated-client-v1",
                    "experiment": "joint-feedback",
                    "task_instruction": "pick",
                    "client_name": "test",
                    "robot_id": "ark-2",
                    "calibration_id": "calibration",
                    "open_baselines": {"left": -3.3, "right": -3.3},
                },
            )
            assert opened.status_code == 200, opened.text
            session_id = opened.json()["session_id"]
            request = {
                "protocol_version": FEEDBACK_PROTOCOL,
                "request_id": 1,
                "sample_monotonic_ns": 1,
                "raw_joint_feedback": [0.0] * 14,
            }
            for field in ("request_id", "sample_monotonic_ns"):
                for invalid in (True, 1.5, "1", 0, -1):
                    rejected = await client.post(
                        f"/arx/v4/sessions/{session_id}/action-chunks",
                        data={"metadata": json.dumps({**request, field: invalid})}, files=files,
                    )
                    assert rejected.status_code == 422
            response = await client.post(
                f"/arx/v4/sessions/{session_id}/action-chunks",
                data={"metadata": json.dumps(request)},
                files=files,
            )
            assert response.status_code == 200, response.text
            payload = response.json()
            assert payload["protocol_version"] == FEEDBACK_PROTOCOL
            assert payload["calibration_version"] == FEEDBACK_VERSION
            assert payload["component_source_offsets"] == {
                "state": 0,
                "arm_action": 1,
                "gripper_action": 1,
            }
            for invalid_id in (1, 3):
                rejected = await client.post(
                    f"/arx/v4/sessions/{session_id}/action-chunks",
                    data={"metadata": json.dumps({**request, "request_id": invalid_id})}, files=files,
                )
                assert rejected.status_code == 409
        recordings = list(tmp_path.rglob("request-*.npz"))
        assert len(recordings) == 1
        with np.load(recordings[0], allow_pickle=False) as recording:
            assert recording["raw_eef_feedback"].size == 0
            assert recording["calibrated_action_chunk"].shape == (30, 14)

    asyncio.run(check())


def test_constant_gripper_normalization_is_finite():
    from tau0_vla.data.robots.unified import UnifiedAssembler
    from tau0_vla.data.stats import NormStats
    stats = NormStats(mean=np.zeros(40), std=np.zeros(40), q01=np.zeros(40), q99=np.zeros(40))
    assembler = UnifiedAssembler(registry_key=ArxCalibratedJoint._unified_registry_key, has_eef_action=False,
        has_eef_state=False, norm_stats={"state": stats,"action":stats})
    out = assembler({"_state_raw": np.zeros(14), "_action_raw": np.zeros((30,14)), "images":{}, "prompt":""})
    assert np.isfinite(out["action"]).all() and np.isfinite(out["state"]).all()
    np.testing.assert_array_equal(out["action"][:,18:20],0)
