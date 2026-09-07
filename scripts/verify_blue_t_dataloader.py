#!/usr/bin/env python3
"""Verify actual anchors, saved contracts, statistics and server/train round trips."""
import argparse
import importlib
import json
from pathlib import Path
import sys
import time

from prepare_blue_t_training import REPO, EVIDENCE, TASKS, MODES, profile, config_name, dataset_path, save, sha256, decode
sys.path[:0] = [str(REPO / "src"), str(REPO)]
import h5py
import numpy as np
import pyarrow.parquet as pq
from scipy.spatial.transform import Rotation
from torch.utils.data import BatchSampler, DistributedSampler

from tau0_vla.data import FinchDataLoader, load_data_spec, encode_payload, restore_action, action_slices, encode_unified_action_prefix
from tau0_vla.data.data_spec import save_data_spec
from tau0_vla.adapters.arx_lift2s.calibrated import VERSION, PROTOCOL, contract, POSE_INDICES
from deploy.arx_calibrated_http import adapt_request


def verify(name, mode, evidence_root=EVIDENCE):
    slug = profile(name, mode)
    evidence = Path(evidence_root) / slug
    ready = json.loads((evidence / "converted.json").read_text())
    cname = config_name(name, mode)
    importlib.import_module(f"configs.{cname}.data")
    loader = FinchDataLoader.from_config_name(cname+"_ft")
    dataset = loader.dataset
    assert len(dataset) == ready["anchors"]
    batches = [len(BatchSampler(DistributedSampler(dataset, num_replicas=16, rank=rank, seed=42), batch_size=8, drop_last=True)) for rank in range(16)]
    assert batches == [ready["expected_batches_per_rank"]]*16
    manifest = json.loads((evidence / "source_manifest.json").read_text())
    spec_path = save_data_spec(cname+"_ft", evidence / "deployment", vlm_model_type="qwen3.5", max_images_per_sample=3)
    spec = load_data_spec(spec_path)
    assert spec.deployment_contract == contract(mode)
    assert spec.action_semantics == contract(mode)["action_semantics"] and spec.action_offset_frames is None
    assert spec.is_eef == (mode == "eef-vr")
    active = list(range(20)) if mode == "eef-vr" else [18,19,*range(24,30),*range(32,38)]
    partial = np.load(next((evidence / "stats").glob("*.npz")))
    assert int(partial["n_frames"]) == len(dataset)
    expected_count = np.zeros(40); expected_count[active] = len(dataset)
    np.testing.assert_array_equal(partial["state_count"], expected_count)
    np.testing.assert_array_equal(partial["action_count"], expected_count*30)
    if name == "BlueT":
        parents=[np.load(next((EVIDENCE/profile(task,mode)/"stats").glob("*.npz"))) for task in TASKS]
        for role in ("state","action"):
            for field in ("sum","sumsq","count"):
                key=role+"_"+field
                np.testing.assert_allclose(partial[key],sum(p[key] for p in parents),rtol=1e-9,atol=1e-6)
    start, checked, max_rotation_error = 0, 0, 0.
    replay_tasks=set()
    for ep in manifest["episodes"]:
        count = max(0, ep["output_frames"]-29)
        n = ep["episode_index"]
        table = pq.read_table(dataset_path(name,mode) / f"data/chunk-000/file-{n:03d}.parquet")
        states = np.asarray(table["observation.state"].to_pylist(), dtype=np.float32)
        actions = np.asarray(table["action"].to_pylist(), dtype=np.float32)
        task=ep["dataset"]
        np.testing.assert_array_equal(np.asarray(table["task_index"]),np.full(len(table),ep.get("task_index",0)))
        for local in sorted({0, count-1}) if count else []:
            sample = dataset[start+local]
            assert sample["state"].shape == (40,) and sample["action"].shape == (30,40)
            assert TASKS[task] in sample["prompt"]
            assert set(sample["images"]) == {"head", "left_wrist", "right_wrist"}
            with h5py.File(ep["source_path"], "r") as f:
                raw = {"protocol_version": PROTOCOL, "calibration_version": VERSION, "experiment": mode,
                    "raw_joint_feedback": f["observations/qpos"][2*local].tolist(), "raw_eef_feedback": f["observations/eef"][2*local].tolist(),
                    "open_baselines": {side: ep["sides"][side]["baseline"] for side in ("left", "right")},
                    "request_id": checked+1, "sample_monotonic_ns": time.monotonic_ns(), "task_instruction": TASKS[task],
                    "timestamp_source": "offline request generation; source collection has no frame timestamps",
                    "source_episode": ep["file"], "source_dataset":task, "source_frame": 2*local}
                if task not in replay_tasks:
                    replay_images = {k: decode(f[f"observations/images/{k}"][2*local]) for k in ("head","left_wrist","right_wrist")}
            payload = adapt_request(raw, sample["images"], spec)
            np.testing.assert_array_equal(payload["state"], states[local])
            encoded = encode_payload(payload,spec)
            for key in ("state_mask", "action_mask"):
                np.testing.assert_array_equal(np.flatnonzero(sample[key]), active)
                np.testing.assert_array_equal(encoded[key], sample[key])
            np.testing.assert_allclose(encoded["state"], sample["state"], atol=2e-6)
            predicted = encode_unified_action_prefix(states[local], actions[local:local+30], spec)
            np.testing.assert_allclose(predicted, sample["action"], atol=2e-6)
            restored = restore_action(predicted,spec,state=encoded["state_abs"])
            split = {key:restored[:,offset:offset+dim] for key,offset,dim in action_slices(spec)}
            a = actions[local:local+30]
            for side,g,j,e in (("left",6,0,14),("right",13,7,20)):
                np.testing.assert_allclose(split[side+"_gripper"][:,0], a[:,g],atol=2e-6)
                if mode == "eef-vr":
                    np.testing.assert_allclose(split[side+"_eef"][:,:3],a[:,e:e+3],atol=2e-6)
                    delta = Rotation.from_euler("xyz",a[:,e+3:e+6]).inv()*Rotation.from_quat(split[side+"_eef"][:,3:])
                    error = float(delta.magnitude().max())
                    max_rotation_error = max(max_rotation_error,error)
                    assert error < 5e-4, error
                else:
                    np.testing.assert_allclose(split[side+"_arm"],a[:,j:j+6],atol=2e-6)
            if task not in replay_tasks:
                with (evidence / f"offline_request_{task}.npz").open("wb") as stream:
                    np.savez_compressed(stream,request_json=json.dumps(raw),**{f"image_{k}":v for k,v in replay_images.items()})
                if checked == 0:
                    (evidence / "offline_request.npz").write_bytes((evidence/f"offline_request_{task}.npz").read_bytes())
                replay_tasks.add(task)
            checked += 1
        start += count
    stats_path = REPO / "configs" / cname / "norm_stats.json"
    assert json.loads(stats_path.read_text()) == json.loads(Path(spec.norm_stats_path).read_text())
    report = {**ready,"validation":"ok","batches_per_rank":batches,"vla_epoch_at_10000":10000/batches[0],
        "episode_boundary_samples_checked":checked,"state_action_active_slots":active,"statistics_anchors_match":True,
        "server_train_encoding_parity":True,"max_rotation_roundtrip_error_radians":max_rotation_error,
        "task_instructions_verified":sorted(replay_tasks),"mixed_sufficient_statistics_match":name=="BlueT",
        "norm_stats_sha256":sha256(stats_path),"saved_spec":str(spec_path)}
    save(evidence / "dataloader_verification.json",report)
    save(evidence / "ready.json",report)
    print(json.dumps(report),flush=True)


if __name__ == "__main__":
    p=argparse.ArgumentParser()
    p.add_argument("name",choices=[*TASKS,"BlueT"])
    p.add_argument("mode",choices=MODES)
    p.add_argument("--evidence-root",type=Path,default=EVIDENCE)
    args=p.parse_args()
    verify(args.name,args.mode,args.evidence_root)
