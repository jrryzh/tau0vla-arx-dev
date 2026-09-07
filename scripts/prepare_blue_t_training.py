#!/usr/bin/env python3
"""Read-only HDF5 -> versioned LeRobot v3, with exhaustive lossless video checks."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import copy
import hashlib
from io import BytesIO
import json
import os
from pathlib import Path
import subprocess
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO / "src"), str(REPO)]

import av
import h5py
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image

from tau0_vla.adapters.arx_lift2s.calibrated import (
    MODES, VERSION, GRIP_INDICES, POSE_NAMES, contract, estimate_open, labels,
)
from tau0_vla.adapters.arx_lift2s.layout import ARX_LIFT2S_JOINT_NAMES

CAMERAS = ("head", "left_wrist", "right_wrist")
TASKS = {"Blue": "Pick up the blue box and place it in its designated position on the board.",
         "T": "Pick up the T-shaped part and place it in its designated position on the board."}
DATA = REPO / "data/0907_blue_t_v1"
EVIDENCE = REPO / "outputs/0907_blue_t_preparation"


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    tmp.replace(path)


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def decode(value):
    with Image.open(BytesIO(np.asarray(value, dtype=np.uint8).tobytes())) as image:
        image.load()
        result = np.asarray(image.convert("RGB"))
    if result.shape != (480, 640, 3):
        raise ValueError(f"invalid image shape: {result.shape}")
    return result


def profile(name, mode):
    return "0907-" + name.lower() + "-" + mode


def config_name(name, mode):
    return "arx_lift2s_" + profile(name, mode).replace("-", "_")


def dataset_path(name, mode):
    return DATA / name / mode


def convert_episode(job):
    name, path_text, number = job
    path = Path(path_text)
    proof = EVIDENCE / name / f"episode_{number:03d}.json"
    if proof.exists():
        row = json.loads(proof.read_text())
        if sha256(path) != row["sha256"]:
            raise ValueError(f"source changed: {path}")
        for relative, digest in row["artifacts"].items():
            if sha256(DATA / relative) != digest:
                raise ValueError(f"output changed: {relative}")
        return row
    before = path.stat()
    digest = sha256(path)
    artifacts = {}
    with h5py.File(path, "r") as source:
        if int(source.attrs["frame_rate"]) != 60:
            raise ValueError("source must be 60 FPS")
        q, eef, cmd = (source[k][()] for k in ("observations/qpos", "observations/eef", "action_poscmd"))
        sides = {side: estimate_open(q[:, i], cmd[:, i]) for side, i in zip(("left", "right"), GRIP_INDICES)}
        baselines = [sides[side]["baseline"] for side in ("left", "right")]
        for mode in MODES:
            indices, state, action = labels(q, eef, cmd, baselines, mode)
            size = len(indices)
            dim = state.shape[1]
            table = pa.table({"observation.state": pa.array(state.tolist(), type=pa.list_(pa.float32(), dim)),
                "action": pa.array(action.tolist(), type=pa.list_(pa.float32(), dim)),
                "timestamp": pa.array(np.arange(size, dtype=np.float32) / 30),
                "frame_index": pa.array(np.arange(size, dtype=np.int64)),
                "episode_index": pa.array(np.full(size, number, np.int64)),
                "index": pa.array(np.arange(size, dtype=np.int64)),
                "task_index": pa.array(np.zeros(size, np.int64))})
            out = dataset_path(name, mode) / f"data/chunk-000/file-{number:03d}.parquet"
            out.parent.mkdir(parents=True, exist_ok=True)
            pq.write_table(table, out)
        video_checks = {}
        for camera in CAMERAS:
            images = source[f"observations/images/{camera}"]
            if len(images) != len(q):
                raise ValueError(f"misaligned camera: {path}/{camera}")
            relative = f"videos/observation.images.{camera}/chunk-000/file-{number:03d}.mp4"
            out = dataset_path(name, MODES[0]) / relative
            out.parent.mkdir(parents=True, exist_ok=True)
            selected_digest = hashlib.sha256()
            selected = set(indices.tolist())
            with av.open(str(out), "w") as container:
                stream = container.add_stream("libx264rgb", rate=30)
                stream.width, stream.height, stream.pix_fmt = 640, 480, "rgb24"
                stream.options = {"crf": "0", "preset": "fast", "threads": "1", "g": "2"}
                for i in range(len(images)):
                    rgb = decode(images[i])  # Every source frame, including discarded frames.
                    if i not in selected:
                        continue
                    selected_digest.update(rgb.tobytes())
                    for packet in stream.encode(av.VideoFrame.from_ndarray(rgb, format="rgb24")):
                        container.mux(packet)
                for packet in stream.encode():
                    container.mux(packet)
            decoded_digest = hashlib.sha256()
            decoded_frames = 0
            with av.open(str(out)) as container:
                stream = container.streams.video[0]
                if stream.average_rate != 30:
                    raise ValueError("output FPS mismatch")
                for i, frame in enumerate(container.decode(stream)):
                    rgb = frame.to_ndarray(format="rgb24")
                    if rgb.shape != (480, 640, 3) or abs(float(frame.pts * frame.time_base) - i / 30) > 1e-5:
                        raise ValueError("video shape/timestamp mismatch")
                    decoded_digest.update(rgb.tobytes())
                    decoded_frames += 1
            if decoded_frames != size or decoded_digest.digest() != selected_digest.digest():
                raise ValueError(f"lossless source/video parity failed: {out}")
            for mode in MODES[1:]:
                target = dataset_path(name, mode) / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                if target.exists():
                    target.unlink()
                os.link(out, target)
            video_checks[camera] = {"all_source_frames_decoded": len(images), "output_frames_decoded": decoded_frames,
                "source_selected_rgb_sha256": selected_digest.hexdigest(), "decoded_video_rgb_sha256": decoded_digest.hexdigest(), "pixel_exact": True}
            video_digest = sha256(out)
            for mode in MODES:
                artifacts[str((dataset_path(name, mode) / relative).relative_to(DATA))] = video_digest
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns) or sha256(path) != digest:
        raise ValueError(f"source changed during conversion: {path}")
    row = {"dataset": name, "episode_index": number, "source_path": str(path.resolve()), "file": path.name,
        "source_episode_number": int(path.stem.split("_")[-1]), "sha256": digest, "bytes": before.st_size,
        "mtime_ns": before.st_mtime_ns, "source_frames": len(q), "output_frames": size, "source_frame_start": 0,
        "source_frame_stride": 2, "source_frame_last": int(indices[-1]), "sides": sides, "videos": video_checks, "artifacts": artifacts}
    save(proof, row)
    print(f"{name} episode={number} source={path.name} frames={size} calibrated/video=ok", flush=True)
    return row


def features(mode):
    result = {f"observation.images.{camera}": {"dtype": "video", "shape": [480,640,3], "names": ["height","width","channels"],
        "info": {"video.height":480,"video.width":640,"video.codec":"h264","video.pix_fmt":"gbrp","video.is_depth_map":False,"video.fps":30,"video.channels":3,"has_audio":False}}
        for camera in CAMERAS}
    for role,key in (("state","observation.state"),("action","action")):
        result[key] = {"dtype":"float32","shape":[14],"names":list(ARX_LIFT2S_JOINT_NAMES),"field_descriptions":{
            f"{role}/joint/position":{"description":"dual-arm joints in radians","dimensions":12,"indices":[*range(6),*range(7,13)]},
            f"{role}/left_effector/position":{"dimensions":1,"indices":[6]},
            f"{role}/right_effector/position":{"dimensions":1,"indices":[13]}}}
    for key in ("timestamp","frame_index","episode_index","index","task_index"):
        result[key] = {"dtype":"float32" if key=="timestamp" else "int64","shape":[1],"names":None}
    for role, key in (("state", "observation.state"), ("action", "action")):
        feat = result[key]
        if mode == "eef-vr":
            feat["shape"] = [26]
            feat["names"] = list(ARX_LIFT2S_JOINT_NAMES) + POSE_NAMES
            feat["field_descriptions"][f"{role}/eef/pose"] = {"description": "dual-arm xyz meters and RPY radians, XYZ extrinsic", "dimensions": 12, "indices": list(range(14, 26))}
        for side in ("left", "right"):
            feat["field_descriptions"][f"{role}/{side}_effector/position"]["description"] = (
                "VR closure intent: 0 open, 1 closed" if role == "action" and mode != "joint-feedback" else "feedback minus episode side full-open baseline, raw travel units")
    for camera in CAMERAS:
        result[f"observation.images.{camera}"]["info"].update({"video.codec": "h264", "video.pix_fmt": "gbrp"})
    return result


def finalize(name, rows):
    rows = sorted(rows, key=lambda r: r["episode_index"])
    total = sum(r["output_frames"] for r in rows)
    anchors = sum(max(0, r["output_frames"] - 29) for r in rows)
    for mode in MODES:
        root = dataset_path(name, mode)
        meta = root / "meta"
        (meta / "episodes/chunk-000").mkdir(parents=True, exist_ok=True)
        ep_rows, offset, all_vectors = [], 0, {"observation.state": [], "action": []}
        for row in rows:
            n, length = row["episode_index"], row["output_frames"]
            path = root / f"data/chunk-000/file-{n:03d}.parquet"
            table = pq.read_table(path)
            table = table.set_column(table.schema.get_field_index("index"), "index", pa.array(np.arange(offset, offset+length, dtype=np.int64)))
            pq.write_table(table, path)
            with h5py.File(row["source_path"], "r") as f:
                indices, state, action = labels(f["observations/qpos"][()], f["observations/eef"][()], f["action_poscmd"][()],
                    [row["sides"][s]["baseline"] for s in ("left", "right")], mode)
            for key, expected in (("observation.state", state), ("action", action)):
                actual = np.stack(table[key].to_pylist()).astype(np.float32)
                np.testing.assert_array_equal(actual, expected)
                all_vectors[key].append(actual)
            ep = {"episode_index": n, "tasks": [TASKS[name]], "length": length, "data/chunk_index": 0,
                "data/file_index": n, "dataset_from_index": offset, "dataset_to_index": offset+length,
                "meta/episodes/chunk_index": 0, "meta/episodes/file_index": 0}
            for camera in CAMERAS:
                prefix = f"videos/observation.images.{camera}"
                ep.update({prefix+"/chunk_index": 0, prefix+"/file_index": n, prefix+"/from_timestamp": 0., prefix+"/to_timestamp": length/30})
            ep_rows.append(ep)
            offset += length
        pq.write_table(pa.Table.from_pylist(ep_rows), meta / "episodes/chunk-000/file-000.parquet")
        pd.DataFrame({"task_index": [0]}, index=[TASKS[name]]).to_parquet(meta / "tasks.parquet")
        stats = {}
        for key, blocks in all_vectors.items():
            array = np.concatenate(blocks).astype(np.float64)
            stats[key] = {"min": array.min(0).tolist(), "max": array.max(0).tolist(), "mean": array.mean(0).tolist(), "std": array.std(0).tolist(), "count": [total]}
        save(meta / "stats.json", stats)
        save(meta / "info.json", {"codebase_version": "v3.0", "robot_type": "ARX_LIFT2s", "total_episodes": len(rows), "total_frames": total,
            "total_tasks": 1, "chunks_size": 1000, "data_files_size_in_mb": 100, "video_files_size_in_mb": 500, "fps": 30,
            "splits": {"train": f"0:{len(rows)}"}, "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
            "video_path": "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4", "features": features(mode)})
        manifest = {**contract(mode), "dataset": name, "task": TASKS[name], "total_episodes": len(rows), "total_frames": total,
            "anchors": anchors, "episodes": rows, "validation_passed": True, "all_component_labels_checked": True, "source_hashes_rechecked": True,
            "calibration_parameters": {"minimum_open_vr": 4.99, "minimum_source_frames": 30, "tail_frames": 10,
                "max_p90_p10": .003, "max_half_median_difference": .0015, "baseline_percentile": 10, "max_platform_drift": .02},
            "calibration_window_definition": "contiguous VR>=4.99 run; final ten feedback frames; compare first/last five of those ten",
            "video_encoding": "lossless libx264rgb crf=0; decoded RGB digest equals selected source RGB digest"}
        save(meta / "arx.json", manifest)
        evidence = EVIDENCE / profile(name, mode)
        save(evidence / "conversion_validation.json", {"validation": "ok", "episodes": len(rows), "frames": total,
            "all_source_images": True, "all_output_images": True, "pixel_exact": True, "all_labels": True, "all_source_sha256": True})
        save(evidence / "source_manifest.json", manifest)
        save(evidence / "converted.json", {"dataset": str(root), "profile": profile(name, mode), "episodes": len(rows), "frames": total, "anchors": anchors,
            "expected_batches_per_rank": ((anchors+15)//16)//8, "expected_vla_epoch": 10000/(((anchors+15)//16)//8)})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=TASKS, required=True)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    import fcntl
    EVIDENCE.mkdir(parents=True, exist_ok=True)
    with (EVIDENCE / f"{args.dataset}.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        source = REPO / "data" / ("L" if args.dataset == "T" else "Blue")
        paths = sorted(source.glob("episode_*.hdf5"), key=lambda p: int(p.stem.split("_")[-1]))
        if len(paths) != (53 if args.dataset == "T" else 52):
            raise ValueError("source episode count changed")
        jobs = [(args.dataset, str(p), i) for i, p in enumerate(paths)]
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            rows = list(pool.map(convert_episode, jobs))
        finalize(args.dataset, rows)
        print(f"{args.dataset}: three datasets fully validated", flush=True)


if __name__ == "__main__":
    main()
