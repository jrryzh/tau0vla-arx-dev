#!/usr/bin/env python3
"""Validate an ARX HDF5-to-LeRobot conversion against its source episodes."""

from __future__ import annotations

import argparse
import json
from io import BytesIO
from pathlib import Path

import av
import h5py
import numpy as np
import pyarrow.dataset as pads
import pyarrow.parquet as pq
from PIL import Image

CAMERAS = ("head", "left_wrist", "right_wrist")
IMAGE_SHAPE = (480, 640, 3)


def _vectors(table, key: str) -> np.ndarray:
    values = table.column(key).to_numpy(zero_copy_only=False)
    return np.stack(values) if values.dtype == object else np.asarray(values)


def _decode_hdf5_rgb(encoded) -> np.ndarray:
    payload = np.asarray(encoded, dtype=np.uint8).tobytes()
    with Image.open(BytesIO(payload)) as image:
        return np.asarray(image.convert("RGB"))


def _validate_source_and_vectors(
    source: Path, manifest: dict, episodes_table, data_table
) -> None:
    state = _vectors(data_table, "observation.state")
    action = _vectors(data_table, "action")
    if state.shape != (manifest["total_frames"], 14) or action.shape != state.shape:
        raise ValueError(
            f"unexpected state/action shapes: {state.shape}, {action.shape}"
        )
    if not np.isfinite(state).all() or not np.isfinite(action).all():
        raise ValueError("converted state/action contains NaN or Inf")

    episode_rows = sorted(
        episodes_table.to_pylist(), key=lambda row: row["episode_index"]
    )
    if len(episode_rows) != len(manifest["episodes"]):
        raise ValueError("episode metadata count does not match manifest")

    for output_episode, (entry, row) in enumerate(
        zip(manifest["episodes"], episode_rows, strict=True)
    ):
        if row["episode_index"] != output_episode:
            raise ValueError(
                f"non-contiguous output episode index: {row['episode_index']}"
            )
        start = int(row["dataset_from_index"])
        end = int(row["dataset_to_index"])
        if end - start != entry["output_frames"]:
            raise ValueError(f"{entry['file']}: output frame count mismatch")

        path = source / entry["file"]
        with h5py.File(path, "r") as root:
            qpos = np.asarray(root["observations/qpos"])
            source_action = np.asarray(root["action"])
            if (
                qpos.shape != (entry["source_frames"], 14)
                or source_action.shape != qpos.shape
            ):
                raise ValueError(f"{path.name}: source qpos/action shape mismatch")
            if not np.isfinite(qpos).all() or not np.isfinite(source_action).all():
                raise ValueError(f"{path.name}: source qpos/action contains NaN or Inf")
            selected = qpos[:: manifest["temporal_stride"]]
            np.testing.assert_array_equal(state[start:end], selected[:-1])
            np.testing.assert_array_equal(action[start:end], selected[1:])

            for camera in CAMERAS:
                images = root[f"observations/images/{camera}"]
                if len(images) != len(qpos):
                    raise ValueError(f"{path.name}: {camera} is not frame-aligned")
                for source_index in range(len(images)):
                    shape = _decode_hdf5_rgb(images[source_index]).shape
                    if shape != IMAGE_SHAPE:
                        raise ValueError(
                            f"{path.name}: {camera}[{source_index}] decoded as {shape}"
                        )
        print(
            f"source_episode={entry['source_episode_number']} validation=ok", flush=True
        )


def _validate_videos(root: Path, expected_frames: int) -> None:
    for camera in CAMERAS:
        paths = sorted(
            (root / "videos" / f"observation.images.{camera}").rglob("*.mp4")
        )
        if not paths:
            raise ValueError(f"no converted videos found for {camera}")
        frames = 0
        for path in paths:
            with av.open(str(path)) as container:
                stream = container.streams.video[0]
                if stream.width != 640 or stream.height != 480:
                    raise ValueError(f"{path}: unexpected video dimensions")
                if float(stream.average_rate) != 30.0:
                    raise ValueError(
                        f"{path}: unexpected video FPS {stream.average_rate}"
                    )
                for frame in container.decode(stream):
                    shape = frame.to_ndarray(format="rgb24").shape
                    if shape != IMAGE_SHAPE:
                        raise ValueError(f"{path}: decoded video frame as {shape}")
                    frames += 1
        if frames != expected_frames:
            raise ValueError(
                f"{camera}: decoded {frames} frames, expected {expected_frames}"
            )
        print(
            f"video_camera={camera} decoded_frames={frames} validation=ok", flush=True
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--expected-episodes", type=int, required=True)
    parser.add_argument("--expected-frames", type=int, required=True)
    parser.add_argument("--expected-task", required=True)
    parser.add_argument("--expected-missing", type=int, nargs="*", default=[])
    args = parser.parse_args()

    info = json.loads((args.dataset / "meta/info.json").read_text())
    manifest = json.loads((args.dataset / "meta/arx.json").read_text())
    if info["fps"] != 30 or manifest["fps"] != 30:
        raise ValueError("dataset is not 30 FPS")
    if info["total_episodes"] != args.expected_episodes:
        raise ValueError("unexpected info.json episode count")
    if info["total_frames"] != args.expected_frames:
        raise ValueError("unexpected info.json frame count")
    if manifest["total_episodes"] != args.expected_episodes:
        raise ValueError("unexpected manifest episode count")
    if manifest["total_frames"] != args.expected_frames:
        raise ValueError("unexpected manifest frame count")
    if manifest["missing_source_episode_numbers"] != args.expected_missing:
        raise ValueError("unexpected missing source episode numbers")
    if manifest["task"] != args.expected_task:
        raise ValueError("unexpected manifest task")
    if manifest["action_semantics"] != "state_t_plus_1":
        raise ValueError("unexpected action semantics")

    features = info["features"]
    expected_camera_keys = {f"observation.images.{camera}" for camera in CAMERAS}
    if not expected_camera_keys.issubset(features):
        raise ValueError("converted dataset is missing camera keys")
    if features["observation.state"]["shape"] != [14] or features["action"][
        "shape"
    ] != [14]:
        raise ValueError("converted dataset does not use native 14D state/action")

    tasks = pq.read_table(args.dataset / "meta/tasks.parquet").to_pydict()
    task_columns = [key for key in tasks if key != "task_index"]
    if len(task_columns) != 1 or tasks[task_columns[0]] != [args.expected_task]:
        raise ValueError(f"unexpected task metadata: {tasks}")

    episodes_table = pads.dataset(
        str(args.dataset / "meta/episodes"), format="parquet"
    ).to_table()
    data_table = pads.dataset(str(args.dataset / "data"), format="parquet").to_table(
        columns=["observation.state", "action"]
    )
    _validate_source_and_vectors(args.source, manifest, episodes_table, data_table)
    _validate_videos(args.dataset, args.expected_frames)
    print("arx_lerobot_conversion_validation=ok")


if __name__ == "__main__":
    main()
