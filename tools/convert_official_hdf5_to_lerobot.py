#!/usr/bin/env python3
"""Convert official ROS2_LIFT_Play HDF5 episodes to LeRobot 0.4.x/v3.

The default state-as-action contract writes action(t)=qpos(t+1) at the output
FPS. Source commands remain available through the explicitly labelled
``joint_position_command`` mode.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from io import BytesIO
from pathlib import Path

import h5py
import numpy as np
from PIL import Image

CAMERA_NAMES = ("head", "left_wrist", "right_wrist")
JOINT_NAMES = tuple(
    [f"left_j{i}" for i in range(6)]
    + ["left_gripper"]
    + [f"right_j{i}" for i in range(6)]
    + ["right_gripper"]
)
ACTION_DIM = 14
EXPECTED_IMAGE_SHAPE = (480, 640, 3)
STATE_AS_ACTION_SEMANTICS = "state_t_plus_1"
SOURCE_ACTION_SEMANTICS = "joint_position_command"
EPISODE_FILE_RE = re.compile(r"^episode_(\d+)\.hdf5$")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def decode_rgb(encoded) -> np.ndarray:
    payload = np.asarray(encoded, dtype=np.uint8).tobytes()
    with Image.open(BytesIO(payload)) as image:
        return np.asarray(image.convert("RGB"))


def episode_number(path: Path) -> int:
    match = EPISODE_FILE_RE.fullmatch(path.name)
    if match is None:
        raise ValueError(
            f"invalid episode filename {path.name!r}; expected episode_<integer>.hdf5"
        )
    return int(match.group(1))


def selected_paths(
    input_dir: Path, start: int, end: int, *, allow_missing: bool = False
) -> list[Path]:
    if start < 0 or end < start:
        raise ValueError("require 0 <= start <= end")
    if not allow_missing:
        paths = [input_dir / f"episode_{index}.hdf5" for index in range(start, end + 1)]
        missing = [str(path) for path in paths if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"missing selected episodes: {missing}")
        return paths

    numbered: dict[int, Path] = {}
    for path in input_dir.glob("episode_*.hdf5"):
        if not path.is_file():
            raise ValueError(f"episode candidate is not a file: {path}")
        number = episode_number(path)
        if number in numbered:
            raise ValueError(
                f"duplicate episode number {number}: {numbered[number].name}, {path.name}"
            )
        numbered[number] = path
    paths = [numbered[number] for number in sorted(numbered) if start <= number <= end]
    if not paths:
        raise FileNotFoundError(
            f"no valid episode files found in requested range [{start}, {end}]"
        )
    return paths


def inspect_episode(path: Path) -> dict:
    with h5py.File(path, "r") as root:
        required = [
            "observations/qpos",
            "observations/qvel",
            "observations/effort",
            "observations/eef",
            "observations/images",
            "action",
        ]
        missing = [key for key in required if key not in root]
        if missing:
            raise ValueError(f"{path.name}: missing keys {missing}")
        qpos = root["observations/qpos"]
        action = root["action"]
        frames = len(action)
        if qpos.shape != (frames, ACTION_DIM) or action.shape != (frames, ACTION_DIM):
            raise ValueError(f"{path.name}: expected qpos/action shape (T, 14)")
        if not np.isfinite(qpos[()]).all() or not np.isfinite(action[()]).all():
            raise ValueError(f"{path.name}: qpos/action contains NaN or Inf")
        image_shapes = {}
        for camera in CAMERA_NAMES:
            key = f"observations/images/{camera}"
            if key not in root or len(root[key]) != frames:
                raise ValueError(f"{path.name}: missing or misaligned camera {camera}")
            indices = sorted({0, frames // 2, frames - 1})
            decoded = [decode_rgb(root[key][index]) for index in indices]
            if len({image.shape for image in decoded}) != 1:
                raise ValueError(
                    f"{path.name}: inconsistent decoded shape for {camera}"
                )
            image_shapes[camera] = list(decoded[0].shape)
            if tuple(decoded[0].shape) != EXPECTED_IMAGE_SHAPE:
                raise ValueError(
                    f"{path.name}: decoded {camera} shape {decoded[0].shape}; "
                    f"expected {EXPECTED_IMAGE_SHAPE} RGB"
                )
        base_max = 0.0
        for key in (
            "observations/robot_base",
            "observations/base_velocity",
            "action_base",
            "action_velocity",
        ):
            if key in root:
                base_max = max(base_max, float(np.max(np.abs(root[key][()]))))
        return {
            "file": path.name,
            "source_episode_number": episode_number(path),
            "sha256": sha256(path),
            "frames": frames,
            "source_action_semantics": str(root.attrs.get("action_semantics", "")),
            "height_command": (
                float(root.attrs["height_command"])
                if "height_command" in root.attrs
                else None
            ),
            "source_task": str(root.attrs.get("task", "")),
            "image_shapes": image_shapes,
            "base_max_abs": base_max,
        }


def temporal_stride(source_fps: int, output_fps: int) -> int:
    if source_fps <= 0 or output_fps <= 0:
        raise ValueError("source/output fps must be positive")
    if output_fps > source_fps or source_fps % output_fps != 0:
        raise ValueError("output fps must be an integer divisor of source fps")
    return source_fps // output_fps


def selected_frame_indices(frames: int, stride: int) -> range:
    return range(0, frames, stride)


def validate_selection(
    paths: list[Path],
    source_fps: int,
    fps: int,
    task: str,
    action_mode: str,
    *,
    requested_start: int | None = None,
    requested_end: int | None = None,
) -> dict:
    stride = temporal_stride(source_fps, fps)
    if not task.strip():
        raise ValueError(
            "--task must be non-empty; source HDF5 task IDs are retained only as "
            "provenance and are not used as training instructions"
        )
    episodes = [inspect_episode(path) for path in paths]
    if action_mode == "source":
        invalid = [
            info["file"]
            for info in episodes
            if info["source_action_semantics"] != SOURCE_ACTION_SEMANTICS
        ]
        if invalid:
            raise ValueError(
                "--action-mode source requires HDF5 action_semantics="
                f"{SOURCE_ACTION_SEMANTICS!r}; invalid episodes: {invalid}"
            )
    action_semantics = (
        STATE_AS_ACTION_SEMANTICS
        if action_mode == "state_t_plus_1"
        else SOURCE_ACTION_SEMANTICS
    )
    action_offset_frames = 1 if action_mode == "state_t_plus_1" else 0
    for info in episodes:
        info["source_frames"] = info.pop("frames")
        selected = len(selected_frame_indices(info["source_frames"], stride))
        info["output_frames"] = selected - action_offset_frames
        if info["output_frames"] <= 0:
            raise ValueError(f"{info['file']}: no frame remains after action alignment")
    shapes = {
        tuple(info["image_shapes"][camera])
        for info in episodes
        for camera in CAMERA_NAMES
    }
    if len(shapes) != 1:
        raise ValueError(f"camera decoded shapes disagree: {sorted(shapes)}")
    selected_numbers = [info["source_episode_number"] for info in episodes]
    if requested_start is None:
        requested_start = min(selected_numbers)
    if requested_end is None:
        requested_end = max(selected_numbers)
    missing_numbers = sorted(
        set(range(requested_start, requested_end + 1)) - set(selected_numbers)
    )
    return {
        "source_format": "official_ros2_lift_play_hdf5",
        "lerobot_version": None,
        "compatible_lerobot_api": "0.4.x",
        "lerobot_format": "v3",
        "source_fps": source_fps,
        "fps": fps,
        "temporal_stride": stride,
        "downsample_method": "uniform_stride_no_interpolation",
        "timestamps_available": False,
        "action_dim": ACTION_DIM,
        "action_semantics": action_semantics,
        "action_offset_frames": action_offset_frames,
        "joint_names": list(JOINT_NAMES),
        "camera_names": list(CAMERA_NAMES),
        "task": task,
        "requested_episode_range": {"start": requested_start, "end": requested_end},
        "selected_source_episode_numbers": selected_numbers,
        "missing_source_episode_numbers": missing_numbers,
        "total_episodes": len(episodes),
        "total_source_frames": sum(info["source_frames"] for info in episodes),
        "total_frames": sum(info["output_frames"] for info in episodes),
        "episodes": episodes,
    }


def convert(args) -> dict:
    paths = selected_paths(
        args.input,
        args.start,
        args.end,
        allow_missing=args.allow_missing_episodes,
    )
    manifest = validate_selection(
        paths,
        args.source_fps,
        args.fps,
        args.task,
        args.action_mode,
        requested_start=args.start,
        requested_end=args.end,
    )
    if args.validate_only:
        return manifest
    if args.output.exists():
        raise FileExistsError(f"output already exists: {args.output}")

    image_shape = tuple(manifest["episodes"][0]["image_shapes"][CAMERA_NAMES[0]])
    features = {
        **{
            f"observation.images.{camera}": {
                "dtype": "video",
                "shape": image_shape,
                "names": ["height", "width", "channels"],
            }
            for camera in CAMERA_NAMES
        },
        "observation.state": {
            "dtype": "float32",
            "shape": (ACTION_DIM,),
            "names": list(JOINT_NAMES),
            "field_descriptions": {
                "state/joint/position": {
                    "description": "dual-arm joint positions in radians",
                    "dimensions": 12,
                    "indices": [*range(0, 6), *range(7, 13)],
                },
                "state/left_effector/position": {
                    "description": "raw left gripper position",
                    "dimensions": 1,
                    "indices": [6],
                },
                "state/right_effector/position": {
                    "description": "raw right gripper position",
                    "dimensions": 1,
                    "indices": [13],
                },
            },
        },
        "action": {
            "dtype": "float32",
            "shape": (ACTION_DIM,),
            "names": list(JOINT_NAMES),
            "field_descriptions": {
                "action/joint/position": {
                    "description": "dual-arm target joint positions in radians",
                    "dimensions": 12,
                    "indices": [*range(0, 6), *range(7, 13)],
                },
                "action/left_effector/position": {
                    "description": "raw left gripper target",
                    "dimensions": 1,
                    "indices": [6],
                },
                "action/right_effector/position": {
                    "description": "raw right gripper target",
                    "dimensions": 1,
                    "indices": [13],
                },
            },
        },
    }

    import lerobot
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    lerobot_version = str(getattr(lerobot, "__version__", "unknown"))
    if not lerobot_version.startswith("0.4."):
        raise RuntimeError(f"expected LeRobot 0.4.x, found {lerobot_version}")
    manifest["lerobot_version"] = lerobot_version

    dataset = LeRobotDataset.create(
        repo_id=args.repo_id,
        root=args.output,
        fps=args.fps,
        robot_type="ARX_LIFT2s",
        features=features,
        use_videos=True,
    )
    for path in paths:
        with h5py.File(path, "r") as root:
            source_frames = len(root["action"])
            indices = list(
                selected_frame_indices(source_frames, manifest["temporal_stride"])
            )
            if args.action_mode == "state_t_plus_1":
                observation_indices = indices[:-1]
                action_indices = indices[1:]
            else:
                observation_indices = indices
                action_indices = indices
            for observation_index, action_index in zip(
                observation_indices, action_indices
            ):
                frame = {
                    "observation.state": np.asarray(
                        root["observations/qpos"][observation_index], dtype=np.float32
                    ),
                    "action": np.asarray(
                        (
                            root["observations/qpos"][action_index]
                            if args.action_mode == "state_t_plus_1"
                            else root["action"][action_index]
                        ),
                        dtype=np.float32,
                    ),
                    "task": args.task,
                }
                for camera in CAMERA_NAMES:
                    image = decode_rgb(
                        root[f"observations/images/{camera}"][observation_index]
                    )
                    if image.shape != EXPECTED_IMAGE_SHAPE:
                        raise ValueError(
                            f"{path.name}: frame {observation_index} camera {camera} "
                            f"decoded as {image.shape}; expected {EXPECTED_IMAGE_SHAPE}"
                        )
                    frame[f"observation.images.{camera}"] = image
                dataset.add_frame(frame)
            dataset.save_episode()
    dataset.finalize()

    sidecar = args.output / "meta" / "arx.json"
    sidecar.parent.mkdir(parents=True, exist_ok=True)
    sidecar.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--start", type=int, required=True)
    parser.add_argument("--end", type=int, required=True)
    parser.add_argument(
        "--allow-missing-episodes",
        action="store_true",
        help=(
            "convert valid episode_<integer>.hdf5 files present in the requested "
            "range and record missing source episode numbers in the manifest"
        ),
    )
    parser.add_argument("--source-fps", type=int, default=60)
    parser.add_argument(
        "--fps", type=int, default=60, help="output fps; must divide source-fps"
    )
    parser.add_argument("--task", required=True)
    parser.add_argument(
        "--action-mode",
        choices=("state_t_plus_1", "source"),
        default="state_t_plus_1",
        help="next-state-as-action target, or an explicitly labelled source command",
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--repo-id")
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    if not args.validate_only and (args.output is None or not args.repo_id):
        parser.error("conversion requires --output and --repo-id")
    return args


if __name__ == "__main__":
    print(json.dumps(convert(parse_args()), indent=2, ensure_ascii=False))
