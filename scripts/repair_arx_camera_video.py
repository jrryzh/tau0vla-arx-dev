#!/usr/bin/env python3
"""Re-encode one failed, single-file camera stream from the hashed ARX source.

Keeps the original AV1/CRF30/GOP2 format and exact 30 FPS frame ordering.
The replacement is decoded and checked before an atomic swap; the original
is retained outside the dataset with a repair audit record.
"""
import argparse
from contextlib import nullcontext
from fractions import Fraction
import hashlib
from io import BytesIO
import json
from pathlib import Path
import shutil
import time

import av
import h5py
import numpy as np
import pyarrow.dataset as pads
from PIL import Image


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--camera", choices=("head", "left_wrist", "right_wrist"), required=True)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--validate-existing", action="store_true", help="Validate a completed replacement without repeating encoding")
    args = parser.parse_args()
    manifest = json.loads((args.dataset / "meta/arx.json").read_text())
    assert manifest["fps"] == 30 and manifest["temporal_stride"] == 2
    assert manifest["action_semantics"] == "state_t_plus_1"
    key = f"observation.images.{args.camera}"
    paths = sorted((args.dataset / "videos" / key).rglob("*.mp4"))
    assert len(paths) == 1, "Only a single-file camera stream is supported"
    target = paths[0]
    args.evidence.mkdir(parents=True, exist_ok=True)
    replacement = args.evidence / f"{args.camera}.replacement.mp4"
    backup = args.evidence / f"{args.camera}.original.mp4"
    report_path = args.evidence / f"{args.camera}.repair.json"
    assert not backup.exists() and not report_path.exists(), "Inspect prior repair before retry"
    rows = sorted(pads.dataset(str(args.dataset / "meta/episodes"), format="parquet").to_table().to_pylist(), key=lambda r: r["episode_index"])
    assert len(rows) == len(manifest["episodes"])
    count = 0
    started = time.time()
    samples = {}
    # Limit SVT worker parallelism; the original encoder logged an internal
    # object-wrapper release error before producing undecodable AV1 packets.
    options = {"g": "2", "crf": "30", "preset": "8", "svtav1-params": "lp=2"}
    if args.validate_existing:
        assert replacement.is_file(), "No completed replacement to validate"
    context = nullcontext(None) if args.validate_existing else av.open(str(replacement), "w", options={"movflags": "faststart"})
    with context as output:
        if output is not None:
            stream = output.add_stream("libsvtav1", 30, options=options)
            stream.width, stream.height, stream.pix_fmt = 640, 480, "yuv420p"
        for ep, row in zip(manifest["episodes"], rows, strict=True):
            source = args.source / ep["file"]
            assert sha256(source) == ep["sha256"], source
            assert row[f"videos/{key}/chunk_index"] == 0
            assert row[f"videos/{key}/file_index"] == 0
            assert abs(row[f"videos/{key}/from_timestamp"] - count / 30) < 0.001
            with h5py.File(source, "r") as root:
                images = root[f"observations/images/{args.camera}"]
                indices = list(range(0, ep["source_frames"], 2))[:-1]
                assert len(indices) == ep["output_frames"]
                for local, index in enumerate(indices):
                    if output is None and local not in (0, len(indices) - 1):
                        count += 1
                        continue
                    with Image.open(BytesIO(np.asarray(images[index], dtype=np.uint8).tobytes())) as im:
                        rgb = im.convert("RGB")
                        assert rgb.size == (640, 480)
                        if local in (0, len(indices) - 1):
                            samples[count] = np.asarray(rgb).copy()
                        frame = av.VideoFrame.from_image(rgb)
                    if output is not None:
                        frame.pts, frame.time_base = count, Fraction(1, 30)
                        output.mux(stream.encode(frame))
                    count += 1
            assert abs(row[f"videos/{key}/to_timestamp"] - count / 30) < 0.001
            print(f"camera={args.camera} episode={ep['source_episode_number']} encoded={count} elapsed={time.time()-started:.1f}s", flush=True)
        if output is not None:
            output.mux(stream.encode())
    assert count == manifest["total_frames"]
    decoded = 0
    errors = []
    with av.open(str(replacement)) as container:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        stream.codec_context.thread_count = 4
        assert stream.average_rate == 30 and stream.codec_context.codec.id == av.Codec("av1", "r").id
        for frame in container.decode(stream):
            assert frame.width == 640 and frame.height == 480
            assert abs(float(frame.pts * frame.time_base) - decoded / 30) < 0.001
            if decoded in samples:
                mae = float(np.abs(frame.to_ndarray(format="rgb24").astype(np.float32) - samples.pop(decoded)).mean())
                assert mae < 12, (decoded, mae)
                errors.append(mae)
            decoded += 1
    assert decoded == count and not samples, (decoded, count)
    original_hash, repaired_hash = sha256(target), sha256(replacement)
    shutil.copy2(target, backup)
    assert sha256(backup) == original_hash
    replacement.replace(target)
    report = {"camera": args.camera, "target": str(target), "backup": str(backup),
              "original_sha256": original_hash, "repaired_sha256": repaired_hash,
              "frames": count, "decoded_frames": decoded, "source_boundary_samples": len(errors),
              "maximum_boundary_rgb_mae": max(errors), "encoder_options": options,
              "all_timestamps_checked": True, "elapsed_seconds": time.time() - started}
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
