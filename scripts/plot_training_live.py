#!/usr/bin/env python3
"""Render a lightweight live dashboard from Tau0-VLA rank-0 text logs."""

from __future__ import annotations

import argparse
import ast
import csv
import json
import os
import re
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


DEFAULT_LOG_DIR = Path(
    "outputs/arx_lift2s_pickplace_h200_formal/"
    "arx_lift2s_pickplace_h200_formal/log"
)
DEFAULT_OUTPUT_DIR = Path("outputs/training_dashboard")
METRIC_RE = re.compile(r"\{[^{}\r\n]*'loss'[^{}\r\n]*\}")
PROGRESS_RE = re.compile(r"(?<!\d)(\d+)/(\d+)(?!\d)")
FIELDS = ("step", "loss", "vla_loss", "grad_norm", "learning_rate", "epoch", "vla_epoch")


def _as_float(value: object) -> float:
    if value is None:
        return float("nan")
    return float(value)


def parse_logs(paths: list[Path]) -> tuple[list[dict[str, float]], int | None]:
    """Parse metric dictionaries, associating each with the preceding progress step."""
    rows_by_step: dict[int, dict[str, float]] = {}
    texts = [path.read_text(encoding="utf-8", errors="replace").replace("\r", "\n") for path in paths]
    # NCCL and other runtime messages contain ratios such as ``12/0`` and
    # ``1/1``.  The tqdm training denominator is the largest denominator in the
    # rank-0 log, so resolve it once and ignore unrelated ratios below.
    denominators = [
        int(match.group(2))
        for text in texts
        for match in PROGRESS_RE.finditer(text)
    ]
    total_steps = max(denominators, default=None)
    for text in texts:
        current_step: int | None = None
        for chunk in text.splitlines():
            progress = [
                match
                for match in PROGRESS_RE.finditer(chunk)
                if total_steps is not None and int(match.group(2)) == total_steps
            ]
            if progress:
                current_step = int(progress[-1].group(1))
            for match in METRIC_RE.finditer(chunk):
                if current_step is None:
                    continue
                try:
                    metrics = ast.literal_eval(match.group(0))
                except (SyntaxError, ValueError):
                    continue
                rows_by_step[current_step] = {
                    "step": current_step,
                    **{name: _as_float(metrics.get(name)) for name in FIELDS[1:]},
                }
    return [rows_by_step[step] for step in sorted(rows_by_step)], total_steps


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        handle.write(text)
        temp_path = Path(handle.name)
    os.replace(temp_path, path)


def write_csv(path: Path, rows: list[dict[str, float]]) -> None:
    lines: list[str] = []
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", newline="", dir=path.parent, delete=False) as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
        temp_path = Path(handle.name)
    os.replace(temp_path, path)


def _moving_average(values: np.ndarray, window: int = 10) -> np.ndarray:
    if len(values) < window:
        return values.copy()
    kernel = np.ones(window, dtype=np.float64) / window
    smoothed = np.convolve(values, kernel, mode="valid")
    return np.concatenate((np.full(window - 1, np.nan), smoothed))


def render_plot(path: Path, rows: list[dict[str, float]], total_steps: int | None) -> None:
    steps = np.asarray([row["step"] for row in rows], dtype=np.float64)
    series = {name: np.asarray([row[name] for row in rows], dtype=np.float64) for name in FIELDS[1:]}

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(2, 2, figsize=(14, 8), constrained_layout=True)

    ax = axes[0, 0]
    ax.plot(steps, series["loss"], color="#4C78A8", alpha=0.28, linewidth=0.8, label="loss")
    ax.plot(steps, _moving_average(series["loss"]), color="#4C78A8", linewidth=2, label="loss (10-log MA)")
    ax.plot(steps, _moving_average(series["vla_loss"]), color="#F58518", linewidth=1.6, label="vla_loss (10-log MA)")
    ax.set_title("Training loss")
    ax.set_xlabel("optimizer step")
    ax.set_ylabel("loss")
    ax.legend(loc="upper right")

    ax = axes[0, 1]
    ax.plot(steps, series["grad_norm"], color="#E45756", alpha=0.35, linewidth=0.8)
    ax.plot(steps, _moving_average(series["grad_norm"]), color="#E45756", linewidth=2)
    ax.axhline(1.0, color="#777777", linestyle="--", linewidth=1, label="clip threshold 1.0")
    ax.set_title("Gradient norm")
    ax.set_xlabel("optimizer step")
    ax.set_ylabel("grad norm")
    ax.legend(loc="upper right")

    ax = axes[1, 0]
    ax.plot(steps, series["learning_rate"], color="#54A24B", linewidth=2)
    ax.set_title("Learning-rate schedule")
    ax.set_xlabel("optimizer step")
    ax.set_ylabel("learning rate")
    ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))

    ax = axes[1, 1]
    ax.plot(steps, series["vla_epoch"], color="#B279A2", linewidth=2, label="VLA data epoch")
    ax.set_title("Dataset passes")
    ax.set_xlabel("optimizer step")
    ax.set_ylabel("vla_epoch")
    ax.legend(loc="upper left")

    if total_steps:
        for ax in axes.flat:
            ax.set_xlim(0, total_steps)

    latest = rows[-1]
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    fig.suptitle(
        "ARX LIFT2s · 16×H200 full-parameter training\n"
        f"step {int(latest['step'])}/{total_steps or '?'} · "
        f"loss {latest['loss']:.6g} · grad norm {latest['grad_norm']:.4g} · updated {timestamp}",
        fontsize=14,
    )

    with tempfile.NamedTemporaryFile(suffix=".png", dir=path.parent, delete=False) as handle:
        temp_path = Path(handle.name)
    fig.savefig(temp_path, dpi=150)
    plt.close(fig)
    os.replace(temp_path, path)


def write_dashboard(path: Path, refresh_seconds: int) -> None:
    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>ARX training dashboard</title>
  <style>
    body {{ margin: 0; background: #111827; color: #e5e7eb; font-family: system-ui, sans-serif; }}
    main {{ max-width: 1500px; margin: 0 auto; padding: 20px; }}
    h1 {{ font-size: 20px; font-weight: 600; }}
    p {{ color: #9ca3af; }}
    img {{ width: 100%; border-radius: 10px; background: white; box-shadow: 0 8px 30px #0008; }}
  </style>
</head>
<body>
  <main>
    <h1>ARX LIFT2s · live training curves</h1>
    <p>Auto-refreshes every {refresh_seconds} seconds. Raw points are also available in metrics.csv.</p>
    <img id="curves" src="training_curves.png" alt="Training curves">
  </main>
  <script>
    const refresh = () => document.getElementById('curves').src = 'training_curves.png?t=' + Date.now();
    setInterval(refresh, {refresh_seconds * 1000});
  </script>
</body>
</html>
"""
    _atomic_text(path, html)


def render_once(log_dir: Path, output_dir: Path, refresh_seconds: int) -> tuple[int, int | None]:
    paths = sorted(log_dir.glob("training_log_nodeIdx000_*.txt"))
    if not paths:
        raise FileNotFoundError(f"No rank-0 training logs found under {log_dir}")
    rows, total_steps = parse_logs(paths)
    if not rows:
        raise RuntimeError(f"No metric records found in {len(paths)} log file(s)")

    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "metrics.csv", rows)
    render_plot(output_dir / "training_curves.png", rows, total_steps)
    latest = rows[-1]
    summary = {
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "log_files": [str(path) for path in paths],
        "records": len(rows),
        "total_steps": total_steps,
        "latest": latest,
    }
    _atomic_text(output_dir / "summary.json", json.dumps(summary, indent=2) + "\n")
    write_dashboard(output_dir / "index.html", refresh_seconds)
    return int(latest["step"]), total_steps


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log-dir", type=Path, default=DEFAULT_LOG_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--watch", action="store_true", help="Refresh until the configured total step is reached")
    parser.add_argument("--interval", type=int, default=15, help="Refresh interval in seconds")
    args = parser.parse_args()
    if args.interval < 5:
        parser.error("--interval must be at least 5 seconds")

    while True:
        step, total_steps = render_once(args.log_dir, args.output_dir, args.interval)
        print(f"rendered step={step}/{total_steps or '?'}", flush=True)
        if not args.watch or (total_steps is not None and step >= total_steps):
            break
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
