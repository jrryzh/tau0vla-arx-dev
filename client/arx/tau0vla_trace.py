"""Low-overhead JSONL tracing and smoothness metrics for Tau0VLA rollout."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from tau0vla_protocol import AdoptionInfo, ScheduledAction


ARM_INDICES = np.asarray([0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12])
GRIPPER_INDICES = np.asarray([6, 13])


class TraceWriter:
    def __init__(self, path: Path | None):
        self.path = path
        self._stream = None
        self._pending = 0
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            self._stream = path.open("w", encoding="utf-8")

    def _write(self, payload: dict[str, Any]) -> None:
        if self._stream is None:
            return
        self._stream.write(json.dumps(payload, separators=(",", ":"), allow_nan=False) + "\n")
        self._pending += 1
        if self._pending >= 30:
            self._stream.flush()
            self._pending = 0

    def metadata(self, **values: Any) -> None:
        self._write({"event": "metadata", **values})

    def adoption(self, request_id: int, info: AdoptionInfo) -> None:
        self._write(
            {
                "event": "adoption",
                "request_id": int(request_id),
                "skipped": info.skipped,
                "blended_steps": info.blended_steps,
                "age_ms": info.age_ms,
                "raw_boundary_jump_max": info.raw_boundary_jump_max,
                "blended_boundary_jump_max": info.blended_boundary_jump_max,
                "rtc_delay": info.rtc_delay,
                "actual_elapsed": info.actual_elapsed,
                "prefix_max_abs_error": info.prefix_max_abs_error,
                "adopted": info.adopted,
                "drop_reason": info.drop_reason,
            }
        )

    def request(
        self,
        *,
        request_id: int,
        control_step: int,
        predicted_delay: int,
        source_index: int,
        action_prefix: np.ndarray,
    ) -> None:
        self._write(
            {
                "event": "request",
                "request_id": int(request_id),
                "control_step": int(control_step),
                "predicted_delay": int(predicted_delay),
                "source_index": int(source_index),
                "action_prefix": np.asarray(action_prefix, dtype=np.float32).tolist(),
            }
        )

    def tick(
        self,
        *,
        monotonic_ns: int,
        control_step: int,
        scheduled: ScheduledAction,
        command: np.ndarray,
        feedback: np.ndarray,
        execute: bool,
    ) -> None:
        self._write(
            {
                "event": "tick",
                "monotonic_ns": int(monotonic_ns),
                "control_step": int(control_step),
                "execute": bool(execute),
                "request_id": scheduled.request_id,
                "source_index": scheduled.source_index,
                "skipped": scheduled.skipped,
                "blend_alpha": scheduled.blend_alpha,
                "round_trip_ms": scheduled.round_trip_ms,
                "raw_action": scheduled.raw_action.tolist(),
                "scheduled_action": scheduled.action.tolist(),
                "command": np.asarray(command, dtype=np.float32).tolist(),
                "feedback": np.asarray(feedback, dtype=np.float32).tolist(),
            }
        )

    def starvation(self, monotonic_ns: int, control_step: int) -> None:
        self._write(
            {"event": "starvation", "monotonic_ns": int(monotonic_ns), "control_step": int(control_step)}
        )

    def close(self) -> None:
        if self._stream is not None:
            self._stream.flush()
            self._stream.close()
            self._stream = None


def _series_metrics(array: np.ndarray, indices: np.ndarray) -> dict[str, float]:
    if len(array) < 2:
        return {"step_p95": 0.0, "step_max": 0.0, "jerk_p95": 0.0, "jerk_max": 0.0}
    step = np.max(np.abs(np.diff(array[:, indices], axis=0)), axis=1)
    jerk = np.max(np.abs(np.diff(array[:, indices], n=2, axis=0)), axis=1) if len(array) >= 3 else np.zeros(1)
    return {
        "step_p95": float(np.percentile(step, 95)),
        "step_max": float(np.max(step)),
        "jerk_p95": float(np.percentile(jerk, 95)),
        "jerk_max": float(np.max(jerk)),
    }


def _boundary_metrics(current: np.ndarray, previous: np.ndarray, indices: np.ndarray) -> dict[str, float]:
    if len(current) == 0:
        return {"p95": 0.0, "max": 0.0}
    delta = np.max(np.abs(current[:, indices] - previous[:, indices]), axis=1)
    return {"p95": float(np.percentile(delta, 95)), "max": float(np.max(delta))}


def analyze_trace(path: str | Path) -> dict[str, Any]:
    events = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
    ticks = [event for event in events if event.get("event") == "tick"]
    adoptions = [event for event in events if event.get("event") == "adoption"]
    delay_misses = sum(
        event.get("event") == "adoption" and event.get("drop_reason") == "delay_miss"
        for event in events
    )
    starvation = sum(event.get("event") == "starvation" for event in events)
    if not ticks:
        return {
            "ticks": 0,
            "adoptions": len(adoptions),
            "delay_misses": delay_misses,
            "starvation": starvation,
        }
    command = np.asarray([row["command"] for row in ticks], dtype=np.float32)
    scheduled = np.asarray([row["scheduled_action"] for row in ticks], dtype=np.float32)
    raw = np.asarray([row["raw_action"] for row in ticks], dtype=np.float32)
    feedback = np.asarray([row["feedback"] for row in ticks], dtype=np.float32)
    request_ids = np.asarray([row["request_id"] for row in ticks])
    boundary = np.flatnonzero(request_ids[1:] != request_ids[:-1]) + 1
    boundary_step = (
        np.max(np.abs(command[boundary] - command[boundary - 1]), axis=1)
        if len(boundary)
        else np.zeros(1, dtype=np.float32)
    )
    raw_jumps = np.asarray([row["raw_boundary_jump_max"] for row in adoptions[1:]], dtype=np.float32)
    blended_jumps = np.asarray([row["blended_boundary_jump_max"] for row in adoptions[1:]], dtype=np.float32)
    tracking = np.max(np.abs(command - feedback), axis=1)
    boundary_previous = command[boundary - 1] if len(boundary) else np.empty((0, command.shape[1]))
    return {
        "ticks": len(ticks),
        "adoptions": len(adoptions),
        "starvation": starvation,
        "delay_misses": delay_misses,
        "boundary_count": int(len(boundary)),
        "boundary_step_p95": float(np.percentile(boundary_step, 95)),
        "boundary_step_max": float(np.max(boundary_step)),
        "raw_boundary_jump_p95": float(np.percentile(raw_jumps, 95)) if len(raw_jumps) else 0.0,
        "blended_boundary_jump_p95": float(np.percentile(blended_jumps, 95)) if len(blended_jumps) else 0.0,
        "boundary": {
            "raw_arm": _boundary_metrics(raw[boundary], boundary_previous, ARM_INDICES),
            "scheduled_arm": _boundary_metrics(scheduled[boundary], boundary_previous, ARM_INDICES),
            "command_arm": _boundary_metrics(command[boundary], boundary_previous, ARM_INDICES),
            "raw_gripper": _boundary_metrics(raw[boundary], boundary_previous, GRIPPER_INDICES),
            "scheduled_gripper": _boundary_metrics(scheduled[boundary], boundary_previous, GRIPPER_INDICES),
            "command_gripper": _boundary_metrics(command[boundary], boundary_previous, GRIPPER_INDICES),
        },
        "arm": _series_metrics(command, ARM_INDICES),
        "gripper": _series_metrics(command, GRIPPER_INDICES),
        "tracking_error_p95": float(np.percentile(tracking, 95)),
        "tracking_error_max": float(np.max(tracking)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", type=Path)
    args = parser.parse_args()
    print(json.dumps(analyze_trace(args.trace), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
