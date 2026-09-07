"""Pure HTTP and action-chunk contract for ARX LIFT2s Tau0VLA inference."""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np


PROTOCOL_VERSION = "arx_lift2s_http_v1"
RTC_PROTOCOL_VERSION = "arx_lift2s_http_v2"
FPS = 30
ACTION_DIM = 14
ACTION_HORIZON = 30
ACTION_SEMANTICS = "state_t_plus_1"
CAMERA_NAMES = ("head", "left_wrist", "right_wrist")
JOINT_NAMES = tuple(
    [f"left_j{i}" for i in range(6)]
    + ["left_gripper"]
    + [f"right_j{i}" for i in range(6)]
    + ["right_gripper"]
)


class ProtocolError(RuntimeError):
    pass


@dataclass(frozen=True)
class Observation:
    qpos: np.ndarray
    images: Mapping[str, bytes]
    sample_monotonic_ns: int


@dataclass(frozen=True)
class ActionChunk:
    actions: np.ndarray
    request_id: int
    sample_monotonic_ns: int
    round_trip_ms: float
    inference_ms: float
    model_id: str
    rtc_delay: int = 0
    action_prefix: np.ndarray | None = None
    request_control_step: int = 0


@dataclass(frozen=True)
class AdoptionInfo:
    skipped: int
    blended_steps: int
    age_ms: float
    raw_boundary_jump_max: float
    blended_boundary_jump_max: float
    rtc_delay: int = 0
    actual_elapsed: int = 0
    prefix_max_abs_error: float = 0.0
    adopted: bool = True
    drop_reason: str | None = None


@dataclass(frozen=True)
class ScheduledAction:
    action: np.ndarray
    raw_action: np.ndarray
    request_id: int
    source_index: int
    skipped: int
    blend_alpha: float
    round_trip_ms: float
    ema_passthrough: bool = False


class RollingDelayEstimator:
    """Conservative rolling RTC delay prediction in 30 Hz action steps."""

    def __init__(
        self,
        round_trip_ms: Sequence[float] = (),
        *,
        margin_ms: float = 100.0,
        window_size: int = 30,
    ):
        if margin_ms < 0 or not math.isfinite(float(margin_ms)):
            raise ValueError("margin_ms must be finite and non-negative")
        if int(window_size) < 1:
            raise ValueError("window_size must be positive")
        from collections import deque

        self.margin_ms = float(margin_ms)
        self._samples = deque(maxlen=int(window_size))
        for value in round_trip_ms:
            self.observe(value)

    @property
    def samples(self) -> tuple[float, ...]:
        return tuple(self._samples)

    def observe(self, elapsed_ms: float) -> None:
        value = float(elapsed_ms)
        if value < 0 or not math.isfinite(value):
            raise ValueError("latency sample must be finite and non-negative")
        self._samples.append(value)

    def predict(self, *, background: bool = True) -> int:
        maximum_ms = max(self._samples, default=0.0)
        steps = int(math.ceil((maximum_ms + self.margin_ms) * FPS / 1000.0))
        return max(1 if background else 0, steps)


class Tau0VLAHttpClient:
    def __init__(
        self,
        base_url: str,
        *,
        request_timeout: float = 5.0,
        max_response_age_ms: float = 2000.0,
        rtc: bool = False,
    ):
        import requests

        self.base_url = base_url.rstrip("/")
        self.request_timeout = float(request_timeout)
        self.max_response_age_ms = float(max_response_age_ms)
        self.session = requests.Session()
        self.session_id: str | None = None
        self.model_id: str | None = None
        self.protocol_version = RTC_PROTOCOL_VERSION if rtc else PROTOCOL_VERSION
        self.api_version = "v2" if rtc else "v1"
        self.rtc_enabled = False
        self.rtc_max_delay = 0

    def health(self) -> dict:
        response = self.session.get(f"{self.base_url}/health", timeout=(1.0, 3.0))
        response.raise_for_status()
        payload = response.json()
        if payload.get("status") != "ok" or payload.get("ready") is not True:
            raise ProtocolError(f"server is not ready: {payload}")
        return payload

    def policy_contract(self) -> dict:
        response = self.session.get(
            f"{self.base_url}/api/{self.api_version}/arx-lift2s/policy-contract", timeout=(1.0, 3.0)
        )
        response.raise_for_status()
        payload = response.json()
        expected = {
            "protocol_version": self.protocol_version,
            "fps": FPS,
            "camera_names": list(CAMERA_NAMES),
            "state_dim": ACTION_DIM,
            "action_dim": ACTION_DIM,
            "action_horizon": ACTION_HORIZON,
            "action_semantics": ACTION_SEMANTICS,
            "joint_names": list(JOINT_NAMES),
        }
        mismatches = {key: (payload.get(key), value) for key, value in expected.items() if payload.get(key) != value}
        if mismatches:
            raise ProtocolError(f"server policy contract mismatch: {mismatches}")
        self.model_id = str(payload.get("model_id", "unknown"))
        if self.api_version == "v2":
            self.rtc_enabled = payload.get("rtc_enabled") is True
            self.rtc_max_delay = int(payload.get("rtc_max_delay", -1))
            if not self.rtc_enabled or not 1 <= self.rtc_max_delay < ACTION_HORIZON:
                raise ProtocolError("server checkpoint does not expose a valid training-time RTC contract")
            if payload.get("rtc_delay_unit") != "action_steps":
                raise ProtocolError("server RTC delay unit mismatch")
        return payload

    def create_session(self, task_instruction: str) -> dict:
        instruction = task_instruction.strip()
        if not instruction:
            raise ProtocolError("task instruction must not be empty")
        response = self.session.post(
            f"{self.base_url}/api/{self.api_version}/arx-lift2s/sessions",
            json={
                "protocol_version": self.protocol_version,
                "task_instruction": instruction,
                "client_name": "arx1",
            },
            timeout=(1.0, 5.0),
        )
        response.raise_for_status()
        payload = response.json()
        if payload.get("protocol_version") != self.protocol_version or not payload.get("session_id"):
            raise ProtocolError("invalid create-session response")
        if self.model_id is not None and payload.get("model_id") != self.model_id:
            raise ProtocolError("model changed between policy-contract and session creation")
        self.session_id = str(payload["session_id"])
        return payload

    def infer(
        self,
        observation: Observation,
        request_id: int,
        *,
        rtc_delay: int = 0,
        action_prefix: np.ndarray | None = None,
        request_control_step: int = 0,
    ) -> ActionChunk:
        if self.session_id is None:
            raise ProtocolError("create_session must be called before infer")
        state = np.asarray(observation.qpos, dtype=np.float32)
        if state.shape != (ACTION_DIM,) or not np.isfinite(state).all():
            raise ProtocolError("observation qpos must be a finite 14-vector")
        missing = set(CAMERA_NAMES) - set(observation.images)
        if missing:
            raise ProtocolError(f"observation is missing cameras: {sorted(missing)}")
        metadata = {
            "protocol_version": self.protocol_version,
            "request_id": int(request_id),
            "sample_monotonic_ns": int(observation.sample_monotonic_ns),
            "observation_state": state.tolist(),
        }
        prefix = np.empty((0, ACTION_DIM), dtype=np.float32)
        if self.api_version == "v2":
            if isinstance(rtc_delay, bool) or not isinstance(rtc_delay, (int, np.integer)):
                raise ProtocolError("rtc_delay must be an integer")
            rtc_delay = int(rtc_delay)
            prefix = (
                np.empty((0, ACTION_DIM), dtype=np.float32)
                if action_prefix is None
                else np.asarray(action_prefix, dtype=np.float32)
            )
            if rtc_delay == 0 and prefix.size == 0:
                prefix = prefix.reshape(0, ACTION_DIM)
            if rtc_delay < 0 or rtc_delay > self.rtc_max_delay:
                raise ProtocolError(f"rtc_delay must be in [0, {self.rtc_max_delay}]")
            if prefix.shape != (rtc_delay, ACTION_DIM) or not np.isfinite(prefix).all():
                raise ProtocolError(f"action_prefix must be a finite [{rtc_delay}, {ACTION_DIM}] array")
            metadata.update(rtc_delay=rtc_delay, action_prefix=prefix.tolist())
        elif rtc_delay or action_prefix is not None:
            raise ProtocolError("RTC metadata cannot be sent through the v1 API")
        files = {
            camera: (f"{camera}.jpg", bytes(observation.images[camera]), "image/jpeg")
            for camera in CAMERA_NAMES
        }
        started = time.monotonic()
        response = self.session.post(
            f"{self.base_url}/api/{self.api_version}/arx-lift2s/sessions/{self.session_id}/action-chunks",
            data={"metadata": json.dumps(metadata, separators=(",", ":"))},
            files=files,
            timeout=(1.0, self.request_timeout),
        )
        round_trip_ms = (time.monotonic() - started) * 1000.0
        response.raise_for_status()
        if round_trip_ms > self.max_response_age_ms:
            raise ProtocolError(f"response age {round_trip_ms:.1f} ms exceeds {self.max_response_age_ms:.1f} ms")
        payload = response.json()
        if payload.get("protocol_version") != self.protocol_version:
            raise ProtocolError("response protocol version mismatch")
        if payload.get("session_id") != self.session_id or int(payload.get("request_id", -1)) != request_id:
            raise ProtocolError("response session/request ID mismatch")
        if int(payload.get("sample_monotonic_ns", -1)) != observation.sample_monotonic_ns:
            raise ProtocolError("response observation timestamp mismatch")
        if payload.get("action_semantics") != ACTION_SEMANTICS:
            raise ProtocolError("response action semantics mismatch")
        if not math.isclose(float(payload.get("action_dt", 0.0)), 1.0 / FPS, abs_tol=1e-8):
            raise ProtocolError("response action_dt mismatch")
        if self.model_id is not None and payload.get("model_id") != self.model_id:
            raise ProtocolError("response model ID changed")
        if self.api_version == "v2" and int(payload.get("rtc_delay", -1)) != rtc_delay:
            raise ProtocolError("response rtc_delay mismatch")
        actions = np.asarray(payload.get("actions"), dtype=np.float32)
        if actions.shape != (ACTION_HORIZON, ACTION_DIM) or not np.isfinite(actions).all():
            raise ProtocolError(f"invalid action chunk shape/content: {actions.shape}")
        inference_ms = float(payload.get("inference_ms", -1.0))
        if inference_ms < 0.0 or not math.isfinite(inference_ms):
            raise ProtocolError("invalid server inference time")
        return ActionChunk(
            actions=actions,
            request_id=request_id,
            sample_monotonic_ns=observation.sample_monotonic_ns,
            round_trip_ms=round_trip_ms,
            inference_ms=inference_ms,
            model_id=str(payload.get("model_id")),
            rtc_delay=rtc_delay,
            action_prefix=prefix.copy(),
            request_control_step=int(request_control_step),
        )


def recommended_replan_steps(
    round_trip_ms: Sequence[float], *, margin_ms: float = 100.0, maximum: int = 15
) -> tuple[int, float]:
    values = np.asarray(round_trip_ms, dtype=np.float64)
    if values.ndim != 1 or len(values) == 0 or not np.isfinite(values).all() or np.any(values < 0):
        raise ValueError("round-trip samples must be a non-empty finite non-negative sequence")
    p99_ms = float(np.percentile(values, 99))
    maximum_window_ms = (ACTION_HORIZON - 1) * 1000.0 / FPS
    if p99_ms + margin_ms >= maximum_window_ms:
        raise ProtocolError(
            f"p99 RTT {p99_ms:.1f} ms plus margin {margin_ms:.1f} ms cannot fit the action horizon"
        )
    available_steps = math.floor(ACTION_HORIZON - (p99_ms + margin_ms) * FPS / 1000.0)
    return max(1, min(int(maximum), available_steps)), p99_ms


class ChunkScheduler:
    """Single-request buffer with time alignment and smooth chunk handoff."""

    def __init__(self, replan_steps: int, blend_steps: int = 6):
        if not 1 <= int(replan_steps) < ACTION_HORIZON:
            raise ValueError(f"replan_steps must be in [1, {ACTION_HORIZON - 1}]")
        if not 0 <= int(blend_steps) < ACTION_HORIZON:
            raise ValueError(f"blend_steps must be in [0, {ACTION_HORIZON - 1}]")
        self.replan_steps = int(replan_steps)
        self.blend_steps = int(blend_steps)
        self._steps: list[ScheduledAction] = []
        self._index = 0
        self._published_since_adopt = 0
        self._last_action: np.ndarray | None = None

    @property
    def remaining(self) -> int:
        return len(self._steps) - self._index

    @property
    def current_source_index(self) -> int:
        """Source index of the next command, or the horizon when exhausted."""
        if self.remaining <= 0:
            return ACTION_HORIZON
        return int(self._steps[self._index].source_index)

    def can_request(self, rtc_delay: int) -> bool:
        delay = int(rtc_delay)
        return delay >= 1 and self.remaining >= delay and self.current_source_index + delay <= ACTION_HORIZON

    def should_request(self, request_pending: bool, rtc_delay: int | None = None) -> bool:
        if rtc_delay is not None:
            return (
                not request_pending
                and self.can_request(rtc_delay)
                and self.current_source_index >= self.replan_steps
            )
        return not request_pending and (
            self.remaining == 0
            or self._published_since_adopt >= self.replan_steps
            # A delayed response may skip most of its prefix. If the usable
            # suffix is already shorter than the normal replan interval,
            # prefetch its successor immediately instead of consuming the
            # short suffix first and guaranteeing starvation.
            or self.remaining <= self.replan_steps
        )

    def adopt(
        self,
        chunk: ActionChunk,
        *,
        initial: bool = False,
        arrival_monotonic_ns: int | None = None,
        actual_elapsed: int | None = None,
    ) -> AdoptionInfo:
        actions = np.asarray(chunk.actions, dtype=np.float32)
        if actions.shape != (ACTION_HORIZON, ACTION_DIM) or not np.isfinite(actions).all():
            raise ProtocolError("cannot adopt an invalid action chunk")
        if actual_elapsed is not None:
            actual_elapsed = int(actual_elapsed)
            if actual_elapsed < 0:
                raise ValueError("actual_elapsed must be non-negative")
            age_ms = actual_elapsed * 1000.0 / FPS
        elif arrival_monotonic_ns is None:
            age_ms = float(chunk.round_trip_ms)
        else:
            age_ms = max(0.0, (int(arrival_monotonic_ns) - int(chunk.sample_monotonic_ns)) / 1_000_000.0)
        # action[0] targets state(t+1), so it remains usable until one full
        # control period has elapsed. Floor avoids discarding a still-future
        # first target on low-latency Ethernet responses.
        skipped = (
            0
            if initial
            else min(
                ACTION_HORIZON - 1,
                actual_elapsed
                if actual_elapsed is not None
                else int(math.floor(age_ms * FPS / 1000.0)),
            )
        )
        rtc_delay = int(chunk.rtc_delay)
        prefix_error = 0.0
        if rtc_delay:
            expected = np.asarray(chunk.action_prefix, dtype=np.float32)
            if expected.shape != (rtc_delay, ACTION_DIM):
                raise ProtocolError("RTC chunk is missing its request action prefix")
            prefix_error = float(np.max(np.abs(actions[:rtc_delay] - expected)))
            if actual_elapsed is None:
                raise ValueError("RTC adoption requires actual_elapsed control ticks")
            if actual_elapsed > rtc_delay:
                return AdoptionInfo(
                    skipped=actual_elapsed,
                    blended_steps=0,
                    age_ms=age_ms,
                    raw_boundary_jump_max=0.0,
                    blended_boundary_jump_max=0.0,
                    rtc_delay=rtc_delay,
                    actual_elapsed=actual_elapsed,
                    prefix_max_abs_error=prefix_error,
                    adopted=False,
                    drop_reason="delay_miss",
                )
        fresh = actions[skipped:]
        old_steps = self._steps[self._index :]
        conditioned_remaining = max(0, rtc_delay - skipped)
        overlap = 0 if initial else min(
            self.blend_steps,
            max(0, len(old_steps) - conditioned_remaining),
            max(0, len(fresh) - conditioned_remaining),
        )
        scheduled: list[ScheduledAction] = []
        for offset, raw_action in enumerate(fresh):
            alpha = 1.0
            action = raw_action.copy()
            blend_offset = offset - conditioned_remaining
            if 0 <= blend_offset < overlap:
                progress = (blend_offset + 1) / overlap
                alpha = progress * progress * (3.0 - 2.0 * progress)
                action = (1.0 - alpha) * old_steps[offset].action + alpha * raw_action
            scheduled.append(
                ScheduledAction(
                    action=np.asarray(action, dtype=np.float32),
                    raw_action=np.asarray(raw_action, dtype=np.float32).copy(),
                    request_id=int(chunk.request_id),
                    source_index=skipped + offset,
                    skipped=skipped,
                    blend_alpha=float(alpha),
                    round_trip_ms=float(chunk.round_trip_ms),
                    ema_passthrough=offset < conditioned_remaining,
                )
            )
        raw_jump = 0.0
        blended_jump = 0.0
        if self._last_action is not None and len(fresh):
            raw_jump = float(np.max(np.abs(fresh[0] - self._last_action)))
            blended_jump = float(np.max(np.abs(scheduled[0].action - self._last_action)))
        self._steps = scheduled
        self._index = 0
        self._published_since_adopt = 0
        return AdoptionInfo(
            skipped=skipped,
            blended_steps=overlap,
            age_ms=age_ms,
            raw_boundary_jump_max=raw_jump,
            blended_boundary_jump_max=blended_jump,
            rtc_delay=rtc_delay,
            actual_elapsed=skipped if actual_elapsed is None else actual_elapsed,
            prefix_max_abs_error=prefix_error,
        )

    def preview(self, count: int, ema: "ActionEMA | None" = None) -> np.ndarray:
        """Preview final commands without advancing the buffer or EMA state."""
        count = int(count)
        if count < 0 or count > self.remaining:
            raise BufferError(f"cannot preview {count} actions with {self.remaining} remaining")
        steps = self._steps[self._index : self._index + count]
        if ema is None:
            return np.asarray([step.action for step in steps], dtype=np.float32).reshape(count, ACTION_DIM)
        return ema.preview(steps)

    preview_actions = preview

    def next_action(self) -> ScheduledAction:
        if self.remaining <= 0:
            raise BufferError("action chunk exhausted")
        step = self._steps[self._index]
        self._index += 1
        self._published_since_adopt += 1
        self._last_action = step.action.copy()
        return step


class ActionEMA:
    """Optional command EMA; alpha=1 keeps the scheduled action unchanged."""

    def __init__(self, arm_alpha: float = 1.0, gripper_alpha: float = 1.0):
        for name, value in (("arm_alpha", arm_alpha), ("gripper_alpha", gripper_alpha)):
            if not 0.0 < float(value) <= 1.0:
                raise ValueError(f"{name} must be in (0, 1]")
        self._alpha = np.full(ACTION_DIM, float(arm_alpha), dtype=np.float32)
        self._alpha[[6, 13]] = float(gripper_alpha)
        self._previous: np.ndarray | None = None

    def reset(self, action: np.ndarray) -> None:
        value = np.asarray(action, dtype=np.float32)
        if value.shape != (ACTION_DIM,) or not np.isfinite(value).all():
            raise ProtocolError("EMA initial action must be a finite 14-vector")
        self._previous = value.copy()

    def apply(self, action: np.ndarray) -> np.ndarray:
        value = np.asarray(action, dtype=np.float32)
        if value.shape != (ACTION_DIM,) or not np.isfinite(value).all():
            raise ProtocolError("EMA input must be a finite 14-vector")
        if self._previous is None:
            filtered = value.copy()
        else:
            filtered = self._alpha * value + (1.0 - self._alpha) * self._previous
        self._previous = filtered.copy()
        return filtered

    def apply_scheduled(self, scheduled: ScheduledAction) -> np.ndarray:
        if scheduled.ema_passthrough:
            self.reset(scheduled.action)
            return scheduled.action.copy()
        return self.apply(scheduled.action)

    def preview(self, steps: Sequence[ScheduledAction] | np.ndarray) -> np.ndarray:
        """Return filtered future commands without modifying EMA state."""
        previous = None if self._previous is None else self._previous.copy()
        result = []
        for item in steps:
            passthrough = isinstance(item, ScheduledAction) and item.ema_passthrough
            value = np.asarray(item.action if isinstance(item, ScheduledAction) else item, dtype=np.float32)
            if value.shape != (ACTION_DIM,) or not np.isfinite(value).all():
                raise ProtocolError("EMA preview input must contain finite 14-vectors")
            if passthrough or previous is None:
                filtered = value.copy()
            else:
                filtered = self._alpha * value + (1.0 - self._alpha) * previous
            result.append(filtered.copy())
            previous = filtered
        return np.asarray(result, dtype=np.float32).reshape(len(result), ACTION_DIM)


__all__ = [
    "ACTION_DIM",
    "ACTION_HORIZON",
    "ACTION_SEMANTICS",
    "ActionEMA",
    "ActionChunk",
    "AdoptionInfo",
    "CAMERA_NAMES",
    "ChunkScheduler",
    "FPS",
    "JOINT_NAMES",
    "Observation",
    "PROTOCOL_VERSION",
    "RTC_PROTOCOL_VERSION",
    "ProtocolError",
    "RollingDelayEstimator",
    "ScheduledAction",
    "Tau0VLAHttpClient",
    "recommended_replan_steps",
]
