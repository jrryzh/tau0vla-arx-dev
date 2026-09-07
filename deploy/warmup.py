"""Shared deploy-time inference warmup helpers."""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from tau0_vla.utils.profiling import cuda_sync

logger = logging.getLogger(__name__)


def _configure_inference_mode(policy, infer_mode: str, max_prefix_len: int) -> None:
    """Apply Tau0VLA inference backend switches without bypassing deploy policy."""
    flow_matching = getattr(getattr(policy, "model", None), "flow_matching", None)
    if flow_matching is None or not hasattr(flow_matching, "config"):
        if infer_mode == "optim":
            logger.warning("--infer-mode optim requested, but policy has no flow_matching module")
        return

    if infer_mode == "eager":
        flow_matching.config.max_prefix_len = 0
    else:
        if max_prefix_len > 0:
            flow_matching.config.max_prefix_len = int(max_prefix_len)
        elif int(getattr(flow_matching.config, "max_prefix_len", 0) or 0) <= 0:
            flow_matching.config.max_prefix_len = 256
    if hasattr(policy.model, "config"):
        policy.model.config.max_prefix_len = flow_matching.config.max_prefix_len

    # Changing static-shape settings invalidates any previously captured graph.
    if hasattr(flow_matching, "_cuda_graph_prefix"):
        flow_matching._cuda_graph_prefix = None
    if hasattr(flow_matching, "_cuda_graph_prefix_signature"):
        flow_matching._cuda_graph_prefix_signature = None
    if hasattr(flow_matching, "_compiled_predict_velocity"):
        flow_matching._compiled_predict_velocity = None
    logger.info(
        "Tau0VLA infer-mode=%s, max_prefix_len=%s",
        infer_mode,
        getattr(flow_matching.config, "max_prefix_len", None),
    )


def _raw_state_dim(data_spec) -> int:
    artifacts_dir = getattr(data_spec, "artifacts_dir", None)
    if artifacts_dir:
        path = Path(artifacts_dir) / "field_descriptions.json"
        if path.exists():
            payload = json.loads(path.read_text(encoding="utf-8"))
            state_fields = payload.get("state", {})
            max_idx = -1
            for desc in state_fields.values():
                if isinstance(desc, dict):
                    indices = desc.get("indices") or []
                    if indices:
                        max_idx = max(max_idx, max(int(i) for i in indices))
            if max_idx >= 0:
                return max_idx + 1
    return int(getattr(data_spec, "state_dim", 0) or 0)


def _dummy_payload(policy, *, seed: int = 42) -> dict[str, Any]:
    data_spec = policy.data_spec
    target_size = getattr(data_spec, "target_size", None) or (224, 224)
    height, width = int(target_size[0]), int(target_size[1])
    rng = np.random.default_rng(seed)

    images = {
        key: rng.integers(0, 256, size=(height, width, 3), dtype=np.uint8)
        for key in tuple(data_spec.cam_keys)
    }
    return {
        "prompt": "Perform the task according to the camera observations.",
        "images": images,
        "state": rng.normal(loc=0.0, scale=0.01, size=_raw_state_dim(data_spec)).astype(np.float32),
        "meta": {},
    }


@torch.no_grad()
def _run_dummy_input_warmup(policy, *, warmup_steps: int = 3, seed: int = 42) -> None:
    """Run deploy-path dummy inference to capture CUDA graphs / compile kernels."""
    if warmup_steps <= 0:
        logger.info("[warmup] skipped: warmup_steps <= 0")
        return
    if not hasattr(getattr(policy, "model", None), "sample_action"):
        logger.info("[warmup] skipped: policy model has no sample_action")
        return

    cpu_rng_state = torch.get_rng_state()
    cuda_rng_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None

    try:
        payload = _dummy_payload(policy, seed=seed)
        image_shape = next(iter(payload["images"].values())).shape if payload["images"] else None
        logger.info(
            "[warmup] starting: steps=%d, cameras=%d, image_shape=%s, raw_state_dim=%d",
            warmup_steps,
            len(payload["images"]),
            image_shape,
            int(np.asarray(payload["state"]).size),
        )
        total_t0 = time.perf_counter()
        rtc_enabled = bool(getattr(policy, "rtc_enabled", False))
        rtc_max_delay = int(getattr(policy, "rtc_max_delay", 0) or 0)
        raw_action_dim = int(np.asarray(payload["state"]).size)
        registry_key = getattr(policy.data_spec, "unified_registry_key", None)
        if registry_key is not None:
            from tau0_vla.data.robots.unified import get_registry_entry

            raw_action_dim = int(get_registry_entry(registry_key).get("action_dim") or raw_action_dim)
        delays = [0]
        if rtc_enabled and rtc_max_delay > 0 and registry_key is not None:
            delays.append(min(rtc_max_delay, max(1, rtc_max_delay // 2)))
        calls = max(warmup_steps, len(delays)) if len(delays) > 1 else warmup_steps
        for idx in range(calls):
            delay = delays[idx % len(delays)]
            step_payload = payload
            if rtc_enabled:
                step_payload = dict(payload)
                step_payload["meta"] = {
                    "rtc_delay": delay,
                    "action_prefix": np.zeros((delay, raw_action_dim), dtype=np.float32),
                }
            cuda_sync()
            step_t0 = time.perf_counter()
            policy.infer(step_payload)
            cuda_sync()
            logger.info(
                "[warmup] step %d/%d (rtc_delay=%d): %.1f ms",
                idx + 1,
                calls,
                delay,
                (time.perf_counter() - step_t0) * 1000,
            )
        logger.info("[warmup] done: total=%.1f ms", (time.perf_counter() - total_t0) * 1000)
    except Exception:
        logger.exception("[warmup] failed; continuing without warmed inference path")
    finally:
        torch.set_rng_state(cpu_rng_state)
        if cuda_rng_states is not None:
            torch.cuda.set_rng_state_all(cuda_rng_states)
        cuda_sync()
