from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Any

import numpy as np


@dataclasses.dataclass(frozen=True)
class CheckpointSpec:
    checkpoint_dir: Path
    train_config_name: str
    checkpoint_config_name: str
    finch_config_name: str | None
    policy_type: str
    routes: tuple[str, ...]
    route: str | None
    finch_config: Any
    model_action_dim: int | None
    manifest: dict[str, Any]
    # Dotted path of the adapter package whose ``deploy_io`` serves this
    # checkpoint, e.g. ``tau0_vla.adapters.g1``. None for checkpoints written
    # before this field existed — deploy falls back to its ``--adapter`` flag.
    adapter_module: str | None = None


def adapter_module_for_config(config: Any) -> str | None:
    """Dotted path of the adapter package a Robot Config comes from.

    An adapter's Robot Config lives in ``<adapter package>.layout``, so dropping
    the module's last segment names the package whose ``deploy_io`` can serve it.
    Returns None for a config defined outside ``tau0_vla.adapters`` — a user's
    own subclass, say — since guessing there would record a path that cannot be
    imported at serve time. Also None for ``None``, so callers can pass an
    optional attribute straight through.
    """
    if config is None:
        return None
    module = type(config).__module__ if not isinstance(config, type) else config.__module__
    if not module.startswith("tau0_vla.adapters."):
        return None
    package = module.rsplit(".", 1)[0]
    # ``tau0_vla.adapters`` itself is not an adapter.
    return package if package != "tau0_vla.adapters" else None


def resolve_routes(finch_config_name: str | None, *, config_modules: list[str] | None = None) -> list[str]:
    if finch_config_name is None:
        return []

    from tau0_vla.data.config import get_config

    # Only discover if the caller explicitly provided modules. Mirrors the
    # training-side contract (``if data_args.config_modules: discover(...)``).
    # A fresh process that lacks the registration will get a clear
    # ``Unknown config 'X'. Available configs: <empty>`` from ``get_config``
    # instead of a misleading "does not auto-discover" from an empty
    # discover call.
    if config_modules:
        from tau0_vla.data.config import discover_config_modules

        discover_config_modules(extra_module_names=config_modules)
    finch_config = get_config(finch_config_name)
    if not hasattr(finch_config, "configs"):
        return [finch_config_name]
    return [str(child if isinstance(child, str) else getattr(child, "name")) for child in finch_config.configs]


def build_policy_manifest(
    *,
    train_config_name: str,
    finch_config_name: str | None,
    policy_type: str = "finch",
    checkpoint_config_name: str | None = None,
    model_action_dim: int | None = None,
    config_modules: list[str] | None = None,
) -> dict[str, object]:
    routes = resolve_routes(finch_config_name, config_modules=config_modules)
    manifest = {
        "version": 1,
        "policy_type": policy_type,
        "train_config_name": train_config_name,
        "checkpoint_config_name": checkpoint_config_name or train_config_name,
        "finch_config_name": finch_config_name,
        "routes": routes,
    }
    if model_action_dim is not None:
        manifest["model_action_dim"] = int(model_action_dim)
    if config_modules:
        manifest["config_modules"] = list(config_modules)
    # Record which adapter serves this checkpoint, so deploy does not have to be
    # told. Best-effort: a config that cannot be resolved here still trains, and
    # deploy falls back to its ``--adapter`` flag.
    adapter_module = None
    if finch_config_name is not None:
        try:
            adapter_module = adapter_module_for_config(
                load_finch_config(
                    str(finch_config_name),
                    route=routes[0] if len(routes) > 1 else None,
                    config_modules=config_modules,
                )
            )
        except Exception:  # noqa: BLE001 - a manifest field is not worth failing a run over
            adapter_module = None
    if adapter_module:
        manifest["adapter_module"] = adapter_module
    return manifest


def save_policy_manifest(
    path: str | Path,
    *,
    train_config_name: str,
    finch_config_name: str | None,
    policy_type: str = "finch",
    checkpoint_config_name: str | None = None,
    model_action_dim: int | None = None,
    config_modules: list[str] | None = None,
) -> dict[str, object]:
    manifest = build_policy_manifest(
        train_config_name=train_config_name,
        finch_config_name=finch_config_name,
        policy_type=policy_type,
        checkpoint_config_name=checkpoint_config_name,
        model_action_dim=model_action_dim,
        config_modules=config_modules,
    )
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True))
    return manifest


def load_policy_manifest(path: str | Path) -> dict[str, Any] | None:
    path = Path(path)
    if path.is_dir():
        path = path / "policy_manifest.json"
    if not path.exists():
        return None
    return json.loads(path.read_text())


def require_policy_manifest(path: str | Path) -> dict[str, Any]:
    manifest = load_policy_manifest(path)
    if manifest is None:
        raise ValueError(f"Checkpoint '{Path(path)}' is missing policy_manifest.json")

    config_name = manifest.get("checkpoint_config_name") or manifest.get("train_config_name")
    if not config_name:
        raise ValueError("policy_manifest.json is missing 'checkpoint_config_name'/'train_config_name'")
    if "finch_config_name" not in manifest:
        raise ValueError("policy_manifest.json is missing 'finch_config_name'")
    if "routes" not in manifest:
        raise ValueError("policy_manifest.json is missing 'routes'")
    return manifest


def load_checkpoint_spec(
    path: str | Path,
    *,
    route: str | None = None,
    resolve_finch_config: bool = True,
) -> CheckpointSpec:
    """Load checkpoint identity and optionally reconstruct its dataset config.

    Serving uses the persisted Data Spec and passes ``resolve_finch_config=False``
    so a deployment does not require the training dataset to be mounted. Offline
    evaluation keeps the default because it needs the live config and dataset.
    """
    checkpoint_dir = Path(path)
    manifest = require_policy_manifest(checkpoint_dir)
    checkpoint_config_name = str(manifest.get("checkpoint_config_name") or manifest.get("train_config_name"))
    train_config_name = str(manifest.get("train_config_name") or checkpoint_config_name)
    finch_config_name = manifest.get("finch_config_name")
    routes = tuple(str(value) for value in manifest.get("routes", []) if value is not None)
    config_modules = [str(value) for value in manifest.get("config_modules", []) if value is not None]
    selected_route = resolve_selected_route(route, list(routes))
    finch_config = None
    if finch_config_name is not None and resolve_finch_config:
        finch_config = load_finch_config(str(finch_config_name), route=selected_route, config_modules=config_modules)
    model_action_dim = manifest.get("model_action_dim")
    # Prefer what the manifest recorded; derive it from the resolved config for
    # checkpoints written before the field existed, so an older run directory
    # still resolves its own adapter instead of falling through to a default.
    adapter_module = manifest.get("adapter_module")
    if adapter_module is None and finch_config is not None:
        adapter_module = adapter_module_for_config(finch_config)
    return CheckpointSpec(
        checkpoint_dir=checkpoint_dir,
        train_config_name=train_config_name,
        checkpoint_config_name=checkpoint_config_name,
        finch_config_name=None if finch_config_name is None else str(finch_config_name),
        policy_type=str(manifest.get("policy_type", "finch")),
        routes=routes,
        route=selected_route,
        finch_config=finch_config,
        model_action_dim=None if model_action_dim is None else int(model_action_dim),
        manifest=manifest,
        adapter_module=None if adapter_module is None else str(adapter_module),
    )


def resolve_selected_route(route: str | None, routes: list[str]) -> str | None:
    if not routes:
        return route
    if len(routes) == 1:
        only_route = routes[0]
        if route is not None and route != only_route:
            raise ValueError(f"Checkpoint only supports route '{only_route}', got '{route}'")
        return None
    if route is None:
        available = ", ".join(routes)
        raise ValueError(f"route is required for this checkpoint. Available routes: {available}")
    if route not in routes:
        available = ", ".join(routes)
        raise ValueError(f"Unknown route '{route}'. Available routes: {available}")
    return route


def load_finch_config(
    finch_config_name: str,
    route: str | None = None,
    *,
    config_modules: list[str] | None = None,
):
    from tau0_vla.data.config import get_config

    # Same contract as ``resolve_routes``: only discover if modules were
    # passed, otherwise rely on the registry already populated by the
    # caller's import-time ``@register_config`` side-effects.
    if config_modules:
        from tau0_vla.data.config import discover_config_modules

        discover_config_modules(extra_module_names=config_modules)
    finch_config = get_config(finch_config_name)
    if not hasattr(finch_config, "configs"):
        return finch_config

    if route is None:
        available = ", ".join(resolve_routes(finch_config_name))
        raise ValueError(
            f"route is required for mixed Finch config '{finch_config_name}'. Available routes: {available}"
        )

    child_lookup = {}
    for child in finch_config.configs:
        child_name = str(child if isinstance(child, str) else getattr(child, "name"))
        child_lookup[child_name] = get_config(child) if isinstance(child, str) else child
    try:
        return child_lookup[route]
    except KeyError as exc:
        available = ", ".join(sorted(child_lookup))
        raise ValueError(f"Unknown route '{route}'. Available routes: {available}") from exc


def align_actions_for_evaluation(
    gt_actions: np.ndarray,
    inferred_actions: np.ndarray,
    *,
    action_components,
) -> np.ndarray:
    from tau0_vla.data import EefPose

    aligned = np.asarray(inferred_actions, dtype=np.float32).copy()
    if not any(isinstance(component, EefPose) for component in action_components):
        return aligned

    pose_dim = sum(7 for component in action_components if getattr(component, "key", None) == "eef_pose")
    if pose_dim == 0:
        return aligned

    gt_pose = np.asarray(gt_actions[:, :pose_dim], dtype=np.float32)
    pred_pose = np.asarray(aligned[:, :pose_dim], dtype=np.float32)
    if gt_pose.shape != pred_pose.shape:
        raise ValueError(f"Shape mismatch for quaternion alignment: gt={gt_pose.shape}, pred={pred_pose.shape}")
    if gt_pose.shape[-1] % 7 != 0:
        raise ValueError(f"Expected xyz+quat pose blocks of 7 dims, got shape={gt_pose.shape}")

    gt_blocks = gt_pose.reshape(*gt_pose.shape[:-1], gt_pose.shape[-1] // 7, 7)
    pred_blocks = pred_pose.reshape(*pred_pose.shape[:-1], pred_pose.shape[-1] // 7, 7)
    sign = np.where(
        np.sum(gt_blocks[..., 3:] * pred_blocks[..., 3:], axis=-1, keepdims=True) < 0.0,
        -1.0,
        1.0,
    ).astype(np.float32)
    pred_blocks[..., 3:] *= sign
    aligned[:, :pose_dim] = pred_blocks.reshape(*pred_pose.shape[:-1], -1)
    return aligned
