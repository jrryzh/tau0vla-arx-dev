#!/usr/bin/env python3
"""Prepare, candidate-test, promote, roll back, and inspect ARX model serving."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import time
from urllib.error import URLError
from urllib.request import ProxyHandler, build_opener


ROOT = Path(__file__).resolve().parents[1]
START_SCRIPT = ROOT / "scripts/start_arx_lift2s_server.sh"
STATE_DIR = Path("/home/xiangchengliu/logs/tau0vla-arx/deployment-state")
LOG_DIR = Path("/home/xiangchengliu/logs/tau0vla-arx")
PRODUCTION_SESSION = "tau0vla-arx-server"
CANDIDATE_SESSION = "tau0vla-arx-candidate"
PRODUCTION_URL = "http://192.168.50.2:8000"
CANDIDATE_URL = "http://127.0.0.1:8001"
OPENER = build_opener(ProxyHandler({}))


def _run(
    *command: str,
    check: bool = True,
    capture: bool = False,
    cwd: str | None = None,
):
    return subprocess.run(
        command,
        check=check,
        text=True,
        capture_output=capture,
        cwd=cwd,
    )


def _health(url: str, timeout: float = 3.0) -> dict | None:
    try:
        with OPENER.open(f"{url}/health", timeout=timeout) as response:
            return json.loads(response.read())
    except (OSError, URLError, ValueError):
        return None


def _wait_health(url: str, expected: dict, timeout: float = 180.0) -> dict:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        health = _health(url)
        if health is not None:
            for key in ("model_id", "checkpoint_sha256"):
                if health.get(key) != expected[key]:
                    raise RuntimeError(
                        f"{url} {key}={health.get(key)!r}, expected {expected[key]!r}"
                    )
            if health.get("ready") is True:
                return health
        time.sleep(2.0)
    raise TimeoutError(f"timed out waiting for {url}/health")


def _manifest(bundle: Path) -> dict:
    bundle = bundle.resolve()
    manifest_path = bundle / "deployment_manifest.json"
    checksum_path = bundle / "SHA256SUMS"
    if not manifest_path.is_file() or not checksum_path.is_file():
        raise FileNotFoundError("bundle requires deployment_manifest.json and SHA256SUMS")
    _run("sha256sum", "-c", "SHA256SUMS", check=True, capture=False, cwd=str(bundle))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    required = ("model_id", "model_sha256", "route", "service_compatibility_commit")
    missing = [key for key in required if not manifest.get(key)]
    if missing:
        raise ValueError(f"deployment manifest is missing {missing}")
    if manifest["model_sha256"] != _sha256(bundle / "model.safetensors"):
        raise ValueError("deployment manifest model_sha256 mismatch")
    commit = str(manifest["service_compatibility_commit"])
    ancestry = subprocess.run(
        ["git", "-C", str(ROOT), "merge-base", "--is-ancestor", commit, "HEAD"]
    )
    if ancestry.returncode != 0:
        raise ValueError(f"service HEAD does not contain compatibility commit {commit}")
    manifest["model_dir"] = str(bundle)
    manifest["checkpoint_sha256"] = manifest["model_sha256"]
    return manifest


def _sha256(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _tmux_exists(session: str) -> bool:
    return subprocess.run(
        ["tmux", "has-session", "-t", session],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    ).returncode == 0


def _stop(session: str) -> None:
    if _tmux_exists(session):
        _run("tmux", "kill-session", "-t", session, check=False)
        time.sleep(1.0)


def _start(config: dict, *, candidate: bool) -> None:
    session = CANDIDATE_SESSION if candidate else PRODUCTION_SESSION
    bind_host = "127.0.0.1" if candidate else "192.168.50.2"
    client_ip = "127.0.0.1" if candidate else "192.168.50.1"
    port = "8001" if candidate else "8000"
    suffix = "candidate" if candidate else "active"
    log_file = LOG_DIR / f"server_{config['model_id']}_{suffix}.log"
    if _tmux_exists(session):
        raise RuntimeError(f"tmux session already exists: {session}")
    command = [
        "tmux",
        "new-session",
        "-d",
        "-s",
        session,
        "env",
        f"MODEL_DIR={config['model_dir']}",
        f"MODEL_ID={config['model_id']}",
        f"CHECKPOINT_SHA256={config['checkpoint_sha256']}",
        f"BIND_HOST={bind_host}",
        f"PORT={port}",
        f"ARX_CLIENT_IP={client_ip}",
        f"TMUX_SESSION={session}",
        f"LOG_FILE={log_file}",
        f"ARX_RECORD_DIR={LOG_DIR / 'requests'}",
        str(START_SCRIPT),
        "--foreground",
    ]
    _run(*command)


def _live_config() -> dict | None:
    if not _tmux_exists(PRODUCTION_SESSION):
        return None
    result = _run(
        "tmux",
        "list-panes",
        "-t",
        PRODUCTION_SESSION,
        "-F",
        "#{pane_pid}",
        capture=True,
    )
    pid = int(result.stdout.strip().splitlines()[0])
    command = Path(f"/proc/{pid}/cmdline").read_bytes().split(b"\0")
    args = [value.decode() for value in command if value]

    def option(name: str) -> str:
        return args[args.index(name) + 1]

    return {
        "model_dir": option("--model"),
        "model_id": option("--model-id"),
        "checkpoint_sha256": option("--checkpoint-sha256"),
    }


def prepare(bundle: Path) -> None:
    manifest = _manifest(bundle)
    print(json.dumps(manifest, indent=2, sort_keys=True))


def candidate(bundle: Path) -> None:
    manifest = _manifest(bundle)
    _stop(CANDIDATE_SESSION)
    _start(manifest, candidate=True)
    health = _wait_health(CANDIDATE_URL, manifest)
    print(json.dumps(health, indent=2, sort_keys=True))


def promote(bundle: Path) -> None:
    manifest = _manifest(bundle)
    candidate_health = _health(CANDIDATE_URL)
    if candidate_health is None:
        raise RuntimeError("candidate is not healthy; run candidate first")
    for key in ("model_id", "checkpoint_sha256"):
        if candidate_health.get(key) != manifest[key]:
            raise RuntimeError("candidate identity does not match requested bundle")
    previous = _live_config()
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    if previous is not None:
        (STATE_DIR / "previous.json").write_text(
            json.dumps(previous, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    _stop(CANDIDATE_SESSION)
    _stop(PRODUCTION_SESSION)
    time.sleep(2.0)
    try:
        _start(manifest, candidate=False)
        health = _wait_health(PRODUCTION_URL, manifest)
    except Exception:
        _stop(PRODUCTION_SESSION)
        if previous is not None:
            _start(previous, candidate=False)
            _wait_health(PRODUCTION_URL, previous)
        raise
    (STATE_DIR / "active.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(health, indent=2, sort_keys=True))


def rollback() -> None:
    path = STATE_DIR / "previous.json"
    if not path.is_file():
        raise FileNotFoundError(f"no rollback state: {path}")
    previous = json.loads(path.read_text(encoding="utf-8"))
    _stop(CANDIDATE_SESSION)
    _stop(PRODUCTION_SESSION)
    time.sleep(2.0)
    _start(previous, candidate=False)
    print(json.dumps(_wait_health(PRODUCTION_URL, previous), indent=2, sort_keys=True))


def status() -> None:
    print(json.dumps({
        "production": _health(PRODUCTION_URL),
        "candidate": _health(CANDIDATE_URL),
        "production_tmux": _tmux_exists(PRODUCTION_SESSION),
        "candidate_tmux": _tmux_exists(CANDIDATE_SESSION),
    }, indent=2, sort_keys=True))


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("prepare", "candidate", "promote"):
        command = sub.add_parser(name)
        command.add_argument("bundle", type=Path)
    sub.add_parser("rollback")
    sub.add_parser("status")
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    globals()[arguments.command](**(
        {"bundle": arguments.bundle} if hasattr(arguments, "bundle") else {}
    ))
