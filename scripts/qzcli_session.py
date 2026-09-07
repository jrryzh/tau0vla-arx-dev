#!/usr/bin/env python3
"""Share a refreshed qzcli login across controllers without exposing credentials."""
from __future__ import annotations

import argparse
import fcntl
from pathlib import Path
import subprocess
import sys
import time

READ_COMMANDS = {"status", "logs", "list", "avail", "workspaces", "res", "resources"}


def ensure_session(credentials: Path, *, force: bool = False) -> None:
    directory = Path.home() / ".qzcli"
    directory.mkdir(exist_ok=True)
    stamp = directory / "arx_session_refreshed"
    with (directory / "arx_session.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if not force and stamp.exists() and time.time() - stamp.stat().st_mtime < 300:
            return
        lines = credentials.read_text().splitlines()
        if len(lines) < 2 or not all(lines[:2]):
            raise RuntimeError("Credential file requires two nonempty lines")
        result = subprocess.run(
            ["qzcli", "login", "--username", lines[0], "--password-stdin"],
            input=lines[1] + "\n", capture_output=True, text=True, timeout=90,
        )
        if result.returncode:
            raise RuntimeError("qzcli login failed; credential details suppressed")
        stamp.touch(mode=0o600)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--credentials", required=True, type=Path)
    parser.add_argument("--ensure", action="store_true")
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    ensure_session(args.credentials)
    if args.ensure:
        return 0
    command = args.command
    if command and command[0] == "--":
        command = command[1:]
    if not command:
        parser.error("qzcli command is required")
    result = subprocess.run(["qzcli", *command], capture_output=True, text=True)
    message = result.stdout + result.stderr
    auth_error = any(word in message.lower() for word in ("401", "cookie 已过期", "cookie失效", "重新登录", "not authenticated"))
    if auth_error and command[0] in READ_COMMANDS:
        ensure_session(args.credentials, force=True)
        result = subprocess.run(["qzcli", *command], capture_output=True, text=True)
    sys.stdout.write(result.stdout)
    sys.stderr.write(result.stderr)
    return result.returncode


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (RuntimeError, subprocess.TimeoutExpired) as exc:
        print(str(exc) if isinstance(exc, RuntimeError) else "qzcli login timed out", file=sys.stderr)
        raise SystemExit(1)
