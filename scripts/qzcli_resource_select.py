#!/usr/bin/env python3
"""Read qzcli's refreshed cache and list H200 groups or select an 8-GPU spec."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _h200(item: dict) -> bool:
    return "H200" in " ".join(str(item.get(key, "")) for key in ("name", "gpu_type", "gpu_type_display")).upper()


def _resources() -> dict:
    path = Path.home() / ".qzcli" / "resources.json"
    if not path.is_file():
        raise SystemExit(f"qzcli resource cache not found: {path}")
    return json.loads(path.read_text())


def _group_ids(spec: dict) -> set[str]:
    values = spec.get("logic_compute_group_ids") or []
    if isinstance(values, str):
        values = [values]
    direct = spec.get("logic_compute_group_id")
    if direct:
        values = [*values, direct]
    return {str(value) for value in values if value}


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("groups")
    spec_parser = sub.add_parser("spec")
    spec_parser.add_argument("workspace")
    spec_parser.add_argument("group")
    args = parser.parse_args()
    resources = _resources()

    if args.command == "groups":
        rows = []
        for workspace_id, workspace in resources.items():
            for group_id, group in workspace.get("compute_groups", {}).items():
                if _h200(group):
                    rows.append((str(workspace_id), str(group_id)))
        for row in sorted(rows):
            print("\t".join(row))
        return

    workspace = resources.get(args.workspace) or {}
    candidates = []
    for spec_id, spec in workspace.get("specs", {}).items():
        if int(spec.get("gpu_count") or 0) != 8:
            continue
        if not _h200(spec):
            continue
        memberships = _group_ids(spec)
        if memberships and args.group not in memberships:
            continue
        candidates.append((str(spec_id), spec))
    if not candidates:
        raise SystemExit(f"no 8-GPU H200 spec found for {args.workspace}/{args.group}")
    spec_id, spec = sorted(candidates, key=lambda item: item[0])[0]
    print(spec_id)
    print(int(spec.get("cpu_count") or 0))
    print(int(spec.get("memory_gb") or 0))


if __name__ == "__main__":
    main()
