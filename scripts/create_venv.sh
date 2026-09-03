#!/usr/bin/env bash
set -euo pipefail

REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
if [ ! -d "$REPO/.venv" ]; then
    /usr/bin/python3 -m venv "$REPO/.venv"
fi
PYTHON_BIN="$REPO/.venv/bin/python" bash "$REPO/scripts/setup.sh"
