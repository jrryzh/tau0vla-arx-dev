#!/usr/bin/env bash
# Run a command and emit the peak per-GPU memory observed on this worker.
set -uo pipefail

if [ "$#" -eq 0 ]; then
    echo "Usage: $0 <command> [args ...]" >&2
    exit 2
fi

SAMPLES=$(mktemp)
"$@" &
COMMAND_PID=$!
while kill -0 "$COMMAND_PID" 2>/dev/null; do
    nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits >> "$SAMPLES" 2>/dev/null || true
    sleep 1
done
wait "$COMMAND_PID"
STATUS=$?
PEAK=$(awk 'BEGIN {max=0} $1+0 > max {max=$1+0} END {print max}' "$SAMPLES")
rm -f "$SAMPLES"
echo "[H200_PEAK_MEMORY] peak_mib=${PEAK:-0}"
exit "$STATUS"
