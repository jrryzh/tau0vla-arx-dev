#!/bin/bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
script_path="${repo_root}/scripts/$(basename "${BASH_SOURCE[0]}")"
: "${MODEL_DIR:=/home/xiangchengliu/models/tau0vla-arx-pickplace-h200-step10000}"
: "${PYTHON_BIN:=/home/xiangchengliu/anaconda3/envs/tau0-vla/bin/python}"
: "${BIND_HOST:=192.168.31.83}"
: "${PORT:=8000}"
: "${ARX_CLIENT_IP:=192.168.31.57}"
: "${MODEL_ID:=tau0vla-arx-pickplace-h200-step10000}"
: "${TMUX_SESSION:=tau0vla-arx-server}"
: "${LOG_DIR:=/home/xiangchengliu/logs/tau0vla-arx}"
: "${CHECKPOINT_SHA256:?Set CHECKPOINT_SHA256 to the verified model.safetensors digest}"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Python interpreter not found: ${PYTHON_BIN}" >&2
  exit 1
fi
if [[ ! -f "${MODEL_DIR}/model.safetensors" ]]; then
  echo "Checkpoint not found: ${MODEL_DIR}/model.safetensors" >&2
  exit 1
fi

if [[ "${1:-}" == "--foreground" ]]; then
  mkdir -p "${LOG_DIR}"
  : "${LOG_FILE:=${LOG_DIR}/server_$(date +%Y%m%d_%H%M%S).log}"
  exec > >(tee -a "${LOG_FILE}") 2>&1
  cd "${repo_root}"
  exec "${PYTHON_BIN}" -m deploy.server \
    --model "${MODEL_DIR}" \
    --host "${BIND_HOST}" \
    --port "${PORT}" \
    --model-id "${MODEL_ID}" \
    --checkpoint-sha256 "${CHECKPOINT_SHA256}" \
    --allow-client-ip "${ARX_CLIENT_IP}" \
    --allow-client-ip "${BIND_HOST}" \
    --allow-client-ip 127.0.0.1 \
    --infer-mode optim \
    --warmup-steps 3
fi

if tmux has-session -t "${TMUX_SESSION}" 2>/dev/null; then
  echo "Refused: tmux session ${TMUX_SESSION} already exists." >&2
  echo "Inspect it with: tmux attach -t ${TMUX_SESSION}" >&2
  exit 1
fi

mkdir -p "${LOG_DIR}"
export MODEL_DIR PYTHON_BIN BIND_HOST PORT ARX_CLIENT_IP MODEL_ID TMUX_SESSION LOG_DIR CHECKPOINT_SHA256
tmux new-session -d -s "${TMUX_SESSION}" "$(printf '%q' "${script_path}") --foreground"
echo "Started ${TMUX_SESSION}; attach with: tmux attach -t ${TMUX_SESSION}"
