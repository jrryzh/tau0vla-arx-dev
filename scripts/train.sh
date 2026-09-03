#!/usr/bin/env bash
# Launch training:  bash scripts/train.sh <config.yaml> [--key value ...]
#
# Reads run_name / project_name / output_dir / notes out of the YAML for logging
# and Weights & Biases, works out the torchrun topology, and forwards everything
# after the config path to train.py untouched.

CONFIG_PATH=$1
EXTRA_ARGS=("${@:2}")  # everything after the config path, forwarded verbatim

if [ -z "$CONFIG_PATH" ]; then
    echo "Usage: ./train.sh <path_to_yaml_config> [--key value ...]"
    echo "Example: ./train.sh configs/my_config.yaml --run_name exp1 --learning_rate 2e-5"
    exit 1
fi

# Base environment
export TZ="${TZ:-UTC}"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True,garbage_collection_threshold:0.97}"
# AUTO_RESUME=1 makes train.py continue from the newest checkpoint in output_dir.
export AUTO_RESUME="${AUTO_RESUME:-1}"
# Collator diagnostic threshold, in tokens. A sample longer than this gets one
# [LONG_SAMPLE] log line — the point is to notice an over-long prompt before it
# hits the model_max_length truncation.
export VLA_LOG_LONG_SAMPLE_LEN="${VLA_LOG_LONG_SAMPLE_LEN:-1600}"
# Invoke torchrun as a Python module so relocated virtual environments do not
# depend on the absolute interpreter path embedded in the torchrun shebang.
PYTHON_BIN="${PYTHON_BIN:-python}"

# Offline by default. On a machine without an outbound route, every Hub lookup
# otherwise blocks for 120-300s per task before giving up. Set these to 0 when
# you do want weights downloaded.
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"

# Multi-node DataLoader construction can take a while; raise the heartbeat
# timeout so a slow start is not mistaken for a dead rank.
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=${TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC:-7200}
export NCCL_TIMEOUT=${NCCL_TIMEOUT:-7200}
export NCCL_SOCKET_FAMILY=AF_INET
# IB: 1 QP per connection — avoids QP explosion / setup overhead at large scale.
export NCCL_IB_QPS_PER_CONNECTION=${NCCL_IB_QPS_PER_CONNECTION:-1}


# === 1. torchrun topology ===
# Single node: GPU count is auto-detected. Multi-node: the scheduler injects
# either PET_* (torchrun elastic) or WORLD_SIZE/RANK. Neither present → 1 node.
# NNODES is exported because train.py reads it back to log the topology.
export NNODES=${PET_NNODES:-${WORLD_SIZE:-1}}
NODE_RANK=${PET_NODE_RANK:-${RANK:-0}}
NPROC_PER_NODE=${NPROC_PER_NODE:-${PET_NPROC_PER_NODE:-$(nvidia-smi -L | wc -l)}}
MASTER_ADDR=${MASTER_ADDR:-${PET_MASTER_ADDR:-127.0.0.1}}
MASTER_PORT=${MASTER_PORT:-${PET_MASTER_PORT:-$(shuf -i 20001-29999 -n 1)}}


# === 2. Pull run_name / project_name / output_dir / notes out of the YAML ===
# `awk '{print $2}'` would truncate a quoted value at its first space, which is
# what `notes` always is. Take everything after the first colon instead, then
# strip surrounding whitespace and one layer of quotes.
_yaml_scalar() {
    sed -n "s/^[[:space:]]*$1:[[:space:]]*//p" "$CONFIG_PATH" \
        | head -n 1 \
        | sed -e 's/[[:space:]]*$//' -e 's/^"\(.*\)"$/\1/' -e "s/^'\(.*\)'$/\1/"
}

PROJECT_NAME=$(_yaml_scalar project_name)
RUN_NAME=$(_yaml_scalar run_name)
BASE_DIR=$(_yaml_scalar output_dir)
NOTES=$(_yaml_scalar notes)

# CLI wins over the YAML for these two.
for ((i = 0; i < ${#EXTRA_ARGS[@]}; i++)); do
    case "${EXTRA_ARGS[$i]}" in
        --run_name)    RUN_NAME="${EXTRA_ARGS[$((i+1))]}"; ((i++)) ;;
        --output_dir)  BASE_DIR="${EXTRA_ARGS[$((i+1))]}"; ((i++)) ;;
    esac
done

PROJECT_NAME=${PROJECT_NAME:-"tau-0-vla"}
RUN_NAME=${RUN_NAME:-$(basename "$CONFIG_PATH" .yaml)}
BASE_DIR=${BASE_DIR:-"outputs"}
NOTES=${NOTES:-"notes"}

# An unedited template would otherwise create a directory literally named
# "<YOUR_RUN_NAME>" here, before Python gets a chance to validate anything.
case "$RUN_NAME$BASE_DIR" in
    *'<YOUR_'*|*'<PATH_TO_YOUR_'*)
        echo "Error: $CONFIG_PATH still carries placeholders." >&2
        echo "  run_name:   $RUN_NAME" >&2
        echo "  output_dir: $BASE_DIR" >&2
        echo "Fill them in, or override with --run_name / --output_dir." >&2
        echo "  grep -rn 'YOUR_' $(dirname "$CONFIG_PATH")" >&2
        exit 1
        ;;
esac

export RUN_NAME

# Log directory
LOG_DIR="${BASE_DIR}/${RUN_NAME}/log"
mkdir -p "$LOG_DIR"

# Weights & Biases; an already-exported value is left alone
export WANDB_PROJECT="${WANDB_PROJECT:-${PROJECT_NAME}}"
export WANDB_NAME="${WANDB_NAME:-${RUN_NAME}}"
export WANDB_NOTES="${WANDB_NOTES:-${NOTES}}"
export WANDB_DIR="${WANDB_DIR:-${BASE_DIR}/${RUN_NAME}}"
mkdir -p "$WANDB_DIR"

export HF_HOME=${HF_HOME:-${HOME}/.cache/}

# === 3. Startup summary ===
echo "[INFO] Launching with torchrun: nnodes=$NNODES, nproc_per_node=$NPROC_PER_NODE, node_rank=$NODE_RANK"
echo "[INFO] Master: $MASTER_ADDR:$MASTER_PORT"
echo "[INFO] Running experiment with config: $CONFIG_PATH"
echo "[INFO] PROJECT_NAME: $PROJECT_NAME"
echo "[INFO] NOTES: $NOTES"
echo "[INFO] HF_HOME: $HF_HOME"
echo "[INFO] WANDB_DIR: $WANDB_DIR"
echo "[INFO] LOG_DIR: $LOG_DIR"
echo "[INFO] AUTO_RESUME: $AUTO_RESUME"
if [ ${#EXTRA_ARGS[@]} -gt 0 ]; then
    echo "[INFO] CLI overrides: ${EXTRA_ARGS[*]}"
fi

# === 4. Go ===
"$PYTHON_BIN" -m torch.distributed.run \
    --nnodes="$NNODES" \
    --node-rank="$NODE_RANK" \
    --nproc-per-node="$NPROC_PER_NODE" \
    --master-addr="$MASTER_ADDR" \
    --master-port="$MASTER_PORT" \
    src/tau0_vla/trainer/train.py "$CONFIG_PATH" "${EXTRA_ARGS[@]}" \
    2>&1 | tee -a "${LOG_DIR}/training_log_nodeIdx$(printf "%03d" ${NODE_RANK})_$(date +"%Y%m%d_%H%M").txt"
