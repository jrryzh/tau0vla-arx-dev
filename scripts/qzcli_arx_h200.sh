#!/usr/bin/env bash
# Guarded ARX H200 workflow with task-specific launch profiles.
#
#   scripts/qzcli_arx_h200.sh auto --profile tool-yipan --credentials /secure/qzcli.txt
#   scripts/qzcli_arx_h200.sh smoke --profile pickplace --credentials /secure/qzcli.txt
set -euo pipefail

MODE=${1:-}
if [ -z "$MODE" ]; then
    echo "Usage: $0 {auto|smoke|formal} [--credentials <file> | --cookie-file <file>] [--profile pickplace|tool-yipan] [--poll-seconds N]" >&2
    exit 2
fi
shift

CREDENTIALS=""
COOKIE_FILE=""
PROFILE=pickplace
POLL_SECONDS=30
while [ "$#" -gt 0 ]; do
    case "$1" in
        --credentials) CREDENTIALS=${2:?missing credentials file}; shift 2 ;;
        --cookie-file) COOKIE_FILE=${2:?missing cookie file}; shift 2 ;;
        --profile) PROFILE=${2:?missing profile}; shift 2 ;;
        --poll-seconds) POLL_SECONDS=${2:?missing poll interval}; shift 2 ;;
        *) echo "Unknown argument: $1" >&2; exit 2 ;;
    esac
done
case "$MODE" in auto|smoke|formal) ;; *) echo "Unknown mode: $MODE" >&2; exit 2 ;; esac
if [ -n "$CREDENTIALS" ] && [ -n "$COOKIE_FILE" ]; then
    echo "Use only one of --credentials and --cookie-file." >&2
    exit 2
fi
if [ -z "$CREDENTIALS" ] && [ -z "$COOKIE_FILE" ] && [ -f "${HOME}/.qzcli/.cookie" ]; then
    COOKIE_FILE=${HOME}/.qzcli/.cookie
fi
if { [ -n "$CREDENTIALS" ] && [ ! -f "$CREDENTIALS" ]; } \
    || { [ -n "$COOKIE_FILE" ] && [ ! -f "$COOKIE_FILE" ]; } \
    || { [ -z "$CREDENTIALS" ] && [ -z "$COOKIE_FILE" ]; }; then
    echo "A username/password credential file or qzcli cookie file is required." >&2
    exit 2
fi

REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
PYTHON_BIN="$REPO/.venv/bin/python"
GPUS_PER_NODE=8
SHM_GI=1200
GLOBAL_BATCH=128
FREE_KIB_REQUIRED=$((900 * 1024 * 1024))
TARGET_WORKSPACE=""
TARGET_GROUP=""

case "$PROFILE" in
    pickplace)
        CONFIG="$REPO/configs/arx_lift2s_pickplace/train_h200.yaml"
        FORMAL_ROOT="$REPO/outputs/arx_lift2s_pickplace_h200_formal"
        FORMAL_RUN=arx_lift2s_pickplace_h200_formal
        SMOKE_ROOT="$REPO/outputs/arx_lift2s_pickplace_h200_smoke"
        MARKER="$REPO/outputs/.arx_h200_smoke_passed"
        STATE_DIR="$REPO/outputs/qzcli_arx_h200"
        SMOKE_ATTEMPTS=("2 8 1" "2 4 2" "2 2 4")
        ;;
    tool-yipan)
        CONFIG="$REPO/configs/arx_lift2s_pickplace_tool_yipan/train_h200.yaml"
        FORMAL_ROOT="$REPO/outputs/arx_lift2s_pickplace_tool_yipan_h200_formal"
        FORMAL_RUN=arx_lift2s_pickplace_tool_yipan_h200_formal
        SMOKE_ROOT="$REPO/outputs/arx_lift2s_pickplace_tool_yipan_h200_smoke"
        MARKER="$REPO/outputs/.arx_h200_tool_yipan_smoke_passed"
        STATE_DIR="$REPO/outputs/qzcli_arx_h200_tool_yipan"
        TARGET_WORKSPACE=ws-21bd7e9f-5f97-4ffa-831e-966a436c7818
        TARGET_GROUP=lcg-d8eb9030-2233-47f7-b8cb-988c3e7c0ec9
        SMOKE_ATTEMPTS=("1 16 1" "2 8 1")
        ;;
    *) echo "Unknown profile: $PROFILE" >&2; exit 2 ;;
esac

FORMAL_DIR="$FORMAL_ROOT/$FORMAL_RUN"
mkdir -p "$STATE_DIR" "$FORMAL_ROOT" "$SMOKE_ROOT"
printf '%s\n' "$$" > "$STATE_DIR/controller.pid"
if [ ! -x "$PYTHON_BIN" ]; then
    echo "Missing project environment: $PYTHON_BIN (run scripts/create_venv.sh first)" >&2
    exit 1
fi

LOGIN_LOG=$(mktemp)
trap 'rm -f "$LOGIN_LOG"' EXIT
if [ -n "$CREDENTIALS" ]; then
    IFS= read -r QZ_USERNAME < "$CREDENTIALS"
    QZ_PASSWORD=$(sed -n '2p' "$CREDENTIALS")
    if [ -z "$QZ_USERNAME" ] || [ -z "$QZ_PASSWORD" ]; then
        echo "Credential file must contain a non-empty username on line 1 and password on line 2." >&2
        exit 2
    fi
    if ! printf '%s\n' "$QZ_PASSWORD" | qzcli login --username "$QZ_USERNAME" --password-stdin >"$LOGIN_LOG" 2>&1; then
        echo "qzcli login failed (credential contents were not printed)." >&2
        exit 1
    fi
    unset QZ_PASSWORD
else
    if ! qzcli cookie --file "$COOKIE_FILE" >"$LOGIN_LOG" 2>&1; then
        echo "qzcli cookie validation failed (cookie contents were not printed)." >&2
        exit 1
    fi
fi
echo "[QZCLI] Login succeeded; refreshing workspace/resource/spec cache."
RESOURCE_CACHE=${HOME}/.qzcli/resources.json
CACHE_MAX_AGE_SECONDS=3600
CACHE_AGE_SECONDS=$(( $(date +%s) - $(stat -c %Y "$RESOURCE_CACHE" 2>/dev/null || echo 0) ))
if [ -s "$RESOURCE_CACHE" ] \
    && [ "$CACHE_AGE_SECONDS" -ge 0 ] \
    && [ "$CACHE_AGE_SECONDS" -le "$CACHE_MAX_AGE_SECONDS" ] \
    && [ -n "$(/usr/bin/python3 "$REPO/scripts/qzcli_resource_select.py" groups)" ]; then
    echo "[QZCLI] Reusing ${CACHE_AGE_SECONDS}s-old H200 resource cache."
else
    qzcli workspaces --update --full --parallel 1
fi

select_resources() {
    local instances=$1 best_ws="" best_group="" best_free=-1 ws group output free
    if [ -n "$TARGET_GROUP" ]; then
        best_ws=$TARGET_WORKSPACE
        best_group=$TARGET_GROUP
        if ! output=$(qzcli avail --workspace "$best_ws" --group "$best_group" --nodes "$instances" --export 2>&1); then
            return 1
        fi
        best_free=$(printf '%s\n' "$output" | sed -n 's/.*[ (]\([0-9][0-9]*\) 空节点.*/\1/p' | tail -n 1)
        [ -n "$best_free" ] && [ "$best_free" -ge "$instances" ] || return 1
    else
        while IFS=$'\t' read -r ws group; do
            [ -n "$ws" ] || continue
            if output=$(qzcli avail --workspace "$ws" --group "$group" --nodes "$instances" --export 2>&1); then
                free=$(printf '%s\n' "$output" | sed -n 's/.*[ (]\([0-9][0-9]*\) 空节点.*/\1/p' | tail -n 1)
                [ -n "$free" ] || continue
                if [ "$free" -ge "$instances" ] && { [ "$free" -gt "$best_free" ] || { [ "$free" -eq "$best_free" ] && [[ "$group" < "$best_group" ]]; }; }; then
                    best_ws=$ws
                    best_group=$group
                    best_free=$free
                fi
            fi
        done < <("/usr/bin/python3" "$REPO/scripts/qzcli_resource_select.py" groups)
        [ -n "$best_group" ] || return 1
    fi
    mapfile -t spec_fields < <("/usr/bin/python3" "$REPO/scripts/qzcli_resource_select.py" spec "$best_ws" "$best_group")
    if [ "${#spec_fields[@]}" -ne 3 ] || [ "${spec_fields[1]}" -le 0 ] || [ "${spec_fields[2]}" -le 0 ]; then
        echo "Invalid H200 8-GPU spec metadata for $best_ws/$best_group" >&2
        return 1
    fi
    SELECTED_WORKSPACE=$best_ws
    SELECTED_GROUP=$best_group
    SELECTED_FREE=$best_free
    SELECTED_SPEC=${spec_fields[0]}
    SELECTED_CPU=${spec_fields[1]}
    SELECTED_MEMORY=${spec_fields[2]}
    echo "[QZCLI] Selected $SELECTED_GROUP in $SELECTED_WORKSPACE ($SELECTED_FREE idle nodes), spec $SELECTED_SPEC."
}

wait_for_resources() {
    local instances=$1
    while ! select_resources "$instances"; do
        echo "[QZCLI] Waiting ${POLL_SECONDS}s for $instances idle H200 node(s)."
        sleep "$POLL_SECONDS"
    done
}

validate_dry_run() {
    local text=$1 instances=$2 batch=$3 accum=$4 world_size
    world_size=$((instances * GPUS_PER_NODE))
    printf '%s' "$text" | "$PYTHON_BIN" "$REPO/scripts/validate_qzcli_payload.py" \
        --compute-group "$SELECTED_GROUP" --spec "$SELECTED_SPEC" --repo "$REPO" \
        --instances "$instances" --gpus-per-node "$GPUS_PER_NODE" --shm-gi "$SHM_GI" \
        --world-size "$world_size" --global-batch "$GLOBAL_BATCH" \
        --per-device-batch "$batch" --gradient-accumulation "$accum"
}

submit_job() {
    local name=$1 command=$2 instances=$3 batch=$4 accum=$5 dry actual
    local -a args=(
        --name "$name" --command "$command"
        --workspace "$SELECTED_WORKSPACE" --compute-group "$SELECTED_GROUP"
        --spec "$SELECTED_SPEC" --gpu-type NVIDIA_H200_SXM_141G
        --cpu "$SELECTED_CPU" --gpus "$GPUS_PER_NODE" --memory "$SELECTED_MEMORY"
        --instances "$instances" --shm "$SHM_GI" --framework pytorch
    )
    dry=$(qzcli create "${args[@]}" --dry-run) || return 1
    validate_dry_run "$dry" "$instances" "$batch" "$accum" || return 1
    actual=$(qzcli create "${args[@]}" --json) || return 1
    printf '%s\n' "$actual" > "$STATE_DIR/${name}_submit.log"
    printf '%s' "$actual" | "$PYTHON_BIN" -c '
import json, sys
lines = [line for line in sys.stdin.read().splitlines() if line.lstrip().startswith("{")]
if not lines:
    raise SystemExit("qzcli create did not return JSON")
print(json.loads(lines[-1])["job_id"])
'
}

job_status() {
    qzcli status "$1" --json 2>/dev/null | "$PYTHON_BIN" -c '
import json, sys
text = sys.stdin.read()
starts = [index for index in range(len(text)) if text.startswith("{\n", index)]
for start in reversed(starts):
    try:
        obj = json.loads(text[start:])
    except json.JSONDecodeError:
        continue
    print(str(obj.get("status", "unknown")).lower())
    break
else:
    print("unknown")
'
}

collect_worker_logs() {
    local job_id=$1 instances=$2 suffix=$3 worker worker_logs combined=""
    for ((index = 0; index < instances; index++)); do
        worker="worker-$index"
        # --raw avoids the interactive renderer's much smaller display window.
        # The server currently caps a response at 1000 records even when a
        # larger tail is requested, so durable train.sh logs are also passed to
        # the smoke validator below.
        worker_logs=$(qzcli logs "$job_id" --pod "${job_id}-${worker}" --tail 10000 --raw 2>&1 || true)
        printf '%s\n' "$worker_logs" > "$STATE_DIR/${job_id}_${worker}_${suffix}.log"
        combined+=$'\n'
        combined+=$worker_logs
    done
    printf '%s\n' "$combined"
}

wait_for_smoke() {
    local job_id=$1 instances=$2 batch=$3 world_size status logs index
    local run_name run_dir status_file
    local -a validator_args training_logs
    world_size=$((instances * GPUS_PER_NODE))
    while true; do
        status=$(job_status "$job_id" || true)
        echo "[SMOKE] job=$job_id status=$status"
        case "$status" in
            *succeeded*) break ;;
            *failed*|*stopped*|*error*)
                logs=$(collect_worker_logs "$job_id" "$instances" failed)
                printf '%s\n' "$logs" > "$STATE_DIR/${job_id}_failed.log"
                return 1
                ;;
        esac
        qzcli logs "$job_id" --tail 80 > "$STATE_DIR/${job_id}_latest.log" 2>&1 || true
        sleep "$POLL_SECONDS"
    done
    logs=$(collect_worker_logs "$job_id" "$instances" success)
    printf '%s\n' "$logs" > "$STATE_DIR/${job_id}_success.log"
    status_file="$STATE_DIR/${job_id}_status.json"
    qzcli status "$job_id" --json > "$status_file"
    run_name="${FORMAL_RUN%_formal}_smoke_${instances}x${GPUS_PER_NODE}_b${batch}"
    run_dir="$SMOKE_ROOT/$run_name"
    validator_args=(
        "$REPO/scripts/validate_h200_smoke.py"
        --status-json "$status_file" --job-id "$job_id"
        --compute-group "$SELECTED_GROUP" --instances "$instances"
        --gpus-per-node "$GPUS_PER_NODE" --shm-gi "$SHM_GI"
        --world-size "$world_size" --global-batch "$GLOBAL_BATCH"
    )
    for ((index = 0; index < instances; index++)); do
        validator_args+=(--worker-log "$STATE_DIR/${job_id}_worker-${index}_success.log")
    done
    mapfile -t training_logs < <(find "$run_dir/log" -maxdepth 1 -type f -name 'training_log_nodeIdx*.txt' | sort)
    for training_log in "${training_logs[@]}"; do
        validator_args+=(--training-log "$training_log")
    done
    if ! "$PYTHON_BIN" "${validator_args[@]}" | tee "$STATE_DIR/${job_id}_validation.log"; then
        return 1
    fi
}

remote_command() {
    local kind=$1 instances=$2 batch=$3 accum=$4
    local world_size run_name output_root auto_resume max_steps save_strategy skip_final_save
    world_size=$((instances * GPUS_PER_NODE))
    if [ "$kind" = smoke ]; then
        run_name="${FORMAL_RUN%_formal}_smoke_${instances}x${GPUS_PER_NODE}_b${batch}"
        output_root=$SMOKE_ROOT
        auto_resume=0
        max_steps=20
        save_strategy=no
        skip_final_save=1
    else
        run_name=$FORMAL_RUN
        output_root=$FORMAL_ROOT
        auto_resume=1
        max_steps=10000
        save_strategy=steps
        skip_final_save=0
    fi
    printf "cd %q && source .venv/bin/activate && EXPECTED_NNODES=%q EXPECTED_GPUS_PER_NODE=%q python scripts/check_h200_worker.py && test \$(df -Pk %q | awk 'NR==2 {print \$4}') -ge %q && export PYTHON_BIN=%q AUTO_RESUME=%q SKIP_FINAL_SAVE=%q REQUIRE_WORLD_SIZE=%q REQUIRE_GLOBAL_BATCH=%q REQUIRE_ALL_TRAINABLE=1 && bash scripts/run_with_gpu_peak.sh bash scripts/train.sh %q --run_name %q --output_dir %q --max_steps %q --per_device_train_batch_size %q --gradient_accumulation_steps %q --save_strategy %q" \
        "$REPO" "$instances" "$GPUS_PER_NODE" "$output_root" "$FREE_KIB_REQUIRED" \
        "$PYTHON_BIN" "$auto_resume" "$skip_final_save" "$world_size" "$GLOBAL_BATCH" "$CONFIG" "$run_name" \
        "$output_root" "$max_steps" "$batch" "$accum" "$save_strategy"
}

run_smoke() {
    local attempt instances batch accum command name job_id logs
    for attempt in "${SMOKE_ATTEMPTS[@]}"; do
        read -r instances batch accum <<< "$attempt"
        wait_for_resources "$instances"
        command=$(remote_command smoke "$instances" "$batch" "$accum")
        name="arx-${PROFILE}-h200-smoke-${instances}x8-b${batch}-$(date +%m%d-%H%M%S)"
        job_id=$(submit_job "$name" "$command" "$instances" "$batch" "$accum")
        echo "[SMOKE] Submitted $job_id (instances=$instances, world_size=$((instances * GPUS_PER_NODE)), batch=$batch, accumulation=$accum)."
        if wait_for_smoke "$job_id" "$instances" "$batch"; then
            printf 'profile=%s\ninstances=%s\nworld_size=%s\nbatch=%s\naccumulation=%s\njob_id=%s\n' \
                "$PROFILE" "$instances" "$((instances * GPUS_PER_NODE))" "$batch" "$accum" "$job_id" > "$MARKER"
            echo "[SMOKE] Accepted: job=$job_id."
            return 0
        fi
        logs="$STATE_DIR/${job_id}_failed.log"
        if [ ! -f "$logs" ] || ! grep -Eiq 'out of memory|cuda.*oom' "$logs"; then
            echo "Smoke failed for a reason other than CUDA OOM; refusing automatic fallback." >&2
            return 1
        fi
        echo "[SMOKE] CUDA OOM detected; retrying the next approved topology."
    done
    echo "All allowed H200 topologies OOMed." >&2
    return 1
}

log_has_step_after_500() {
    "$PYTHON_BIN" - "$1" <<'PY'
import re, sys
text = open(sys.argv[1], errors="replace").read()
steps = [int(value) for value in re.findall(r"['\"]global_step['\"]\s*:\s*(\d+)", text)]
raise SystemExit(0 if steps and max(steps) > 500 else 1)
PY
}

run_formal() {
    if [ ! -f "$MARKER" ]; then
        echo "Formal submission requires an accepted smoke marker: $MARKER" >&2
        return 1
    fi
    local marker_profile instances world_size batch accum command name job_id status checkpoint latest
    marker_profile=$(sed -n 's/^profile=//p' "$MARKER")
    instances=$(sed -n 's/^instances=//p' "$MARKER")
    world_size=$(sed -n 's/^world_size=//p' "$MARKER")
    batch=$(sed -n 's/^batch=//p' "$MARKER")
    accum=$(sed -n 's/^accumulation=//p' "$MARKER")
    # Accept smoke markers produced by the original fixed 2x8 pickplace
    # workflow. New markers always carry the full topology contract.
    if [ "$PROFILE" = pickplace ] && [ -z "$marker_profile" ]; then
        marker_profile=pickplace
        instances=${instances:-2}
        world_size=${world_size:-16}
        accum=${accum:-$(sed -n 's/^accum=//p' "$MARKER")}
    fi
    if [ "$marker_profile" != "$PROFILE" ] || [ -z "$instances" ] || [ -z "$world_size" ] \
        || [ -z "$batch" ] || [ -z "$accum" ] \
        || [ "$world_size" -ne $((instances * GPUS_PER_NODE)) ] \
        || [ $((world_size * batch * accum)) -ne "$GLOBAL_BATCH" ]; then
        echo "Invalid or mismatched smoke marker: $MARKER" >&2
        return 1
    fi
    if [ "$(df -Pk "$FORMAL_ROOT" | awk 'NR==2 {print $4}')" -lt "$FREE_KIB_REQUIRED" ]; then
        echo "Formal output filesystem has less than 900 GiB free." >&2
        return 1
    fi
    wait_for_resources "$instances"
    command=$(remote_command formal "$instances" "$batch" "$accum")
    name="arx-${PROFILE}-h200-formal-10k-$(date +%m%d-%H%M%S)"
    job_id=$(submit_job "$name" "$command" "$instances" "$batch" "$accum")
    printf '%s\n' "$job_id" > "$STATE_DIR/formal_job_id"
    echo "[FORMAL] Submitted $job_id. Waiting for checkpoint-500 and a later optimization step."
    echo "[FORMAL] https://qz.sii.edu.cn/jobs/distributedTrainingDetail/$job_id?spaceId=$SELECTED_WORKSPACE"
    checkpoint="$FORMAL_DIR/checkpoint-500"
    latest="$STATE_DIR/${job_id}_latest.log"
    while true; do
        qzcli logs "$job_id" --tail 300 > "$latest" 2>&1 || true
        if "$PYTHON_BIN" "$REPO/scripts/validate_checkpoint.py" "$checkpoint" \
            --world-size "$world_size" --expected-step 500 > "$STATE_DIR/checkpoint_500_validation.log" 2>&1 \
            && log_has_step_after_500 "$latest"; then
            cat "$STATE_DIR/checkpoint_500_validation.log"
            echo "[FORMAL] checkpoint-500 accepted and training continued beyond step 500 for job $job_id."
            return 0
        fi
        status=$(job_status "$job_id" || true)
        echo "[FORMAL] job=$job_id status=$status checkpoint_500_or_continuation=pending"
        case "$status" in
            *failed*|*stopped*|*error*|*succeeded*)
                qzcli logs "$job_id" --tail 1000 > "$STATE_DIR/${job_id}_terminal.log" 2>&1 || true
                echo "Formal job became terminal before checkpoint-500 continuation was verified." >&2
                return 1
                ;;
        esac
        sleep "$POLL_SECONDS"
    done
}

case "$MODE" in
    smoke) run_smoke ;;
    formal) run_formal ;;
    auto) run_smoke && run_formal ;;
esac
