#!/usr/bin/env bash
# Guarded ARX H200 workflow:
#   scripts/qzcli_arx_h200.sh auto   --credentials /secure/qzcli.txt
#   scripts/qzcli_arx_h200.sh smoke  --credentials /secure/qzcli.txt
#   scripts/qzcli_arx_h200.sh formal --credentials /secure/qzcli.txt
set -euo pipefail

MODE=${1:-}
if [ -z "$MODE" ]; then
    echo "Usage: $0 {auto|smoke|formal} --credentials <file> [--poll-seconds N]" >&2
    exit 2
fi
shift

CREDENTIALS=""
POLL_SECONDS=30
while [ "$#" -gt 0 ]; do
    case "$1" in
        --credentials) CREDENTIALS=${2:?missing credentials file}; shift 2 ;;
        --poll-seconds) POLL_SECONDS=${2:?missing poll interval}; shift 2 ;;
        *) echo "Unknown argument: $1" >&2; exit 2 ;;
    esac
done
case "$MODE" in auto|smoke|formal) ;; *) echo "Unknown mode: $MODE" >&2; exit 2 ;; esac
if [ -z "$CREDENTIALS" ] || [ ! -f "$CREDENTIALS" ]; then
    echo "Credential file is required (line 1 username, line 2 password)." >&2
    exit 2
fi
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
PYTHON_BIN="$REPO/.venv/bin/python"
CONFIG="$REPO/configs/arx_lift2s_pickplace/train_h200.yaml"
FORMAL_ROOT="$REPO/outputs/arx_lift2s_pickplace_h200_formal"
FORMAL_RUN=arx_lift2s_pickplace_h200_formal
FORMAL_DIR="$FORMAL_ROOT/$FORMAL_RUN"
SMOKE_ROOT="$REPO/outputs/arx_lift2s_pickplace_h200_smoke"
MARKER="$REPO/outputs/.arx_h200_smoke_passed"
STATE_DIR="$REPO/outputs/qzcli_arx_h200"
mkdir -p "$STATE_DIR" "$FORMAL_ROOT" "$SMOKE_ROOT"
printf '%s\n' "$$" > "$STATE_DIR/controller.pid"
if [ ! -x "$PYTHON_BIN" ]; then
    echo "Missing project environment: $PYTHON_BIN (run scripts/create_venv.sh first)" >&2
    exit 1
fi

IFS= read -r QZ_USERNAME < "$CREDENTIALS"
QZ_PASSWORD=$(sed -n '2p' "$CREDENTIALS")
if [ -z "$QZ_USERNAME" ] || [ -z "$QZ_PASSWORD" ]; then
    echo "Credential file must contain a non-empty username on line 1 and password on line 2." >&2
    exit 2
fi
LOGIN_LOG=$(mktemp)
trap 'rm -f "$LOGIN_LOG"' EXIT
if ! printf '%s\n' "$QZ_PASSWORD" | qzcli login --username "$QZ_USERNAME" --password-stdin >"$LOGIN_LOG" 2>&1; then
    echo "qzcli login failed (credential contents were not printed)." >&2
    exit 1
fi
unset QZ_PASSWORD
echo "[QZCLI] Login succeeded; refreshing workspace/resource/spec cache."
# A full refresh fans out across every workspace and can trigger HTTP 429 when
# qzcli uses its default parallelism.  Resource discovery is startup-only, so
# prefer the slower deterministic serial path here.
RESOURCE_CACHE=${HOME}/.qzcli/resources.json
CACHE_MAX_AGE_SECONDS=3600
CACHE_AGE_SECONDS=$(( $(date +%s) - $(stat -c %Y "$RESOURCE_CACHE" 2>/dev/null || echo 0) ))
if [ -s "$RESOURCE_CACHE" ] \
    && [ "$CACHE_AGE_SECONDS" -ge 0 ] \
    && [ "$CACHE_AGE_SECONDS" -le "$CACHE_MAX_AGE_SECONDS" ] \
    && [ -n "$(/usr/bin/python3 "$REPO/scripts/qzcli_resource_select.py" groups)" ]; then
    # The login immediately above succeeded, and this cache was produced by a
    # full refresh in the current workflow. Reusing it avoids repeating a slow
    # all-workspace historical scan every time an idle-node wait is restarted.
    echo "[QZCLI] Reusing ${CACHE_AGE_SECONDS}s-old H200 resource cache."
else
    qzcli workspaces --update --full --parallel 1
fi

select_resources() {
    local best_ws="" best_group="" best_free=-1 ws group output free
    while IFS=$'\t' read -r ws group; do
        [ -n "$ws" ] || continue
        if output=$(qzcli avail --workspace "$ws" --group "$group" --nodes 2 --export 2>&1); then
            free=$(printf '%s\n' "$output" | sed -n 's/.*(\([0-9][0-9]*\) 空节点).*/\1/p' | tail -n 1)
            [ -n "$free" ] || continue
            if [ "$free" -gt "$best_free" ] || { [ "$free" -eq "$best_free" ] && [[ "$group" < "$best_group" ]]; }; then
                best_ws=$ws
                best_group=$group
                best_free=$free
            fi
        fi
    done < <("/usr/bin/python3" "$REPO/scripts/qzcli_resource_select.py" groups)
    if [ -z "$best_group" ]; then
        echo "No H200 compute group currently has two fully idle nodes." >&2
        return 1
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
    echo "[QZCLI] Selected H200 group $SELECTED_GROUP in $SELECTED_WORKSPACE ($SELECTED_FREE idle nodes), spec $SELECTED_SPEC."
}

validate_dry_run() {
    local text=$1
    printf '%s' "$text" | "$PYTHON_BIN" "$REPO/scripts/validate_qzcli_payload.py" \
        --compute-group "$SELECTED_GROUP" --spec "$SELECTED_SPEC" --repo "$REPO"
}

submit_job() {
    local name=$1 command=$2 dry actual
    local -a args=(
        --name "$name" --command "$command"
        --workspace "$SELECTED_WORKSPACE" --compute-group "$SELECTED_GROUP"
        --spec "$SELECTED_SPEC" --gpu-type NVIDIA_H200_SXM_141G
        --cpu "$SELECTED_CPU" --gpus 8 --memory "$SELECTED_MEMORY"
        --instances 2 --shm 1200 --framework pytorch
    )
    dry=$(qzcli create "${args[@]}" --dry-run) || return 1
    validate_dry_run "$dry" || return 1
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

wait_for_smoke() {
    local job_id=$1 status logs worker worker_logs
    while true; do
        status=$(job_status "$job_id" || true)
        echo "[SMOKE] job=$job_id status=$status"
        case "$status" in
            *succeeded*) break ;;
            *failed*|*stopped*|*error*)
                qzcli logs "$job_id" --tail 500 > "$STATE_DIR/${job_id}_failed.log" 2>&1 || true
                return 1
                ;;
        esac
        qzcli logs "$job_id" --tail 80 > "$STATE_DIR/${job_id}_latest.log" 2>&1 || true
        sleep "$POLL_SECONDS"
    done
    logs=""
    for worker in worker-0 worker-1; do
        worker_logs=$(qzcli logs "$job_id" --pod "${job_id}-${worker}" --tail 10000 2>&1)
        printf '%s\n' "$worker_logs" > "$STATE_DIR/${job_id}_${worker}_success.log"
        logs+=$'\n'
        logs+=$worker_logs
        if ! printf '%s\n' "$worker_logs" | grep -q '\[H200_PREFLIGHT\]'; then
            echo "Smoke logs do not prove H200 preflight for $worker." >&2
            return 1
        fi
    done
    printf '%s\n' "$logs" > "$STATE_DIR/${job_id}_success.log"
    if ! printf '%s\n' "$logs" | grep -q 'Distributed contract verified: world_size=16'; then
        echo "Smoke logs do not prove world_size=16/global batch 128." >&2
        return 1
    fi
    if ! printf '%s\n' "$logs" | grep -q "'grad_norm':" || ! printf '%s\n' "$logs" | grep -q "'train_runtime':"; then
        echo "Smoke logs do not contain both optimization metrics and the completed 20-step summary." >&2
        return 1
    fi
    if printf '%s\n' "$logs" | grep -Eiq 'out of memory|\bnan\b|nccl.*(timeout|error)|traceback'; then
        echo "Smoke logs contain an OOM/NaN/NCCL/Python failure signature." >&2
        return 1
    fi
}

remote_command() {
    local kind=$1 batch=$2 accum=$3 run_name output_root auto_resume max_steps save_strategy
    if [ "$kind" = smoke ]; then
        run_name="arx_lift2s_pickplace_h200_smoke_b${batch}"
        output_root=$SMOKE_ROOT
        auto_resume=0
        max_steps=20
        save_strategy=no
    else
        run_name=$FORMAL_RUN
        output_root=$FORMAL_ROOT
        auto_resume=1
        max_steps=10000
        save_strategy=steps
    fi
    printf "cd %q && source .venv/bin/activate && EXPECTED_NNODES=2 EXPECTED_GPUS_PER_NODE=8 python scripts/check_h200_worker.py && test \$(df -Pk %q | awk 'NR==2 {print \$4}') -ge 786432000 && export PYTHON_BIN=%q AUTO_RESUME=%q REQUIRE_WORLD_SIZE=16 REQUIRE_GLOBAL_BATCH=128 REQUIRE_ALL_TRAINABLE=1 && bash scripts/train.sh %q --run_name %q --output_dir %q --max_steps %q --per_device_train_batch_size %q --gradient_accumulation_steps %q --save_strategy %q" \
        "$REPO" "$output_root" "$PYTHON_BIN" "$auto_resume" "$CONFIG" "$run_name" "$output_root" "$max_steps" "$batch" "$accum" "$save_strategy"
}

run_smoke() {
    local batch accum command name job_id logs
    for pair in "8 1" "4 2" "2 4"; do
        read -r batch accum <<< "$pair"
        command=$(remote_command smoke "$batch" "$accum")
        name="arx-h200-smoke-b${batch}-$(date +%m%d-%H%M%S)"
        job_id=$(submit_job "$name" "$command")
        echo "[SMOKE] Submitted $job_id (micro-batch=$batch, accumulation=$accum)."
        if wait_for_smoke "$job_id"; then
            printf 'batch=%s\naccum=%s\njob_id=%s\n' "$batch" "$accum" "$job_id" > "$MARKER"
            echo "[SMOKE] Accepted: job=$job_id, micro-batch=$batch, accumulation=$accum."
            return 0
        fi
        logs="$STATE_DIR/${job_id}_failed.log"
        if [ ! -f "$logs" ] || ! grep -Eiq 'out of memory|cuda.*oom' "$logs"; then
            echo "Smoke failed for a reason other than OOM; refusing automatic fallback." >&2
            return 1
        fi
        echo "[SMOKE] OOM detected; retrying while preserving global batch 128."
    done
    echo "All allowed H200 batch combinations OOMed." >&2
    return 1
}

run_formal() {
    if [ ! -f "$MARKER" ]; then
        echo "Formal submission requires an accepted smoke marker: $MARKER" >&2
        return 1
    fi
    local batch accum command name job_id status checkpoint
    batch=$(sed -n 's/^batch=//p' "$MARKER")
    accum=$(sed -n 's/^accum=//p' "$MARKER")
    if [ -z "$batch" ] || [ -z "$accum" ] || [ $((16 * batch * accum)) -ne 128 ]; then
        echo "Invalid smoke marker; effective global batch is not 128." >&2
        return 1
    fi
    if [ "$(df -Pk "$FORMAL_ROOT" 2>/dev/null | awk 'NR==2 {print $4}')" -lt 786432000 ]; then
        echo "Formal output filesystem has less than 750 GiB free." >&2
        return 1
    fi
    command=$(remote_command formal "$batch" "$accum")
    name="arx-h200-formal-10k-$(date +%m%d-%H%M%S)"
    job_id=$(submit_job "$name" "$command")
    printf '%s\n' "$job_id" > "$STATE_DIR/formal_job_id"
    echo "[FORMAL] Submitted $job_id. Waiting for checkpoint-500."
    echo "[FORMAL] https://qz.sii.edu.cn/jobs/distributedTrainingDetail/$job_id?spaceId=$SELECTED_WORKSPACE"
    checkpoint="$FORMAL_DIR/checkpoint-500"
    while true; do
        if "$PYTHON_BIN" "$REPO/scripts/validate_checkpoint.py" "$checkpoint" --world-size 16 > "$STATE_DIR/checkpoint_500_validation.log" 2>&1; then
            cat "$STATE_DIR/checkpoint_500_validation.log"
            echo "[FORMAL] checkpoint-500 accepted for job $job_id."
            return 0
        fi
        status=$(job_status "$job_id" || true)
        echo "[FORMAL] job=$job_id status=$status checkpoint_500=pending"
        case "$status" in
            *failed*|*stopped*|*error*|*succeeded*)
                qzcli logs "$job_id" --tail 1000 > "$STATE_DIR/${job_id}_terminal.log" 2>&1 || true
                echo "Formal job became terminal before a valid checkpoint-500 was found." >&2
                return 1
                ;;
        esac
        qzcli logs "$job_id" --tail 100 > "$STATE_DIR/${job_id}_latest.log" 2>&1 || true
        sleep "$POLL_SECONDS"
    done
}

while ! select_resources; do
    echo "[QZCLI] Waiting ${POLL_SECONDS}s for two idle H200 nodes."
    sleep "$POLL_SECONDS"
done
case "$MODE" in
    smoke) run_smoke ;;
    formal) run_formal ;;
    auto) run_smoke && run_formal ;;
esac
