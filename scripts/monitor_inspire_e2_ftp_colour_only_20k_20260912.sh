#!/usr/bin/env bash
set -euo pipefail

RUNNER="${GROOT_MONITOR_RUNNER:-/home/alex/Development/Isaac-GR00T/scripts/run_inspire_e2_ftp_colour_only_20k_20260912.sh}"
OUTPUT_DIR="${GROOT_MONITOR_OUTPUT_DIR:-/home/alex/Development/Models/inspire/inspire_e2_ftp_colour_only_rgb_redcup_pick_put_82eps_bf16_b32a1_h32_20k_20260912T150904Z}"
TRAIN_LOG="${GROOT_MONITOR_TRAIN_LOG:-/home/alex/Development/logs/groot/training/inspire_e2_ftp_colour_only_rgb_20k_20260912T150904Z.log}"
MONITOR_LOG="${GROOT_MONITOR_LOG:-/home/alex/Development/logs/groot/training/inspire_e2_ftp_colour_only_rgb_20k_20260912T150904Z.monitor.log}"
CURRENT_SERVICE="${GROOT_MONITOR_CURRENT_SERVICE:-groot-inspire-e2-ftp-colour-only-rgb-20k-20260912.service}"
MAX_RESTARTS="${GROOT_MONITOR_MAX_RESTARTS:-2}"
CHECK_COUNT="${GROOT_MONITOR_CHECK_COUNT:-6}"
CHECK_INTERVAL_SECONDS="${GROOT_MONITOR_CHECK_INTERVAL_SECONDS:-7200}"
STALL_WINDOW_SECONDS="${GROOT_MONITOR_STALL_WINDOW_SECONDS:-1800}"
LOW_GPU_THRESHOLD_PERCENT="${GROOT_MONITOR_LOW_GPU_THRESHOLD_PERCENT:-5}"
LOW_GPU_SAMPLE_COUNT="${GROOT_MONITOR_LOW_GPU_SAMPLE_COUNT:-3}"
LOW_GPU_SAMPLE_INTERVAL_SECONDS="${GROOT_MONITOR_LOW_GPU_SAMPLE_INTERVAL_SECONDS:-2}"
SERVICE_STOP_WAIT_SECONDS="${GROOT_MONITOR_SERVICE_STOP_WAIT_SECONDS:-30}"
PYTHON_BIN="${GROOT_MONITOR_PYTHON_BIN:-/home/alex/Development/Isaac-GR00T/.venv/bin/python}"
NVIDIA_SMI_BIN="${GROOT_MONITOR_NVIDIA_SMI_BIN:-nvidia-smi}"
SYSTEMCTL_BIN="${GROOT_MONITOR_SYSTEMCTL_BIN:-systemctl}"
SYSTEMD_RUN_BIN="${GROOT_MONITOR_SYSTEMD_RUN_BIN:-systemd-run}"

monitor_log() {
    printf '%s\n' "$*" >>"${MONITOR_LOG}"
}

checkpoint_is_complete() {
    local checkpoint="$1"
    "${PYTHON_BIN}" - "${checkpoint}" <<'PY'
import json
import pathlib
import sys

checkpoint = pathlib.Path(sys.argv[1])
required = [
    "trainer_state.json",
    "model.safetensors.index.json",
    "optimizer.pt",
    "scheduler.pt",
    "rng_state.pth",
]
if any(
    not (checkpoint / name).is_file() or (checkpoint / name).stat().st_size == 0
    for name in required
):
    raise SystemExit(1)
step = int(checkpoint.name.rsplit("-", 1)[1])
state = json.loads((checkpoint / "trainer_state.json").read_text())
if int(state["global_step"]) != step:
    raise SystemExit(1)
index = json.loads((checkpoint / "model.safetensors.index.json").read_text())
for shard in set(index["weight_map"].values()):
    path = checkpoint / shard
    if not path.is_file() or path.stat().st_size == 0:
        raise SystemExit(1)
PY
}

checkpoint_step() {
    local checkpoint_name="${1##*/}"
    if [[ "${checkpoint_name}" =~ ^checkpoint-([0-9]+)$ ]]; then
        printf '%s\n' "${BASH_REMATCH[1]}"
        return 0
    fi
    return 1
}

latest_complete_checkpoint() {
    local name
    while IFS= read -r name; do
        if checkpoint_is_complete "${OUTPUT_DIR}/${name}" 2>/dev/null; then
            printf '%s\n' "${OUTPUT_DIR}/${name}"
            return 0
        fi
    done < <(
        find "${OUTPUT_DIR}" -mindepth 1 -maxdepth 1 -type d -name 'checkpoint-[0-9]*' \
            -printf '%f\n' 2>/dev/null | sort -t- -k2,2nr
    )
    return 1
}

quarantine_incomplete_newer_checkpoints() {
    local selected_checkpoint="$1"
    local selected_step checkpoint step quarantine_batch target

    selected_step="$(checkpoint_step "${selected_checkpoint}")"
    checkpoint_is_complete "${selected_checkpoint}"
    quarantine_batch=""

    while IFS= read -r checkpoint; do
        step="$(checkpoint_step "${checkpoint}")" || continue
        if (( 10#${step} <= 10#${selected_step} )); then
            continue
        fi
        if checkpoint_is_complete "${checkpoint}" 2>/dev/null; then
            monitor_log "Safety refusal: complete newer checkpoint appeared at ${checkpoint}; no retry launched."
            return 1
        fi
        if [[ -z "${quarantine_batch}" ]]; then
            quarantine_batch="${OUTPUT_DIR}.checkpoint_quarantine/$(date -u +%Y%m%dT%H%M%SZ)-$$"
            mkdir -p "${quarantine_batch}"
        fi
        target="${quarantine_batch}/${checkpoint##*/}"
        if [[ -e "${target}" ]]; then
            monitor_log "Safety refusal: quarantine target already exists at ${target}; no retry launched."
            return 1
        fi
        mv -- "${checkpoint}" "${target}"
        monitor_log "Resume safety: moved incomplete newer ${checkpoint} to recoverable quarantine ${target}."
    done < <(
        find "${OUTPUT_DIR}" -mindepth 1 -maxdepth 1 -type d -name 'checkpoint-[0-9]*' \
            -print 2>/dev/null | sort -t- -k2,2n
    )

    # The runner uses Trainer's bare auto-resume. Re-check that the selected
    # checkpoint is valid and no higher checkpoint directory remains visible.
    checkpoint_is_complete "${selected_checkpoint}"
    while IFS= read -r checkpoint; do
        step="$(checkpoint_step "${checkpoint}")" || continue
        if (( 10#${step} > 10#${selected_step} )); then
            monitor_log "Safety refusal: newer checkpoint still visible at ${checkpoint}; no retry launched."
            return 1
        fi
    done < <(
        find "${OUTPUT_DIR}" -mindepth 1 -maxdepth 1 -type d -name 'checkpoint-[0-9]*' \
            -print 2>/dev/null
    )
}

log_size_bytes() {
    if [[ -f "${TRAIN_LOG}" ]]; then
        stat -c '%s' "${TRAIN_LOG}"
    else
        printf '0\n'
    fi
}

current_attempt_log_offset() {
    local marker
    marker="$(
        LC_ALL=C grep -Eabo 'run_mode=(fresh|resume)' "${TRAIN_LOG}" 2>/dev/null \
            | tail -n 1 || true
    )"
    if [[ "${marker}" =~ ^([0-9]+): ]]; then
        printf '%s\n' "${BASH_REMATCH[1]}"
    else
        printf '0\n'
    fi
}

failure_tail_from_offset() {
    local start_offset="$1"
    local log_size

    [[ "${start_offset}" =~ ^[0-9]+$ ]] || return 1
    [[ -f "${TRAIN_LOG}" ]] || return 0
    log_size="$(log_size_bytes)"
    if (( start_offset > log_size )); then
        start_offset="$(current_attempt_log_offset)"
    fi
    tail -c "+$((start_offset + 1))" "${TRAIN_LOG}" 2>/dev/null \
        | tail -c 2000000 | tr '\r' '\n' || true
}

latest_training_activity_epoch() {
    local latest=0 candidate raw_timestamp

    if [[ -f "${TRAIN_LOG}" ]]; then
        candidate="$(stat -c '%Y' "${TRAIN_LOG}")"
        (( candidate > latest )) && latest="${candidate}"
    fi
    if [[ -d "${OUTPUT_DIR}" ]]; then
        while IFS= read -r raw_timestamp; do
            candidate="${raw_timestamp%%.*}"
            [[ "${candidate}" =~ ^[0-9]+$ ]] || continue
            (( candidate > latest )) && latest="${candidate}"
        done < <(
            find "${OUTPUT_DIR}" -mindepth 1 -maxdepth 2 \
                \( -type d -o -type f \) -printf '%T@\n' 2>/dev/null
        )
    fi
    printf '%s\n' "${latest}"
}

gpu_repeatedly_low() {
    local sample utilization
    LOW_GPU_SAMPLES=""

    (( LOW_GPU_SAMPLE_COUNT > 0 )) || return 1
    for ((sample = 1; sample <= LOW_GPU_SAMPLE_COUNT; sample++)); do
        utilization="$(
            "${NVIDIA_SMI_BIN}" --id=0 --query-gpu=utilization.gpu \
                --format=csv,noheader,nounits 2>/dev/null | head -n 1 | tr -d '[:space:]'
        )" || return 1
        [[ "${utilization}" =~ ^[0-9]+$ ]] || return 1
        LOW_GPU_SAMPLES+="${utilization}%"
        if (( sample < LOW_GPU_SAMPLE_COUNT )); then
            LOW_GPU_SAMPLES+=","
        fi
        (( utilization < LOW_GPU_THRESHOLD_PERCENT )) || return 1
        if (( sample < LOW_GPU_SAMPLE_COUNT )); then
            sleep "${LOW_GPU_SAMPLE_INTERVAL_SECONDS}"
        fi
    done
}

training_looks_stalled() {
    local now_epoch="${1:-$(date +%s)}"

    LAST_ACTIVITY_EPOCH="$(latest_training_activity_epoch)"
    LAST_INACTIVITY_SECONDS=0
    LOW_GPU_SAMPLES=""
    if (( LAST_ACTIVITY_EPOCH <= 0 || now_epoch < LAST_ACTIVITY_EPOCH )); then
        return 1
    fi
    LAST_INACTIVITY_SECONDS=$((now_epoch - LAST_ACTIVITY_EPOCH))
    (( LAST_INACTIVITY_SECONDS >= STALL_WINDOW_SECONDS )) || return 1
    gpu_repeatedly_low
}

service_is_active() {
    "${SYSTEMCTL_BIN}" --user is-active --quiet "${CURRENT_SERVICE}"
}

stop_confirmed_stall() {
    local waited=0

    if ! "${SYSTEMCTL_BIN}" --user stop "${CURRENT_SERVICE}"; then
        monitor_log "Stall recovery refused: systemd could not stop ${CURRENT_SERVICE}; no retry launched."
        return 1
    fi
    while service_is_active; do
        if (( waited >= SERVICE_STOP_WAIT_SECONDS )); then
            monitor_log "Stall recovery refused: ${CURRENT_SERVICE} remained active after ${SERVICE_STOP_WAIT_SECONDS}s; no retry launched."
            return 1
        fi
        sleep 1
        waited=$((waited + 1))
    done
}

archive_output_for_fresh_start() {
    local archived_output

    if [[ ! -e "${OUTPUT_DIR}" ]]; then
        printf '\n'
        return 0
    fi
    archived_output="${OUTPUT_DIR}.failed_start_$(date -u +%Y%m%dT%H%M%SZ)-$$"
    if [[ -e "${archived_output}" ]]; then
        monitor_log "Safety refusal: archive target already exists at ${archived_output}; no retry launched."
        return 1
    fi
    mv -- "${OUTPUT_DIR}" "${archived_output}"
    printf '%s\n' "${archived_output}"
}

snapshot() {
    local check_number="$1"
    {
        echo
        echo "===== check ${check_number}/${CHECK_COUNT}: $(date --iso-8601=seconds) ====="
        "${SYSTEMCTL_BIN}" --user show "${CURRENT_SERVICE}" \
            -p ActiveState -p SubState -p Result -p ExecMainStatus -p MainPID \
            2>&1 || true
        "${NVIDIA_SMI_BIN}" \
            --query-gpu=timestamp,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw \
            --format=csv,noheader,nounits 2>&1 || true
        df -h /home/alex/Development 2>&1 || true
        find "${OUTPUT_DIR}" -mindepth 1 -maxdepth 1 -type d -name 'checkpoint-*' \
            -printf '%f\n' 2>/dev/null | sort -V || true
        echo "Latest progress:"
        tail -c 2000000 "${TRAIN_LOG}" 2>/dev/null | tr '\r' '\n' \
            | grep -E '[0-9]+/20000|loss|Training completed|Traceback|CUDA out of memory|OutOfMemory|NCCL|worker.*(killed|exited)|No space left' \
            | tail -n 12 || true
    } >>"${MONITOR_LOG}" 2>&1
}

main() {
    local restart_count=0 check_number fatal_tail mode archived_output
    local retry_unit next_restart recovery_reason now_epoch attempt_log_offset latest_checkpoint
    local -a env_args

    mkdir -p "$(dirname "${MONITOR_LOG}")"
    attempt_log_offset="$(current_attempt_log_offset)"
    monitor_log "Failure scan baseline: byte ${attempt_log_offset} of ${TRAIN_LOG}; older attempts are excluded."

    for ((check_number = 1; check_number <= CHECK_COUNT; check_number++)); do
        sleep "${CHECK_INTERVAL_SECONDS}"
        snapshot "${check_number}"

        if checkpoint_is_complete "${OUTPUT_DIR}/checkpoint-20000" 2>/dev/null \
            && grep -aq 'Training completed' "${TRAIN_LOG}"; then
            monitor_log "Result: COMPLETED at check ${check_number}."
            continue
        fi

        recovery_reason=""
        if service_is_active; then
            now_epoch="$(date +%s)"
            if ! training_looks_stalled "${now_epoch}"; then
                monitor_log "Result: RUNNING at check ${check_number}; conservative stall criteria not met (last training-file activity ${LAST_INACTIVITY_SECONDS}s ago)."
                continue
            fi
            recovery_reason="confirmed_stall"
            monitor_log "Result: CONFIRMED STALL at check ${check_number}; no training-log/checkpoint activity for ${LAST_INACTIVITY_SECONDS}s and GPU samples were ${LOW_GPU_SAMPLES} (<${LOW_GPU_THRESHOLD_PERCENT}%)."
            if (( restart_count >= MAX_RESTARTS )); then
                monitor_log "Result: RUNNING/STALLED; automatic restart limit reached, so the active service was left untouched."
                continue
            fi
            if ! stop_confirmed_stall; then
                continue
            fi
        fi

        fatal_tail="$(failure_tail_from_offset "${attempt_log_offset}")"
        if [[ "${recovery_reason}" != "confirmed_stall" ]] \
            && ! grep -Eqi 'CUDA out of memory|OutOfMemoryError|NCCL.*(error|timeout)|DataLoader worker.*(killed|exited)|Connection reset by peer' <<<"${fatal_tail}"; then
            monitor_log "Result: STOPPED; no whitelisted easy-repair signature in the current failure interval (from byte ${attempt_log_offset}). No automatic restart."
            continue
        fi
        if (( restart_count >= MAX_RESTARTS )); then
            monitor_log "Result: STOPPED; automatic restart limit reached."
            continue
        fi

        next_restart=$((restart_count + 1))
        retry_unit="groot-inspire-e2-ftp-colour-only-rgb-20k-20260912-retry${next_restart}"
        env_args=()
        if grep -Eqi 'CUDA out of memory|OutOfMemoryError' <<<"${fatal_tail}"; then
            env_args+=(GROOT_GLOBAL_BATCH_SIZE=16 GROOT_GRADIENT_ACCUMULATION_STEPS=2)
        fi
        if grep -Eqi 'DataLoader worker.*(killed|exited)' <<<"${fatal_tail}"; then
            env_args+=(GROOT_DATALOADER_NUM_WORKERS=1)
        fi

        if latest_checkpoint="$(latest_complete_checkpoint)"; then
            mode="resume"
            if ! quarantine_incomplete_newer_checkpoints "${latest_checkpoint}"; then
                monitor_log "Result: STOPPED; checkpoint quarantine verification failed. No automatic restart."
                continue
            fi
            monitor_log "Easy repair: resuming from verified ${latest_checkpoint} as ${retry_unit}."
        else
            mode="fresh"
            if ! archived_output="$(archive_output_for_fresh_start)"; then
                continue
            fi
            if [[ -n "${archived_output}" ]]; then
                monitor_log "Easy repair: no complete checkpoint; archived output to ${archived_output}."
            else
                monitor_log "Easy repair: no complete checkpoint and no output directory; starting fresh."
            fi
        fi

        # Capture the exact byte boundary before this launch. Future failure scans
        # cannot reuse OOM/worker/NCCL text emitted by an earlier attempt.
        attempt_log_offset="$(log_size_bytes)"
        if "${SYSTEMD_RUN_BIN}" --user --unit="${retry_unit}" --collect --property=Type=exec \
            /usr/bin/systemd-inhibit --what=sleep:idle --mode=block --who=GR00T \
            --why=inspire-e2-ftp-colour-only-rgb-20k-retry \
            /usr/bin/env "${env_args[@]}" /usr/bin/bash "${RUNNER}" "${mode}" \
            >>"${MONITOR_LOG}" 2>&1; then
            restart_count="${next_restart}"
            CURRENT_SERVICE="${retry_unit}.service"
            monitor_log "Retry launched: ${CURRENT_SERVICE}; failure scan reset to byte ${attempt_log_offset}."
        else
            monitor_log "Retry launch failed for ${retry_unit}; no service switch recorded."
        fi
    done

    monitor_log "Monitoring window finished: $(date --iso-8601=seconds)"
}

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
    main "$@"
fi
