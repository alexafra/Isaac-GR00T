#!/usr/bin/env bash
set -euo pipefail

RUNNER="/home/alex/Development/Isaac-GR00T/scripts/run_inspire_e2_ftp_colour_only_20k_20260912.sh"
OUTPUT_DIR="/home/alex/Development/Models/inspire_e2_ftp_colour_only_rgb_redcup_pick_put_82eps_bf16_b32a1_h32_20k_20260912T150904Z"
TRAIN_LOG="/home/alex/Development/logs/groot/training/inspire_e2_ftp_colour_only_rgb_20k_20260912T150904Z.log"
MONITOR_LOG="/home/alex/Development/logs/groot/training/inspire_e2_ftp_colour_only_rgb_20k_20260912T150904Z.monitor.log"
CURRENT_SERVICE="groot-inspire-e2-ftp-colour-only-rgb-20k-20260912.service"
MAX_RESTARTS=2

mkdir -p "$(dirname "${MONITOR_LOG}")"

checkpoint_is_complete() {
    local checkpoint="$1"
    /home/alex/Development/Isaac-GR00T/.venv/bin/python - "${checkpoint}" <<'PY'
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
if any(not (checkpoint / name).is_file() for name in required):
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

latest_complete_checkpoint() {
    local name
    while IFS= read -r name; do
        if checkpoint_is_complete "${OUTPUT_DIR}/${name}"; then
            printf '%s\n' "${OUTPUT_DIR}/${name}"
            return 0
        fi
    done < <(
        find "${OUTPUT_DIR}" -mindepth 1 -maxdepth 1 -type d -name 'checkpoint-[0-9]*' \
            -printf '%f\n' 2>/dev/null | sort -t- -k2,2nr
    )
    return 1
}

snapshot() {
    local check_number="$1"
    {
        echo
        echo "===== check ${check_number}/6: $(date --iso-8601=seconds) ====="
        systemctl --user show "${CURRENT_SERVICE}" \
            -p ActiveState -p SubState -p Result -p ExecMainStatus -p MainPID \
            2>&1 || true
        nvidia-smi \
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

restart_count=0
for check_number in 1 2 3 4 5 6; do
    sleep 2h
    snapshot "${check_number}"

    if checkpoint_is_complete "${OUTPUT_DIR}/checkpoint-20000" 2>/dev/null \
        && grep -aq 'Training completed' "${TRAIN_LOG}"; then
        echo "Result: COMPLETED at check ${check_number}." >>"${MONITOR_LOG}"
        continue
    fi

    if systemctl --user is-active --quiet "${CURRENT_SERVICE}"; then
        echo "Result: HEALTHY/RUNNING at check ${check_number}." >>"${MONITOR_LOG}"
        continue
    fi

    fatal_tail="$(tail -c 2000000 "${TRAIN_LOG}" 2>/dev/null | tr '\r' '\n' || true)"
    if ! grep -Eqi 'CUDA out of memory|OutOfMemoryError|NCCL.*(error|timeout)|DataLoader worker.*(killed|exited)|Connection reset by peer' <<<"${fatal_tail}"; then
        echo "Result: STOPPED; no whitelisted easy-repair signature. No automatic restart." >>"${MONITOR_LOG}"
        continue
    fi
    if (( restart_count >= MAX_RESTARTS )); then
        echo "Result: STOPPED; automatic restart limit reached." >>"${MONITOR_LOG}"
        continue
    fi

    restart_count=$((restart_count + 1))
    retry_unit="groot-inspire-e2-ftp-colour-only-rgb-20k-20260912-retry${restart_count}"
    env_args=()
    if grep -Eqi 'CUDA out of memory|OutOfMemoryError' <<<"${fatal_tail}"; then
        env_args+=(GROOT_GLOBAL_BATCH_SIZE=16 GROOT_GRADIENT_ACCUMULATION_STEPS=2)
    fi
    if grep -Eqi 'DataLoader worker.*(killed|exited)' <<<"${fatal_tail}"; then
        env_args+=(GROOT_DATALOADER_NUM_WORKERS=1)
    fi

    if latest_checkpoint="$(latest_complete_checkpoint)"; then
        mode="resume"
        echo "Easy repair: resuming from ${latest_checkpoint} as ${retry_unit}." >>"${MONITOR_LOG}"
    else
        mode="fresh"
        archived_output="${OUTPUT_DIR}.failed_start_$(date -u +%Y%m%dT%H%M%SZ)"
        mv "${OUTPUT_DIR}" "${archived_output}"
        echo "Easy repair: no complete checkpoint; archived output to ${archived_output}." >>"${MONITOR_LOG}"
    fi

    systemd-run --user --unit="${retry_unit}" --collect --property=Type=exec \
        /usr/bin/systemd-inhibit --what=sleep:idle --mode=block --who=GR00T \
        --why=inspire-e2-ftp-colour-only-rgb-20k-retry \
        /usr/bin/env "${env_args[@]}" /usr/bin/bash "${RUNNER}" "${mode}" \
        >>"${MONITOR_LOG}" 2>&1
    CURRENT_SERVICE="${retry_unit}.service"
done

echo "Monitoring window finished: $(date --iso-8601=seconds)" >>"${MONITOR_LOG}"
