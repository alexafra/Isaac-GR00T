#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="/home/alex/Development/Isaac-GR00T"
DATASET_ROOT="/home/alex/Development/Datasets/lerobot2/inspire_e2_ftp_colour_only_test_20260912/train"
OUTPUT_DIR="/home/alex/Development/Models/inspire_e2_ftp_colour_only_rgb_redcup_pick_put_82eps_bf16_b32a1_h32_20k_20260912T150904Z"
LOG_DIR="/home/alex/Development/logs/groot/training"
LOG_FILE="${LOG_DIR}/inspire_e2_ftp_colour_only_rgb_20k_20260912T150904Z.log"
LOCK_FILE="/tmp/groot-inspire-e2-ftp-colour-only-20k-20260912.lock"
GLOBAL_BATCH_SIZE="${GROOT_GLOBAL_BATCH_SIZE:-32}"
GRADIENT_ACCUMULATION_STEPS="${GROOT_GRADIENT_ACCUMULATION_STEPS:-1}"
DATALOADER_NUM_WORKERS="${GROOT_DATALOADER_NUM_WORKERS:-4}"

mode="${1:-fresh}"
if [[ "${mode}" != "fresh" && "${mode}" != "resume" ]]; then
    echo "Usage: $0 [fresh|resume]" >&2
    exit 2
fi

exec 9>"${LOCK_FILE}"
if ! flock -n 9; then
    echo "Another Inspire E2/FTP colour-only training runner holds ${LOCK_FILE}." >&2
    exit 3
fi

test -f "${DATASET_ROOT}/meta/info.json"
test -f "${DATASET_ROOT}/meta/stats.json"
test -f "${DATASET_ROOT}/meta/relative_stats.json"
test -f "${REPO_ROOT}/examples/UnitreeG1/g1_inspire_headonly_config.py"
test -d "/home/alex/Development/Models/GR00T-N1.7-3B"
mkdir -p "${LOG_DIR}"

"${REPO_ROOT}/.venv/bin/python" - "${DATASET_ROOT}" <<'PY'
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
info = json.loads((root / "meta/info.json").read_text())
modality = json.loads((root / "meta/modality.json").read_text())
assert info["codebase_version"] == "v2.1"
assert info["robot_type"] == "Unitree_G1_Inspire_HeadOnly"
assert (info["total_episodes"], info["total_frames"], info["fps"]) == (82, 18794, 30)
assert info["features"]["observation.state"]["shape"] == [26]
assert info["features"]["action"]["shape"] == [26]
video_features = {
    key
    for key, value in info["features"].items()
    if isinstance(value, dict) and value.get("dtype") == "video"
}
assert video_features == {"observation.images.ego_view"}
end_effector = info["end_effector"]
assert (
    end_effector["type"],
    end_effector["protocol"],
    end_effector["hand_dof"],
) == ("inspire", "ftp", 6)
assert set(modality["video"]) == {"ego_view"}
PY

resume_args=()
if [[ "${mode}" == "fresh" ]]; then
    if [[ -e "${OUTPUT_DIR}" ]]; then
        echo "Refusing fresh launch because output already exists: ${OUTPUT_DIR}" >&2
        exit 4
    fi
else
    test -d "${OUTPUT_DIR}"
    compgen -G "${OUTPUT_DIR}/checkpoint-*" >/dev/null
    resume_args+=(--resume-from-checkpoint)
fi

export CUDA_VISIBLE_DEVICES=0
export PATH="/home/alex/Development/Isaac-GR00T/.venv/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NO_ALBUMENTATIONS_UPDATE=1
export HF_HOME="/home/alex/.cache/huggingface"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export HF_HUB_DISABLE_TELEMETRY=1
export GROOT_HF_LOCAL_FIRST=1
export GROOT_PATCH_MISTRAL=1
unset HF_TOKEN HUGGING_FACE_HUB_TOKEN

cd "${REPO_ROOT}"
{
    echo "run_mode=${mode}"
    echo "dataset=${DATASET_ROOT}"
    echo "output=${OUTPUT_DIR}"
    echo "visual_input=RGB colour only"
    echo "end_effector=Inspire RH56E2 / Unitree inspire_ftp"
    echo "global_batch_size=${GLOBAL_BATCH_SIZE}"
    echo "gradient_accumulation_steps=${GRADIENT_ACCUMULATION_STEPS}"
    echo "dataloader_num_workers=${DATALOADER_NUM_WORKERS}"
    date --iso-8601=seconds

    .venv/bin/python -m gr00t.experiment.launch_finetune \
        --base-model-path /home/alex/Development/Models/GR00T-N1.7-3B \
        --dataset-path "${DATASET_ROOT}" \
        --embodiment-tag NEW_EMBODIMENT \
        --modality-config-path examples/UnitreeG1/g1_inspire_headonly_config.py \
        --no-tune-llm \
        --no-tune-visual \
        --tune-vision-patch-embed \
        --tune-projector \
        --load-bf16 \
        --num-gpus 1 \
        --output-dir "${OUTPUT_DIR}" \
        --global-batch-size "${GLOBAL_BATCH_SIZE}" \
        --gradient-accumulation-steps "${GRADIENT_ACCUMULATION_STEPS}" \
        --dataloader-num-workers "${DATALOADER_NUM_WORKERS}" \
        --episode-sampling-rate 0.1 \
        --optim adafactor \
        --learning-rate 1e-4 \
        --warmup-ratio 0.05 \
        --max-steps 20000 \
        --save-steps 5000 \
        --save-total-limit 2 \
        --skip-final-model-save \
        --color-jitter-params brightness 0.20 contrast 0.15 saturation 0.10 hue 0.0 \
        "${resume_args[@]}"
} 2>&1 | tee -a "${LOG_FILE}"
