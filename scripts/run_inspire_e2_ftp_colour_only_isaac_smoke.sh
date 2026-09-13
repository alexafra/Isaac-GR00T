#!/usr/bin/env bash
set -euo pipefail

GROOT_ROOT="/home/alex/Development/Isaac-GR00T"
SIM_ROOT="/home/alex/Development/unitree_sim_isaaclab"
CLIENT_ROOT="/home/alex/Development/unitree_lerobot"
DATASET_ROOT="/home/alex/Development/Datasets/lerobot2/inspire_e2_ftp_colour_only_test_20260912/train"
MODEL_ROOT="/home/alex/Development/Models/inspire_e2_ftp_colour_only_rgb_redcup_pick_put_82eps_bf16_b32a1_h32_20k_20260912T150904Z"
CHECKPOINT_STEP="${CHECKPOINT_STEP:-15000}"
if [[ "${CHECKPOINT_STEP}" != "15000" && "${CHECKPOINT_STEP}" != "20000" ]]; then
    echo "NOT READY: CHECKPOINT_STEP must be 15000 or 20000." >&2
    exit 2
fi
CHECKPOINT="${MODEL_ROOT}/checkpoint-${CHECKPOINT_STEP}"

usage() {
    cat <<'EOF'
Usage:
  bash scripts/run_inspire_e2_ftp_colour_only_isaac_smoke.sh check
  bash scripts/run_inspire_e2_ftp_colour_only_isaac_smoke.sh sim
  bash scripts/run_inspire_e2_ftp_colour_only_isaac_smoke.sh server
  bash scripts/run_inspire_e2_ftp_colour_only_isaac_smoke.sh client-once
  bash scripts/run_inspire_e2_ftp_colour_only_isaac_smoke.sh client-run

Run sim, server, and client in three separate terminals. Before either client
mode, disconnect/isolate every physical Unitree robot network from this host.
Checkpoint 15000 is the default because it had the lower validation MSE. Set
CHECKPOINT_STEP=20000 for an explicit comparison run.
EOF
}

require_file() {
    if [[ ! -f "$1" ]]; then
        echo "NOT READY: missing file: $1" >&2
        exit 1
    fi
}

require_executable() {
    if [[ ! -x "$1" ]]; then
        echo "NOT READY: missing executable: $1" >&2
        exit 1
    fi
}

check_ready() {
    require_executable "${GROOT_ROOT}/.venv/bin/python"
    require_executable "/home/alex/miniconda3/envs/unitree_lerobot/bin/python"
    require_file "${SIM_ROOT}/sim_main.py"
    require_file "${CLIENT_ROOT}/unitree_lerobot/eval_robot/eval_groot_g1.py"
    require_file "${DATASET_ROOT}/meta/info.json"
    require_file "${DATASET_ROOT}/meta/modality.json"
    require_file "${DATASET_ROOT}/meta/tasks.jsonl"
    require_file "${CHECKPOINT}/trainer_state.json"
    require_file "${CHECKPOINT}/model.safetensors.index.json"
    require_file "${CHECKPOINT}/optimizer.pt"
    require_file "${CHECKPOINT}/scheduler.pt"
    require_file "${CHECKPOINT}/rng_state.pth"

    "${GROOT_ROOT}/.venv/bin/python" - "${DATASET_ROOT}" "${CHECKPOINT}" "${CHECKPOINT_STEP}" <<'PY'
import json
from pathlib import Path
import sys

dataset = Path(sys.argv[1])
checkpoint = Path(sys.argv[2])
expected_step = int(sys.argv[3])
info = json.loads((dataset / "meta/info.json").read_text())
modality = json.loads((dataset / "meta/modality.json").read_text())
tasks = {
    json.loads(line)["task"]
    for line in (dataset / "meta/tasks.jsonl").read_text().splitlines()
    if line.strip()
}
trainer_state = json.loads((checkpoint / "trainer_state.json").read_text())
index = json.loads((checkpoint / "model.safetensors.index.json").read_text())

assert trainer_state["global_step"] == expected_step
assert info["robot_type"] == "Unitree_G1_Inspire_HeadOnly"
assert info["features"]["observation.state"]["shape"] == [26]
assert info["features"]["action"]["shape"] == [26]
assert info["end_effector"]["type"] == "inspire"
assert info["end_effector"]["protocol"] == "ftp"
assert info["end_effector"]["hand_dof"] == 6
assert tasks == {"pick up the red cup.", "put down the red cup."}
video_features = {
    key
    for key, value in info["features"].items()
    if isinstance(value, dict) and value.get("dtype") == "video"
}
assert video_features == {"observation.images.ego_view"}
assert set(modality["video"]) == {"ego_view"}

for shard in set(index["weight_map"].values()):
    path = checkpoint / shard
    assert path.is_file() and path.stat().st_size > 0, path

print(
    "READY: FTP-labelled 26D RGB-only dataset and "
    f"checkpoint-{expected_step} are complete."
)
PY
}

mode="${1:-}"
case "${mode}" in
    check)
        check_ready
        ;;
    sim)
        cd "${SIM_ROOT}"
        # Isaac Lab relies on environment variables installed by conda activation.
        source /home/alex/miniconda3/etc/profile.d/conda.sh
        conda activate unitree_sim_env
        exec python sim_main.py \
            --device cuda:0 \
            --enable_cameras \
            --task Isaac-PickPlace-Cylinder-G129-Inspire-Joint \
            --enable_inspire_dds \
            --robot_type g129
        ;;
    server)
        check_ready
        cd "${GROOT_ROOT}"
        export HF_HOME="/home/alex/.cache/huggingface"
        export HF_HUB_OFFLINE=1
        export TRANSFORMERS_OFFLINE=1
        export HF_DATASETS_OFFLINE=1
        export HF_HUB_DISABLE_TELEMETRY=1
        export GROOT_HF_LOCAL_FIRST=1
        export GROOT_PATCH_MISTRAL=1
        unset HF_TOKEN HUGGING_FACE_HUB_TOKEN
        export CUDA_VISIBLE_DEVICES=0
        export NO_ALBUMENTATIONS_UPDATE=1
        exec .venv/bin/python -m gr00t.eval.run_gr00t_server \
            --model-path "${CHECKPOINT}" \
            --embodiment-tag NEW_EMBODIMENT \
            --deployment-dataset-path "${DATASET_ROOT}" \
            --device cuda:0 \
            --host 127.0.0.1 \
            --port 5555
        ;;
    client-once|client-run)
        check_ready
        if [[ "${mode}" == "client-once" ]]; then
            horizon=1
            chunks=1
        else
            horizon=8
            chunks=20
        fi
        cat <<'EOF'
SAFETY: this client will publish on IsaacLab DDS domain 1 using physical robot
topic names. Disconnect/isolate every physical Unitree network before continuing.
The terminal still requires the exact SIMULATE confirmation. q releases during
every armed phase. During active policy motion, s enters powered HOLD. During
blocking startup, initialization, Warmup2, and Return-to-Start, s safely cancels
by using the same orderly-release path as q because powered HOLD cannot be
acknowledged while the child is synchronously executing those transitions.
EOF
        cd "${CLIENT_ROOT}"
        exec /home/alex/miniconda3/envs/unitree_lerobot/bin/python \
            -m unitree_lerobot.eval_robot.eval_groot_g1 \
            --end-effector inspire-ftp \
            --sim \
            --actuate \
            --task pick-red-cup \
            --policy-host 127.0.0.1 \
            --policy-port 5555 \
            --image-host 127.0.0.1 \
            --confirm-sim-network-isolated \
            --initialization xr-home \
            --no-warmup1 \
            --warmup2 \
            --no-future-goal-warmup2 \
            --no-return-to-start \
            --command-conditioning xr \
            --inference-mode synchronous \
            --execution-horizon "${horizon}" \
            --max-chunks "${chunks}"
        ;;
    *)
        usage >&2
        exit 2
        ;;
esac
