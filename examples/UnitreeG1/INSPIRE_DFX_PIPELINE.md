# G1 Inspire DFX policy pipeline

This is the supported path for collecting, filtering, converting, training, and
serving a Unitree G1 policy with Inspire DFX hands. Dex3 remains a separate,
fully supported 28-dimensional embodiment. Never combine Dex3 and Inspire
episodes in one dataset or checkpoint.

## Data contract

| Layer | Order | Size | Hand unit |
| --- | --- | ---: | --- |
| DFX DDS wire | right hand, left hand | 6 + 6 | normalized open fraction |
| Raw JSON / LeRobot / model | left arm, right arm, left hand, right hand | 7 + 7 + 6 + 6 = 26 | arms in radians; hands in normalized open fraction |

Each Inspire hand is ordered `pinky, ring, middle, index, thumb bend, thumb
rotation`. Hand values are bounded to `[0, 1]`, where `0` is fully closed and
`1` is fully open. Model arm actions are relative during training and decoded
back to absolute joint targets by the server. Hand actions remain absolute.

## 1. Collect

The robot, XR input, camera server, and physical safety controls must be ready.

```bash
cd /home/alex/Development/xr_teleoperate/teleop
/home/alex/miniconda3/envs/tv/bin/python -u teleop_hand_and_arm.py \
  --frequency 30 \
  --input-mode hand \
  --arm G1_29 \
  --ee inspire_dfx \
  --record \
  --headless \
  --task-dir utils/data/ \
  --task-name "stack_three_cups_inspire_dfx_YYYYMMDD" \
  --task-goal "stack the three red cups." \
  --motion \
  --img-server-ip 192.168.123.164 \
  --network-interface enp132s0
```

New episode JSON records measured frame timing, DFX subscriber gaps, and DFX
lost-counter/reset/malformed-state diagnostics.

## 2. Reject bad raw episodes before conversion

Run this only after teleoperation has stopped:

```bash
cd /home/alex/Development/xr_teleoperate
/home/alex/miniconda3/envs/tv/bin/python -m teleop.utils.episode_quality \
  teleop/utils/data/stack_three_cups_inspire_dfx_YYYYMMDD \
  --max-frame-gap-s 0.075 \
  --min-measured-fps 29 \
  --json-manifest /tmp/stack_three_cups_inspire_quality.json \
  --tsv-manifest /tmp/stack_three_cups_inspire_quality.tsv

jq -e '.counts.reject == 0 and .counts.unknown == 0' \
  /tmp/stack_three_cups_inspire_quality.json >/dev/null
```

The report never modifies an episode and refuses to overwrite a manifest.
`clean` requires complete timing and hand diagnostics with zero anomalies;
missing legacy diagnostics are `unknown`. If only a subset is clean, construct
a disposable conversion copy from the manifest's `clean` list.

## 3. Preflight and convert a disposable split copy

The conversion wrapper replaces the directory passed to it. Do not use the
only raw copy. A split parent contains `train`, `validation`, and `test`; for
training, the resulting dataset path is its `train` directory.

```bash
RAW_COPY=/home/alex/Development/Datasets/raw/inspire_stack_three_cups_split_copy
REPO_ID=inspire_stack_three_cups_30hz

bash /home/alex/Development/scripts/convert_to_lerobot2.sh \
  --preflight-only \
  --end-effector inspire-dfx \
  --include-surface-normals \
  "$RAW_COPY" "$REPO_ID" 6

bash /home/alex/Development/scripts/convert_to_lerobot2.sh \
  --end-effector inspire-dfx \
  --include-surface-normals \
  "$RAW_COPY" "$REPO_ID" 6
```

Every frame requires `raw_depth_0`; surface normals additionally require
aligned `depth_0`. The preflight verifies exact DFX provenance and native 26D
state/action values before replacement.

## 4. Generate statistics

```bash
cd /home/alex/Development/Isaac-GR00T
TRAIN_DATASET=/home/alex/Development/Datasets/raw/inspire_stack_three_cups_split_copy/train

uv run --no-sync python -m gr00t.data.stats \
  --dataset-path "$TRAIN_DATASET" \
  --embodiment-tag NEW_EMBODIMENT \
  --modality-config-path examples/UnitreeG1/g1_inspire_headonly_config.py
```

All seven visual recipes share the same state/action statistics.

## 5. Train

Choose one of these modality configurations:

| Recipe | Configuration |
| --- | --- |
| RGB | `g1_inspire_headonly_config.py` |
| RGB + gray depth, separate views | `g1_inspire_head_3_channel_gray_depth_config.py` |
| Gray depth only | `g1_inspire_head_depth_gray_only_config.py` |
| RGB + depth early fusion | `g1_inspire_head_4_channel_gray_depth_fusion_config.py` |
| RGB + normals, separate views | `g1_inspire_head_3_channel_surface_normals_config.py` |
| Normals only | `g1_inspire_head_surface_normals_only_config.py` |
| RGB + normals early fusion | `g1_inspire_head_6_channel_surface_normals_fusion_config.py` |

Example for six-channel RGB + surface-normal fusion:

```bash
cd /home/alex/Development/Isaac-GR00T
TRAIN_DATASET=/home/alex/Development/Datasets/raw/inspire_stack_three_cups_split_copy/train
BASE_MODEL=/home/alex/Development/Models/GR00T-N1.7-3B
OUTPUT_DIR=/home/alex/Development/Models/inspire_dfx_normals_6ch_25k

CUDA_VISIBLE_DEVICES=0 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
NO_ALBUMENTATIONS_UPDATE=1 \
uv run --no-sync python -m gr00t.experiment.launch_finetune \
  --base-model-path "$BASE_MODEL" \
  --dataset-path "$TRAIN_DATASET" \
  --embodiment-tag NEW_EMBODIMENT \
  --modality-config-path examples/UnitreeG1/g1_inspire_head_6_channel_surface_normals_fusion_config.py \
  --no-tune-llm \
  --no-tune-visual \
  --tune-vision-patch-embed \
  --vision-patch-embed-init rgb_mean \
  --tune-projector \
  --load-bf16 \
  --num-gpus 1 \
  --output-dir "$OUTPUT_DIR" \
  --global-batch-size 32 \
  --gradient-accumulation-steps 1 \
  --dataloader-num-workers 4 \
  --episode-sampling-rate 0.1 \
  --optim adafactor \
  --learning-rate 1e-4 \
  --warmup-ratio 0.05 \
  --max-steps 25000 \
  --save-steps 5000 \
  --save-total-limit 5 \
  --color-jitter-params brightness 0.20 contrast 0.15 saturation 0.10 hue 0.0
```

Use `--vision-patch-embed-init rgb_mean` only for the two early-fusion
recipes. Do not pass it to the five non-fusion recipes. Every episode must be
long enough to provide the configured 32-action horizon.

## 6. Serve a checkpoint

```bash
cd /home/alex/Development/Isaac-GR00T
CHECKPOINT=/home/alex/Development/Models/inspire_dfx_normals_6ch_25k/checkpoint-25000
TRAIN_DATASET=/home/alex/Development/Datasets/raw/inspire_stack_three_cups_split_copy/train

.venv/bin/python -m gr00t.eval.run_gr00t_server \
  --model-path "$CHECKPOINT" \
  --embodiment-tag NEW_EMBODIMENT \
  --device cuda \
  --host 127.0.0.1 \
  --port 5555 \
  --deployment-dataset-path "$TRAIN_DATASET"
```

Do not pass a modality configuration to the server; it loads the processor
saved with the checkpoint. The deployment dataset path must match the training
path recorded in the checkpoint.

## 7. Validate in publisher-free shadow mode

This uses real DDS state subscribers and the camera, but creates no robot
command publisher:

```bash
cd /home/alex/Development/unitree_lerobot
/home/alex/miniconda3/envs/unitree_lerobot/bin/python \
  -m unitree_lerobot.eval_robot.eval_groot_g1 \
  --end-effector inspire-dfx \
  --task stack-three-cups \
  --policy-host 127.0.0.1 \
  --policy-port 5555 \
  --image-host 192.168.123.164 \
  --network-interface enp132s0 \
  --initialization measured \
  --no-warmup1 \
  --no-warmup2 \
  --no-future-goal-warmup2 \
  --no-return-to-start \
  --no-gravity-feedforward \
  --inference-mode rtc \
  --execution-horizon 8 \
  --max-chunks 2 \
  --command-conditioning xr
```

The client validates robot type, 26D shapes, exact 7/7/6/6 partitions, joint
names, DFX provenance, video contract, and action semantics before accepting a
policy. RGB, separate-view, fusion, depth-only, and normals-only checkpoints
are supported.

## Live deployment status

The current committed client supports Inspire DFX shadow validation but still
fails closed for Inspire `--actuate`. Live support requires a separate guarded
DFX command backend. It will follow teleop's combined 100 Hz DFX position
writer while retaining the policy client's arm release, feedback freshness,
lost-counter, range, slew, HOLD, and heartbeat protections.

The DFX protocol has no explicit hand stop acknowledgement: shutdown stops
publishing and relies on the bridge's command-lease timeout, as existing teleop
does. Real actuation must remain gated by `--allow-unqualified-real`, physical
emergency-stop control, a restrained robot, and direct supervision.
