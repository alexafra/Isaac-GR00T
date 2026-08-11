#!/usr/bin/env bash
set -euo pipefail

cd "$HOME/Development/Isaac-GR00T"

VALIDATION_DATASET="$HOME/Development/Datasets/lerobot2/combined_atomic_only_09_08_2/validation"

if (($#)); then
    MODEL_RUNS=("$@")
else
    MODEL_RUNS=(
        "$HOME/Development/Models/c_d1_batch_32_acc_1_10k_chunk_16_0908_2_atomic"
        "$HOME/Development/Models/c_only_batch_32_acc_1_10k_chunk_16_0908_2_10k_atomic"
    )
fi

for run_dir in "${MODEL_RUNS[@]}"; do
    evaluation_dir="$run_dir/evaluation"
    for csv_name in metrics_by_checkpoint.csv metrics_per_joint.csv; do
        if [[ ! -f "$evaluation_dir/$csv_name" ]]; then
            echo "Error: missing $evaluation_dir/$csv_name" >&2
            exit 1
        fi
    done

    echo "Regenerating summary graphs from CSVs: $evaluation_dir"
    NO_ALBUMENTATIONS_UPDATE=1 \
    uv run --no-sync python -m scripts.analysis_tools.evaluate_checkpoints \
        --run-dir "$run_dir" \
        --dataset-path "$VALIDATION_DATASET" \
        --output-dir "$evaluation_dir" \
        --plots-only \
        --skip-trajectory-plots
done

echo "Finished regenerating evaluation summary graphs."
