#!/bin/bash 

set -euo pipefail

source ../zero_shot_config.sh
source activate proteingym_env

NUM_DATASETS=$(($(wc -l < $DMS_reference_file_path_indels) - 1))

export Progen2_model_name_or_path="$HOME/ProteinGym/checkpoints/progen2-base"
export output_scores_folder="${DMS_output_score_folder_indels}/Progen2/base"

GPUS=(2)
NUM_GPUS=${#GPUS[@]}

run_worker() {
    local worker_id=$1
    local gpu_id=$2

    for ((i=worker_id; i<NUM_DATASETS; i+=NUM_GPUS)); do
        echo "[GPU ${gpu_id}] Running DMS index $i"

        CUDA_VISIBLE_DEVICES=${gpu_id} python ../../proteingym/baselines/progen2/compute_fitness.py \
            --Progen2_model_name_or_path "${Progen2_model_name_or_path}" \
            --DMS_reference_file_path "${DMS_reference_file_path_indels}" \
            --DMS_data_folder "${DMS_data_folder_indels}" \
            --DMS_index "$i" \
            --output_scores_folder "${output_scores_folder}" \
            --indel_mode
    done
}

for worker_id in "${!GPUS[@]}"; do
    gpu_id="${GPUS[$worker_id]}"
    run_worker "$worker_id" "$gpu_id" > "../../logs/progen2_base_base/DMS_zero_shot_indels/gpu_${gpu_id}.log" 2>&1 &
done

wait
echo "All jobs finished."