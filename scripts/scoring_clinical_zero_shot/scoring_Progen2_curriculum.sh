#!/bin/bash 

set -euo pipefail

source ../zero_shot_config.sh

NUM_DATASETS=$(($(wc -l < $clinical_reference_file_path_subs) - 1))

export run_name="descriptive-run-name" # for example, swissprot-100%-100k-2epochs
export Progen2_model_name_or_path="../../checkpoints/progen2-base"
experiment_category="curriculum"
export output_scores_folder="${clinical_output_score_folder_subs}Progen2/${experiment_category}-${run_name}"
run_timestamp=$(date +"%Y_%m_%d_%H_%M_%S")
export log_folder="../../logs/zero_shot_clinical_substitutions/Progen2/${experiment_category}-${run_name}-${run_timestamp}"

mkdir -p "${output_scores_folder}" "${log_folder}"
echo "Writing scores to ${output_scores_folder}"

if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    IFS=',' read -r -a GPUS <<< "${CUDA_VISIBLE_DEVICES}"
else
    # Select GPUs with low memory usage (likely idle)
    mapfile -t GPUS < <(
        nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits \
        | awk -F',' '$2 < 500 {print $1}'
    )
fi

if (( ${#GPUS[@]} == 0 )); then
    echo "No sufficiently idle GPUs found. Exiting."
    exit 1
fi

echo "Using GPUs: ${GPUS[*]}"

NUM_GPUS=${#GPUS[@]}

run_worker() {
    local worker_id=$1
    local gpu_id=$2

    for ((i=worker_id; i<NUM_DATASETS; i+=NUM_GPUS)); do
        echo "[GPU ${gpu_id}] Running clinical index $i"

        CUDA_VISIBLE_DEVICES=${gpu_id} python ../../proteingym/baselines/progen2/compute_fitness.py \
            --Progen2_model_name_or_path "${Progen2_model_name_or_path}" \
            --DMS_reference_file_path "${clinical_reference_file_path_subs}" \
            --DMS_data_folder "${clinical_data_folder_subs}" \
            --DMS_index "$i" \
            --output_scores_folder "${output_scores_folder}"
    done
}

for worker_id in "${!GPUS[@]}"; do
    gpu_id="${GPUS[$worker_id]}"
    run_worker "$worker_id" "$gpu_id" > "${log_folder}/gpu_${gpu_id}.log" 2>&1 &
done

wait
echo "Logs written to ${log_folder}"
echo "All jobs finished."
