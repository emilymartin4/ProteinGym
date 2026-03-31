#!/bin/bash 

set -euo pipefail

source ../zero_shot_config.sh

NUM_DATASETS=$(($(wc -l < $DMS_reference_file_path_subs) - 1))

export Progen2_model_name_or_path="/home/mila/n/noah.elrimawi-fine/projects/progen/progen2/progen_swissprot_final"
export output_scores_folder="${DMS_output_score_folder_subs}/Progen2/curriculum"
run_timestamp=$(date +"%Y_%m_%d_%H_%M_%S")
export log_folder="../../logs/progen2_curriculum/DMS_zero_shot_substitutions_${run_timestamp}"

mkdir -p "${output_scores_folder}" "${log_folder}"

if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    IFS=',' read -r -a GPUS <<< "${CUDA_VISIBLE_DEVICES}"
elif command -v nvidia-smi >/dev/null 2>&1; then
    mapfile -t GPUS < <(nvidia-smi --query-gpu=index --format=csv,noheader)
else
    GPUS=(0)
fi

if (( ${#GPUS[@]} == 0 )); then
    echo "No GPUs available." >&2
    exit 1
fi

NUM_GPUS=${#GPUS[@]}

run_worker() {
    local worker_id=$1
    local gpu_id=$2

    for ((i=worker_id; i<NUM_DATASETS; i+=NUM_GPUS)); do
        echo "[GPU ${gpu_id}] Running DMS index $i"

        CUDA_VISIBLE_DEVICES=${gpu_id} python ../../proteingym/baselines/progen2/compute_fitness.py \
            --Progen2_model_name_or_path "${Progen2_model_name_or_path}" \
            --DMS_reference_file_path "${DMS_reference_file_path_subs}" \
            --DMS_data_folder "${DMS_data_folder_subs}" \
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
