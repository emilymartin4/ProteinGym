#!/bin/bash
#SBATCH --job-name=progen2-overnight
#SBATCH --partition=long
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:a100l:5
#SBATCH --cpus-per-task=20
#SBATCH --mem=96G
#SBATCH --time=5-00:00:00
#SBATCH --output=/home/mila/n/noah.elrimawi-fine/scratch/logs/progen2_overnight_%j.out
#SBATCH --error=/home/mila/n/noah.elrimawi-fine/scratch/logs/progen2_overnight_%j.err

set -euo pipefail

cd /home/mila/n/noah.elrimawi-fine/projects/ProteinGym

source /network/scratch/n/noah.elrimawi-fine/miniconda3/etc/profile.d/conda.sh
conda activate proteingym_env

mkdir -p /home/mila/n/noah.elrimawi-fine/scratch/logs

echo "Job ${SLURM_JOB_ID} started on $(hostname) at $(date)"
echo "Allocated CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"

run_task() {
    local gpu_ids=$1
    local script_path=$2
    local task_name=$3

    echo "Starting ${task_name} on GPU(s) ${gpu_ids}"
    CUDA_VISIBLE_DEVICES="${gpu_ids}" bash "${script_path}" &
}

run_task "0,1" "scripts/scoring_DMS_zero_shot/scoring_Progen2_substitutions.sh" "dms_substitutions"
run_task "2" "scripts/scoring_clinical_zero_shot/scoring_Progen2.sh" "clinical_substitutions"
run_task "3" "scripts/scoring_DMS_zero_shot/scoring_Progen2_indels.sh" "dms_indels"
run_task "4" "scripts/scoring_clinical_zero_shot/scoring_Progen2_indels.sh" "clinical_indels"

wait

echo "Job ${SLURM_JOB_ID} finished at $(date)"
