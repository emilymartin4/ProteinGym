#!/bin/bash

source ../zero_shot_config.sh

export input_scoring_files_folder=/network/scratch/n/noah.elrimawi-fine/ProteinGym/zero_shot_clinical_indels_scores/Progen2/base
export output_performance_file_folder=../../benchmarks/clinical_zero_shot
export clinical_reference_file_path=${clinical_reference_file_path_indels}

python ../../proteingym/performance_clinical_benchmarks.py \
    --input_scoring_files_folder ${input_scoring_files_folder} \
    --output_performance_file_folder ${output_performance_file_folder} \
    --clinical_reference_file_path ${clinical_reference_file_path} \
    --indel_mode
