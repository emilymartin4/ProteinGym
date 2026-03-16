#!/bin/bash

source ../zero_shot_config.sh

export input_scoring_files_folder="../../../.cache/ProteinGym/zero_shot_clinical_substitutions_scores/Progen2/base" # change to path of downloaded data
export output_performance_file_folder=../../benchmarks/clinical_zero_shot
export clinical_reference_file_path=${clinical_reference_file_path_subs}

python ../../proteingym/performance_clinical_benchmarks.py \
    --input_scoring_files_folder ${input_scoring_files_folder} \
    --output_performance_file_folder ${output_performance_file_folder} \
    --clinical_reference_file_path ${clinical_reference_file_path}
