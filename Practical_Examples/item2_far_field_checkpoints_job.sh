#!/bin/bash
set -e
source ~/miniconda3/etc/profile.d/conda.sh
conda activate omar
cd ~/OMAR/Practical_Examples

python -u -m omar_pfem.evaluate_b3_checkpoints_far_region \
    --run_dir /scratch/gole3157/pfem_run --n_samples 10 \
    --out_json /scratch/gole3157/pfem_run/b3_rabczuk_reeval_v2.json \
    --results_out /scratch/gole3157/pfem_run/b3_far_region_checkpoints.json
