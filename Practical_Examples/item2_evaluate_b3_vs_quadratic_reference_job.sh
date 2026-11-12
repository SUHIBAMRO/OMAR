#!/bin/bash
set -e
source ~/miniconda3/etc/profile.d/conda.sh
conda activate omar
cd ~/OMAR/Practical_Examples

mkdir -p /scratch/gole3157/pfem_run
python -u -m omar_pfem.evaluate_b3_checkpoints_vs_quadratic \
    --run_dir /scratch/gole3157/pfem_run --n_samples 10 \
    --resolution 41 25 50 \
    --out_json /scratch/gole3157/pfem_run/b3_quadratic_reference.json \
    --results_out /scratch/gole3157/pfem_run/b3_quadratic_checkpoints.json
