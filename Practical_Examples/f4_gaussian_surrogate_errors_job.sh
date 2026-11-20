#!/bin/bash
# F4: surrogate errors of the B3 groove stresses under the Gaussian-kernel
# region definition (verification agent gk2_* QoIs), every u-only checkpoint,
# against the CACHED 950,400-element reference of the re-evaluation run (no new
# FEM solve; one forward pass per checkpoint and sample).
# Submit (as for the earlier B3 evaluation jobs):
#   bsub -gpu "num=1:mode=exclusive_process:gmem=75G" -q BatchGPU \
#        -o f4_gaussian.%J.out -e f4_gaussian.%J.err < f4_gaussian_surrogate_errors_job.sh
set -e
source ~/miniconda3/etc/profile.d/conda.sh
conda activate omar
cd ~/OMAR/Practical_Examples

python -u -m omar_pfem.evaluate_b3_checkpoints_gaussian \
    --run_dir /scratch/gole3157/pfem_run --n_samples 10 \
    --out_json /scratch/gole3157/pfem_run/b3_rabczuk_reeval_v2.json \
    --results_out /scratch/gole3157/pfem_run/b3_gaussian_checkpoints.json
