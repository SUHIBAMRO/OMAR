#!/bin/bash
set -e
source ~/miniconda3/etc/profile.d/conda.sh
conda activate omar
cd ~/OMAR/Practical_Examples

mkdir -p /scratch/gole3157/pfem_run
python -u -m omar_pfem.data.run_z_refine_gpu_sweep \
    --out_json /scratch/gole3157/pfem_run/item_1a_z_refine_gpu_sweep.json
