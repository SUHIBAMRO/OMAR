#!/bin/bash
set -e
source ~/miniconda3/etc/profile.d/conda.sh
conda activate omar
cd ~/OMAR/Practical_Examples

mkdir -p /scratch/gole3157/pfem_run
python -u -m omar_pfem.data.compute_b3_scf \
    --Ntheta 41 --Nr 25 --Nz 50 --groove_z_beta 18.0 \
    --out_json /scratch/gole3157/pfem_run/item_1b_scf.json
