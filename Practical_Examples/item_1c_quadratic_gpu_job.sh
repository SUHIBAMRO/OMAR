#!/bin/bash
set -e
source ~/miniconda3/etc/profile.d/conda.sh
conda activate omar
cd ~/OMAR/Practical_Examples

mkdir -p /scratch/gole3157/pfem_run
python -u -c "
import json
import torch
from omar_pfem.data.solve_quadratic_hex_groove_stress import solve_and_measure_groove_region_quadratic

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'device={device}', flush=True)
resolutions = [(13,12,19), (17,16,25), (21,18,31), (25,20,36)]
results = []
for Ntheta, Nr, Nz in resolutions:
    r = solve_and_measure_groove_region_quadratic(Ntheta, Nr, Nz, device=device)
    print(f\"n_elem={r['n_elem']} n_region_gauss={r['n_region_gauss']} \"
          f\"avg|sigma_xx|={r['region_avg_sigma_xx']:.4f} max|sigma_xx|={r['region_max_sigma_xx']:.4f} \"
          f\"avg|sigma|_F={r['region_avg_sigma_frob']:.4f} time={r['elapsed_s']:.2f}s\", flush=True)
    results.append(r)

with open('/scratch/gole3157/pfem_run/item_1c_quadratic_gpu_sweep.json', 'w') as f:
    json.dump(results, f, indent=2)
print('wrote /scratch/gole3157/pfem_run/item_1c_quadratic_gpu_sweep.json')
"
