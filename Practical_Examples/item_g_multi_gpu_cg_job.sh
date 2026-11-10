#!/bin/bash
set -e
source ~/miniconda3/etc/profile.d/conda.sh
conda activate omar
cd ~/OMAR/Practical_Examples

# Item G, real multi-GPU step: run the CORRECTNESS check first, at small
# scale, exactly as the CPU/gloo version was checked -- never trust a
# timing number from a distributed run whose own correctness hasn't been
# confirmed on THIS hardware/backend. Only once this prints PASS does the
# bigger timing run below mean anything.
echo "=== Step 1: correctness check (small, fast) ==="
python -u -m omar_pfem.distributed_domain_decomposition_cg --n_ranks 2 --n_side 40

echo ""
echo "=== Step 2: real timing comparison (bigger problem, real GPU work) ==="
python -u -m omar_pfem.distributed_domain_decomposition_cg --n_ranks 2 --n_side 160
