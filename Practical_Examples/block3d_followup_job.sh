#!/bin/bash
# OPTIONAL, prepared 2026-11-23 (round J), for authorized MANUAL submission only (no
# agent submits jobs). Runs the follow-up level k = 12 of the clamped 3D block that the
# agent chose but declined within its own budget (300 s, 8 GB; estimate about 980 s on
# this machine and at least 15 GB). Predictions were committed before this job exists:
# verification_agent/block3d_followup_preregistration.json.
#   cost: estimated 0.3-2 h; memory at least 15 GB (direct solver, 194k unknowns)
#   bsub -n 8 -R "rusage[mem=48G]" -q <cpu-queue> -W 4:00 \
#        -o block3d_followup.%J.out -e block3d_followup.%J.err < block3d_followup_job.sh
# Result: verification_agent/clamped_block_3d_followup.json (the pre-registered ladder
# block3d_ladder.json is not changed); then, on any machine:
#   python -m omar_pfem.verification_agent.block3d_followup --score
set -e
source ~/miniconda3/etc/profile.d/conda.sh
conda activate omar
cd ~/OMAR/Practical_Examples
python -u -m omar_pfem.verification_agent.followup clamped_block_3d strain_energy \
    --budget-s 14400 --budget-gb 46
