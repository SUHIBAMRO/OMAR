#!/bin/bash
# OPTIONAL, prepared 2026-11-23, for authorized MANUAL submission only (no agent
# submits jobs). Re-solves the spline levels 1-4 of the B3 groove (verification_agent/
# iga_b3.py) so that the cylindrical Gaussian averages gk2trunc_cyl_rr/tt, added to
# iga_b3.qois() on 2026-11-23, exist for the cross-check of the two B3 quantities that
# still have no independent value (gk2_cyl_rr_avg, gk2_cyl_tt_avg). The Cartesian
# results of these levels are already stored and must be reproduced (checked below).
#   cost: about 5-8 hours (displacement levels 1-3 are already stored in
#   iga_b3_cyl_result.json and are skipped; P1 level 4 alone 2.9 h on 4 cores), 15 GB suffice
#   bsub -n 8 -R "rusage[mem=16G]" -q <cpu-queue> -W 12:00 \
#        -o iga_b3_cyl.%J.out -e iga_b3_cyl.%J.err < iga_b3_cyl_job.sh
# Results go to verification_agent/iga_b3_cyl_result.json (the stored file is untouched).
# BEFORE submitting (on the login node, once): bring the clone up to date with
#   cd ~/OMAR && git fetch origin claude/claude-code-question-d307wp && \
#   git checkout claude/claude-code-question-d307wp && git pull --ff-only
# Full instructions: advisor_feedback/2026-11-23_cluster_jobs.md
set -e
source ~/miniconda3/etc/profile.d/conda.sh
conda activate omar
cd ~/OMAR/Practical_Examples
export VERIF_LINSOLVE=amg
python -u - <<'EOF'
import json, os
from omar_pfem.verification_agent import iga_b3 as I
stored = json.load(open(I.OUT))
out = os.path.join(os.path.dirname(I.OUT), "iga_b3_cyl_result.json")
I.OUT = out                                   # write to a new file, keep the stored one
cache = json.load(open(out)) if os.path.exists(out) else {}
for form in ("fbar_p1", "disp"):
    for lv in (1, 2, 3, 4):
        row = I.run(lv, 2, cache, form)
        key = f"L{lv}:p2" + ("" if form == "disp" else ":fbar_p1")
        d = abs(row["gk2trunc_sigma_xx_avg"] / stored[key]["gk2trunc_sigma_xx_avg"] - 1)
        print(key, "reproduces stored gk2trunc_sigma_xx to", d, flush=True)
        assert d < 1e-6, "does not reproduce the stored solve"
EOF
