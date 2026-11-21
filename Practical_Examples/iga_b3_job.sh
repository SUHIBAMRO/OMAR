#!/bin/bash
# IGA-type splines on the B3 groove (verification_agent/iga_b3.py): the next
# levels that do not fit the 15 GB of the development machine.
#   level 5, p = 2: 607,104 unknowns, about 25 GB, about 5-6 hours (AMG-CG)
#   level 3, p = 3: 154,791 unknowns, about 20 GB, about 10 hours
# CPU only (no GPU needed). The queue name is an assumption: use the cluster's
# CPU batch queue, e.g.
#   bsub -n 8 -R "rusage[mem=32G]" -q <cpu-queue> -W 24:00 \
#        -o iga_b3.%J.out -e iga_b3.%J.err < iga_b3_job.sh
# Results are appended to verification_agent/iga_b3_result.json (keys L5:p2, L3:p3).
set -e
source ~/miniconda3/etc/profile.d/conda.sh
conda activate omar
cd ~/OMAR/Practical_Examples
export VERIF_LINSOLVE=amg
python -u - <<'EOF'
from omar_pfem.verification_agent import iga_b3 as I
I.LEVELS[5] = (60, 30, 100)
cache = I.json.load(open(I.OUT)) if I.os.path.exists(I.OUT) else {}
I.run(5, 2, cache)
I.run(3, 3, cache)
EOF
