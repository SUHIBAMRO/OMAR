# =====================================================================
#  CELL -- torch-fem's own accuracy at the SAME low resolutions the NO
#  was actually zero-shot tested at (Timon round-10, item 1: "a fair
#  comparison between our NO in relevant QoIs and norms versus a
#  'suitable' GPU native (coarsest) FEM simulation which achieves a
#  comparable or better accuracy").
#
#  WHY THIS EXISTS: point7a_results/zeroshot_B1_neo_hookean.json already
#  gives the NO's own real, known accuracy (mean_rel_L2_vs_fine_reference)
#  at N = 13, 17, 25, 29, 37, 41, 49 -- ranging 5.2%-9.7%. What was
#  missing was the SAME metric (rel_L2 vs a fine reference) for a native
#  FEM solve at THOSE SAME resolutions, so the two numbers are directly
#  comparable (same norm, same convention). The existing
#  torchfem_convergence_vs_fine_reference_full.json only goes down to
#  N=51, where torch-fem is already at 4.9e-4 (0.049%) -- far more
#  accurate than the NO's best number, but that comparison is NOT at a
#  matched N, so it doesn't yet answer Timon's question directly. This
#  cell fills in exactly the resolutions the NO study used.
#
#  COST: these are all SMALL solves (fewer DOF than N=51, which itself
#  took 3.42s). The one expensive object -- the fine ~10M-DOF (N=2236)
#  reference -- is RESUMED from "ours" own already-converged checkpoint
#  on Drive (pfem_ckpt/fine_B1_neo_hookean_Q4_N2236.pt), not re-solved.
#  Expect a few minutes total, dominated by Colab/Drive/checkpoint
#  loading overhead, not by the actual small solves.
#
#  RESUMABLE: run_convergence_study checks its own out_json and skips
#  resolutions already present. Writing into the SAME out_json path
#  already used for N=51..1401 so the new low-N rows join that one
#  series (needed for a single, honest convergence-rate fit across all
#  points, coarse and fine together).
# =====================================================================
import json
import os
import subprocess
import sys


def run(cmd):
    print('$', ' '.join(str(c) for c in cmd), flush=True)
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                          stderr=subprocess.STDOUT, text=True, bufsize=1)
    for line in p.stdout:
        print(line, end='', flush=True)
    p.wait()
    if p.returncode != 0:
        raise subprocess.CalledProcessError(p.returncode, cmd)


from google.colab import drive
drive.mount('/content/drive')

REPO = '/content/OMAR'
if not os.path.isdir(REPO):
    run(['git', 'clone', '-b', 'claude/claude-code-question-d307wp',
         'https://github.com/SUHIBAMRO/OMAR.git', REPO])
else:
    run(['git', '-C', REPO, 'fetch', 'origin', 'claude/claude-code-question-d307wp'])
    run(['git', '-C', REPO, 'checkout', 'claude/claude-code-question-d307wp'])
    run(['git', '-C', REPO, 'reset', '--hard', 'origin/claude/claude-code-question-d307wp'])

run([sys.executable, '-m', 'pip', 'install', '-q', 'torch-fem'])

WORK = f'{REPO}/Practical_Examples'
os.chdir(WORK)
sys.path.insert(0, WORK)

for _mod_name in list(sys.modules):
    if _mod_name == 'omar_pfem' or _mod_name.startswith('omar_pfem.'):
        del sys.modules[_mod_name]

import torch
print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available()
      else 'NONE -- Runtime > Change runtime type > GPU required')

# Cheap CPU-scale sanity check before spending any real GPU time.
run([sys.executable, '-m', 'omar_pfem.torchfem_comparison', '11', '1e-8'])

R = '/content/drive/MyDrive/pfem_run'
OUT_JSON = f'{R}/torchfem_convergence_vs_fine_reference.json'
os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)

CHECKPOINT_DIR = '/content/drive/MyDrive/pfem_ckpt'

# Exactly the resolutions the NO's own zero-shot study
# (point7a_results/zeroshot_B1_neo_hookean.json) was tested at, plus a
# few coarser points below that range to see where torch-fem's own
# error actually crosses into the NO's 5-10% band.
RESOLUTIONS = [6, 9, 11, 13, 17, 21, 25, 29, 33, 37, 41, 45, 49]

from omar_pfem.torchfem_comparison import run_convergence_study
rows = run_convergence_study(RESOLUTIONS, OUT_JSON, checkpoint_dir=CHECKPOINT_DIR,
                              fine_N=2236, tol=1e-8)

print('\nDone. Results:', OUT_JSON)

# Side-by-side against the NO's own known zero-shot accuracy at the
# same resolutions.
NO_JSON = f'{REPO}/Practical_Examples/omar_pfem/point7a_results/zeroshot_B1_neo_hookean.json'
if os.path.exists(NO_JSON):
    with open(NO_JSON) as f:
        no_rows = {r['N']: r for r in json.load(f)['rows']}
    print('\nN      torchfem l2_rel   NO mean_rel_L2_vs_fine   torchfem wall_s')
    for row in rows:
        N = row['N']
        no = no_rows.get(N)
        no_str = f"{no['mean_rel_L2_vs_fine_reference']:.4f}" if no else "(NO not tested here)"
        print(f"{N:<6} {row['l2_rel']:<17.3e} {no_str:<24} {row['torchfem_wall_clock_s']:.2f}")
else:
    print(f"\nNO comparison file not found at {NO_JSON} -- printing torch-fem's own rows only:")
    for row in rows:
        print(row)
