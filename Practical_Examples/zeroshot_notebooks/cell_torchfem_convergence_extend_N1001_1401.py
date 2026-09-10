# =====================================================================
#  CELL -- extends Round6_TorchFEM_Convergence_vs_Fine_Reference.ipynb's
#  own study to N=1001/1401, now that N=701's real float64 peak memory
#  is known (17.7GB, ~1.83x the old float32 number) and this session's
#  actual GPU is an 80GB A100 -- comfortably enough headroom projected
#  for both (~34GB at N=1001, ~67GB at N=1401), not the small-GPU risk
#  the first notebook was deliberately conservative about.
#
#  RESUMABLE, SAME out_json: run_convergence_study skips N=51-701 (
#  already done) and solves only the two new resolutions -- so this
#  is safe to run in the SAME Drive folder as the first notebook, not
#  a separate/duplicate result file.
#
#  RESULT SO FAR (N=51-701, real A100 run 2026-09-10): torch-fem's own
#  L2_rel/H1_semi_rel against the shared fine ~10M-DOF reference are
#  IDENTICAL (to every printed digit) to "ours" own already-published
#  numbers at every N -- both solvers converge to the SAME discretized
#  FE solution, as expected for two correct implementations of the
#  same element formulation. This is the strongest possible answer to
#  Timon's accuracy question and satisfies his "comparable accuracy
#  and mesh convergence" prerequisite before any timing claim.
# =====================================================================
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
print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NONE')

R = '/content/drive/MyDrive/pfem_run'
OUT_JSON = f'{R}/torchfem_convergence_vs_fine_reference.json'
CHECKPOINT_DIR = '/content/drive/MyDrive/pfem_ckpt'

# N=51-701 already in OUT_JSON from the first notebook -- these two are
# the only ones actually solved this run. N=1001 first, then 1401,
# printed as each finishes so a disconnect after 1001 still leaves a
# usable partial result.
RESOLUTIONS = [51, 101, 201, 401, 701, 1001, 1401]

from omar_pfem.torchfem_comparison import run_convergence_study
rows = run_convergence_study(RESOLUTIONS, OUT_JSON, checkpoint_dir=CHECKPOINT_DIR,
                              fine_N=2236, tol=1e-8)

print('\nDone. Results:', OUT_JSON)

import json
OURS_JSON = f'{REPO}/Practical_Examples/omar_pfem/highdof_stress_qoi_results/high_dof_stress_qoi_B1_neo_hookean_mgv_N701_1001_1401.json'
if os.path.exists(OURS_JSON):
    with open(OURS_JSON) as f:
        ours_rows = {r['N']: r for r in json.load(f)['orders']['Q4']['rows']}
    print('\nN      ours L2_rel      torchfem L2_rel   ours H1_rel      torchfem H1_rel   torchfem wall_s  torchfem peak_MB')
    for row in rows:
        N = row['N']
        o = ours_rows.get(N)
        if o:
            print(f"{N:<6} {o['l2_rel_error']:<14.3e} {row['l2_rel']:<17.3e} "
                  f"{o['h1_semi_rel_error']:<16.3e} {row['h1_semi_rel']:<17.3e} "
                  f"{row['torchfem_wall_clock_s']:<16.2f} {row['torchfem_peak_mem_mb']}")
        else:
            print(f"{N:<6} (no 'ours' row at this N)")
