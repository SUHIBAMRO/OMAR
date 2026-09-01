# =====================================================================
#  Per-member consistency of the MMS operator across its own family
#
#  PROJECT_STATUS.md, point 9: "the operator's spread over the family is
#  not recoverable -- mms_operator*.json stores only the family MEAN, no
#  stdev, so 'is the operator consistent across the family, or merely
#  consistent on average?' cannot be answered from what is on disk."
#
#  This does NOT retrain anything. It loads the three checkpoints the
#  N=9/17/33 operator runs already wrote, rebuilds the exact same test
#  family and normalization those runs used (same seeds), and runs the 16
#  test members through the model one more time each, keeping every
#  member's row instead of collapsing it to a mean.
#
#  COST: three forward-pass sweeps of 16 samples each, no training loop.
#  The N=9/17 runs' OWN dataset-build step measured under 2s; N=33 is
#  larger but still just mesh assembly, not FEM solves or optimization.
#  Expect well under a minute per mesh.
#
#  If a checkpoint is missing, that mesh is skipped and reported, not
#  treated as fatal -- the other two can still answer the question.
# =====================================================================
import json
import os
import subprocess
import sys

REPO = '/content/OMAR'
BRANCH = 'claude/claude-code-question-d307wp'
CKPT = {
    9: '/content/drive/MyDrive/pfem_run/mms/operator_rate/N9/model_best.pt',
    17: '/content/drive/MyDrive/pfem_run/mms/operator_N17/model_best.pt',
    33: '/content/drive/MyDrive/pfem_run/mms/operator_rate/N33/model_best.pt',
}
OUT = '/content/drive/MyDrive/pfem_run/mms/operator_per_member'

from google.colab import drive
drive.mount('/content/drive')

import torch


def run(cmd):
    print('$', ' '.join(str(c) for c in cmd), flush=True)
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                         stderr=subprocess.STDOUT, text=True, bufsize=1)
    for line in p.stdout:
        print(line, end='', flush=True)
    p.wait()
    if p.returncode != 0:
        raise subprocess.CalledProcessError(p.returncode, cmd)


if not os.path.isdir(REPO):
    run(['git', 'clone', '-b', BRANCH,
         'https://github.com/SUHIBAMRO/OMAR.git', REPO])
else:
    run(['git', '-C', REPO, 'fetch', 'origin', BRANCH])
    run(['git', '-C', REPO, 'reset', '--hard', f'origin/{BRANCH}'])
run(['git', '-C', REPO, 'log', '--oneline', '-1'])

WORK = f'{REPO}/Practical_Examples'
os.chdir(WORK)
if WORK not in sys.path:
    sys.path.insert(0, WORK)
os.makedirs(OUT, exist_ok=True)
print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none (CPU)')

results = {}
for N, ckpt in CKPT.items():
    print('\n' + '=' * 70)
    if not os.path.exists(ckpt):
        print(f'[N={N}] checkpoint not found at {ckpt} -- skipping')
        continue
    oj = f'{OUT}/mms_operator_per_member_N{N}.json'
    print(f'[N={N}] re-scoring {ckpt}')
    print('=' * 70)
    cmd = [sys.executable, '-u', '-m', 'omar_pfem.mms_operator_per_member',
           '--N', str(N), '--checkpoint', ckpt, '--out_json', oj]
    if not torch.cuda.is_available():
        cmd.append('--cpu')
    run(cmd)
    results[N] = json.load(open(oj))

print('\n' + '=' * 78)
print('OPERATOR CONSISTENCY ACROSS THE FAMILY, ALL MESHES THAT SCORED')
print('=' * 78)
print(f"{'N':>4}{'metric':<16}{'mean':>12}{'std/mean':>12}")
for N, r in results.items():
    for m, s in r['summary'].items():
        print(f"{N:>4}{m:<16}{s['mean']:>12.4e}{s['std_over_mean']:>12.4f}")

print('\nCompare std/mean above against Q4_spread_stdev_over_mean in')
print('point9_results/mms_family_fem_B1_neo_hookean.json -- same 16-member')
print('family, same three meshes, same quantity. Send Claude the printed')
print('table (or the JSONs under', OUT + ')')
print('=' * 78)
