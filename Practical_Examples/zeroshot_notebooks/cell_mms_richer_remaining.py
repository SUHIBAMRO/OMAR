# =====================================================================
#  CELL -- richer-family MMS study (Timon round-8 point 6), Arruda-
#  Boyce. Neo-Hookean's own richer-family run already exists
#  (point9_results/mms_richer_B1_neo_hookean.json); Mooney-Rivlin is
#  running separately on CPU at the same time (already partway through
#  when this was written), so this notebook covers Arruda-Boyce only --
#  no point duplicating the same work on both CPU and GPU at once.
#  This closes the gap Omar caught: the richer sine/cosine family and
#  the three-material extension had never been run together before.
#
#  Pure numerics, no neural network involved: this verifies the FEM
#  SOLVER itself against a manufactured (exact, closed-form) solution.
#  No GPU strictly required, but this runs on GPU when available since
#  the larger meshes (N=33, both element orders) benefit from it.
#
#  RESUMABLE: mms_study.py checks its own --out_json and skips rows
#  already present, so a disconnect loses at most the row in progress.
# =====================================================================
import os, subprocess, sys

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

WORK = f'{REPO}/Practical_Examples'
os.chdir(WORK)
sys.path.insert(0, WORK)

for _mod_name in list(sys.modules):
    if _mod_name == 'omar_pfem' or _mod_name.startswith('omar_pfem.'):
        del sys.modules[_mod_name]

import torch
print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NONE (CPU)')

R = '/content/drive/MyDrive/pfem_run'
OUT = f'{R}/mms_richer'
os.makedirs(OUT, exist_ok=True)

material = 'arruda_boyce'
out_json = f'{OUT}/mms_richer_B1_{material}.json'
run([
    sys.executable, '-u', '-m', 'omar_pfem.mms_study',
    '--material', material, '--richer_family',
    '--out_json', out_json,
])

print('\nDone. Fetch the JSON from', out_json)
