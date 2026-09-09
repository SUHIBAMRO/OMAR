# =====================================================================
#  CELL -- measure the DD-NO's own real inference latency (Timon
#  round-8, point 7's break-even Comparison B currently ASSUMES the
#  data-driven operator's per-sample inference cost equals the
#  physics-informed operator's, since both use the identical
#  Transolver_Irregular_Mesh architecture -- true by construction of
#  the forward pass (inference time depends on the computational graph
#  traversed, not on which loss trained the weights), but never
#  actually measured for the DD-NO checkpoint specifically until now.
#
#  Standalone, no retraining, no risk to any existing result:
#  measure_inference_latency.py only loads an already-trained
#  checkpoint and times its forward pass -- same script, same protocol
#  (n_repeats/warmup) that produced the physics-informed operator's own
#  Table 7 number for this case (4.586 ms/sample, B1 x Neo-Hookean).
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
print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available()
      else 'NONE -- results would not be comparable to the GPU-measured Table 7 number')

R = '/content/drive/MyDrive/pfem_run'
CKPT = f'{R}/data_driven/B1_neo_hookean/model_best.pt'
DATA = f'{R}/results/datasets/B1_neo_hookean/hyperelastic_training_data_q4.npz'
OUT_JSON = f'{R}/data_driven/B1_neo_hookean/inference_latency_dd_no.json'

assert os.path.exists(CKPT), f'DD-NO checkpoint not found: {CKPT}'
assert os.path.exists(DATA), f'dataset not found: {DATA}'

run([
    sys.executable, '-u', '-m', 'omar_pfem.measure_inference_latency',
    '--geometry', 'B1', '--material', 'neo_hookean',
    '--checkpoint', CKPT,
    '--dataset', DATA,
    '--ntrain', '800', '--ntest', '200',
    '--out_json', OUT_JSON,
])

import json
result = json.load(open(OUT_JSON))
print('\n' + '=' * 60)
print('DD-NO inference latency:', result.get('inference_ms_per_sample', result))
print('Compare against the physics-informed operator\'s own Table 7 number')
print('for this same case (B1 x Neo-Hookean): 4.586 ms/sample.')
print('If close, this confirms Comparison B\'s "equal inference cost" assumption;')
print('if not, Table 21a/the break-even write-up needs the real DD-NO number instead.')
