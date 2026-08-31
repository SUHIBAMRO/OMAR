# =====================================================================
#  THE TEST THAT SPLITS THE LAST TWO CANDIDATES
#
#  WHERE THIS STANDS. Everything measurable has been measured and five
#  candidates are closed:
#
#      the load                     repaired and verified mesh-independent
#      loss_force_norm              added; necessary, not sufficient
#      batch size                   ruled out at matched steps (0.9888/0.9444)
#      the data and the functional  Pi's minimum is AT uv_exact, W/U = 2.000
#      input normalisation          falsified (0.9910 against 0.9986)
#      the Dirichlet ramp           EXONERATED by the run just done
#
#  The ramp was the leading structural candidate and it is now dead. The
#  raw field the mask demands to reproduce uv_exact is essentially the
#  same on both geometries:
#
#      quantity                      B1        B2      ratio
#      rms(raw)/rms(output)         1.72x     2.00x    1.16x
#      peak/rms of the raw          2.44      2.46     1.01x
#
#  Against a 15x gap in final error and a 3x gap in roughness, a 16%
#  difference in what the mask asks for explains nothing. B2's two ramps
#  are not the obstacle.
#
#  WHAT IS LEFT, AND WHY THIS IS THE RIGHT TEST. Two differences remain
#  between the failing runs and the B2 model that DOES work -- the 9.11%
#  result b2_accuracy_search.py got from train_B2.py on the same geometry,
#  the same architecture and the same energy:
#
#      (a) the data family. data_generate_B2 draws (E, nu, p) from a 2-D
#          Gaussian random field in (theta, r). ParametricFieldB2 uses two
#          Fourier harmonics in THETA ALONE -- every field is constant
#          along each radius.
#      (b) joint training. The 9.11% run trains at ONE resolution. The
#          zero-shot trainer trains at N=21 and N=33 together, which is
#          the whole point of the study.
#
#  Guessing between them is what the last five rounds did. This separates
#  them instead, with the cheapest experiment that can:
#
#      arm A   B2, train at N=21 ALONE
#      arm B   B2, train at N=33 ALONE
#
#  Same data, same everything else. Only the joint-training half removed.
#
#      BOTH REACH ~0.07  ->  joint training is the fault. The data family
#                            is fine and the trainer is fine one mesh at a
#                            time; something about combining two meshes in
#                            one model breaks it. That is a finding worth
#                            reporting on its own, since resolution
#                            invariance is the claim under test.
#
#      BOTH STAY ~1.0    ->  joint training is NOT the fault, and the
#                            parametric family is the last thing standing.
#                            The next step would be to regenerate the B2
#                            cache from data_generate_B2's GRF, which is
#                            hours of FEM and should only be spent once
#                            this arm has pointed at it.
#
#      ONE OF EACH       ->  a resolution-specific problem, and the mesh
#                            that fails is where to look.
#
#  Either way this run ends a guess rather than starting one.
#
#  ~13 min per arm on an A100, so under half an hour. Neither arm touches
#  the real output directory.
# =====================================================================
import json
import os
import shutil
import subprocess
import sys

REPO = '/content/OMAR'
BRANCH = 'claude/claude-code-question-d307wp'
R = '/content/drive/MyDrive/pfem_run'
SRC = f'{R}/zeroshot_B2_neo_hookean'
WORKDIR = f'{R}/b2_single_resolution'
JOINT = 0.9986        # the 21+33 run already on Drive
B1_REF = '0.0658 to 0.0827'

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
assert torch.cuda.is_available(), 'no GPU -- Runtime -> Change runtime type'
print('GPU:', torch.cuda.get_device_name(0))

cache = f'{SRC}/samples_cache.pt'
assert os.path.exists(cache), f'no sample cache at {cache}'

# The joint run trains on 400 per resolution, so a single-resolution arm on
# 400 sees HALF the samples and half the optimizer steps per epoch. Epochs
# are doubled so the two arms and the joint run see comparable optimizer
# work; the comparison is against the joint run's 0.9986, and starving an
# arm of steps would make a failure uninterpretable.
ARMS = [('21', 4000), ('33', 4000)]

print('\n' + '=' * 78)
print('THE ARMS')
print('=' * 78)
print(f'  joint      N=21 and N=33   2000 epochs   -> already run, {JOINT}')
for res, ep in ARMS:
    print(f'  single     N={res:<13} {ep} epochs   -> this run')
print('\n  the single arms get double the epochs because a single resolution')
print('  is half the training set, so optimizer steps stay comparable.')

for res, epochs in ARMS:
    d = f'{WORKDIR}/N{res}'
    os.makedirs(d, exist_ok=True)
    dst = f'{d}/samples_cache.pt'
    if not os.path.exists(dst):
        print(f'\ncopying the sample cache into {d} '
              f'(nothing writes to {SRC}) ...')
        shutil.copy2(cache, dst)
    if os.path.exists(f'{d}/metrics_history.json'):
        print(f'\n[N={res}] already run -- skipping')
        continue
    print('\n' + '=' * 78)
    print(f'[N={res} alone] {epochs} epochs')
    print('=' * 78)
    run([sys.executable, '-u', '-m',
         'omar_pfem.resolution_invariance_zeroshot',
         'train', '--geometry', 'B2', '--material', 'neo_hookean',
         '--train_resolutions', res,
         '--n_train_per_res', '400', '--n_val_per_res', '100',
         '--epochs', str(epochs), '--validate_every', '50',
         '--batch_size', '8', '--out_dir', d])

# ---- verdict ---------------------------------------------------------
print('\n' + '=' * 78)
print('RESULT')
print('=' * 78)
best = {}
for res, _ in ARMS:
    h = json.load(open(f'{WORKDIR}/N{res}/metrics_history.json'))
    b = min(h, key=lambda e: e['combined_val_error'])
    best[res] = b['combined_val_error']
    print(f"  N={res} alone   best {b['combined_val_error']:.4f} at epoch "
          f"{b['epoch']} ({b['opt_steps']:,} steps)   "
          f"last {h[-1]['combined_val_error']:.4f}")
print(f"  N=21 and 33     {JOINT:.4f}   (the joint run on Drive)")
print(f"\n  the three B1 cases reach {B1_REF} on this metric.")
print()

lo, hi = min(best.values()), max(best.values())
if hi < 0.3:
    print('JOINT TRAINING IS THE FAULT. Both single-resolution arms learn and')
    print('the joint run does not. The data family is fine, the trainer is')
    print('fine one mesh at a time, and combining two meshes in one B2 model')
    print('is what breaks it -- while B1 does exactly that and is fine.')
    print('That is a reportable finding in its own right, because resolution')
    print('invariance is the property under test. Next: why does joining the')
    print('two meshes hurt B2 and not B1? The per-node load scale differing')
    print('1.98x between B2 meshes against B1\'s 1.26x is the first thing to')
    print('look at, and it is now a targeted question rather than a guess.')
elif lo > 0.8:
    print('JOINT TRAINING IS NOT THE FAULT. Both arms fail alone, so the')
    print('problem is present at a single resolution and combining meshes')
    print('adds nothing to it. THE PARAMETRIC FAMILY IS THE LAST CANDIDATE')
    print('STANDING: ParametricFieldB2 varies with theta alone, where the')
    print('9.11% recipe draws its fields from a 2-D GRF in (theta, r).')
    print('Testing that means regenerating the B2 cache from')
    print('data_generate_B2 -- HOURS of FEM. Do not start it without')
    print('deciding it is worth the time: B2 zero-shot is one cell of one')
    print('table and the report is honest without it.')
else:
    print(f'SPLIT: {lo:.4f} against {hi:.4f}. One resolution trains and the')
    print('other does not, which is a resolution-specific problem and')
    print('narrower than either candidate above. The failing mesh is where')
    print('to look, and its sample cache is the first thing to inspect.')
print('=' * 78)
print('\nNothing was written to', SRC)
