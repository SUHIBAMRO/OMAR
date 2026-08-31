# =====================================================================
#  EXPERIMENT — does standardizing the input channels fix B2 zero-shot?
#
#  WHAT THE B1 CONTROL ESTABLISHED. The same probe was run on both arms.
#  B1, same trainer and same protocol, reaches 0.066; B2 sits at ~1.0.
#
#      quantity                        B1              B2
#      descent captured                100%            47%   (37-60%)
#      relative L2                     0.030-0.058     0.45-0.95
#      correlation with uv_exact       +0.997..+0.9996 +0.87 down to -0.11
#      amplitude ratio                 0.977-1.019     0.23-0.61
#      prediction variability          0.332 / 0.157   0.134 / 0.100
#        against target variability    0.346 / 0.159   0.641 / 0.310
#      roughness*                      1.01x           3.00x  (1.67-4.80)
#      rms(f)/rms(E) in the input      6.1e-4 - 8.2e-4 2.3e-5 - 7.2e-5
#
#      * (U_pred/U_exact) / (amplitude ratio)^2. A prediction as smooth as
#        the truth has U scaling with amplitude squared, so this is 1.0.
#        B1 lands on 1.01 across all eight samples. B2 lands on 3.00: its
#        field carries three times the strain energy its size warrants --
#        it is ROUGH, not merely small. Rescaling would not fix it, and the
#        Pi(s * pred) scan confirms that: s = 1.0 on six of eight.
#
#  So B2's model descends halfway down a CORRECTLY SPECIFIED Pi (the
#  functional test proved the minimum is at uv_exact, W/U = 2.000) and
#  lands on a rough field that is nearly the same whatever it is shown.
#
#  THE ONE CANDIDATE THE CONTROL SINGLES OUT. Of everything measured, the
#  input channels are where B1 and B2 differ by an order of magnitude:
#  the two channels carrying the LOAD sit 2.3e-05 to 7.2e-05 below the one
#  carrying STIFFNESS on B2, against 6.1e-04 to 8.2e-04 on B1 -- ten to
#  thirty times quieter -- and they are nonzero on 3-5% of nodes. On top
#  of that the per-node load scale changes by 1.98x between B2's two
#  training meshes (measured with ONE fixed seed, total load constant to
#  0.001%), so the same physical loading is presented to the network as
#  two different numbers.
#
#  BE CLEAR ABOUT WHAT THIS IS. Neither the B1 nor the B2 runs reported so
#  far used any input normalization -- train_B1 has the hook and it is a
#  documented no-op by default. So this is NOT the difference between the
#  two arms. It is a candidate REMEDY for a condition that is measurably
#  much worse on B2. It may not work.
#
#  WHAT WOULD FALSIFY IT: if the normalized run also lands near 1.0, the
#  input scaling is not the obstacle and the next place to look is the
#  Dirichlet ramp (B2 has two ramps, x/R_out and y/R_out, against B1's
#  single y/Ly) or the parametric family itself.
#
#  THE ARMS, matched in every respect except the flag:
#      A  --normalize_inputs 0   the run already on Drive, 0.9986
#      B  --normalize_inputs 1   this experiment
#
#  ~15 min on a GPU. Neither arm touches the real output directory.
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
WORKDIR = f'{R}/b2_input_norm'
BASELINE = 0.9986          # arm A, already run; not repeated

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
D = f'{WORKDIR}/norm'
os.makedirs(D, exist_ok=True)
if not os.path.exists(f'{D}/samples_cache.pt'):
    print(f'\ncopying the sample cache into {D} (nothing writes to {SRC}) ...')
    shutil.copy2(cache, f'{D}/samples_cache.pt')

# Same protocol as the run that produced 0.9986, one flag different.
if not os.path.exists(f'{D}/metrics_history.json'):
    run([sys.executable, '-u', '-m', 'omar_pfem.resolution_invariance_zeroshot',
         'train', '--geometry', 'B2', '--material', 'neo_hookean',
         '--train_resolutions', '21,33',
         '--n_train_per_res', '400', '--n_val_per_res', '100',
         '--epochs', '2000', '--validate_every', '25', '--batch_size', '8',
         '--normalize_inputs', '1',
         '--out_dir', D])
else:
    print(f'\n{D}/metrics_history.json exists -- skipping the training run.')

# ---- verdict ---------------------------------------------------------
print('\n' + '=' * 78)
print('RESULT')
print('=' * 78)
h = json.load(open(f'{D}/metrics_history.json'))
best = min(h, key=lambda e: e['combined_val_error'])
print(f"  --normalize_inputs 0   {BASELINE:.4f}   (the run already on Drive)")
print(f"  --normalize_inputs 1   {best['combined_val_error']:.4f}   "
      f"at epoch {best['epoch']} ({best['opt_steps']:,} steps), "
      f"last {h[-1]['combined_val_error']:.4f}")
print(f"\n  the three B1 cases reach 0.0658 to 0.0827 on this metric.")
print()
if best['combined_val_error'] < 0.3:
    print('IT WORKS. The input scaling was the obstacle. The three B2 cases')
    print('should be retrained with --normalize_inputs 1 and re-evaluated, and')
    print('the checkpoints must carry their input_norm.json with them -- a')
    print('model trained on standardized inputs and evaluated on raw ones')
    print('gives plausible-looking garbage rather than an error. The eval')
    print('path loads it automatically from beside the checkpoint.')
elif best['combined_val_error'] < 0.9 * BASELINE:
    print('IT HELPS BUT DOES NOT RESOLVE IT. Better than 0.9986, and still')
    print('nowhere near the 0.07 B1 reaches. Something else is also wrong and')
    print('this arm does not identify it.')
else:
    print('IT IS NOT THE INPUT SCALING. That was the falsification stated')
    print('before the run, so this is a real answer and not a dead end: the')
    print('next candidates are the Dirichlet ramp (B2 has TWO ramps, x/R_out')
    print('and y/R_out, against B1\'s single y/Ly, and the two vanish on')
    print('different edges) and the parametric family itself.')
print('=' * 78)
print('\nNothing was written to', SRC)
print('Send this block over before any further B2 training.')
