# =====================================================================
#  CELL — Does input normalization fix the OOD degradation?
#  (Timon round-6, point 1, second half: "If this diagnosis suggests a
#   relatively straightforward mitigation, such as changing the training
#   range or normalization, it would be useful to test it.")
#
#  Section 8.6 diagnosed the degradation as entering ENTIRELY through the
#  material channel, and named normalizing that channel as the cheapest
#  candidate fix: one training run, no new FEM data. This cell runs it.
#
#  Two stages, both resumable:
#    1. train B1 x Neo-Hookean again under the IDENTICAL protocol, with
#       --normalize_inputs 1 as the only difference           (~45-60 min, GPU)
#    2. re-run the same progressive OOD sweep on the new checkpoint and
#       compare curve against curve                            (~20-40 min)
#
#  Stage 2 is fast because the FEM references are cached: they do not depend
#  on the checkpoint, so the ~190 solves the baseline sweep already did are
#  reused instead of repeated. Point CACHE at the same directory every time.
#
#  GPU: yes for stage 1. Any GPU will do -- this is FP32 training, not the
#  FP64 solver sweep, so T4/L4 are fine here.
# =====================================================================
import os, subprocess, sys, json

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
BRANCH = 'claude/claude-code-question-d307wp'
if not os.path.isdir(REPO):
    run(['git', 'clone', '-b', BRANCH, 'https://github.com/SUHIBAMRO/OMAR.git', REPO])
else:
    run(['git', '-C', REPO, 'fetch', 'origin', BRANCH])
    run(['git', '-C', REPO, 'checkout', BRANCH])
    run(['git', '-C', REPO, 'reset', '--hard', f'origin/{BRANCH}'])

WORK = f'{REPO}/Practical_Examples'
os.chdir(WORK)
sys.path.insert(0, WORK)

import torch
print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NONE')

R = '/content/drive/MyDrive/pfem_run'
BASE_DIR = f'{R}/results/B1_neo_hookean'            # the existing, raw-input model
NORM_DIR = f'{R}/results/B1_neo_hookean_inputnorm'  # the new, normalized model
CACHE = f'{R}/ood_progressive/fem_cache'            # shared FEM references
OUT = f'{R}/ood_progressive'
os.makedirs(OUT, exist_ok=True)

# The dataset MUST be the same .npz the baseline B1 x Neo-Hookean model was
# trained on, or stage 2 compares two models trained on different data and
# attributes the difference to normalization. The first path below is the one
# the point-7b runs actually loaded from ("Loaded 800 train / 200 test from
# .../results/datasets/..."), so it is the known-good location; the rest are
# older layouts. If none exists the cell SEARCHES Drive rather than guessing
# a third time, and prints what it found so the choice is visible.
import glob
NAME = 'hyperelastic_training_data_q4.npz'
CANDIDATES = [f'{R}/results/datasets/B1_neo_hookean/{NAME}',
              f'{R}/datasets/B1_neo_hookean/{NAME}',
              f'{R}/datasets_archive/B1_neo_hookean/{NAME}']
DATASET = next((p for p in CANDIDATES if os.path.exists(p)), None)
if DATASET is None:
    found = sorted(glob.glob(f'{R}/**/B1_neo_hookean/{NAME}', recursive=True))
    assert found, (
        'training dataset not found. Tried:\n  ' + '\n  '.join(CANDIDATES) +
        f'\nand a recursive search of {R} for */B1_neo_hookean/{NAME}, which '
        'matched nothing.\nRun the main training notebook first, or set '
        'DATASET by hand to the .npz the baseline model was trained on.')
    print('none of the expected paths exist; found by search:')
    for p in found:
        print('   ', p, f'({os.path.getsize(p) / 1e6:.1f} MB)')
    DATASET = found[0]
print('dataset:', DATASET, f'({os.path.getsize(DATASET) / 1e6:.1f} MB)')

# The protocol MUST match the original run exactly, or the comparison measures
# the protocol instead of the normalization. These are Table 4's values, the
# same ones PFEM_Training_Colab.ipynb passes for every production case.
PROTOCOL = dict(ntrain='800', ntest='200', epochs='2000', batch_size='8',
                validate_every='25', early_stop_patience='8',
                early_stop_min_delta='1e-4')
print('protocol:', PROTOCOL)

# Pre-flight. Stage 2 needs the BASELINE checkpoint, and checking for it only
# after stage 1 would throw away an hour of GPU training before saying so.
# Everything stage 2 depends on that stage 1 does not produce is checked here,
# before any training starts.
_base = next((f'{BASE_DIR}/{n}' for n in ('model_best.pt', 'model_final.pt')
              if os.path.exists(f'{BASE_DIR}/{n}')), None)
assert _base is not None, (
    f'baseline checkpoint not found in {BASE_DIR}\n'
    'Stage 2 compares the normalized model against it, so there is no point '
    'spending an hour on stage 1 without it. This is the same checkpoint '
    'Round6_OOD_Progressive used; check that directory name first.')
assert not os.path.exists(f'{BASE_DIR}/input_norm.json'), (
    f'{BASE_DIR} contains an input_norm.json, so it is NOT the raw-input '
    'model the comparison needs as its baseline.')
print('baseline checkpoint:', _base,
      f'({os.path.getsize(_base) / 1e6:.1f} MB)')

# ---------------------------------------------------------------- stage 1
if os.path.exists(f'{NORM_DIR}/model_final.pt') or os.path.exists(f'{NORM_DIR}/EARLY_STOPPED'):
    print(f'\n[stage 1] {NORM_DIR} already trained -- skipping.')
else:
    print('\n[stage 1] training B1 x Neo-Hookean with --normalize_inputs 1\n')
    run([sys.executable, '-u', '-m', 'omar_pfem.train_B1',
         '--path', DATASET, '--material', 'neo_hookean',
         '--ntrain', PROTOCOL['ntrain'], '--ntest', PROTOCOL['ntest'],
         '--epochs', PROTOCOL['epochs'], '--batch_size', PROTOCOL['batch_size'],
         '--validate_every', PROTOCOL['validate_every'],
         '--save_every', PROTOCOL['validate_every'],
         '--early_stop_patience', PROTOCOL['early_stop_patience'],
         '--early_stop_min_delta', PROTOCOL['early_stop_min_delta'],
         '--print_every', '999999',
         '--normalize_inputs', '1',          # <-- the ONLY difference
         '--out_dir', NORM_DIR])

CKPT_NORM = f'{NORM_DIR}/model_best.pt'
if not os.path.exists(CKPT_NORM):
    CKPT_NORM = f'{NORM_DIR}/model_final.pt'
assert os.path.exists(CKPT_NORM), f'no checkpoint produced in {NORM_DIR}'
assert os.path.exists(f'{NORM_DIR}/input_norm.json'), (
    'input_norm.json was not written -- the run did not actually normalize')
print('\nnormalized checkpoint:', CKPT_NORM)
print('input_norm.json:', json.load(open(f'{NORM_DIR}/input_norm.json')))

# ---------------------------------------------------------------- stage 2
# The baseline sweep is re-run too, with the cache, so both curves come from
# the same code path on the same machine. It is cheap once the cache is warm.
CKPT_BASE = _base   # resolved and checked in the pre-flight above

for tag, ckpt in (('baseline', CKPT_BASE), ('inputnorm', CKPT_NORM)):
    out_json = f'{OUT}/ood_progressive_B1_neo_hookean_{tag}.json'
    print(f'\n[stage 2/{tag}] -> {out_json}')
    run([sys.executable, '-u', '-m', 'omar_pfem.ood_progressive',
         '--geometry', 'B1', '--material', 'neo_hookean',
         '--checkpoint', ckpt,
         '--N', '21',
         '--shifts', '0,0.5,1.0,1.5,2.0,2.5,3.0',
         '--factors', 'material,loading,both',
         '--n_samples', '10',
         '--cache_dir', CACHE,
         '--out_json', out_json])

# ---------------------------------------------------------------- compare
print('\n' + '=' * 78)
print('DID IT HELP?  degradation vs the in-distribution baseline, per curve')
print('=' * 78)

def load(tag):
    with open(f'{OUT}/ood_progressive_B1_neo_hookean_{tag}.json') as f:
        d = json.load(f)
    return d, {(r['factor'], r['shift_sigma']): r for r in d['rows']}

db, B = load('baseline')
dn, Nn = load('inputnorm')

base_id = next(r['mean_rel_L2'] for r in db['rows'] if r['factor'] == 'baseline')
norm_id = next(r['mean_rel_L2'] for r in dn['rows'] if r['factor'] == 'baseline')
print(f'\nIn-distribution error:  raw {base_id:.4f}   normalized {norm_id:.4f}   '
      f'({(norm_id / base_id - 1) * 100:+.1f}%)')
print('(if this got worse, normalization cost accuracy on the job it was '
      'trained for, and that trade has to be reported too)\n')

hdr = f"{'factor':<10}{'k':>5}{'raw err':>10}{'norm err':>10}{'raw x':>8}{'norm x':>8}{'change':>9}"
print(hdr); print('-' * len(hdr))
for factor in ('material', 'loading', 'both'):
    for k in (0.5, 1.0, 1.5, 2.0, 2.5, 3.0):
        rb, rn = B.get((factor, k)), Nn.get((factor, k))
        if not rb or not rn:
            continue
        xb = rb['mean_rel_L2'] / base_id
        xn = rn['mean_rel_L2'] / norm_id
        print(f"{factor:<10}{k:>5}{rb['mean_rel_L2']:>10.4f}{rn['mean_rel_L2']:>10.4f}"
              f"{xb:>7.2f}x{xn:>7.2f}x{(xn / xb - 1) * 100:>8.0f}%")

mb = B.get(('material', 3.0)); mn = Nn.get(('material', 3.0))
if mb and mn:
    xb, xn = mb['mean_rel_L2'] / base_id, mn['mean_rel_L2'] / norm_id
    print(f'\nHEADLINE — material shift at k=3: {xb:.2f}x raw vs {xn:.2f}x normalized.')
    if xn < xb * 0.8:
        print('  Normalization materially reduces the degradation. Worth reporting as a fix.')
    elif xn > xb * 1.2:
        print('  Normalization makes it WORSE. Report that; it is still an answer.')
    else:
        print('  Essentially unchanged. Standardizing is an affine rescaling, so a')
        print('  shifted E is still outside the range the network saw -- this is the')
        print('  outcome the mechanism in 8.6 predicts. Report it as a tested-and-')
        print('  did-not-work mitigation, which is what Timon asked for, and note')
        print('  that the untested remaining candidate is predicting a scaled')
        print('  quantity such as u*E rather than u.')

print(f'\nJSONs written to {OUT}/ -- commit both into '
      f'Practical_Examples/omar_pfem/point6_results/')
