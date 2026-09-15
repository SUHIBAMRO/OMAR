# =====================================================================
#  FOLLOW-UP to Test_TF32_Training.ipynb (2026-09-15): that test rejected
#  TF32 for training because the loss TRAJECTORY diverged from the fp32
#  baseline (mean relative per-step difference 121%). But that decision
#  rule was arguably too strict for stochastic optimization: TWO fp32
#  runs with DIFFERENT seeds also produce different intermediate loss
#  trajectories (that is normal for Adam -- different batch order/init
#  walks a different path through a non-convex landscape) while still
#  converging to similarly good final models. Trajectory MATCHING was
#  never actually the thing that matters for production; FINAL VALIDATION
#  ACCURACY is. The first test never checked that.
#
#  Omar's own question (2026-09-15): can this be fixed/salvaged, even for
#  a modest (~1 hour on an ~11-hour job, i.e. the 1.12x measured before)
#  speedup -- worth it if real, not worth it if it costs accuracy.
#
#  THE RIGHT CONTROLLED EXPERIMENT: run THREE trainings, not two --
#    A: fp32, seed=1234   (original baseline)
#    B: fp32, seed=5678   (SAME precision, DIFFERENT seed -- this measures
#       the training's OWN natural run-to-run variation, the noise floor
#       any two honest fp32 runs already have, with no TF32 involved)
#    C: TF32, seed=1234   (same seed as A -- isolates the TF32 effect)
#  Compare final validation error: if |A-C| is roughly the same size as
#  |A-B| (TF32's deviation is no bigger than two fp32 runs already differ
#  by), TF32 is not meaningfully harmful -- it is just one more source of
#  the same kind of noise stochastic training always has. If |A-C| is
#  much larger than |A-B|, TF32 is a real, extra source of harm beyond
#  normal variance and should stay rejected.
#
#  Runs longer than the first test (3000 steps, not 200) so the
#  validation-error comparison means something -- still cheap at N=21
#  (measured ~33ms/step -> ~3300 steps in under 2 minutes -- see below,
#  the 3 runs together take a few minutes on an A100). READ-ONLY against
#  the same finished job's cache; writes nothing back; separate tab.
# =====================================================================
import os
os.environ['JAX_PLATFORMS'] = 'cpu'

import sys
import time
import random

from google.colab import drive
drive.mount('/content/drive')

REPO = '/content/OMAR'
if not os.path.isdir(REPO):
    import subprocess
    subprocess.run(['git', 'clone', '-b', 'claude/claude-code-question-d307wp',
                     'https://github.com/SUHIBAMRO/OMAR.git', REPO], check=True)
else:
    import subprocess
    subprocess.run(['git', '-C', REPO, 'fetch', 'origin', 'claude/claude-code-question-d307wp'], check=True)
    subprocess.run(['git', '-C', REPO, 'checkout', 'claude/claude-code-question-d307wp'], check=True)
    subprocess.run(['git', '-C', REPO, 'reset', '--hard', 'origin/claude/claude-code-question-d307wp'], check=True)

WORK = f'{REPO}/Practical_Examples'
os.chdir(WORK)
sys.path.insert(0, WORK)

for _mod_name in list(sys.modules):
    if _mod_name == 'omar_pfem' or _mod_name.startswith('omar_pfem.'):
        del sys.modules[_mod_name]

import subprocess
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q',
                'einops', 'timm', 'h5py', 'jax', 'tqdm'], check=True)

import argparse
import numpy as np
import torch

assert torch.cuda.is_available(), 'this cell needs a real GPU'
print('GPU:', torch.cuda.get_device_name(0))

from omar_pfem.resolution_invariance_zeroshot import build_model, loss_and_pred, mesh_tensors_of, evaluate_resolution

CACHE_PATH = '/content/drive/MyDrive/pfem_run/zeroshot_B1_neo_hookean_multires/samples_cache.pt'
assert os.path.exists(CACHE_PATH), f'finished job cache not found: {CACHE_PATH}'
cache = torch.load(CACHE_PATH, weights_only=False, map_location='cpu')
N = 21  # smallest resolution in the cache -> cheapest real steps
train_samples = cache['train_samples'][N]
val_samples = cache['val_samples'][N]
print(f'loaded {len(train_samples)} train + {len(val_samples)} val real N={N} samples '
      f'from a FINISHED job (read-only, this run writes nothing back to it)')

device = torch.device('cuda')
dtype = torch.float32
N_STEPS = 3000
VAL_EVERY = 300
BATCH_SIZE = 8

args = argparse.Namespace(
    geometry='B1', material='neo_hookean',
    model='Transolver_Irregular_Mesh', n_hidden=256, n_layers=4, n_heads=8,
    mlp_ratio=2, dropout=0.1, unified_pos=0, ref=16, slice_num=128, fun_dim=4,
    use_soft_dirichlet=1, Ly=1.0, mode='plane_strain',
    loss_force_norm=0, grad_clip=0.0, lr=2e-3, weight_decay=0.0,
)

mesh_tensors = mesh_tensors_of(args.geometry, train_samples[0], device, dtype)


def run_training(use_tf32, seed, n_steps=N_STEPS, val_every=VAL_EVERY, batch_size=BATCH_SIZE):
    orig_precision = torch.get_float32_matmul_precision()
    torch.set_float32_matmul_precision('high' if use_tf32 else orig_precision)
    try:
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        model = build_model(args, device)
        opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        model.train()

        losses, val_history = [], []
        n = len(train_samples)
        order = np.random.permutation(n)
        pos = 0
        t0_total = time.time()
        for step in range(1, n_steps + 1):
            if pos + batch_size > n:
                order = np.random.permutation(n)
                pos = 0
            idx = order[pos:pos + batch_size]
            pos += batch_size

            samples = [train_samples[i] for i in idx]
            E_batch = torch.tensor(np.stack([s['E_node'] for s in samples]), device=device, dtype=dtype)
            nu_batch = torch.tensor(np.stack([s['nu_node'] for s in samples]), device=device, dtype=dtype)
            force_batch = torch.tensor(np.stack([s['node_forces'] for s in samples]), device=device, dtype=dtype)

            Pi, _, _, _, _ = loss_and_pred(args.geometry, mesh_tensors, model,
                                            E_batch, nu_batch, force_batch, args, dtype)
            loss = Pi.mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            losses.append(float(loss.item()))

            if step % val_every == 0 or step == n_steps:
                per_component, combined = evaluate_resolution(
                    args.geometry, val_samples, mesh_tensors, model, args, device, dtype, both=True)
                model.train()  # evaluate_resolution leaves the model in eval() mode
                val_history.append({'step': step, 'per_component': per_component, 'combined': combined})
        total_time = time.time() - t0_total
        return {'losses': losses, 'val_history': val_history, 'total_time': total_time}
    finally:
        torch.set_float32_matmul_precision(orig_precision)


print(f'\n=== Run A: fp32, seed=1234 (baseline) ===')
run_A = run_training(use_tf32=False, seed=1234)
print(f'  total={run_A["total_time"]:.1f}s  final val per_component='
      f'{run_A["val_history"][-1]["per_component"]:.4e}')

print(f'\n=== Run B: fp32, seed=5678 (SAME precision, DIFFERENT seed -- '
      f'measures natural run-to-run variation) ===')
run_B = run_training(use_tf32=False, seed=5678)
print(f'  total={run_B["total_time"]:.1f}s  final val per_component='
      f'{run_B["val_history"][-1]["per_component"]:.4e}')

print(f'\n=== Run C: TF32, seed=1234 (same seed as A -- isolates the TF32 effect) ===')
run_C = run_training(use_tf32=True, seed=1234)
print(f'  total={run_C["total_time"]:.1f}s  final val per_component='
      f'{run_C["val_history"][-1]["per_component"]:.4e}')

val_A = run_A['val_history'][-1]['per_component']
val_B = run_B['val_history'][-1]['per_component']
val_C = run_C['val_history'][-1]['per_component']
diff_AB = abs(val_A - val_B)   # natural fp32-vs-fp32 seed variation (noise floor)
diff_AC = abs(val_A - val_C)   # TF32's own deviation from the same-seed fp32 run
ratio = diff_AC / max(diff_AB, 1e-12)
speedup = run_A['total_time'] / run_C['total_time']

print('\n' + '=' * 70)
print('VALIDATION ERROR (per_component, the metric every reported number in '
      'this project uses) at each checkpoint:')
print('=' * 70)
print(f"{'step':<8}{'A (fp32,1234)':<18}{'B (fp32,5678)':<18}{'C (TF32,1234)':<18}")
for a, b, c in zip(run_A['val_history'], run_B['val_history'], run_C['val_history']):
    print(f"{a['step']:<8}{a['per_component']:<18.4e}{b['per_component']:<18.4e}{c['per_component']:<18.4e}")

print('\n' + '=' * 70)
print('SUMMARY')
print('=' * 70)
print(f'final val per_component: A(fp32,1234)={val_A:.4e}  B(fp32,5678)={val_B:.4e}  '
      f'C(TF32,1234)={val_C:.4e}')
print(f'natural fp32-vs-fp32 seed variation |A-B| = {diff_AB:.4e}  (the noise floor -- '
      f'this exists even with NO TF32 involved)')
print(f"TF32's own deviation |A-C| = {diff_AC:.4e}")
print(f'ratio |A-C| / |A-B| = {ratio:.2f}  (near 1 means TF32 is no worse than ordinary '
      f'seed noise; much bigger than 1 means TF32 is a real extra source of harm)')
print(f'speedup (wall-clock, {N_STEPS} steps, N={N}, batch_size={BATCH_SIZE}): {speedup:.2f}x')

safe = ratio <= 2.0
verdict = ('SAFE -- TF32 deviation is within (or near) normal run-to-run noise; the '
           f'{speedup:.2f}x speedup can be trusted' if safe else
           'DO NOT USE -- TF32 pushes final accuracy meaningfully further off than two '
           'ordinary fp32 runs already differ by; the earlier rejection stands')
print(f'\nVERDICT: {verdict}')
