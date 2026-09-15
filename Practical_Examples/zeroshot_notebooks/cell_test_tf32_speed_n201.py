# =====================================================================
#  FOLLOW-UP #2 to the TF32-training tests (2026-09-15). What we know so
#  far:
#    - Test_TF32_Training_Convergence.ipynb (N=21, 3000 steps): TF32 did
#      NOT destabilize training -- same-seed fp32 vs TF32 stayed close
#      and both smooth (final val per_component 0.686 vs 0.633). Safety
#      looks fine at this resolution.
#    - BUT the measured speedup was 1.00x -- essentially NONE, unlike the
#      1.12x seen in the very first (200-step) test, which was most
#      likely just warm-up noise, not a real effect.
#    - N=21 is the SMALLEST/cheapest resolution in the multi-res set --
#      its matmuls (attention over ~21x21 nodes) may simply be too small
#      to be tensor-core/GEMM-bound, so TF32 has nothing to speed up
#      there, safe or not.
#
#  THIS TEST asks the more relevant question directly: does TF32 give a
#  real per-step speedup at N=201, the LARGEST resolution in the
#  multi-res set (21,33,101,201) and the one that actually dominates a
#  training epoch's wall-clock cost, since its matmuls are much bigger?
#  If TF32 helps nowhere in this resolution range, there is no
#  meaningful time to save and the whole idea is correctly abandoned. If
#  it helps here, that is where an hour-scale saving on an ~11 hour job
#  would actually come from.
#
#  Pure timing focus (not full accuracy) -- accuracy at same precision
#  was already checked properly at N=21 and the underlying mechanism
#  (TF32 reduces matmul mantissa precision) does not depend on N, so a
#  short stability sanity check (no NaN/Inf, comparable loss magnitude)
#  is enough here, not a full 3-way accuracy study.
#
#  Still READ-ONLY against the same finished job's cache; still a
#  separate tab; writes nothing back.
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

from omar_pfem.resolution_invariance_zeroshot import build_model, loss_and_pred, mesh_tensors_of

CACHE_PATH = '/content/drive/MyDrive/pfem_run/zeroshot_B1_neo_hookean_multires/samples_cache.pt'
assert os.path.exists(CACHE_PATH), f'finished job cache not found: {CACHE_PATH}'
cache = torch.load(CACHE_PATH, weights_only=False, map_location='cpu')
N = 201  # LARGEST resolution in the multi-res set -> the one that actually
          # dominates a training epoch's wall-clock, and where GEMMs are
          # big enough for TF32 tensor cores to plausibly matter.
train_samples = cache['train_samples'][N]
print(f'loaded {len(train_samples)} real N={N} training samples from a FINISHED job '
      f'(read-only, this run writes nothing back to it)')

device = torch.device('cuda')
dtype = torch.float32
SEED = 1234
N_WARMUP = 10
N_TIMED = 100
BATCH_SIZE = 8

args = argparse.Namespace(
    geometry='B1', material='neo_hookean',
    model='Transolver_Irregular_Mesh', n_hidden=256, n_layers=4, n_heads=8,
    mlp_ratio=2, dropout=0.1, unified_pos=0, ref=16, slice_num=128, fun_dim=4,
    use_soft_dirichlet=1, Ly=1.0, mode='plane_strain',
    loss_force_norm=0, grad_clip=0.0, lr=2e-3, weight_decay=0.0,
)

mesh_tensors = mesh_tensors_of(args.geometry, train_samples[0], device, dtype)


def run_training(use_tf32, seed, n_warmup=N_WARMUP, n_timed=N_TIMED, batch_size=BATCH_SIZE):
    orig_precision = torch.get_float32_matmul_precision()
    torch.set_float32_matmul_precision('high' if use_tf32 else orig_precision)
    try:
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        model = build_model(args, device)
        opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        model.train()

        n = len(train_samples)
        order = np.random.permutation(n)
        pos = 0
        losses, step_times = [], []
        n_steps = n_warmup + n_timed
        for step in range(n_steps):
            if pos + batch_size > n:
                order = np.random.permutation(n)
                pos = 0
            idx = order[pos:pos + batch_size]
            pos += batch_size

            samples = [train_samples[i] for i in idx]
            E_batch = torch.tensor(np.stack([s['E_node'] for s in samples]), device=device, dtype=dtype)
            nu_batch = torch.tensor(np.stack([s['nu_node'] for s in samples]), device=device, dtype=dtype)
            force_batch = torch.tensor(np.stack([s['node_forces'] for s in samples]), device=device, dtype=dtype)

            torch.cuda.synchronize(device)
            t0 = time.time()
            Pi, _, _, _, _ = loss_and_pred(args.geometry, mesh_tensors, model,
                                            E_batch, nu_batch, force_batch, args, dtype)
            loss = Pi.mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            torch.cuda.synchronize(device)
            dt = time.time() - t0

            if step >= n_warmup:  # exclude warm-up (cuDNN/kernel-cache autotuning) from the timing
                step_times.append(dt)
            losses.append(float(loss.item()))
        return {'losses': losses, 'step_times': step_times}
    finally:
        torch.set_float32_matmul_precision(orig_precision)


print(f'\nRunning {N_WARMUP} warm-up + {N_TIMED} timed real training steps at N={N}, '
      f'baseline float32 precision...')
baseline = run_training(use_tf32=False, seed=SEED)
mean_fp32 = float(np.mean(baseline['step_times']))
print(f'  mean_step={1000*mean_fp32:.1f}ms  median_step={1000*np.median(baseline["step_times"]):.1f}ms  '
      f'final_loss={baseline["losses"][-1]:.6e}')

print(f'\nRunning {N_WARMUP} warm-up + {N_TIMED} timed real training steps at N={N}, '
      f"TF32 (torch.set_float32_matmul_precision('high'))...")
tf32 = run_training(use_tf32=True, seed=SEED)
mean_tf32 = float(np.mean(tf32['step_times']))
print(f'  mean_step={1000*mean_tf32:.1f}ms  median_step={1000*np.median(tf32["step_times"]):.1f}ms  '
      f'final_loss={tf32["losses"][-1]:.6e}')

speedup = mean_fp32 / mean_tf32
any_nonfinite = not (np.all(np.isfinite(baseline['losses'])) and np.all(np.isfinite(tf32['losses'])))
same_order = 0.1 < (tf32['losses'][-1] / max(abs(baseline['losses'][-1]), 1e-12)) < 10

print('\n' + '=' * 70)
print('SUMMARY')
print('=' * 70)
print(f'N={N} (largest resolution in the multi-res set) per-step wall-clock:')
print(f'  fp32: {1000*mean_fp32:.1f}ms/step   TF32: {1000*mean_tf32:.1f}ms/step   '
      f'speedup={speedup:.2f}x')
print(f'any NaN/Inf: {any_nonfinite}   final losses same order of magnitude: {same_order} '
      f'(fp32={baseline["losses"][-1]:.4e}, tf32={tf32["losses"][-1]:.4e})')

if any_nonfinite:
    verdict = 'DO NOT USE -- non-finite loss encountered'
elif speedup < 1.05:
    verdict = (f'NO MEANINGFUL SPEEDUP at N={N} either ({speedup:.2f}x) -- TF32 does not '
               f'help training anywhere in this resolution range; correctly abandon the idea')
else:
    verdict = (f'REAL SPEEDUP at N={N} ({speedup:.2f}x) and no stability red flag here -- '
               f'worth estimating the actual time this would save across a full multi-res '
               f'epoch (N=201 batches dominate epoch cost) before deciding to adopt it')
print(f'\nVERDICT: {verdict}')
