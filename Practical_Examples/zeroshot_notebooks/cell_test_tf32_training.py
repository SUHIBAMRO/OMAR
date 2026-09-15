# =====================================================================
#  QUICK TEST -- does TF32 matmul precision
#  (torch.set_float32_matmul_precision('high')) help TRAINING, not just
#  inference?
#
#  Already confirmed for INFERENCE only (no_inference_torch_compile.py,
#  profile_no_inference_n1401.py): TF32 alone gave a real 4.67x speedup
#  at N=1401, with output relative difference checked against the
#  strict-fp32 eager baseline before being trusted. Training was never
#  tested -- a single forward pass is a much weaker correctness check
#  than ~hundreds of Adam steps, where small per-step numerical
#  differences could in principle compound into a different trajectory.
#
#  DESIGN, so the comparison is fair and reproducible:
#    - loads an ALREADY-COMPLETE, finished sample cache
#      (zeroshot_B1_neo_hookean_multires/samples_cache.pt) READ-ONLY --
#      this job is fully done (see PROJECT_STATUS.md), so reading it
#      cannot interfere with any notebook still running.
#    - uses only the N=21 samples from that cache (smallest resolution
#      present, so each step is cheap and the sweep finishes in minutes).
#    - runs the SAME real training step sequence (loss_and_pred -> Adam)
#      used by resolution_invariance_zeroshot.py's own cmd_train, TWICE:
#      once at baseline float32, once with TF32 enabled.
#    - resets torch/numpy/random seeds to the SAME value immediately
#      before EACH run (model init, dropout masks, and batch shuffling
#      all draw from these), so the only thing that differs between the
#      two runs is the matmul precision setting itself -- not the model
#      weights, not the batch order, not which units get dropped out.
#    - writes NOTHING to the real out_dir; pure diagnostic, separate tab.
#
#  DECISION RULE: TF32 is "SAFE to use for training" only if (a) it
#  never produces NaN/Inf, and (b) the two loss trajectories track each
#  other closely (mean relative step-to-step difference below 5%) over
#  the whole run, not just at the first few steps.
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
N = 21  # smallest resolution in the cache -> cheapest real steps
train_samples = cache['train_samples'][N]
print(f'loaded {len(train_samples)} real N={N} training samples from a FINISHED job '
      f'(read-only, this run writes nothing back to it)')

device = torch.device('cuda')
dtype = torch.float32
SEED = 1234
N_STEPS = 200
BATCH_SIZE = 8

args = argparse.Namespace(
    geometry='B1', material='neo_hookean',
    model='Transolver_Irregular_Mesh', n_hidden=256, n_layers=4, n_heads=8,
    mlp_ratio=2, dropout=0.1, unified_pos=0, ref=16, slice_num=128, fun_dim=4,
    use_soft_dirichlet=1, Ly=1.0, mode='plane_strain',
    loss_force_norm=0, grad_clip=0.0, lr=2e-3, weight_decay=0.0,
)

mesh_tensors = mesh_tensors_of(args.geometry, train_samples[0], device, dtype)


def run_training(use_tf32, n_steps=N_STEPS, batch_size=BATCH_SIZE):
    orig_precision = torch.get_float32_matmul_precision()
    torch.set_float32_matmul_precision('high' if use_tf32 else orig_precision)
    try:
        # Same seed for BOTH runs, reset right here: fixes model init,
        # dropout masks, and batch shuffling identically across the two
        # runs, so TF32 is the only variable that differs.
        torch.manual_seed(SEED)
        np.random.seed(SEED)
        random.seed(SEED)

        model = build_model(args, device)
        opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        model.train()

        losses, step_times = [], []
        n = len(train_samples)
        order = np.random.permutation(n)
        pos = 0
        t0_total = time.time()
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
            step_times.append(time.time() - t0)

            losses.append(float(loss.item()))
        total_time = time.time() - t0_total
        return {'losses': losses, 'step_times': step_times, 'total_time': total_time}
    finally:
        torch.set_float32_matmul_precision(orig_precision)


print(f'\nRunning {N_STEPS} real training steps at baseline float32 precision...')
baseline = run_training(use_tf32=False)
print(f'  total={baseline["total_time"]:.2f}s  '
      f'mean_step={1000*np.mean(baseline["step_times"]):.1f}ms  '
      f'final_loss={baseline["losses"][-1]:.6e}')

print(f'\nRunning {N_STEPS} real training steps with TF32 '
      f"(torch.set_float32_matmul_precision('high'))...")
tf32 = run_training(use_tf32=True)
print(f'  total={tf32["total_time"]:.2f}s  '
      f'mean_step={1000*np.mean(tf32["step_times"]):.1f}ms  '
      f'final_loss={tf32["losses"][-1]:.6e}')

speedup = baseline['total_time'] / tf32['total_time']
losses_base = np.array(baseline['losses'])
losses_tf32 = np.array(tf32['losses'])
any_nonfinite = not (np.all(np.isfinite(losses_base)) and np.all(np.isfinite(losses_tf32)))
rel_diff_per_step = np.abs(losses_tf32 - losses_base) / np.clip(np.abs(losses_base), 1e-12, None)
mean_rel_diff = float(rel_diff_per_step.mean())
max_rel_diff = float(rel_diff_per_step.max())
final_rel_diff = float(rel_diff_per_step[-1])

print('\n' + '=' * 70)
print('SUMMARY')
print('=' * 70)
print(f'speedup (wall-clock, {N_STEPS} steps, N={N}, batch_size={BATCH_SIZE}): {speedup:.2f}x')
print(f'any NaN/Inf in either loss trajectory: {any_nonfinite}')
print(f'loss trajectory relative difference: mean={mean_rel_diff:.4e}  '
      f'max={max_rel_diff:.4e}  at_final_step={final_rel_diff:.4e}')
print('\nloss every 20 steps (fp32 vs TF32):')
for s in range(0, N_STEPS, 20):
    print(f'  step {s:>4}: fp32={losses_base[s]:.6e}  tf32={losses_tf32[s]:.6e}  '
          f'rel_diff={rel_diff_per_step[s]:.3e}')

safe = (not any_nonfinite) and (mean_rel_diff < 0.05)
verdict = ('SAFE to use for training' if safe else
           'DO NOT USE for training -- loss trajectories diverge or went non-finite')
print(f'\nVERDICT: {verdict}')
if safe:
    print(f'If adopted: add torch.set_float32_matmul_precision("high") once, near the '
          f'top of cmd_train in resolution_invariance_zeroshot.py (or at the top of the '
          f'notebook cell before training starts), matching how it is already applied for '
          f'inference in no_inference_torch_compile.py.')
