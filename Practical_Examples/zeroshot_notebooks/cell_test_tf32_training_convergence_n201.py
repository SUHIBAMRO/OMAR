# =====================================================================
#  FOLLOW-UP #3 to the TF32-training tests (2026-09-15). What we know:
#    - N=21, 3000 steps (Test_TF32_Training_Convergence.ipynb): same-seed
#      fp32 vs TF32 tracked closely and stably (final val per_component
#      0.686 vs 0.633) -- but wall-clock speedup was 1.00x, i.e. NONE.
#    - N=201, 100 timed steps (Test_TF32_Speed_N201.ipynb): a REAL 2.10x
#      per-step speedup (547.0ms fp32 vs 260.5ms TF32) -- N=201 is the
#      LARGEST resolution in the multi-res set and the one that actually
#      dominates a training epoch's wall-clock, so this is where an hour
#      -scale (or more) real saving would come from. BUT that same test's
#      final loss after 100 steps was noticeably further apart than at
#      N=21 (fp32=0.0697 vs TF32=0.386, ~5.5x) -- a bigger short-term gap
#      than N=21 showed at a comparable step count, though N=21's own gap
#      also closed by step 3000. Not yet known whether N=201's gap closes
#      the same way or is a real, persistent problem at this resolution.
#
#  THIS TEST answers that directly: the same controlled 3-way experiment
#  used at N=21 (fp32/seedA, fp32/seedB, TF32/seedA), run AT N=201, with
#  enough steps to see whether the short-term gap closes as training
#  continues.
#
#  A NOTE ON THE N=21 RESULT'S OWN LIMITATION, carried over here: at N=21,
#  the fp32-vs-fp32 seed-variation control (run B) itself spiked to a much
#  worse validation error mid-training (7.24 at step 2100, recovering only
#  partially to 2.10 by step 3000) -- a real instability unrelated to
#  TF32, which made the "natural noise floor" used to judge TF32 larger
#  and less trustworthy than intended. This script flags the same thing
#  automatically if it happens again here (checks whether run B's own
#  validation history has a similar blow-up), and reports the full
#  per-checkpoint table so a real judgement can be made by eye, not just
#  from a single ratio that could be thrown off by one outlier run.
#
#  Fewer steps than the N=21 test (each N=201 step costs ~15-20x more
#  wall-clock): N_STEPS=1200, VAL_EVERY=200 -> ~11 min for the two fp32
#  runs (A, B) and ~5 min for the TF32 run (C) at the measured per-step
#  costs, roughly 25-30 minutes total including validation overhead.
#  READ-ONLY against the same finished job's cache; writes nothing back;
#  separate tab.
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
N = 201  # LARGEST resolution -- where the real 2.10x speedup showed up,
          # and where any real time saving would actually come from.
train_samples = cache['train_samples'][N]
val_samples = cache['val_samples'][N]
print(f'loaded {len(train_samples)} train + {len(val_samples)} val real N={N} samples '
      f'from a FINISHED job (read-only, this run writes nothing back to it)')

device = torch.device('cuda')
dtype = torch.float32
N_STEPS = 1200
VAL_EVERY = 200
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
                print(f'  [step {step}/{n_steps}] val_per_component={per_component:.4e}  '
                      f'train_loss={loss.item():.4e}', flush=True)
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
diff_AB = abs(val_A - val_B)
diff_AC = abs(val_A - val_C)
ratio = diff_AC / max(diff_AB, 1e-12)
speedup = run_A['total_time'] / run_C['total_time']

# Same honest check the N=21 test's own result exposed a need for: is run B
# itself a clean "normal noise" example, or did IT blow up (which would make
# |A-B| an unreliable yardstick, same issue found at N=21)?
b_vals = [h['per_component'] for h in run_B['val_history']]
b_blew_up = max(b_vals) > 3 * np.median(b_vals)
if b_blew_up:
    print(f"\n*** WARNING: run B's own validation history shows a blow-up "
          f"(max={max(b_vals):.3e} vs median={np.median(b_vals):.3e}), same pattern "
          f"seen at N=21 -- |A-B| may be inflated by an unrelated fp32 instability "
          f"event, not a clean 'normal noise' baseline. Judge mainly from the direct "
          f"A-vs-C table below, not from the ratio alone. ***")

print('\n' + '=' * 70)
print('VALIDATION ERROR (per_component) at each checkpoint:')
print('=' * 70)
print(f"{'step':<8}{'A (fp32,1234)':<18}{'B (fp32,5678)':<18}{'C (TF32,1234)':<18}")
for a, b, c in zip(run_A['val_history'], run_B['val_history'], run_C['val_history']):
    print(f"{a['step']:<8}{a['per_component']:<18.4e}{b['per_component']:<18.4e}{c['per_component']:<18.4e}")

print('\n' + '=' * 70)
print('SUMMARY')
print('=' * 70)
print(f'final val per_component: A(fp32,1234)={val_A:.4e}  B(fp32,5678)={val_B:.4e}  '
      f'C(TF32,1234)={val_C:.4e}')
print(f'natural fp32-vs-fp32 seed variation |A-B| = {diff_AB:.4e}'
      f'{"  (POSSIBLY INFLATED, see warning above)" if b_blew_up else ""}')
print(f"TF32's own deviation |A-C| = {diff_AC:.4e}")
print(f'ratio |A-C| / |A-B| = {ratio:.2f}')
print(f'speedup (wall-clock, {N_STEPS} steps, N={N}, batch_size={BATCH_SIZE}): {speedup:.2f}x')

a_vs_c_close = diff_AC < 0.5 * val_A  # C within 50% of A's own final value -- a direct,
                                      # ratio-independent sanity check that does not
                                      # depend on run B being well-behaved
safe = a_vs_c_close and (b_blew_up or ratio <= 2.0)
verdict = ('SAFE -- TF32 tracks the same-seed fp32 run closely enough at N=201 too; the '
           f'{speedup:.2f}x speedup can be trusted here' if safe else
           'DO NOT USE -- TF32 final accuracy at N=201 diverges too far from the same-seed '
           'fp32 run to trust the speedup')
print(f'\nVERDICT: {verdict}')
