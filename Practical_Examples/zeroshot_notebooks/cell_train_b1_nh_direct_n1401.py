# =====================================================================
#  CELL -- train a NEW B1 x Neo-Hookean checkpoint DIRECTLY at N=1401
#  (single resolution, at the actual deployment resolution), as an
#  ablation against the multi-resolution (N=21,33,101,201 -> zero-shot
#  N=1401) checkpoint. Timon round-11 point 4.
#
#  WHY THIS IS DIFFERENT FROM THE MULTI-RES CHECKPOINT: that one is
#  never trained AT N=1401 at all -- it only sees N=21,33,101,201 during
#  training and is evaluated zero-shot (no retraining) at N=1401. This
#  ablation instead trains directly on N=1401 itself, answering "if you
#  just train where you'll deploy, how does that compare on cost,
#  memory, accuracy, and inference time?" Per Timon's own explicit
#  instruction, the zero-shot number stays a SEPARATE, untouched result
#  -- this is not a replacement for it.
#
#  SAMPLE COUNT: 100 train + 20 val (NOT the multi-res checkpoint's own
#  400+100) -- Omar's own explicit choice, recommended here given the
#  per-sample ground-truth cost at N=1401 available at the time (see
#  CORRECTION below for why that number was wrong): appropriate for an
#  ablation/comparison study, not the headline result, regardless.
#
#  CORRECTION (2026-09-14, found on a real GPU run of this exact cell):
#  the COST ESTIMATE below (58.54s/sample) was measured under a DIFFERENT
#  condition than what this cell actually runs, and is wrong by ~10x.
#  58.54s comes from assembled_direct_convergence_production_N401_1401.json,
#  a single-shot solve (nsteps=1, one full-load Newton solve). But
#  build_sample_b1_fast (called by THIS cell's --fast_solver 1 path) uses
#  nsteps=10 (resolution_invariance_zeroshot.py's own hardcoded default),
#  because a prior finding (2026-09-12) showed the single-shot solve does
#  NOT converge at N=1401 with random per-sample material fields (Newton
#  starting cold at full load stalls) -- 10 incremental load steps are
#  needed for real convergence, not optional. The real, measured rate on
#  this run's own console output was 587.2s/sample -- 587.2/58.54 = 10.03,
#  matching nsteps=10 almost exactly (each load step costs about as much
#  as one full single-shot solve). The fast solver (solve_assembled_direct)
#  IS being used correctly here -- this was a benchmark/apples-to-oranges
#  mismatch in the cost ESTIMATE, not a bug in the solver or a missed
#  speedup opportunity.
#
#  SPEEDUP FOUND AND APPLIED (2026-09-14, Test_FewerLoadSteps_N1401.ipynb,
#  real A100 run): nsteps=10 is not actually required -- every one of its
#  own 10 steps converged far tighter than the 1e-7 tolerance ever needed,
#  suggesting fewer/larger steps would still work. Tested nsteps=5 and
#  nsteps=3 directly (3 seeds each, independent post-hoc convergence check,
#  not guessed): BOTH fully converged, and nsteps=3 was not only 1.64x
#  faster (358.1s/sample mean vs. 588.9s/sample for nsteps=10, same 3
#  seeds) but its relative residuals were actually TIGHTER (worst 4.459e-12
#  vs. nsteps=10's own 3.859e-10). This cell now passes --nsteps 3.
#  Mixing nsteps across samples already generated at nsteps=10 and new ones
#  at nsteps=3 is safe -- it only changes the SOLUTION METHOD used to reach
#  convergence, not the converged physical solution itself.
#
#  COST ESTIMATE (data generation only, before training) -- CORRECTED TWICE:
#    120 samples x ~358s/sample ~= 42,960s ~= 11.9 GPU-hours (with nsteps=3)
#  (the original 58.54s-based estimate undercounted by ~10x; the first
#  correction, 587s/sample at nsteps=10, was itself ~1.64x higher than
#  necessary once nsteps=3 was verified safe)
#  Training cost itself is unknown in advance (single-resolution N=1401
#  training has never been run) -- tracked for real via write_manifest,
#  not guessed.
#
#  RESUMABLE: resolution_invariance_zeroshot.py's own train subcommand
#  already checkpoints per-validation-event and resumes generation in
#  chunks -- same discipline as every other training cell in this
#  project.
# =====================================================================
#
#  Defensive fix (2026-09-14, found on a sibling notebook): force JAX
#  onto CPU before any import. omar_pfem.data.materials unconditionally
#  imports omar_pfem.data.material_models_jax, and JAX's own default
#  behavior on first touching a GPU is to preallocate ~90% of it for
#  the life of the process, invisible to torch.cuda's own memory stats.
#  This case is Neo-Hookean only (whose own conversion doesn't actually
#  use JAX), but the import happens regardless of material -- applied
#  here anyway since it costs nothing and removes any risk.
import os
os.environ['JAX_PLATFORMS'] = 'cpu'
# Third real OOM fix attempt (2026-09-15), cheap and zero-risk: even after
# batch_size=1 + grad_checkpoint=1, a real GPU run still OOM'd, but got
# much further this time -- all the way through the full forward pass and
# into loss.backward() itself, failing needing 7.49 GiB with "7.42 GiB
# free" reported yet only 71.81/79.25 GiB actually in use. That gap (a
# request smaller than the reported free amount still failing) is the
# classic signature of allocator FRAGMENTATION, not genuine exhaustion --
# the error message itself names the fix: PYTORCH_CUDA_ALLOC_CONF=
# expandable_segments:True lets the CUDA caching allocator grow existing
# reserved segments instead of demanding a fresh contiguous block, which
# is exactly what a request that size, that close to the fragmented
# reserved-but-unallocated 11.09 GiB, needs. Pure allocator strategy, no
# effect on computed values -- must be set before the child subprocess's
# own `import torch`, which is why it's set on THIS process's environ
# here: subprocess.Popen (used by run() below) inherits it automatically.
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

import json
import subprocess
import sys
import time

_started = time.time()


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

run([sys.executable, '-m', 'pip', 'install', '-q',
     'einops', 'timm', 'h5py', 'jax', 'tqdm'])
run([sys.executable, '-m', 'pip', 'install', '-q', 'torch-sla'])
run([sys.executable, '-m', 'pip', 'install', '-q', 'nvmath-python[cu12]==0.9.0'])

WORK = f'{REPO}/Practical_Examples'
os.chdir(WORK)
sys.path.insert(0, WORK)

for _mod_name in list(sys.modules):
    if _mod_name == 'omar_pfem' or _mod_name.startswith('omar_pfem.'):
        del sys.modules[_mod_name]

import torch
assert torch.cuda.is_available(), 'this cell needs a real GPU'
print('GPU:', torch.cuda.get_device_name(0))

from torch_sla.backends import is_cudss_available
if not is_cudss_available():
    raise RuntimeError("cuDSS not available -- data generation needs the real direct solver.")
print('cuDSS available.')

OUT_NAME = 'zeroshot_B1_neo_hookean_direct_n1401'
OUT = f'/content/drive/MyDrive/pfem_run/{OUT_NAME}'
os.makedirs(OUT, exist_ok=True)
print('OUT =', OUT)
for f in sorted(os.listdir(OUT)):
    print('  found:', f)

N_TRAIN = 100
N_VAL = 20

# nsteps=3 verified SAFE 2026-09-14 (Test_FewerLoadSteps_N1401.ipynb, real
# A100 run): 3 seeds all converged_likely=True at N=1401, relative residuals
# actually TIGHTER than nsteps=10's own (worst 4.459e-12 vs 3.859e-10), and
# 1.64x faster (358.1s/sample mean vs 588.9s/sample for nsteps=10 on the
# SAME 3 seeds). Mixing nsteps across samples in the same cache is safe --
# it only changes the SOLUTION METHOD, not the converged physical solution
# -- so switching mid-run does not invalidate the samples already generated
# at nsteps=10.
NSTEPS = 3

print(f'\nEstimated data-generation cost: {N_TRAIN + N_VAL} samples x ~358s/sample '
      f'(real, measured, nsteps={NSTEPS}, verified SAFE on a real A100 run -- '
      f'Test_FewerLoadSteps_N1401.ipynb, 2026-09-14) '
      f'~= {(N_TRAIN + N_VAL) * 358.0 / 3600:.2f} GPU-hours.\n')

# ---- Step 1: generate FEM ground truth at N=1401 (fast solver) ----
run([sys.executable, '-u', '-m', 'omar_pfem.resolution_invariance_zeroshot', 'train',
     '--geometry', 'B1', '--material', 'neo_hookean',
     '--train_resolutions', '1401',
     '--n_train_per_res', str(N_TRAIN), '--n_val_per_res', str(N_VAL),
     '--fast_solver', '1', '--nsteps', str(NSTEPS),
     '--gen_chunk', '10', '--stop_after_generation',
     '--out_dir', OUT])

# ---- Step 2: train (same protocol as the multi-res checkpoint, single resolution) ----
#
# BATCH SIZE FIX (2026-09-15, found on a real GPU run of this exact cell):
# --batch_size 8 (the multi-res checkpoint's own value, fine up to N=201)
# OOM'd immediately on the very first forward pass at N=1401 --
# torch.OutOfMemoryError trying to allocate 14.98 GiB with only 2.95 GiB
# free out of 79.25 GiB total (A100). Root cause, confirmed by the numbers
# themselves: generate_grid_Q4(Lx, Ly, 1401, 1401) builds a FULL 1402x1402
# grid, i.e. ~1.966 MILLION nodes per sample -- N=1401 is a per-edge count,
# not a node count. A single (batch, n_nodes, n_hidden) activation tensor
# at batch=8, n_hidden=256 is 1,965,604 * 8 * 256 * 4 bytes ~= 15.0 GiB --
# matches the failing allocation almost exactly, and that is only ONE such
# tensor near the very start of the model (the embedding stage); TRAINING
# (unlike the inference-only accuracy/timing checks already done at
# N=1401) additionally needs every intermediate activation kept alive for
# backward, so the real requirement is many times that single number. This
# is not a memory-cleanup bug (no leftover cache from generation carries
# over -- this is a fresh subprocess) -- it is simply that batch_size=8 at
# this node count cannot fit on one 80GB GPU with gradients retained.
# Dropping to batch_size=1 (real GPU run, 2026-09-15) was NOT enough on
# its own -- OOM'd again, deeper in the model (inside the 2nd
# transformer block's MLP/GELU), trying to allocate 3.74 GiB with 77.49
# GiB already in use. Confirms the earlier prediction: even ONE sample's
# worth of activations, kept alive across all n_layers=4 blocks for
# backward, is too much at N=1401's ~1.966M nodes -- an MLP hidden
# tensor alone (n_hidden*mlp_ratio = 512) is
# 1,965,604 * 512 * 4 bytes ~= 4.0 GiB, matching the failing allocation,
# and there are several such tensors PER block, times 4 blocks, all
# retained simultaneously in the naive (no-checkpoint) backward graph.
# batch_size=1 was already the floor, so the real fix is GRADIENT
# CHECKPOINTING (added to Model/Transolver_Irregular_Mesh.py and wired
# through as --grad_checkpoint, default 0/off everywhere else): trades
# recompute for memory by NOT keeping each block's activations alive,
# recomputing them during backward instead. This does not change the
# computed gradients (checkpoint reproduces the exact same forward
# exactly, it is not an approximation) -- only memory and step time
# (expect roughly 1.3-2x slower per step from the extra recompute, on
# top of the batch_size=1 slowdown already expected). Untested at
# N=1401 until this run happens; the already-generated samples_cache.pt
# is untouched and reused as-is, so this only re-runs the training step.
_train_started = time.time()
run([sys.executable, '-u', '-m', 'omar_pfem.resolution_invariance_zeroshot', 'train',
     '--geometry', 'B1', '--material', 'neo_hookean',
     '--train_resolutions', '1401',
     '--n_train_per_res', str(N_TRAIN), '--n_val_per_res', str(N_VAL),
     '--fast_solver', '1', '--nsteps', str(NSTEPS),
     '--epochs', '2000', '--validate_every', '25', '--batch_size', '1',
     '--grad_checkpoint', '1',
     '--early_stop_patience', '8', '--lr', '2e-3',
     '--out_dir', OUT])
_train_wall_clock = time.time() - _train_started

# ---- Step 3: real accuracy/QoI check at N=1401 vs. real FEM ground truth ----
from omar_pfem.measure_inference_latency import build_model
from omar_pfem.no_accuracy_at_n1401 import run_accuracy_degradation_sweep
import argparse

model_args = argparse.Namespace(
    model='Transolver_Irregular_Mesh', n_hidden=256, n_layers=4, n_heads=8,
    mlp_ratio=2, dropout=0.1, unified_pos=0, ref=16, slice_num=128, fun_dim=4,
    use_soft_dirichlet=1, Lx=1.0, Ly=1.0, R_out=2.0,
)
CKPT = f'{OUT}/model_best.pt'
assert os.path.exists(CKPT), f'checkpoint not found after training: {CKPT}'
model = build_model(model_args, device=torch.device('cuda')).to(torch.float32)
model.load_state_dict(torch.load(CKPT, map_location='cuda'))
print('\nLoaded newly-trained direct-N1401 checkpoint:', CKPT)

ACC_JSON = f'{OUT}/no_accuracy_degradation_sweep_direct_n1401.json'
rows = run_accuracy_degradation_sweep(model, model_args, [1401], ACC_JSON, torch.device('cuda'),
                                       material='neo_hookean')
r1401 = rows[0]
print('\n' + '=' * 70)
print('DIRECT-N1401 CHECKPOINT -- accuracy at N=1401 vs. real FEM ground truth')
print('=' * 70)
print(f"disp_rel_L2={r1401['fp32']['disp_rel_L2']:.4e}  "
      f"L2_rel={r1401['fp32']['L2_rel']:.4e}  "
      f"H1_semi_rel={r1401['fp32']['H1_semi_rel']:.4e}")

# ---- Step 4: inference timing (eager fp32, same protocol as every other timing number) ----
from omar_pfem.resolution_invariance_zeroshot import build_sample_b1
from omar_pfem.no_inference_torch_compile import profile_with_torch_compile

sample, _ = build_sample_b1(1401, seed=0, material='neo_hookean', Lx=model_args.Lx,
                             Ly=model_args.Ly, solve_fem=False)
timing_result = profile_with_torch_compile(sample, model, model_args, torch.device('cuda'),
                                            torch.float32, n_repeats=200, n_warmup=20,
                                            compile_warmup=5, try_tf32=True)

# ---- Step 5: peak GPU memory during training (best-effort: re-read from metrics_history.json) ----
metrics_path = f'{OUT}/metrics_history.json'
peak_mem_mb = None
if os.path.exists(metrics_path):
    metrics = json.load(open(metrics_path))
    mem_vals = [m.get('gpu_peak_mem_mb') for m in metrics if m.get('gpu_peak_mem_mb') is not None]
    if mem_vals:
        peak_mem_mb = max(mem_vals)

report = {
    'note': "Direct-N1401 ablation (Timon round-11 point 4) vs. the multi-res "
            "(N=21,33,101,201 -> zero-shot N=1401) checkpoint. The zero-shot number "
            "is a SEPARATE, untouched result -- this is not a replacement for it.",
    'n_train': N_TRAIN, 'n_val': N_VAL,
    'training_wall_clock_s': _train_wall_clock,
    'training_peak_mem_mb': peak_mem_mb,
    'accuracy_at_N1401': {
        'disp_rel_L2': r1401['fp32']['disp_rel_L2'],
        'L2_rel': r1401['fp32']['L2_rel'],
        'H1_semi_rel': r1401['fp32']['H1_semi_rel'],
    },
    'inference_eager_ms_per_sample': timing_result.get('eager_ms_per_sample'),
    'inference_compile_tf32_ms_per_sample': timing_result.get('compiled_tf32_ms_per_sample'),
    'comparison_multires_checkpoint': {
        'training_wall_clock_s': 41881.28,
        'disp_rel_L2_at_N1401': 0.0585,
        'inference_eager_ms_per_sample': 2292.1,
        'inference_compile_tf32_ms_per_sample': 394.0,
        'note': "Already-verified numbers, reused here for the side-by-side "
                "comparison, not re-measured in this cell.",
    },
}

print('\n' + '=' * 70)
print('COMPARISON -- direct-N1401 (this run) vs. multi-res (already verified)')
print('=' * 70)
print(f"  Training cost:   direct={_train_wall_clock:.1f}s ({_train_wall_clock/3600:.2f}h)   "
      f"multi-res=41881.28s (11.63h)")
print(f"  Accuracy @N1401: direct disp_rel_L2={r1401['fp32']['disp_rel_L2']:.4f}   "
      f"multi-res disp_rel_L2=0.0585")
print(f"  Inference eager: direct={timing_result.get('eager_ms_per_sample'):.1f}ms   "
      f"multi-res=2292.1ms")
if peak_mem_mb is not None:
    print(f"  Training peak mem: direct={peak_mem_mb:.1f}MB")

OUT_JSON = f'{OUT}/direct_n1401_vs_multires_comparison.json'
with open(OUT_JSON, 'w') as f:
    json.dump(report, f, indent=2)

try:
    from omar_pfem.run_manifest import write_manifest
    write_manifest(OUT, kind='train_b1_nh_direct_n1401', args={'n_train': N_TRAIN, 'n_val': N_VAL},
                    started_at=_started, results=report, outputs=[OUT_JSON, ACC_JSON],
                    notes="Timon round-11 point 4: direct-N1401 ablation vs. the multi-res "
                          "checkpoint, reduced sample count (100+20, not 400+100) per Omar's "
                          "own explicit choice given the real per-sample N=1401 ground-truth cost.")
except Exception as e:
    print(f'[manifest] not recorded: {e}')

print('\nSaved:', OUT_JSON)
