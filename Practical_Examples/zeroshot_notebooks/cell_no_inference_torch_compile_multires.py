# =====================================================================
#  CELL -- re-verification of point 3 (profiling / torch.compile / TF32)
#  against the NEW multi-res checkpoint, built 2026-09-14.
#
#  WHY THIS EXISTS: the original torch.compile/TF32 measurement
#  (cell_no_inference_torch_compile.py) used the OLD checkpoint (trained
#  on N=21,33 only) -- correct, but not the new multi-res checkpoint the
#  draft's point 1 now recommends. Timing should be checkpoint-
#  independent (same architecture/compute graph, only the weights
#  differ) -- but this project verifies that instead of assuming it,
#  exactly as was already done once for the wrong-vs-correct-checkpoint
#  question. This cell is that verification, pointed at the multi-res
#  checkpoint instead.
#
#  Identical logic to cell_no_inference_torch_compile.py otherwise.
#  Loads the checkpoint by its KNOWN path and verifies its sha256
#  against the fingerprint already recorded in
#  no_accuracy_degradation_sweep_multires.json before trusting it.
# =====================================================================
import json
import os
import subprocess
import sys
import time


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
sys.path.insert(0, f'{WORK}/report_builders')

for _mod_name in list(sys.modules):
    if _mod_name == 'omar_pfem' or _mod_name.startswith('omar_pfem.'):
        del sys.modules[_mod_name]

import torch
assert torch.cuda.is_available(), 'this cell needs a real GPU'
print('GPU:', torch.cuda.get_device_name(0))
print('torch:', torch.__version__)

R = '/content/drive/MyDrive/pfem_run'
OUT = f'{R}/zeroshot_B1_neo_hookean_multires'
CKPT = f'{OUT}/model_best.pt'
assert os.path.exists(CKPT), f'multi-res checkpoint not found: {CKPT}'

from omar_pfem.resolve_b1_checkpoint import sha256_of
EXPECTED_FP = 'cb318c4694d820152d018bcb3a2caa6be4528f3654a59cc8aa2482e9cd495f86'
actual_fp = sha256_of(CKPT)
print(f'{CKPT}\n  sha256={actual_fp}')
assert actual_fp == EXPECTED_FP, (
    f'Checkpoint fingerprint mismatch! Expected {EXPECTED_FP}, got {actual_fp}. '
    f'Refusing to proceed with an unverified checkpoint.')
print('Checkpoint identity verified by fingerprint -- this IS the multi-res retrained model.')

device = torch.device('cuda')
dtype = torch.float32

from omar_pfem.resolution_invariance_zeroshot import build_sample_b1
from omar_pfem.measure_inference_latency import build_model
from omar_pfem.no_inference_torch_compile import profile_with_torch_compile
import argparse

args = argparse.Namespace(
    model='Transolver_Irregular_Mesh', n_hidden=256, n_layers=4, n_heads=8,
    mlp_ratio=2, dropout=0.1, unified_pos=0, ref=16, slice_num=128, fun_dim=4,
    use_soft_dirichlet=1, Lx=1.0, Ly=1.0, R_out=2.0,
)

model = build_model(args, device)
state_dict = torch.load(CKPT, map_location=device)
model.load_state_dict(state_dict)
model.eval()
print('Checkpoint loaded.')

N_TEST = 1401
print(f'\nBuilding N={N_TEST} sample...')
sample, _ = build_sample_b1(N_TEST, seed=0, material='neo_hookean', Lx=args.Lx, Ly=args.Ly,
                             solve_fem=False)

print('\nRunning eager baseline + torch.compile + TF32 attempts (multi-res checkpoint, '
      'this can take ~15-20 min -- the FIRST compiled call triggers real compilation, '
      'not counted in the timing)...')
_started = time.time()
result = profile_with_torch_compile(sample, model, args, device, dtype,
                                     n_repeats=200, n_warmup=20, compile_warmup=5,
                                     try_tf32=True)

OUT_JSON = f'{OUT}/no_inference_torch_compile_N1401_multires.json'
with open(OUT_JSON, 'w') as f:
    json.dump(result, f, indent=2)
print('Saved:', OUT_JSON)

try:
    from omar_pfem.run_manifest import write_manifest
    write_manifest(
        os.path.dirname(os.path.abspath(OUT_JSON)) or '.',
        kind='no_inference_torch_compile_multires', args={'checkpoint': CKPT, 'N': N_TEST},
        started_at=_started, results=result, outputs=[OUT_JSON],
        notes="Per Timon's own note, 2026-09-14, to record the exact git "
              "commit + setup for every run considered final.")
except Exception as e:
    print(f'[manifest] not recorded: {e}')

print('\n' + '=' * 70)
print('RESULT -- torch.compile attempt at N=1401 (multi-res checkpoint)')
print('=' * 70)
print(json.dumps(result, indent=2))
if result['compile_succeeded']:
    print(f"\ntorch.compile: {result['compiled_ms_per_sample']:.4f} ms/sample vs. eager "
          f"{result['eager_ms_per_sample']:.4f} ms/sample -- {result['speedup_vs_eager']:.2f}x, "
          f"output relative difference vs. eager: {result['compiled_vs_eager_rel_diff']:.3e}")
else:
    print(f"\ntorch.compile did not produce a usable result on this model: {result['error']}")

if result['eager_tf32_ms_per_sample'] is not None:
    print(f"\neager + TF32: {result['eager_tf32_ms_per_sample']:.4f} ms/sample -- "
          f"{result['speedup_tf32_vs_eager']:.2f}x vs. strict-fp32 eager, output relative "
          f"difference: {result['eager_tf32_vs_eager_rel_diff']:.3e}")
    if result['compiled_tf32_ms_per_sample'] is not None:
        print(f"torch.compile + TF32: {result['compiled_tf32_ms_per_sample']:.4f} ms/sample -- "
              f"{result['speedup_compiled_tf32_vs_eager']:.2f}x vs. strict-fp32 eager, output "
              f"relative difference: {result['compiled_tf32_vs_eager_rel_diff']:.3e}")
elif result['tf32_error'] is not None:
    print(f"\nTF32 test did not produce a usable result: {result['tf32_error']}")

# ---- Compare against the OLD checkpoint's own numbers ----
OLD_JSON = f'{R}/no_inference_torch_compile_N1401.json'
if os.path.exists(OLD_JSON):
    with open(OLD_JSON) as f:
        old = json.load(f)
    print('\n' + '=' * 70)
    print('OLD (N=21,33 checkpoint) vs. NEW (multi-res checkpoint) -- inference timing')
    print('=' * 70)
    for key, label in [('eager_ms_per_sample', 'eager fp32'),
                        ('compiled_ms_per_sample', 'torch.compile'),
                        ('eager_tf32_ms_per_sample', 'eager+TF32'),
                        ('compiled_tf32_ms_per_sample', 'compile+TF32')]:
        old_v = old.get(key)
        new_v = result.get(key)
        if old_v is not None and new_v is not None:
            rel_diff = abs(new_v - old_v) / old_v
            print(f"  {label:<16} old={old_v:>10.4f} ms   new={new_v:>10.4f} ms   "
                  f"rel_diff={rel_diff:.3e}")
