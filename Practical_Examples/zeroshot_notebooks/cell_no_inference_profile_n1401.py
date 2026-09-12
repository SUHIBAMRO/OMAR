# =====================================================================
#  CELL -- detailed profiling of the NO's own inference time at N=1401
#  (Timon round-10, item 3): "I am a bit surprised how slow the NO is at
#  inference for N=1401 while FEM scales better ... Could you please
#  check this before considering the 2.29s as the final inference
#  number. Can you also report the pure GPU forward-pass time after
#  warm-up, excluding data transfer/preprocessing, the precision used,
#  and the peak GPU memory. Ideally, can you provide some profiling to
#  find out where the most inference time is spent."
#
#  Builds on the already-measured 2.29s number (Round6_NO_Inference_vs_
#  TorchFEM_N1401.ipynb) without changing that measurement's own
#  methodology -- adds precision, peak memory, a torch.profiler
#  breakdown of the forward pass, and a bf16-autocast check (diagnostic
#  only, NOT an accuracy claim) to see whether an obvious optimization
#  narrows the gap before 2.29s is treated as final.
# =====================================================================
import json
import os
import subprocess
import sys


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
assert torch.cuda.is_available(), 'this cell needs a real GPU -- peak-memory/profiler numbers on CPU would mean nothing'
print('GPU:', torch.cuda.get_device_name(0))

R = '/content/drive/MyDrive/pfem_run'
# BUG FOUND 2026-09-12: a hardcoded path here ('results/checkpoints/
# B1_neo_hookean/model_best.pt', which never existed on Drive) silently
# fell back to 'data_driven/B1_neo_hookean/model_best.pt' -- a COMPLETELY
# DIFFERENT model (train_data_driven.py's own data-driven-loss baseline
# from the round-5/6 comparison study). Same architecture as the
# data-driven model, so this profiling cell's TIMING numbers are likely
# unaffected either way -- but re-run to be sure now that accuracy
# elsewhere was found to be corrupted by this same bug. Fixed properly
# this time: resolve by CONTENT (sha256), verified against the zero-shot
# study's own already-trusted checkpoint fingerprint, not by guessing a
# path -- see resolve_b1_checkpoint.py's own docstring.
from omar_pfem.resolve_b1_checkpoint import resolve_b1_neo_hookean_checkpoint
CKPT, _ckpt_fp = resolve_b1_neo_hookean_checkpoint(R)
print(f'Resolved checkpoint (verified by fingerprint): {CKPT}')

device = torch.device('cuda')
dtype = torch.float32

from omar_pfem.resolution_invariance_zeroshot import build_sample_b1
from omar_pfem.measure_inference_latency import build_model
from omar_pfem.profile_no_inference_n1401 import profile_inference_detailed
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
print(f'\nBuilding N={N_TEST} sample (solve_fem=False -- no FEM ground truth needed for timing)...')
sample, _ = build_sample_b1(N_TEST, seed=0, material='neo_hookean', Lx=args.Lx, Ly=args.Ly,
                             solve_fem=False)
print(f'  mesh: {sample["xy"].shape[0]} nodes, {sample["quad"].shape[0]} elements')

print('\nProfiling (pure forward time, peak memory, torch.profiler breakdown, bf16 check)...')
result = profile_inference_detailed(sample, model, args, device, dtype,
                                     n_repeats=200, n_warmup=20,
                                     profiler_repeats=30, try_bf16=True)

OUT_JSON = f'{R}/no_inference_profile_N1401.json'
OUT_TXT = f'{R}/no_inference_profile_N1401_profiler_table.txt'

profiler_table = result.pop('profiler_table_top20_by_cuda_time')
with open(OUT_TXT, 'w') as f:
    f.write(profiler_table)
print('Saved profiler table:', OUT_TXT)

with open(OUT_JSON, 'w') as f:
    json.dump(result, f, indent=2)
print('Saved:', OUT_JSON)

# ---- Printed analysis --------------------------------------------------
print('\n' + '=' * 70)
print('ANALYSIS')
print('=' * 70)
print(f'  Pure GPU forward-pass time (after warm-up, excl. data transfer/preprocessing):')
print(f'    {result["inference_ms_per_sample"]:.4f} ms/sample ({result["gpu_name"]})')
print(f'  Precision used: {result["precision"]}')
print(f'  Peak GPU memory during inference: {result["peak_memory_mb"]:.1f} MB')
print()
print('  torch.profiler breakdown (top 20 ops by CUDA time, saved in full to '
      f'{OUT_TXT}):')
print(profiler_table)
if result['bf16_autocast_ms_per_sample'] is not None:
    speedup = result['inference_ms_per_sample'] / result['bf16_autocast_ms_per_sample']
    print(f'\n  bf16 autocast: {result["bf16_autocast_ms_per_sample"]:.4f} ms/sample '
          f'({speedup:.2f}x vs. fp32), relative difference vs. fp32 output: '
          f'{result["bf16_autocast_rel_diff_vs_fp32"]:.3e}')
    print('  (diagnostic only -- NOT a validated accuracy claim; a real accuracy check '
          'would need to be run against ground truth, not just fp32-vs-bf16 self-consistency.)')
print('\n  Compare inference_ms_per_sample above to the already-recorded 2,286.69 ms/sample '
      '(Round6_NO_Inference_vs_TorchFEM_N1401.ipynb) -- it should match closely; this cell '
      'measures the SAME thing, just with more instrumentation around it, not a different call.')
