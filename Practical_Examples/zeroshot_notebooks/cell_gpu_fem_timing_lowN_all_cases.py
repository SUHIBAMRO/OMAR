# =====================================================================
#  CELL -- GPU-native FEM solver timing across the FULL LOW_N sweep, all
#  six cases (Timon's newest email, item 1: finalize the low-N accuracy
#  comparison + add a new QoI-accuracy-threshold vs. required-FEM-
#  resolution vs. break-even table).
#
#  WHY THIS IS NEEDED: gpu_fem_benchmark.py has so far only ever been run
#  at a SINGLE N (N=11, one case) for the accuracy-matched break-even
#  calc (cell_break_even_accuracy_matched.py). The new threshold table
#  needs, for each case and each QoI-accuracy level, "what FEM resolution
#  N is required" -- which in turn needs FEM's own per-sample cost AT
#  EVERY N in the LOW_N sweep (not just N=11), so a break-even point can
#  be computed at whichever N actually satisfies each threshold.
#
#  Reuses gpu_fem_benchmark.py's OWN already-validated build_batch_b1/b2
#  + its exact warm-up/timing convention directly (no subprocess-per-N --
#  that would pay a fresh CUDA-init cost 96 times over for nothing); one
#  untimed warm-up + 3 timed repeats per (case, N), batch_size=1 (the
#  single-sample latency this project's break-even calcs have always
#  used).
#
#  COST: 6 cases x 16 N x (1 warmup + 3 repeats) = 384 solves, all at
#  N<=49 (small meshes) -- expect a few minutes total, not GPU-hours.
# =====================================================================
import os
os.environ['JAX_PLATFORMS'] = 'cpu'

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

WORK = f'{REPO}/Practical_Examples'
os.chdir(WORK)
sys.path.insert(0, WORK)

for _mod_name in list(sys.modules):
    if _mod_name == 'omar_pfem' or _mod_name.startswith('omar_pfem.'):
        del sys.modules[_mod_name]

import numpy as np
import torch
assert torch.cuda.is_available(), 'this cell needs a real GPU'
print('GPU:', torch.cuda.get_device_name(0))

from omar_pfem.gpu_fem_benchmark import build_batch_b1, build_batch_b2

R = '/content/drive/MyDrive/pfem_run'
OUT_DIR = f'{R}/break_even'
os.makedirs(OUT_DIR, exist_ok=True)

device = torch.device('cuda')
dtype = torch.float64  # matches gpu_fem_benchmark.py's own convention

CASES = [
    ('B1', 'neo_hookean'),
    ('B1', 'mooney_rivlin'),
    ('B1', 'arruda_boyce'),
    ('B2', 'neo_hookean'),
    ('B2', 'mooney_rivlin'),
    ('B2', 'arruda_boyce'),
]
LOW_N = [3, 4, 5, 6, 9, 11, 13, 17, 21, 25, 29, 33, 37, 41, 45, 49]
N_REPEATS = 3
BATCH_SIZE = 1  # single-sample latency, matches every break-even calc in this project


def time_one(build_fn, N, material):
    # Untimed warm-up (own fresh batch, not one of the timed repeats) -- same
    # convention as gpu_fem_benchmark.main()'s own _run_one_batch.
    warmup_solve, _ = build_fn(N, BATCH_SIZE, material, device, dtype, seed0=999_000)
    warmup_solve()
    torch.cuda.synchronize(device)

    times_s = []
    for r in range(N_REPEATS):
        solve, n_nodes = build_fn(N, BATCH_SIZE, material, device, dtype, seed0=1000 * r + 1)
        torch.cuda.synchronize(device)
        t0 = time.perf_counter()
        solve()
        torch.cuda.synchronize(device)
        times_s.append(time.perf_counter() - t0)
    median_s = float(np.median(times_s))
    return {"n_nodes": n_nodes, "median_batch_time_s": median_s,
            "per_sample_ms": 1000.0 * median_s / BATCH_SIZE,
            "all_repeat_times_s": times_s}


all_out_jsons = []
for geometry, material in CASES:
    case_id = f'{geometry}_{material}'
    build_fn = build_batch_b1 if geometry == 'B1' else build_batch_b2
    print('\n' + '#' * 78)
    print(f'# {case_id}  (GPU-native FEM timing, N=3..49, bs={BATCH_SIZE})')
    print('#' * 78)

    rows = []
    for N in LOW_N:
        r = time_one(build_fn, N, material)
        r['N'] = N
        r['batch_size'] = BATCH_SIZE
        rows.append(r)
        print(f"  N={N:<4} n_nodes={r['n_nodes']:<7} per_sample={r['per_sample_ms']:9.3f} ms  "
              f"(repeats: {[f'{t:.4f}' for t in r['all_repeat_times_s']]})")

    out_json = f'{OUT_DIR}/gpu_fem_timing_lowN_{case_id}.json'
    report = {"geometry": geometry, "material": material, "batch_size": BATCH_SIZE,
              "n_repeats": N_REPEATS, "low_N": LOW_N, "rows": rows}
    with open(out_json, 'w') as f:
        json.dump(report, f, indent=2)
    all_out_jsons.append(out_json)
    print(f'Saved: {out_json}')

print('\nAll six cases done.')

try:
    from omar_pfem.run_manifest import write_manifest
    write_manifest(OUT_DIR, kind='gpu_fem_timing_lowN_all_cases',
                    args={'low_N': LOW_N, 'batch_size': BATCH_SIZE, 'n_repeats': N_REPEATS,
                          'cases': [c[0] + '_' + c[1] for c in CASES]},
                    started_at=_started, results={'cases': [c[0] + '_' + c[1] for c in CASES]},
                    outputs=all_out_jsons,
                    notes="GPU-native FEM (gpu_fem_solver.py, via gpu_fem_benchmark.py's own "
                          "validated build_batch_b1/b2) timed at EVERY LOW_N resolution "
                          "(previously only measured at N=11, one case), needed as the FEM-"
                          "cost input to Timon's new QoI-accuracy-threshold vs. required-FEM-"
                          "resolution vs. break-even table.")
except Exception as e:
    print(f'[manifest] not recorded: {e}')

print('\nDone.')
