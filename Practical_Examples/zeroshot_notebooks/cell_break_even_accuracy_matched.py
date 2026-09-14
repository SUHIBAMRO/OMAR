# =====================================================================
#  CELL -- accuracy-matched break-even (Timon round-10 point 5), built
#  2026-09-14 once point 1 was fully numerically settled for the new
#  multi-res checkpoint.
#
#  WHY THIS IS DIFFERENT FROM break_even_analysis.py: that script matches
#  FEM and the NO at the SAME resolution N, varying only GPU batch size --
#  a fair matched-hardware comparison, but not what "coarsest suitable
#  FEM" means. Here the whole point is that the two run at DIFFERENT N:
#  the NO predicts at N=1401 (its real deployment resolution), while FEM
#  only needs N=11 to match the NEW checkpoint's own accuracy there
#  (from no_peak_stress_fixed_location_multires.json's crossover -- N=11
#  is the coarsest suitable FEM for NO(multi-res)@N=1401, bound by
#  tangent energy). This is the actual "does the NO pay for its own
#  training cost" question Timon's point 5 is asking, not a matched-N
#  throughput comparison (that one's answer is already in point 2).
#
#  Uses the ALREADY-VERIFIED NO N=1401 numbers from point 3 (re-measured
#  with the corrected checkpoint, see PROJECT_STATUS.md 2026-09-13):
#    eager fp32:      2292.1 ms/sample
#    compile+TF32:     394.0 ms/sample (best case, ~3.2e-3 rel. accuracy cost)
#  and the real training_seconds for the multi-res checkpoint (confirmed
#  2026-09-14 from its own metrics_history.json): 41881.28s.
#
#  This cell measures the ONE missing number: GPU-FEM per-sample cost at
#  N=11, bs=1 -- cheap, single N, should take well under a minute.
# =====================================================================
import json
import os
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

import torch
assert torch.cuda.is_available(), 'this cell needs a real GPU'
print('GPU:', torch.cuda.get_device_name(0))

R = '/content/drive/MyDrive/pfem_run'
OUT_JSON = f'{R}/break_even/fem_timing_N11.json'
os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)

run([sys.executable, '-u', '-m', 'omar_pfem.gpu_fem_benchmark',
     '--geometry', 'B1', '--material', 'neo_hookean', '--N', '11',
     '--batch_sizes', '1', '--n_repeats', '3', '--out_json', OUT_JSON])

with open(OUT_JSON) as f:
    fem_data = json.load(f)
fem_ms = next(r['per_sample_ms'] for r in fem_data['rows'] if r['batch_size'] == 1)

# Already-verified NO numbers, N=1401, bs=1 (point 3, re-measured with the
# corrected checkpoint 2026-09-13 -- see PROJECT_STATUS.md):
NO_EAGER_MS = 2292.1
NO_COMPILE_TF32_MS = 394.0
TRAINING_SECONDS = 41881.28  # confirmed 2026-09-14, metrics_history_multires.json

print('\n' + '=' * 78)
print('ACCURACY-MATCHED BREAK-EVEN -- NO@N=1401 vs. torch-fem@N=11 (coarsest')
print('suitable FEM for the multi-res checkpoint at N=1401, bound by tangent')
print('energy -- see no_peak_stress_fixed_location_multires.json)')
print('=' * 78)
print(f'  torch-fem @ N=11, bs=1: {fem_ms:.3f} ms/sample')
print(f'  NO @ N=1401, bs=1 (eager fp32):        {NO_EAGER_MS:.1f} ms/sample')
print(f'  NO @ N=1401, bs=1 (compile+TF32):      {NO_COMPILE_TF32_MS:.1f} ms/sample')
print()
for label, no_ms in [('eager fp32', NO_EAGER_MS), ('compile+TF32', NO_COMPILE_TF32_MS)]:
    saving_ms = fem_ms - no_ms
    speedup = fem_ms / no_ms
    print(f'  [{label}] speedup vs. accuracy-matched FEM: {speedup:.3f}x '
          f'(saving {saving_ms:+.1f} ms/sample)')
    if saving_ms > 0:
        be = TRAINING_SECONDS / (saving_ms / 1000.0)
        print(f'    -> break-even after {be:.0f} samples '
              f'({be * no_ms / 1000.0 / 3600.0:.2f} GPU-hours of NO inference)')
    else:
        print(f'    -> NO NEVER breaks even: accuracy-matched FEM (N=11) is already '
              f'CHEAPER per sample than the NO at N=1401. Training cost is never repaid '
              f'against this baseline, however many samples are solved.')
print('=' * 78)

report = {
    "note": "Accuracy-matched break-even, NOT matched-N/matched-batch (that is "
            "break_even_analysis.py's own comparison, already used for point 2). "
            "FEM at its coarsest-suitable N (per the multi-res checkpoint's own "
            "no_peak_stress_fixed_location_multires.json crossover at N=1401) vs. "
            "the NO's own real inference cost at N=1401.",
    "fem_N": 11, "fem_ms_per_sample": fem_ms,
    "no_N": 1401,
    "no_eager_ms_per_sample": NO_EAGER_MS,
    "no_compile_tf32_ms_per_sample": NO_COMPILE_TF32_MS,
    "training_seconds": TRAINING_SECONDS,
}
OUT_SUMMARY = f'{R}/break_even/accuracy_matched_break_even_N1401.json'
with open(OUT_SUMMARY, 'w') as f:
    json.dump(report, f, indent=2)

try:
    from omar_pfem.run_manifest import write_manifest
    write_manifest(
        os.path.dirname(os.path.abspath(OUT_SUMMARY)) or '.',
        kind='break_even_accuracy_matched', args={'fem_N': 11, 'no_N': 1401},
        started_at=_started, results=report, outputs=[OUT_JSON, OUT_SUMMARY],
        notes="Per Timon's own note, 2026-09-14, to record the exact git "
              "commit + setup for every run considered final.")
except Exception as e:
    print(f'[manifest] not recorded: {e}')
print('\nSaved:', OUT_JSON, 'and', OUT_SUMMARY)
