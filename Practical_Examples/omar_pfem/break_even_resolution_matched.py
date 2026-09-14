"""
Resolution-MATCHED break-even (Timon round-11, point 1), built 2026-09-14.

WHY THIS IS DIFFERENT FROM break_even_accuracy_matched.py: that script
compares the NO at N=1401 against the CHEAPEST FEM mesh that matches its
own accuracy (N=11, a near-degenerate mesh) -- the "what's the cheapest
FEM you could get away with" question. Timon explicitly asked to ALSO
keep the opposite comparison: NO and FEM both run AT THE SAME N=1401 --
the "if you insist on FEM at full resolution, does the NO still win"
question, which he correctly expects to look much more favorable to the
NO, since FEM's own cost grows steeply with N while the NO's does not.

NEEDS NO NEW GPU RUN. Every number below is already a real, independently
verified GPU measurement sitting in this project's own JSON files:
  - torch-fem @ N=1401, matched FP64/1e-8 precision: 133.83 s/sample
    (real A100 run, round-9 headline result -- see
    torchfem_convergence_vs_fine_reference_full.json and
    no_inference_vs_torchfem_N1401.json). torch-fem, not "ours" own
    GPU-native solver, is used here deliberately: it is this project's
    own established, already-verified "fast, competent FEM" reference at
    this N (round-9's whole point was that torch-fem beats "ours" own
    solver 204-306x at matched precision -- using "ours" own slower
    solver here would not be the fair FEM baseline the round-9 story
    itself already settled on).
  - NO @ N=1401, re-verified against the multi-res checkpoint
    (2026-09-14, no_inference_torch_compile_N1401_multires.json):
    eager fp32 2,292.1 ms/sample, compile+TF32 394.0 ms/sample.
  - Training wall-clock for the multi-res checkpoint (confirmed
    2026-09-14, metrics_history_multires.json): 41,881.28 s.
"""
import json
import os
import time

_started = time.time()

TORCHFEM_N1401_MS = 133.83 * 1000.0  # 133.83 s/sample, matched FP64/1e-8
NO_EAGER_MS = 2292.1
NO_COMPILE_TF32_MS = 394.0
TRAINING_SECONDS = 41881.28

print('=' * 78)
print('RESOLUTION-MATCHED BREAK-EVEN -- NO vs. torch-fem, BOTH AT N=1401')
print('(Timon round-11, point 1: kept alongside the accuracy-matched one,')
print(' not instead of it -- these answer two different questions.)')
print('=' * 78)
print(f'  torch-fem @ N=1401, matched precision: {TORCHFEM_N1401_MS:.1f} ms/sample')
print(f'  NO @ N=1401 (eager fp32):               {NO_EAGER_MS:.1f} ms/sample')
print(f'  NO @ N=1401 (compile+TF32):             {NO_COMPILE_TF32_MS:.1f} ms/sample')
print()

rows = []
for label, no_ms in [('eager fp32', NO_EAGER_MS), ('compile+TF32', NO_COMPILE_TF32_MS)]:
    saving_ms = TORCHFEM_N1401_MS - no_ms
    speedup = TORCHFEM_N1401_MS / no_ms
    be_samples = TRAINING_SECONDS / (saving_ms / 1000.0)
    be_gpu_hours = be_samples * no_ms / 1000.0 / 3600.0
    rows.append({
        'label': label, 'no_ms_per_sample': no_ms,
        'speedup_vs_torchfem_N1401': speedup,
        'break_even_samples': be_samples,
        'break_even_no_gpu_hours': be_gpu_hours,
    })
    print(f'  [{label}] speedup vs. torch-fem @ N=1401: {speedup:.1f}x '
          f'(saving {saving_ms:+.1f} ms/sample)')
    print(f'    -> break-even after {be_samples:.0f} samples '
          f'({be_gpu_hours:.3f} GPU-hours of NO inference, vs. '
          f'{TRAINING_SECONDS/3600.0:.2f} GPU-hours of training)')
print('=' * 78)

report = {
    'note': "Resolution-MATCHED break-even (NO and FEM both solved/evaluated "
            "at N=1401), kept ALONGSIDE break_even_accuracy_matched_N1401.json "
            "(NOT a replacement) per Timon's round-11 point 1. Uses torch-fem's "
            "own matched-precision N=1401 number (this project's established "
            "fast-FEM reference at this N, per the round-9 headline result), "
            "not 'ours' own GPU-native solver (round-9 showed torch-fem beats "
            "'ours' 204-306x at this N, so it is not a fair/representative FEM "
            "baseline for this specific comparison). No new GPU run was needed: "
            "every number reused here is already an independently verified "
            "measurement from an earlier real GPU run (see 'sources' below).",
    'sources': {
        'torchfem_N1401_ms_per_sample': 'torchfem_convergence_vs_fine_reference_full.json (round-9, real A100 run)',
        'no_N1401_ms_per_sample': 'no_inference_torch_compile_N1401_multires.json (2026-09-14, multi-res checkpoint, re-verified)',
        'training_seconds': 'metrics_history_multires.json (2026-09-14, confirmed real training wall-clock)',
    },
    'torchfem_N1401_ms_per_sample': TORCHFEM_N1401_MS,
    'training_seconds': TRAINING_SECONDS,
    'rows': rows,
}

OUT_JSON = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'break_even_resolution_matched_N1401.json')
with open(OUT_JSON, 'w') as f:
    json.dump(report, f, indent=2)

try:
    from omar_pfem.run_manifest import write_manifest
    write_manifest(
        os.path.dirname(OUT_JSON), kind='break_even_resolution_matched',
        args={'N': 1401}, started_at=_started, results=report, outputs=[OUT_JSON],
        notes="Timon round-11 point 1: keep both break-even comparisons. "
              "Pure recombination of already-verified numbers, no new GPU run.")
except Exception as e:
    print(f'[manifest] not recorded: {e}')

print('\nSaved:', OUT_JSON)
