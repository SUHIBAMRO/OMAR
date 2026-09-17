# =====================================================================
#  CELL -- multi-QoI crossover analysis (Timon round-11 point 2), the
#  five (geometry, material) cases beyond B1xNeo-Hookean, which already
#  has this analysis in the Report ("coarsest suitable finite-element
#  mesh", bound by the tangent-energy norm at N=11 -- see Table
#  18-R10e). Timon's own question: which QoI/norm actually determines
#  that crossover, and is it the SAME QoI for every case, or different
#  ones? Answered here directly from real GPU numbers for all five
#  remaining cases, not assumed to generalize from B1xNeo-Hookean alone.
#
#  WHAT THIS REUSES, UNCHANGED: each case's own NO accuracy sweep at
#  N=1401 (and every other resolution) already exists on Drive from
#  task #22 (no_accuracy_degradation_sweep_{geometry}_{material}.json)
#  -- no need to recompute the operator side at all. Only torch-fem's
#  own QoI sweep at LOW N (the same 3-49 range B1xNeo-Hookean's own
#  crossover used) is new here.
#
#  REAL BUG FOUND AND FIXED while building this cell (2026-09-16,
#  committed before this notebook was written): run_qoi_study accepts a
#  `geometry` argument but every existing call site so far only ever
#  used the default (B1), so a real bug never surfaced -- its own
#  E_fn/nu_fn were hardcoded to AnalyticFieldB1 regardless of geometry,
#  which would have silently sampled B1's own material field at B2's
#  polar query points and produced wrong numbers with no error. Fixed
#  to select AnalyticFieldB1/B2 by geometry, and to skip the reaction-
#  resultant QoI for B2 (explicitly "B1 only" elsewhere in this
#  project -- B2's own fixed boundary has no established reaction
#  convention). Verified on real CPU compute at N=7 for both geometries
#  before this cell was ever pointed at a real GPU sweep: B1's own
#  numbers unchanged, B2's are sane and non-crashing.
#
#  COST: the 16-point low-N sweep itself is cheap (each solve smaller
#  than the ones the resolution-matched break-even cell already ran at
#  N=1401) -- but each of these 5 cases ALSO needs its own fresh fine
#  reference (none of these 5 materials/geometries has ever had one
#  computed before; only B1xNeo-Hookean's fine_B1_neo_hookean_Q4_N2236.pt
#  already exists on Drive). The default fine_N=2236 used elsewhere in
#  this project (Table 6a's own ~10M-DOF reference) is drastically
#  oversized for what this cell actually needs: the low-N sweep only
#  goes up to N=49, and this project's own established safety margin is
#  fine_N >= 4x the largest N under test (4*49=196) -- so fine_N=201 is
#  already comfortably past that margin, at a small fraction of
#  N=2236's cost (~15 min class per Table 6a's own recorded N=201
#  timing, vs. an estimated 30-48h for a FRESH N=2236 solve, extrapo-
#  lated from N=1401's real fresh-solve time of 27257.4s / ~7.6h scaled
#  by DOF). REAL BUG CAUGHT before any of the 5 cases got past the
#  first Newton iteration of their fine solve (Omar noticed the run had
#  started a from-scratch N=2236 solve and asked about it): fixed here
#  by using fine_N=201 instead of the function's own fine_N=2236
#  default. Expect low tens of minutes per case, not tens of hours.
# =====================================================================
import os
os.environ['JAX_PLATFORMS'] = 'cpu'  # same defensive fix as every other cell here

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

run([sys.executable, '-m', 'pip', 'install', '-q', 'torch-fem'])

WORK = f'{REPO}/Practical_Examples'
os.chdir(WORK)
sys.path.insert(0, WORK)

for _mod_name in list(sys.modules):
    if _mod_name == 'omar_pfem' or _mod_name.startswith('omar_pfem.'):
        del sys.modules[_mod_name]

import torch
assert torch.cuda.is_available(), 'this cell needs a real GPU'
print('GPU:', torch.cuda.get_device_name(0))

import jax
jax_devices = jax.devices()
print(f'JAX devices: {jax_devices}')
assert all(d.platform == 'cpu' for d in jax_devices), (
    f'JAX resolved to a non-CPU backend ({jax_devices}) -- refusing to proceed, '
    f'see cell_resolution_matched_break_even_all_cases.py for why this matters.')
print('JAX confirmed CPU-only -- safe to proceed with the full GPU for torch-fem.')

from omar_pfem.torchfem_comparison import run_qoi_study

R = '/content/drive/MyDrive/pfem_run'
CHECKPOINT_DIR = '/content/drive/MyDrive/pfem_ckpt'
device = torch.device('cuda')

# Same low-N range B1xNeo-Hookean's own crossover (already in the Report,
# Table 18-R10e) used, so every case's own "coarsest suitable N" is
# directly comparable across cases.
LOW_N = [3, 4, 5, 6, 9, 11, 13, 17, 21, 25, 29, 33, 37, 41, 45, 49]

# (geometry, material, NO's own existing accuracy-sweep JSON on Drive --
# already computed by task #22, reused here unchanged)
CASES = [
    ('B1', 'mooney_rivlin', f'{R}/no_accuracy_degradation_sweep_B1_mooney_rivlin.json'),
    ('B1', 'arruda_boyce', f'{R}/no_accuracy_degradation_sweep_B1_arruda_boyce.json'),
    ('B2', 'neo_hookean', f'{R}/no_accuracy_degradation_sweep_B2_neo_hookean.json'),
    ('B2', 'mooney_rivlin', f'{R}/no_accuracy_degradation_sweep_B2_mooney_rivlin.json'),
    ('B2', 'arruda_boyce', f'{R}/no_accuracy_degradation_sweep_B2_arruda_boyce.json'),
]

# Metric pairs: (NO's own field name, torch-fem's field name in run_qoi_study's
# row, human label) -- identical convention to the already-published
# B1xNeo-Hookean crossover, so results are directly comparable. Reaction
# resultant is B1-only (see run_qoi_study's own fix above), added per-case
# below only when both sides actually have it.
METRIC_PAIRS_COMMON = [
    ('L2_rel', 'l2_rel', 'L2'),
    ('H1_semi_rel', 'h1_semi_rel', 'H1 semi-norm'),
    ('energy_rel', 'energy_norm_rel', 'tangent energy'),
    ('P_peak_rel_err', 'peak_stress_rel_err', 'peak PK1 stress'),
]
METRIC_PAIR_REACTION = ('reaction_resultant_rel_err', 'reaction_resultant_rel_err', 'reaction resultant')

all_case_results = {}
for geometry, material, no_json in CASES:
    case_id = f'{geometry}_{material}'
    print('\n' + '#' * 78)
    print(f'# {case_id}')
    print('#' * 78)

    out_json = f'{R}/torchfem_full_qoi_low_N_{case_id}.json'
    # fine_N=201, not the function's own fine_N=2236 default -- see the
    # cost note at the top of this file for why 201 is already well past
    # this project's own 4x-safety-margin rule for a LOW_N sweep topping
    # out at 49, at a small fraction of 2236's cost.
    fem_rows = run_qoi_study(LOW_N, out_json, geometry=geometry, material=material,
                              checkpoint_dir=CHECKPOINT_DIR, fine_N=201)

    if not os.path.exists(no_json):
        print(f'  *** {case_id}: NO accuracy sweep not found at {no_json} -- '
              f'skipping crossover for this case, torch-fem QoI sweep is still saved. ***')
        all_case_results[case_id] = {'error': f'missing {no_json}'}
        continue

    with open(no_json) as f:
        no_data = json.load(f)
    no_rows = {r['N']: r for r in no_data['rows']}
    if 1401 not in no_rows:
        print(f'  *** {case_id}: N=1401 not present in {no_json} -- skipping crossover. ***')
        all_case_results[case_id] = {'error': 'N=1401 missing from NO sweep'}
        continue

    metric_pairs = list(METRIC_PAIRS_COMMON)
    if geometry == 'B1':
        metric_pairs.append(METRIC_PAIR_REACTION)

    no_row = no_rows[1401]
    print(f'\n  NO at N=1401 ({case_id}):')
    per_metric_crossover = {}
    for no_key, fem_key, label in metric_pairs:
        no_val = no_row.get(no_key)
        if no_val is None:
            continue
        match = next((fr for fr in sorted(fem_rows, key=lambda r: r['N'])
                      if fr.get(fem_key) is not None and fr[fem_key] <= no_val), None)
        if match:
            per_metric_crossover[label] = match['N']
            print(f'    {label:<18} NO={no_val:.3e}  ->  torch-fem matches at N={match["N"]}')
        else:
            print(f'    {label:<18} NO={no_val:.3e}  ->  no torch-fem point in {LOW_N} is coarse enough')

    if per_metric_crossover:
        coarsest_suitable = max(per_metric_crossover.values())
        binding_metric = [k for k, v in per_metric_crossover.items() if v == coarsest_suitable]
        print(f'  => COARSEST SUITABLE FEM for {case_id}@N=1401: N={coarsest_suitable} '
              f'(binding metric: {", ".join(binding_metric)})')
        all_case_results[case_id] = {
            'per_metric_crossover': per_metric_crossover,
            'coarsest_suitable_N': coarsest_suitable,
            'binding_metric': binding_metric,
        }
    else:
        print(f'  => no crossover found for {case_id} within N in {LOW_N}')
        all_case_results[case_id] = {'per_metric_crossover': {}, 'coarsest_suitable_N': None}

OUT_SUMMARY = f'{R}/qoi_crossover_remaining_cases_summary.json'
with open(OUT_SUMMARY, 'w') as f:
    json.dump({'low_N': LOW_N, 'cases': all_case_results}, f, indent=2)

try:
    from omar_pfem.run_manifest import write_manifest
    write_manifest(R, kind='qoi_crossover_remaining_cases',
                    args={'low_N': LOW_N, 'cases': [f'{g}_{m}' for g, m, _ in CASES]},
                    started_at=_started, results=all_case_results, outputs=[OUT_SUMMARY],
                    notes='Timon round-11 point 2, the 5 remaining cases beyond '
                          'B1xNeo-Hookean (already answered in the Report, Table 18-R10e).')
except Exception as e:
    print(f'[manifest] not recorded: {e}')

print('\n' + '=' * 78)
print('SUMMARY -- coarsest suitable FEM per case, and which QoI binds it')
print('=' * 78)
for case_id, r in all_case_results.items():
    if r.get('error'):
        print(f'  {case_id:<20} SKIPPED: {r["error"]}')
        continue
    if r.get('coarsest_suitable_N') is None:
        print(f'  {case_id:<20} no crossover found within N in {LOW_N}')
        continue
    print(f'  {case_id:<20} coarsest suitable N={r["coarsest_suitable_N"]:<5} '
          f'binding: {", ".join(r["binding_metric"])}')

print('\nSaved:', OUT_SUMMARY)
