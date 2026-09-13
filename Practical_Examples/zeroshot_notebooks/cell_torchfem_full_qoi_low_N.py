# =====================================================================
#  CELL -- torch-fem's FULL QoI set (L2, H1, energy, peak stress,
#  per-component stress, reaction resultant) at the SAME low N the NO
#  was tested at (Timon round-10, item 1, refined 2026-09-13).
#
#  WHY THIS EXISTS: the crossover found so far (torch-fem N=3-4 already
#  matches the NO's best L2_rel) used ONLY the L2 norm. A second-opinion
#  review made the correct point: L2 alone doesn't prove FEM is a
#  "suitable" replacement in EVERY sense Timon cares about -- a coarse
#  FEM mesh could match displacement (L2) while still being far worse at
#  gradients (H1), energy, peak stress, or reaction forces. The
#  "coarsest suitable FEM" needs to match or beat the NO in ALL of
#  these simultaneously, not just L2.
#
#  This reuses `run_qoi_study` (torchfem_comparison.py) UNCHANGED --
#  built 2026-09-10 for Timon round-9 item 9 ("What about all QoIs,
#  particularly for large DOFs?") and already used at N=1001/1401. No
#  new solver code: same fine ~10M-DOF reference (RESUMED from the
#  already-converged checkpoint, not re-solved), same
#  compute_l2_h1_errors / compute_tangent_energy_error /
#  compute_peak_stress_error / pk1_component_errors_at_point /
#  compute_reaction_resultant_error machinery already validated for the
#  high-N FEM-vs-FEM tables. Only NEW here: pointing it at the LOW N
#  range (3-49) instead of 1001/1401.
#
#  COST: cheap -- every N here is smaller than N=51, which the earlier
#  low-N sweep measured at 3.42s for the simpler L2/H1-only version;
#  the extra QoIs (energy Hessian-vector product, one peak-stress
#  location + a few point evaluations, one reaction assembly) add only
#  a small constant per N, not a new expensive solve.
#
#  RESUMABLE: run_qoi_study skips any N already present in its output.
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

run([sys.executable, '-m', 'pip', 'install', '-q', 'torch-fem'])

WORK = f'{REPO}/Practical_Examples'
os.chdir(WORK)
sys.path.insert(0, WORK)

for _mod_name in list(sys.modules):
    if _mod_name == 'omar_pfem' or _mod_name.startswith('omar_pfem.'):
        del sys.modules[_mod_name]

import torch
print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available()
      else 'NONE -- Runtime > Change runtime type > GPU required')

R = '/content/drive/MyDrive/pfem_run'
OUT_JSON = f'{R}/torchfem_full_qoi_low_N.json'
os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)

CHECKPOINT_DIR = '/content/drive/MyDrive/pfem_ckpt'

# Same resolutions the NO's own accuracy sweep used (13-49) plus the
# coarser points (3-11) that already beat the NO on L2 alone -- need the
# full QoI set at those too, to see whether they hold up everywhere.
RESOLUTIONS = [3, 4, 5, 6, 9, 11, 13, 17, 21, 25, 29, 33, 37, 41, 45, 49]

from omar_pfem.torchfem_comparison import run_qoi_study
rows = run_qoi_study(RESOLUTIONS, OUT_JSON, checkpoint_dir=CHECKPOINT_DIR,
                      fine_N=2236, tol=1e-8)

print('\nDone. Results:', OUT_JSON)

# ---- Multi-QoI crossover against the NO's own already-committed sweep ----
NO_JSON = f'{REPO}/Practical_Examples/omar_pfem/no_accuracy_degradation_sweep_B1_neo_hookean.json'
if os.path.exists(NO_JSON):
    with open(NO_JSON) as f:
        no_rows = {r['N']: r for r in json.load(f)['rows']}
    fem_rows = sorted(rows, key=lambda r: r['N'])

    # Metric pairs: (NO's own field name in its 'fp32' dict, FEM's field
    # name in run_qoi_study's row) -- both already the SAME definition
    # (relative error against a fine/exact reference), so directly
    # comparable without any rescaling.
    METRIC_PAIRS = [
        ('L2_rel', 'l2_rel', 'L2'),
        ('H1_semi_rel', 'h1_semi_rel', 'H1 semi-norm'),
        ('energy_rel', 'energy_norm_rel', 'tangent energy'),
        ('P_peak_rel_err', 'peak_stress_rel_err', 'peak PK1 stress'),
        ('reaction_resultant_rel_err', 'reaction_resultant_rel_err', 'reaction resultant'),
    ]

    print('\n' + '=' * 78)
    print('MULTI-QoI CROSSOVER -- coarsest torch-fem N matching the NO at EACH metric,')
    print('then the coarsest N that satisfies ALL of them at once')
    print('=' * 78)
    for N in sorted(no_rows):
        no_row = no_rows[N]
        print(f"\nNO at N={N}:")
        per_metric_crossover = {}
        for no_key, fem_key, label in METRIC_PAIRS:
            no_val = no_row['fp32'].get(no_key)
            if no_val is None:
                continue
            match = next((fr for fr in fem_rows if fr.get(fem_key) is not None
                          and fr[fem_key] <= no_val), None)
            if match:
                per_metric_crossover[label] = match['N']
                print(f"    {label:<18} NO={no_val:.3e}  ->  torch-fem matches at N={match['N']}")
            else:
                print(f"    {label:<18} NO={no_val:.3e}  ->  NO torch-fem point here is coarse "
                      f"enough (need N < {fem_rows[0]['N']})")
        if per_metric_crossover:
            coarsest_suitable = max(per_metric_crossover.values())
            binding_metric = [k for k, v in per_metric_crossover.items() if v == coarsest_suitable]
            print(f"  => COARSEST SUITABLE FEM for NO@N={N}: N={coarsest_suitable} "
                  f"(binding metric: {', '.join(binding_metric)})")
else:
    print(f"\nNO comparison file not found at {NO_JSON} -- printing torch-fem's own rows only:")
    for row in rows:
        print(row)
