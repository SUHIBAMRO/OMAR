# =====================================================================
#  CELL -- the NO's own peak-stress error, defined the SAME way torch-fem's
#  is (fixed physical location + value from a fine reference), not the
#  coarse-mesh's own sample-max -- fixes a real metric-definition mismatch
#  found 2026-09-13.
#
#  WHY THIS EXISTS: the multi-QoI crossover notebook compared the NO's own
#  P_peak_rel_err (from evaluate_no_accuracy_at_n1401 -- max stress over
#  the COARSE mesh's OWN gauss points, both predicted and "exact", so it
#  is limited by how many points that mesh even has) directly against
#  torch-fem's peak_stress_rel_err (compute_peak_stress_error -- a FIXED
#  physical location x_star and value peak_ref, located ONCE from a much
#  finer reference, exactly the target a real engineering "peak stress"
#  QoI should be). These are NOT the same quantity despite the shared
#  name, so comparing them directly (as the crossover did) was comparing
#  two different things.
#
#  This computes the NO's peak-stress error the way torch-fem's is
#  computed: find_fine_peak_stress locates x_star/peak_ref ONCE from a
#  fine ground truth (N=1401, solved via solve_b1_fast_gpu -- the same
#  already-verified fast path everything else here uses), then
#  compute_peak_stress_error (the EXACT SAME function torch-fem's sweep
#  calls) evaluates the NO's own prediction at that fixed point, at every
#  resolution.
#
#  NOT a full unification with torch-fem's own number: this uses
#  ParametricFieldB1 (matching every other NO-accuracy result), while
#  torch-fem's sweep used AnalyticFieldB1 -- the two numbers describe the
#  SAME KIND of metric on two different (but analogous) problems, not an
#  identical one. A full unification would need torch-fem re-solved
#  against the same ParametricFieldB1 realization -- a bigger follow-up,
#  not done here.
#
#  COST: cheap. CPU-smoke-tested locally (N=5,7,9 against a tiny
#  fine_N_for_peak=21, random-init model) before this cell was written --
#  confirmed no crashes and correct resume-skip behavior. The one new
#  expensive-looking step (solving N=1401 for peak-location) is the SAME
#  ground-truth solve `Round6_NO_Accuracy_Degradation_Sweep.ipynb`
#  already did -- fast with the assembled+direct backend.
#
#  RESUMABLE: skips any N already present in the output JSON.
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
    raise RuntimeError(
        "cuDSS is NOT available after installing nvmath-python[cu12] -- the ground-truth "
        "solve below would silently fall back to an iterative solver, not the direct one "
        "this notebook is meant to use. Check the pip install output above for the real error.")
print('cuDSS (real direct solver on CUDA) is available.')

R = '/content/drive/MyDrive/pfem_run'
from omar_pfem.resolve_b1_checkpoint import resolve_b1_neo_hookean_checkpoint
CKPT, _ckpt_fp = resolve_b1_neo_hookean_checkpoint(R)
print(f'Resolved checkpoint (verified by fingerprint): {CKPT}')

device = torch.device('cuda')

from omar_pfem.measure_inference_latency import build_model
from omar_pfem.no_accuracy_at_n1401 import run_no_peak_stress_fixed_location
import argparse

args = argparse.Namespace(
    model='Transolver_Irregular_Mesh', n_hidden=256, n_layers=4, n_heads=8,
    mlp_ratio=2, dropout=0.1, unified_pos=0, ref=16, slice_num=128, fun_dim=4,
    use_soft_dirichlet=1, Lx=1.0, Ly=1.0, R_out=2.0,
)
model = build_model(args, device).to(torch.float32)
model.load_state_dict(torch.load(CKPT, map_location=device))
print('Checkpoint loaded, cast to float32.')

RESOLUTIONS = [13, 17, 21, 25, 29, 33, 37, 41, 45, 49, 101, 201, 401, 701, 1001, 1401]

OUT_JSON = f'{R}/no_peak_stress_fixed_location.json'
rows = run_no_peak_stress_fixed_location(model, args, RESOLUTIONS, OUT_JSON, device,
                                          fine_N_for_peak=1401)

print('\n' + '=' * 70)
print('RESULT -- NO peak-stress error (fixed location, matching torch-fem\'s own definition)')
print('=' * 70)
for r in rows:
    print(f"  N={r['N']:<6} peak_stress_rel_err={r['peak_stress_rel_err']:.3e}")
print(json.dumps(rows, indent=2))

# ---- Re-run the multi-QoI crossover with this corrected peak-stress metric ----
FEM_JSON = f'{REPO}/Practical_Examples/omar_pfem/torchfem_full_qoi_low_N_result.json'
NO_JSON = f'{REPO}/Practical_Examples/omar_pfem/no_accuracy_degradation_sweep_B1_neo_hookean.json'
if os.path.exists(FEM_JSON) and os.path.exists(NO_JSON):
    with open(FEM_JSON) as f:
        fem_rows = sorted(json.load(f)['rows'], key=lambda r: r['N'])
    with open(NO_JSON) as f:
        no_rows = {r['N']: r for r in json.load(f)['rows']}
    no_peak = {r['N']: r['peak_stress_rel_err'] for r in rows}

    METRIC_PAIRS = [
        ('L2_rel', 'l2_rel', 'L2'),
        ('H1_semi_rel', 'h1_semi_rel', 'H1 semi-norm'),
        ('energy_rel', 'energy_norm_rel', 'tangent energy'),
        ('reaction_resultant_rel_err', 'reaction_resultant_rel_err', 'reaction resultant'),
    ]
    print('\n' + '=' * 78)
    print('CORRECTED MULTI-QoI CROSSOVER (peak stress now fixed-location, both sides)')
    print('=' * 78)
    for N in sorted(no_rows):
        if N not in no_peak:
            continue
        print(f"\nNO at N={N}:")
        per_metric_crossover = {}
        for no_key, fem_key, label in METRIC_PAIRS:
            no_val = no_rows[N]['fp32'].get(no_key)
            match = next((fr for fr in fem_rows if fr.get(fem_key) is not None
                          and fr[fem_key] <= no_val), None)
            if match:
                per_metric_crossover[label] = match['N']
                print(f"    {label:<18} NO={no_val:.3e}  ->  torch-fem matches at N={match['N']}")
        no_peak_val = no_peak[N]
        match = next((fr for fr in fem_rows if fr.get('peak_stress_rel_err') is not None
                      and fr['peak_stress_rel_err'] <= no_peak_val), None)
        if match:
            per_metric_crossover['peak PK1 stress'] = match['N']
            print(f"    {'peak PK1 stress':<18} NO={no_peak_val:.3e}  ->  torch-fem matches at N={match['N']}")
        else:
            print(f"    {'peak PK1 stress':<18} NO={no_peak_val:.3e}  ->  torch-fem needs N > "
                  f"{fem_rows[-1]['N']} (slow-converging, possibly near a domain-corner "
                  f"singularity -- see PROJECT_STATUS.md)")
        if per_metric_crossover:
            coarsest_suitable = max(per_metric_crossover.values())
            binding = [k for k, v in per_metric_crossover.items() if v == coarsest_suitable]
            print(f"  => COARSEST SUITABLE FEM for NO@N={N}: N={coarsest_suitable} "
                  f"(binding metric: {', '.join(binding)})")
else:
    print(f"\n(FEM full-QoI sweep or NO accuracy sweep not found locally -- pull "
          f"{FEM_JSON} and {NO_JSON} from Drive/commit them first for the crossover table)")

print('\nSaved:', OUT_JSON)
