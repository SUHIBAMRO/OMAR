# =====================================================================
#  CELL -- Timon round-12 point 1: "one final accuracy-versus-resolution
#  table using the final retrained checkpoints only. For each
#  resolution, please compare FEM and NO against the same fine
#  reference in displacement L2, H1/energy norm, reaction force and
#  stress" -- with Cauchy stress (not PK1) as the main engineering
#  quantity, avoiding a bare pointwise maximum (mesh-dependent/singular
#  at corners even for FEM) in favor of a FIXED physical region around
#  the stress concentration, reporting a robust local statistic
#  (weighted average / 95th-99th percentile / mean of the top 1%) with
#  the true max still available separately.
#
#  Runs BOTH sides (torch-fem's own run_qoi_study -- reusing "ours" own
#  matrix-free solver as the FEM comparison, exactly as every other QoI
#  table in this project already does -- and the NO's own forward pass
#  via the new run_no_region_cauchy_fixed_location[_b2]) against the
#  SAME fine reference and the SAME fixed region for all six
#  (geometry, material) cases, using each case's own FINAL checkpoint
#  (the retrained multi-res one where it exists, the original where it
#  does not -- B2xArruda-Boyce only ever had one).
#
#  SCOPE, stated explicitly rather than silently assumed: resolutions
#  are the LOW_N range [3..49] already established and cost-verified
#  earlier this week (fine_N=201 is comfortably past this project's own
#  4x-safety-margin rule for a sweep topping out at 49). Deliberately
#  does NOT extend to N=1401 here -- that would need a fresh fine_N=2236
#  reference for the FIVE cases that have never had one (the exact
#  "~150-240h of unnecessary GPU time" mistake caught and fixed earlier
#  this week for a different notebook), which is a much bigger, separate
#  ask that should be scoped and cost-estimated on its own before
#  running, not silently bundled into this cell.
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
from omar_pfem.no_accuracy_at_n1401 import (
    run_no_region_cauchy_fixed_location, run_no_region_cauchy_fixed_location_b2)
from omar_pfem.measure_inference_latency import build_model
from omar_pfem.gpu_memory_monitor import GPUMemoryMonitor
import argparse

R = '/content/drive/MyDrive/pfem_run'
CHECKPOINT_DIR = '/content/drive/MyDrive/pfem_ckpt'
device = torch.device('cuda')

LOW_N = [3, 4, 5, 6, 9, 11, 13, 17, 21, 25, 29, 33, 37, 41, 45, 49]
FINE_N = 201  # see the scope note above -- comfortably safe and cheap for LOW_N up to 49

# (geometry, material, FINAL checkpoint dir on Drive -- retrained where it
# exists, per PROJECT_STATUS.md's own record of every retrain this week)
CASES = [
    ('B1', 'neo_hookean', f'{R}/zeroshot_B1_neo_hookean_multires/model_best.pt'),
    ('B1', 'mooney_rivlin', f'{R}/zeroshot_B1_mooney_rivlin_multires/model_best.pt'),
    ('B1', 'arruda_boyce', f'{R}/zeroshot_B1_arruda_boyce_multires/model_best.pt'),
    ('B2', 'neo_hookean', f'{R}/zeroshot_B2_neo_hookean_multires/model_best.pt'),
    ('B2', 'mooney_rivlin', f'{R}/zeroshot_B2_mooney_rivlin_multires/model_best.pt'),
    ('B2', 'arruda_boyce', f'{R}/zeroshot_B2_arruda_boyce_fixedsel/model_best.pt'),
]

BASE_ARGS = dict(model='Transolver_Irregular_Mesh', n_hidden=256, n_layers=4, n_heads=8,
                  mlp_ratio=2, dropout=0.1, unified_pos=0, ref=16, slice_num=128, fun_dim=4,
                  use_soft_dirichlet=1, Lx=1.0, Ly=1.0, R_in=1.0, R_out=2.0)

all_results = {}
gpu_monitor = GPUMemoryMonitor(device, interval_s=1.0)
gpu_monitor.__enter__()
for geometry, material, ckpt_path in CASES:
    case_id = f'{geometry}_{material}'
    gpu_monitor.mark(case_id)
    print('\n' + '#' * 78)
    print(f'# {case_id}  (checkpoint: {ckpt_path})')
    print('#' * 78)
    assert os.path.exists(ckpt_path), f'{case_id}: checkpoint not found at {ckpt_path}'

    args = argparse.Namespace(**BASE_ARGS)
    model = build_model(args, device)
    state_dict = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    print('Checkpoint loaded.')

    # ---- FEM side: L2/H1/energy/reaction/PK1-peak/region-Cauchy, "ours" solver ----
    fem_json = f'{R}/round12_fem_qoi_{case_id}.json'
    fem_rows = run_qoi_study(LOW_N, fem_json, geometry=geometry, material=material,
                              checkpoint_dir=CHECKPOINT_DIR, fine_N=FINE_N)

    # ---- NO side: region-Cauchy (new) ----
    no_cauchy_json = f'{R}/round12_no_region_cauchy_{case_id}.json'
    fn = run_no_region_cauchy_fixed_location if geometry == 'B1' else run_no_region_cauchy_fixed_location_b2
    no_cauchy_rows = fn(model, args, LOW_N, no_cauchy_json, device, material=material,
                         fine_N_for_peak=FINE_N)

    # ---- NO side: L2/H1/energy/reaction -- reuse the already-existing accuracy
    # sweep if present on Drive (task #22), else this case simply has no
    # pre-existing L2/H1/energy/reaction row set to merge against; region-Cauchy
    # above is always computed fresh regardless. ----
    no_acc_json = f'{R}/no_accuracy_degradation_sweep_{case_id}.json'
    no_acc_by_N = {}
    if os.path.exists(no_acc_json):
        with open(no_acc_json) as f:
            no_acc_by_N = {r['N']: r['fp32'] for r in json.load(f)['rows']}
        print(f'  Loaded existing NO L2/H1/energy/reaction sweep from {no_acc_json} '
              f'({len(no_acc_by_N)} rows) -- reused unchanged, not recomputed.')
    else:
        print(f'  *** {no_acc_json} not found -- this case\'s combined table will have '
              f'region-Cauchy only for the NO side, no L2/H1/energy/reaction. ***')

    combined = []
    fem_by_N = {r['N']: r for r in fem_rows}
    no_cauchy_by_N = {r['N']: r for r in no_cauchy_rows}
    for N in LOW_N:
        fem_r = fem_by_N.get(N, {})
        no_c = no_cauchy_by_N.get(N, {})
        no_a = no_acc_by_N.get(N, {})
        combined.append({
            'N': N,
            'fem_l2_rel': fem_r.get('l2_rel'), 'no_l2_rel': no_a.get('L2_rel'),
            'fem_h1_semi_rel': fem_r.get('h1_semi_rel'), 'no_h1_semi_rel': no_a.get('H1_semi_rel'),
            'fem_energy_rel': fem_r.get('energy_norm_rel'), 'no_energy_rel': no_a.get('energy_rel'),
            'fem_reaction_rel_err': fem_r.get('reaction_resultant_rel_err'),
            'no_reaction_rel_err': no_a.get('reaction_resultant_rel_err'),
            'fem_cauchy_avg_rel_err': fem_r.get('region_cauchy_avg_rel_err'),
            'no_cauchy_avg_rel_err': no_c.get('region_cauchy_avg_rel_err'),
            'fem_cauchy_p99_rel_err': fem_r.get('region_cauchy_p99_rel_err'),
            'no_cauchy_p99_rel_err': no_c.get('region_cauchy_p99_rel_err'),
            'fem_cauchy_top1pct_rel_err': fem_r.get('region_cauchy_top1pct_rel_err'),
            'no_cauchy_top1pct_rel_err': no_c.get('region_cauchy_top1pct_rel_err'),
            'fem_cauchy_max_rel_err': fem_r.get('region_cauchy_max_rel_err'),
            'no_cauchy_max_rel_err': no_c.get('region_cauchy_max_rel_err'),
        })
        print(f"  N={N}: FEM/NO L2={fem_r.get('l2_rel')}/{no_a.get('L2_rel')}  "
              f"cauchy_avg={fem_r.get('region_cauchy_avg_rel_err')}/{no_c.get('region_cauchy_avg_rel_err')}  "
              f"cauchy_p99={fem_r.get('region_cauchy_p99_rel_err')}/{no_c.get('region_cauchy_p99_rel_err')}")

    all_results[case_id] = combined
    out_combined = f'{R}/round12_final_accuracy_cauchy_{case_id}.json'
    with open(out_combined, 'w') as f:
        json.dump({'geometry': geometry, 'material': material, 'checkpoint': ckpt_path,
                   'fine_N': FINE_N, 'rows': combined}, f, indent=2)
    print(f'Saved: {out_combined}')

gpu_monitor.__exit__(None, None, None)
GPU_MEM_FIG = f'{R}/round12_final_accuracy_cauchy_gpu_memory.png'
gpu_monitor.save_plot(GPU_MEM_FIG, title='GPU memory over time -- Round 12 final accuracy + Cauchy sweep, all 6 cases')

OUT_SUMMARY = f'{R}/round12_final_accuracy_cauchy_summary.json'
with open(OUT_SUMMARY, 'w') as f:
    json.dump({'low_N': LOW_N, 'fine_N': FINE_N, 'cases': all_results,
               'gpu_memory_peak_mb': gpu_monitor.peak_mb(),
               'gpu_memory_device_total_mb': gpu_monitor.device_total_mb}, f, indent=2)

try:
    from omar_pfem.run_manifest import write_manifest
    write_manifest(R, kind='round12_final_accuracy_cauchy', args={'low_N': LOW_N, 'fine_N': FINE_N},
                    started_at=_started, results={'cases': list(all_results)},
                    outputs=[OUT_SUMMARY, GPU_MEM_FIG],
                    notes="Timon round-12 point 1: final accuracy-vs-resolution table, FEM+NO, "
                          "using each case's own final (retrained where applicable) checkpoint, "
                          "with the new fixed-region Cauchy-stress QoI replacing a bare pointwise "
                          "maximum.")
except Exception as e:
    print(f'[manifest] not recorded: {e}')

print('\nAll six cases done. Saved:', OUT_SUMMARY)
