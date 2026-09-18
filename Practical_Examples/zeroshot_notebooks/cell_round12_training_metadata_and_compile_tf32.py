# =====================================================================
#  CELL -- Timon round-12 points 2 and 3, all six cases.
#
#  POINT 2 (the easy half only -- the numbers already exist, this just
#  gathers and reports them): a table of every training run's own
#  resolutions, sample counts, epochs/steps, wall-clock, and cost per
#  sample/step, read from each case's own metrics_history.json already
#  saved on Drive (nothing retrained here). Peak GPU memory during
#  TRAINING was never instrumented for any of these runs -- honestly
#  reported as "not measured" rather than guessed. Timon's own further
#  suggestion in the same point (a controlled re-run matching sample
#  counts/budget between the direct-N1401 and multi-res recipes) is
#  EXPLICITLY skipped here per his own words ("I think we should not
#  waste time on the fine resolution training here but only for a
#  complex geometry problem where resolution might matter") -- this
#  cell only reports what already exists.
#
#  POINT 3: use the compile+TF32 optimized inference number for the
#  paper. Already measured for B1xNeo-Hookean (the flagship case);
#  profile_with_torch_compile itself was B1-only until now, so this
#  extends the same measurement (profile_with_torch_compile_b2, new
#  this round) to the other five cases, using each case's own FINAL
#  checkpoint, so every case has a real compile+TF32 number instead of
#  only the flagship.
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

for _mod_name in list(sys.modules):
    if _mod_name == 'omar_pfem' or _mod_name.startswith('omar_pfem.'):
        del sys.modules[_mod_name]

import torch
assert torch.cuda.is_available(), 'this cell needs a real GPU'
print('GPU:', torch.cuda.get_device_name(0))

R = '/content/drive/MyDrive/pfem_run'
device = torch.device('cuda')

# =====================================================================
# POINT 2: training-cost summary table
# =====================================================================
print('\n' + '#' * 78)
print('# POINT 2: training-cost summary (reading existing metrics_history.json files)')
print('#' * 78)

# (case_id, metrics_history path, n_train_per_res, resolutions, n_samples_total)
# n_train_per_res=400/resolution x 4 resolutions for every multi-res case (the
# protocol documented in the Report); the direct-N1401 ablation used 100
# samples at N=1401 alone. B2xArruda-Boyce has no retrain at all -- its own
# ORIGINAL (fixed-selection) run is reported instead, same "resolutions,
# samples" convention, so the table is honest about what that case actually
# is (a single-protocol run, not a multi-res one).
TRAINING_RUNS = [
    ('B1_neo_hookean (multi-res)', f'{R}/zeroshot_B1_neo_hookean_multires/metrics_history.json',
     [21, 33, 101, 201], 400),
    ('B1_neo_hookean (direct-N1401 ablation)',
     f'{R}/zeroshot_B1_neo_hookean_direct_n1401/metrics_history.json', [1401], 100),
    ('B1_mooney_rivlin (multi-res)', f'{R}/zeroshot_B1_mooney_rivlin_multires/metrics_history.json',
     [21, 33, 101, 201], 400),
    ('B1_arruda_boyce (multi-res)', f'{R}/zeroshot_B1_arruda_boyce_multires/metrics_history.json',
     [21, 33, 101, 201], 400),
    ('B2_neo_hookean (multi-res)', f'{R}/zeroshot_B2_neo_hookean_multires/metrics_history.json',
     [21, 33, 101, 201], 400),
    ('B2_mooney_rivlin (multi-res)', f'{R}/zeroshot_B2_mooney_rivlin_multires/metrics_history.json',
     [21, 33, 101, 201], 400),
    ('B2_arruda_boyce (original, no retrain)', f'{R}/zeroshot_B2_arruda_boyce_fixedsel/metrics_history.json',
     [21, 33], 400),
]

training_summary = []
for label, path, resolutions, n_per_res in TRAINING_RUNS:
    if not os.path.exists(path):
        print(f'  *** {label}: {path} NOT FOUND on Drive -- skipped, not guessed. ***')
        training_summary.append({'case': label, 'found': False})
        continue
    with open(path) as f:
        history = json.load(f)
    last = history[-1]
    n_total_samples = n_per_res * len(resolutions)
    wall_s = last['cumulative_wall_clock_s']
    row = {
        'case': label, 'found': True,
        'resolutions': resolutions, 'n_samples_per_res': n_per_res,
        'n_samples_total': n_total_samples,
        'final_epoch': last['epoch'], 'opt_steps': last['opt_steps'],
        'wall_clock_s': wall_s, 'wall_clock_h': wall_s / 3600.0,
        'cost_per_sample_s': wall_s / n_total_samples,
        'cost_per_opt_step_s': wall_s / last['opt_steps'],
        'peak_gpu_memory': 'not instrumented during training -- not guessed',
    }
    training_summary.append(row)
    print(f"  {label}: resolutions={resolutions} samples/res={n_per_res} "
          f"(total={n_total_samples}) epoch={row['final_epoch']} "
          f"opt_steps={row['opt_steps']} wall_clock={row['wall_clock_h']:.2f}h "
          f"cost/sample={row['cost_per_sample_s']:.2f}s cost/step={row['cost_per_opt_step_s']:.4f}s")

OUT_TRAINING = f'{R}/round12_training_cost_summary.json'
with open(OUT_TRAINING, 'w') as f:
    json.dump({'runs': training_summary}, f, indent=2)
print('Saved:', OUT_TRAINING)

# =====================================================================
# POINT 3: compile+TF32 inference for all six cases
# =====================================================================
print('\n' + '#' * 78)
print('# POINT 3: compile+TF32 inference timing, all six cases (final checkpoints)')
print('#' * 78)

from omar_pfem.measure_inference_latency import build_model
from omar_pfem.resolution_invariance_zeroshot import build_sample_b1, build_sample_b2
from omar_pfem.no_inference_torch_compile import (
    profile_with_torch_compile, profile_with_torch_compile_b2)
from omar_pfem.gpu_memory_monitor import GPUMemoryMonitor
import argparse

CHECKPOINTS = [
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
N_TEST = 1401

inference_summary = []
gpu_monitor = GPUMemoryMonitor(device, interval_s=0.5)
gpu_monitor.__enter__()
for geometry, material, ckpt_path in CHECKPOINTS:
    case_id = f'{geometry}_{material}'
    gpu_monitor.mark(case_id)
    print(f'\n--- {case_id} ---')
    if not os.path.exists(ckpt_path):
        print(f'  *** checkpoint not found: {ckpt_path} -- skipped ***')
        inference_summary.append({'case': case_id, 'found': False})
        continue

    args = argparse.Namespace(**BASE_ARGS)
    model = build_model(args, device)
    state_dict = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    if geometry == 'B1':
        sample, _ = build_sample_b1(N_TEST, seed=0, material=material, Lx=args.Lx, Ly=args.Ly,
                                     solve_fem=False)
        result = profile_with_torch_compile(sample, model, args, device, torch.float32,
                                             n_repeats=200, n_warmup=20, compile_warmup=5,
                                             try_tf32=True)
    else:
        sample, _ = build_sample_b2(N_TEST, seed=0, material=material, R_in=args.R_in,
                                     R_out=args.R_out, solve_fem=False)
        result = profile_with_torch_compile_b2(sample, model, args, device, torch.float32,
                                                n_repeats=200, n_warmup=20, compile_warmup=5,
                                                try_tf32=True)

    row = {'case': case_id, 'found': True, 'checkpoint': ckpt_path, **result}
    inference_summary.append(row)
    print(f"  eager={result['eager_ms_per_sample']:.2f}ms  "
          f"compile={result.get('compiled_ms_per_sample')}  "
          f"compile+TF32={result.get('compiled_tf32_ms_per_sample')}")

    out_json_case = f'{R}/round12_compile_tf32_{case_id}.json'
    with open(out_json_case, 'w') as f:
        json.dump(row, f, indent=2)

gpu_monitor.__exit__(None, None, None)
GPU_MEM_FIG = f'{R}/round12_compile_tf32_gpu_memory.png'
gpu_monitor.save_plot(GPU_MEM_FIG, title='GPU memory over time -- compile+TF32 profiling, all 6 cases')

OUT_INFERENCE = f'{R}/round12_compile_tf32_summary.json'
with open(OUT_INFERENCE, 'w') as f:
    json.dump({'runs': inference_summary, 'gpu_memory_peak_mb': gpu_monitor.peak_mb(),
               'gpu_memory_device_total_mb': gpu_monitor.device_total_mb}, f, indent=2)
print('\nSaved:', OUT_INFERENCE)

try:
    from omar_pfem.run_manifest import write_manifest
    write_manifest(R, kind='round12_training_metadata_and_compile_tf32', args={'N_TEST': N_TEST},
                    started_at=_started,
                    results={'training_runs': len(training_summary), 'inference_runs': len(inference_summary)},
                    outputs=[OUT_TRAINING, OUT_INFERENCE, GPU_MEM_FIG],
                    notes="Timon round-12 points 2 (training-cost summary, existing data only, "
                          "NOT the controlled re-run he himself deprioritized) and 3 "
                          "(compile+TF32 inference, extended from B1xNeo-Hookean-only to all six cases).")
except Exception as e:
    print(f'[manifest] not recorded: {e}')

print('\nDone.')
