# =====================================================================
#  CELL -- B3 (rocking rubber-mount bushing, groove feature) mesh
#  convergence, extended to GPU resolutions well beyond what was
#  CPU-tested locally (up to 3,240 elements). Omar's own explicit
#  instruction (2026-09-21): do not designate a final fine reference,
#  or trust the 5%/2%/1% required-resolution table, until the fixed-
#  region Cauchy-stress QoI stops changing meaningfully between the
#  LAST few resolutions tested -- the CPU run showed it still rising
#  at 3,240 elements (1.56 -> 2.30 -> 2.53 -> 2.75), so this extends
#  the ladder far past that point.
#
#  Reuses omar_pfem.data.mesh_convergence_B3's own already-verified
#  solve_case/compare_to_reference/scalar_qoi_rel_errors/
#  find_required_resolutions/print_threshold_table functions UNCHANGED
#  -- this cell only extends the resolution ladder and moves execution
#  to GPU (device='cuda'), nothing about the physics/geometry/BCs/QoI
#  definitions changes.
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

from omar_pfem.data.mesh_convergence_B3 import (
    solve_case, compare_to_reference, scalar_qoi_rel_errors,
    find_required_resolutions, print_threshold_table)

device = torch.device('cuda')

R = '/content/drive/MyDrive/pfem_run'
os.makedirs(f'{R}/b3', exist_ok=True)

# Extended ladder: starts where the CPU study left off (600 elements)
# and goes to 243,360 for the fine reference -- large enough that if
# region-Cauchy-stress genuinely plateaus somewhere in this range, we
# will see it; if it does NOT plateau even here, that itself is a real,
# important finding to report back (not silently ignored).
RESOLUTIONS = [(13, 6, 11), (21, 10, 19), (29, 14, 27), (37, 18, 35),
               (45, 22, 43), (53, 26, 51), (65, 32, 63)]
FINE_RESOLUTION = (81, 40, 79)

print('\nSolving the fine reference first...')
ref = solve_case(*FINE_RESOLUTION, device=device, verbose=True)
print(f"  fine reference: {ref['n_elements']} elements, {ref['elapsed_s']:.2f}s")

rows = []
for Ntheta, Nr, Nz in RESOLUTIONS:
    r = solve_case(Ntheta, Nr, Nz, device=device)
    l2_rel, h1_rel = compare_to_reference(r, ref)
    r['disp_l2_rel'] = l2_rel
    r['grad_h1_rel'] = h1_rel
    rows.append(r)
    print(f"\n({Ntheta},{Nr},{Nz})  elements={r['n_elements']}  time={r['elapsed_s']:.2f}s")
    print(f"  disp_L2_rel={l2_rel*100:.3f}%  gradF_H1_rel={h1_rel*100:.3f}%")
    print(f"  region(n={r['n_region']}): avg_sigma_xx={r['region_avg_sigma_xx']:.4f}  "
          f"p99={r['region_p99_sigma_xx']:.4f}  (true_max={r['region_true_max_sigma_xx']:.4f}, secondary)")
    print(f"  equilibrium: force_rel_residual={r['force_rel_residual']:.2e}  "
          f"moment_rel_residual={r['moment_rel_residual']:.2e}")

print('\n' + '=' * 90)
print('Region-Cauchy-stress convergence check (THE thing Omar asked to verify '
      'before trusting any reference):')
for r in rows:
    print(f"  ({r['Ntheta']:>2},{r['Nr']:>2},{r['Nz']:>2})  n_elem={r['n_elements']:>6}  "
          f"region_avg_sxx={r['region_avg_sigma_xx']:8.4f}  region_p99_sxx={r['region_p99_sigma_xx']:8.4f}  "
          f"n_region={r['n_region']}")
print(f"  FINE REF ({FINE_RESOLUTION[0]},{FINE_RESOLUTION[1]},{FINE_RESOLUTION[2]})  "
      f"n_elem={ref['n_elements']:>6}  region_avg_sxx={ref['region_avg_sigma_xx']:8.4f}  "
      f"region_p99_sxx={ref['region_p99_sigma_xx']:8.4f}  n_region={ref['n_region']}")

print('\nRelative change in region_avg_sigma_xx between successive rows above '
      '(including the jump to the fine reference) -- READ THIS BEFORE trusting '
      'any threshold table below:')
all_for_trend = rows + [dict(ref, Ntheta=FINE_RESOLUTION[0], Nr=FINE_RESOLUTION[1], Nz=FINE_RESOLUTION[2])]
for a, b in zip(all_for_trend[:-1], all_for_trend[1:]):
    rel = abs(b['region_avg_sigma_xx'] - a['region_avg_sigma_xx']) / abs(a['region_avg_sigma_xx'])
    print(f"  ({a['Ntheta']},{a['Nr']},{a['Nz']}) -> ({b['Ntheta']},{b['Nr']},{b['Nz']}): {rel*100:.2f}% change")

scalar_qoi_rel_errors(rows, ref)
results = find_required_resolutions(rows)
print_threshold_table(results)

print('\nIf the region-Cauchy relative change above has NOT dropped to a few '
      'percent by the last row, this fine reference is STILL NOT sufficient '
      'for that QoI specifically, and the threshold table above should be '
      'treated as provisional for region_avg/region_p99 (the field-based '
      'disp_L2/gradF_H1 numbers, and the global energy/force/moment numbers, '
      'are much less sensitive to this and can likely be trusted already).')

report = {
    'resolutions': RESOLUTIONS, 'fine_resolution': FINE_RESOLUTION,
    'rows': [{k: v for k, v in r.items() if not k.startswith('_')} for r in rows],
    'fine_reference': {k: v for k, v in ref.items() if not k.startswith('_')},
}
out_json = f'{R}/b3/mesh_convergence_extended.json'
with open(out_json, 'w') as f:
    json.dump(report, f, indent=2)
print('\nSaved:', out_json)

try:
    from omar_pfem.run_manifest import write_manifest
    write_manifest(f'{R}/b3', kind='b3_gpu_mesh_convergence',
                    args={'resolutions': RESOLUTIONS, 'fine_resolution': FINE_RESOLUTION},
                    started_at=_started, results={'n_rows': len(rows)}, outputs=[out_json],
                    notes="Extended B3 mesh-convergence study on GPU, per Omar's own explicit "
                          "instruction (2026-09-21) to keep refining until the region-Cauchy-"
                          "stress QoI plateaus before designating a final fine reference or "
                          "trusting the 5%/2%/1% required-resolution table.")
except Exception as e:
    print(f'[manifest] not recorded: {e}')

print('\nDone.')
