# =====================================================================
#  CELL -- B3 (rocking rubber-mount bushing, groove feature) mesh
#  convergence, extended to GPU resolutions well beyond what was
#  CPU-tested locally (up to 3,240 elements).
#
#  REVISED 2026-09-21 (Omar's own 11-point technical review, before any
#  email to Timon) -- two things changed from the first GPU run this
#  cell already produced (that run reached a 243,360-element reference,
#  (81,40,79), whose own last-step relative change was ~0.87% region-avg
#  / ~1.17% region-p99 -- too close to the 1% threshold being claimed):
#    1. Uses the REWRITTEN omar_pfem.data.mesh_convergence_B3 (full
#       Cauchy-tensor field error, volume-weighted quadrature-based
#       region sampling with a reliability gate, verified-nonzero
#       reaction force with moment framed as primary, "total strain
#       energy" instead of "tangent energy") -- compare_to_reference now
#       returns THREE values (l2_rel, h1_rel, cauchy_field_rel), not two.
#    2. Adds ONE additional, FINER reference at (97,48,95) (~424,128
#       elements) specifically to check whether the region-Cauchy-stress
#       QoI's own relative change drops below ~0.5-0.7% going from the
#       old 243,360-element reference to this new one -- if so, the 1%
#       claim for that QoI is much stronger; if not, this cell reports
#       "1% not reached / provisional" explicitly rather than silently
#       keeping the weaker claim. The (97,48,95) mesh becomes the new
#       "official" fine reference for the required-resolution table
#       below (strictly more trustworthy than (81,40,79)), and the two
#       references' own region-stress values are compared directly and
#       printed BEFORE anything else, exactly as Omar asked.
#
#  Otherwise unchanged: reuses solve_case/compare_to_reference/
#  scalar_qoi_rel_errors/find_required_resolutions/print_threshold_table
#  from mesh_convergence_B3.py UNCHANGED in their own interface -- this
#  cell only extends the resolution ladder, adds the second fine
#  reference, and moves execution to GPU (device='cuda'); nothing about
#  the physics/geometry/BCs/QoI definitions changes here.
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

# Real, external, currently-active PyPI issue (2026-09-23): pyvista
# 0.49+ unconditionally imports IPython.core.guarded_eval, a module
# that only exists from IPython>=8.8 -- Colab ships IPython 7.34, so a
# fresh `pip install torch-fem` (which pulls in whatever pyvista is
# "latest" at install time, unpinned) can suddenly start failing with
# ModuleNotFoundError as soon as PyPI's latest pyvista crosses 0.49,
# even though the exact same install command worked earlier the same
# day. Pinning below 0.49 avoids the broken import path entirely.
run([sys.executable, '-m', 'pip', 'install', '-q', 'pyvista<0.49'])
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
# and goes up to the OLD 243,360-element reference resolution (kept in
# the ladder itself so its own convergence trend, already reported
# before, stays visible) -- the NEW, finer 424,128-element mesh is
# solved SEPARATELY below as the second reference, not as just another
# ladder row.
RESOLUTIONS = [(13, 6, 11), (21, 10, 19), (29, 14, 27), (37, 18, 35),
               (45, 22, 43), (53, 26, 51), (65, 32, 63)]
OLD_FINE_RESOLUTION = (81, 40, 79)      # 243,360 elements -- the earlier reference
NEW_FINE_RESOLUTION = (97, 48, 95)      # ~424,128 elements -- the new, finer reference

print('\nSolving the OLD fine reference (81,40,79, 243,360 elements)...')
ref_old = solve_case(*OLD_FINE_RESOLUTION, device=device, verbose=True)
print(f"  OLD reference: {ref_old['n_elements']} elements, {ref_old['elapsed_s']:.2f}s, "
      f"region samples (quadrature points)={ref_old['n_region']} "
      f"(p99 reliable={ref_old['region_p99_reliable']})")

print('\nSolving the NEW, finer fine reference (97,48,95, ~424,128 elements) -- '
      'this is the whole point of this GPU run: check whether the region-Cauchy '
      'stress QoI has actually settled down by this resolution...')
ref_new = solve_case(*NEW_FINE_RESOLUTION, device=device, verbose=True)
print(f"  NEW reference: {ref_new['n_elements']} elements, {ref_new['elapsed_s']:.2f}s, "
      f"region samples (quadrature points)={ref_new['n_region']} "
      f"(p99 reliable={ref_new['region_p99_reliable']})")

print('\n' + '=' * 90)
print("OLD (243,360-el) vs NEW (424,128-el) reference -- THE check Omar asked for "
      "before trusting the 1% claim on region-Cauchy stress:")
d_avg = abs(ref_new['region_avg_sigma_xx'] - ref_old['region_avg_sigma_xx']) / abs(ref_old['region_avg_sigma_xx'])
print(f"  region_avg_sigma_xx: OLD={ref_old['region_avg_sigma_xx']:.4f}  "
      f"NEW={ref_new['region_avg_sigma_xx']:.4f}  relative change={d_avg*100:.3f}%")
if ref_old['region_p99_reliable'] and ref_new['region_p99_reliable']:
    d_p99 = abs(ref_new['region_p99_sigma_xx'] - ref_old['region_p99_sigma_xx']) / abs(ref_old['region_p99_sigma_xx'])
    print(f"  region_p99_sigma_xx: OLD={ref_old['region_p99_sigma_xx']:.4f}  "
          f"NEW={ref_new['region_p99_sigma_xx']:.4f}  relative change={d_p99*100:.3f}%")
else:
    d_p99 = None
    print("  region_p99_sigma_xx: NOT compared -- one or both references have an "
          "unreliable p99 (fewer than the minimum reliable quadrature-point count)")
_, _, cauchy_field_old_vs_new = compare_to_reference(ref_old, ref_new)
print(f"  full Cauchy-tensor field error (OLD relative to NEW): {cauchy_field_old_vs_new*100:.3f}%")

if d_avg < 0.007:
    print(f"\n  ==> region_avg relative change ({d_avg*100:.3f}%) is BELOW 0.5-0.7% -- "
          f"the earlier 1% claim for this QoI is now substantially stronger.")
else:
    print(f"\n  ==> region_avg relative change ({d_avg*100:.3f}%) is STILL ABOVE 0.5-0.7% -- "
          f"per Omar's own explicit instruction, the 1% claim for region-Cauchy stress "
          f"should be written up as '1% not reached / provisional', not as achieved.")

# The NEW, finer reference is strictly more trustworthy, so it becomes
# the fine reference used for the required-resolution table below.
ref = ref_new
FINE_RESOLUTION = NEW_FINE_RESOLUTION

rows = []
for Ntheta, Nr, Nz in RESOLUTIONS:
    r = solve_case(Ntheta, Nr, Nz, device=device)
    l2_rel, h1_rel, cauchy_field_rel = compare_to_reference(r, ref)
    r['disp_l2_rel'] = l2_rel
    r['grad_h1_rel'] = h1_rel
    r['cauchy_field_rel'] = cauchy_field_rel
    rows.append(r)
    print(f"\n({Ntheta},{Nr},{Nz})  elements={r['n_elements']}  time={r['elapsed_s']:.2f}s")
    print(f"  disp_L2_rel={l2_rel*100:.3f}%  gradF_H1_rel={h1_rel*100:.3f}%  "
          f"cauchy_field_rel={cauchy_field_rel*100:.3f}%")
    print(f"  total_strain_energy={r['total_strain_energy']:.6e}")
    print(f"  reaction_moment_y (PRIMARY)={r['reaction_moment_y']:.6e}  "
          f"reaction_force (secondary)={r['reaction_force']}")
    p99_str = f"{r['region_p99_sigma_xx']:.4f}" if r['region_p99_reliable'] else "NOT RELIABLE (too few samples)"
    print(f"  region(n={r['n_region']} quadrature points, volume-weighted): "
          f"avg_sigma_xx={r['region_avg_sigma_xx']:.4f}  p99={p99_str}  "
          f"(true_max={r['region_true_max_sigma_xx']:.4f}, secondary)")
    print(f"  equilibrium: force_rel_residual={r['force_rel_residual']:.2e} (FORCE scale)  "
          f"moment_rel_residual={r['moment_rel_residual']:.2e} (MOMENT scale)")

print('\n' + '=' * 90)
print('Region-Cauchy-stress convergence check across the WHOLE ladder (including '
      'both fine references) -- READ THIS BEFORE trusting any threshold table below:')
all_for_trend = rows + [
    dict(ref_old, Ntheta=OLD_FINE_RESOLUTION[0], Nr=OLD_FINE_RESOLUTION[1], Nz=OLD_FINE_RESOLUTION[2]),
    dict(ref_new, Ntheta=NEW_FINE_RESOLUTION[0], Nr=NEW_FINE_RESOLUTION[1], Nz=NEW_FINE_RESOLUTION[2]),
]
for a, b in zip(all_for_trend[:-1], all_for_trend[1:]):
    rel = abs(b['region_avg_sigma_xx'] - a['region_avg_sigma_xx']) / abs(a['region_avg_sigma_xx'])
    print(f"  ({a['Ntheta']},{a['Nr']},{a['Nz']}) -> ({b['Ntheta']},{b['Nr']},{b['Nz']}): {rel*100:.2f}% change")

scalar_qoi_rel_errors(rows, ref)
results = find_required_resolutions(rows)
print_threshold_table(results)

print(f"\nThis table is now computed against the NEW, finer ({ref['n_elements']}-element) "
      f"reference. The OLD-vs-NEW region-avg relative change reported above "
      f"({d_avg*100:.3f}%) is the direct answer to whether that reference is "
      f"trustworthy enough for the region-Cauchy 1% claim specifically -- if it is "
      f"still above 0.5-0.7%, treat that row of the table as provisional.")

report = {
    'resolutions': RESOLUTIONS,
    'old_fine_resolution': OLD_FINE_RESOLUTION, 'new_fine_resolution': NEW_FINE_RESOLUTION,
    'old_vs_new_region_avg_rel_change': d_avg,
    'old_vs_new_region_p99_rel_change': d_p99,
    'old_vs_new_cauchy_field_rel': cauchy_field_old_vs_new,
    'rows': [{k: v for k, v in r.items() if not k.startswith('_')} for r in rows],
    'old_fine_reference': {k: v for k, v in ref_old.items() if not k.startswith('_')},
    'new_fine_reference': {k: v for k, v in ref_new.items() if not k.startswith('_')},
}
out_json = f'{R}/b3/mesh_convergence_extended.json'
with open(out_json, 'w') as f:
    json.dump(report, f, indent=2, default=lambda x: x.tolist() if hasattr(x, 'tolist') else str(x))
print('\nSaved:', out_json)

try:
    from omar_pfem.run_manifest import write_manifest
    write_manifest(f'{R}/b3', kind='b3_gpu_mesh_convergence',
                    args={'resolutions': RESOLUTIONS, 'old_fine_resolution': OLD_FINE_RESOLUTION,
                          'new_fine_resolution': NEW_FINE_RESOLUTION},
                    started_at=_started, results={'n_rows': len(rows), 'old_vs_new_region_avg_rel_change': d_avg},
                    outputs=[out_json],
                    notes="Extended B3 mesh-convergence study on GPU, rerun after Omar's own "
                          "11-point technical review (2026-09-21): full Cauchy-tensor field "
                          "error, volume-weighted quadrature-based region sampling with a "
                          "reliability gate, and a NEW finer (~424,128-element) reference added "
                          "specifically to check whether the region-Cauchy-stress QoI's own "
                          "relative change drops below 0.5-0.7% relative to the earlier "
                          "243,360-element reference.")
except Exception as e:
    print(f'[manifest] not recorded: {e}')

print('\nDone.')
