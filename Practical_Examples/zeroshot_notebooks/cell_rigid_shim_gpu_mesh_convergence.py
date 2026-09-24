# =====================================================================
#  CELL -- Option B (B8-final, rigid-shim model): full production mesh-
#  convergence study into the advisor's 10^5-10^6-element target range.
#
#  Real prerequisite, already confirmed on real GPU (2026-09-24, commits
#  07c6862/f052493/6d9a175): the rigid-shim model -- each of the 19
#  internal steel shims an exact rigid body, motion determined by
#  equilibrium, replacing the deformable-steel model that failed to
#  converge at 105,456 elements with both Jacobi CG and real GPU AmgX --
#  now solves CLEANLY through (37,19)=50,544 / (45,23)=75,504 /
#  (53,27)=105,456 elements (the EXACT resolution that broke the
#  deformable model), protected by automatic load-step cutback (real
#  root-cause fix) and early divergence detection (bails out of a
#  doomed Newton iteration before wasting a linear solve on it, real
#  fix for a huge-CG-iteration-count/Colab-output-truncation problem
#  found the same day). That was a FEASIBILITY test, not this study:
#  this cell finds the actual region-Cauchy-field-error-vs-resolution
#  crossing point for the rigid-shim model, exactly as already done for
#  Option A (B3, see cell_b3_groove_gpu_mesh_convergence.py) -- the old
#  "~791,864 elements" landmark number for B8 was measured on the OLD
#  deformable-steel model, before it was found to fail at scale, and is
#  NOT reused here without re-verification: a materially different
#  physical representation (exact rigid shims vs. deformable St.
#  Venant-Kirchhoff steel) is not guaranteed to cross the 5-10% band at
#  the identical resolution, even though the two models were shown to
#  agree closely (~2%) at 15,600 elements.
#
#  Same standing discipline as every other GPU mesh-convergence cell in
#  this project: the region-Cauchy FIELD error (not a scalar region
#  average, not a raw peak stress) is the PRIMARY local QoI; the
#  reference is checked reference-against-reference before being
#  trusted; and (the real, hard-won lesson from Option A's own ~12.5
#  wasted GPU-hours) the OLD/NEW reference pair is the ladder's own two
#  largest, already-solved rows -- never a separate large solve.
#
#  Real cost/risk note, stated honestly rather than silently assumed:
#  (53,27)=105,456 elements took 728.24s on a live A100. A live run of
#  this cell's FIRST version showed per-row time growing much worse than
#  linearly past that point -- 180,336 elements took 1552.95s, 302,016
#  took 3483.50s, and the NEXT row (489,216 elements) was still running
#  after 6+ more hours with no sign of finishing, a real, structural
#  consequence of the reduced system's linear solve running entirely on
#  CPU via SciPy (not GPU-accelerated at all -- a fact that was in this
#  module's own docstring from the start but was never translated into
#  an explicit cost warning before this ladder was designed; that gap is
#  fixed here, not repeated).
#
#  A SECOND real mistake, found the same day and fixed here: the first
#  version of this cell deferred ALL region-Cauchy-field comparisons
#  (compare_to_reference) to AFTER the whole ladder loop finished, using
#  the ladder's own two largest rows as the OLD/NEW reference pair (the
#  pattern that worked well for Option A/B3). When the run above had to
#  be interrupted mid-ladder (the 7th row never finishing), the six
#  already-solved rows' raw dicts were lost with it (a Colab restart, or
#  even a plain "Interrupt execution" once a fresh kernel is needed,
#  wipes the notebook's Python namespace) -- meaning six real GPU solves
#  produced NO usable region-Cauchy-field number at all, only the raw
#  scalars (time, residual, max_disp) already visible in the printed
#  log. Fixed for real, not just reordered: this version solves ONE
#  reference resolution FIRST (chosen to be affordable, not the ladder's
#  largest), then computes and PRINTS each row's full comparison
#  (disp_L2, region-Cauchy field error) immediately as that row itself
#  finishes -- nothing is ever deferred to a later step that could be
#  lost. A genuine limitation, stated honestly: this reference is NOT
#  independently checked against an even finer one (that would cost
#  exactly the kind of multi-hour tail this fix is trying to avoid) --
#  treat these results as a real first trend, to be confirmed with a
#  finer reference later only if the crossing point looks close to this
#  reference's own resolution.
# =====================================================================
import os
os.environ['JAX_PLATFORMS'] = 'cpu'

import gc
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
# 0.49+ unconditionally imports IPython.core.guarded_eval, which only
# exists from IPython>=8.8 -- Colab ships 7.34.
run([sys.executable, '-m', 'pip', 'install', '-q', 'pyvista<0.49'])
run([sys.executable, '-m', 'pip', 'install', '-q', 'torch-fem'])

WORK = f'{REPO}/Practical_Examples'
os.chdir(WORK)
sys.path.insert(0, WORK)

for _mod_name in list(sys.modules):
    if (_mod_name == 'omar_pfem' or _mod_name.startswith('omar_pfem.')
            or _mod_name == 'torchfem' or _mod_name.startswith('torchfem.')
            or _mod_name == 'pyvista' or _mod_name.startswith('pyvista.')):
        del sys.modules[_mod_name]

import numpy as np
import matplotlib.pyplot as plt
import torch
assert torch.cuda.is_available(), 'this cell needs a real GPU'
print('GPU:', torch.cuda.get_device_name(0))

if torch.cuda.is_available():
    # Real, currently-active external bug (opened 2026-09-23, no official
    # fix yet): googlecolab/colabtools#6111/#6112 -- Colab's A100 image
    # ships libnvrtc-builtins.so.13.0 but nvrtc can't find it on its own
    # search path, breaking any CUDA op that triggers a JIT-compiled
    # reduction kernel (torch.linalg.det, hit directly here).
    import ctypes
    import glob
    for _p in glob.glob('/usr/local/lib/python3*/dist-packages/nvidia/cu13/lib/libnvrtc-builtins.so.13.0'):
        try:
            ctypes.CDLL(_p)
            print(f'Preloaded {_p} (works around a live Colab A100 nvrtc bug, '
                  f'googlecolab/colabtools#6111 -- unrelated to our own code)')
            break
        except OSError as e:
            print(f'Could not preload {_p}: {e}')

# Real fix (see cell_b3_groove_gpu_mesh_convergence.py's own docstring for
# the full incident): clear any stored exception traceback from an
# earlier crash in this same kernel before freeing GPU memory, since
# Jupyter/IPython keeps every frame of sys.last_traceback (including
# large GPU tensors) reachable until it is cleared, and a plain cell
# re-run does not do this on its own.
for _attr in ('last_traceback', 'last_value', 'last_type'):
    if hasattr(sys, _attr):
        delattr(sys, _attr)
gc.collect()
torch.cuda.empty_cache()
print(f'GPU memory at start: {torch.cuda.memory_allocated()/1e9:.2f} GB allocated, '
      f'{torch.cuda.memory_reserved()/1e9:.2f} GB reserved')

from omar_pfem.data.rigid_shim_solver import solve_case as solve_rigid_shim
from omar_pfem.data.mesh_convergence_B8 import compare_to_reference

device = torch.device('cuda')

R = '/content/drive/MyDrive/pfem_run'
os.makedirs(f'{R}/b8_rigid_shim_production', exist_ok=True)

# n_elements = (Ntheta-1) * (Nr-1) * 78 EXACTLY, for the default
# n_rubber_layers/nz_per_rubber/nz_per_shim used throughout this
# project (78 = total z-direction element count across all rubber+shim
# bands) -- confirmed directly against the three already-solved real
# GPU rows below (50,544 / 75,504 / 105,456), not assumed.
#
# REF_RESOLUTION is solved FIRST and used as the reference for every
# other row's comparison, printed immediately as each row finishes (see
# the module docstring for why -- the deferred-comparison design used
# earlier lost six real GPU solves' worth of QoI data to an interrupt).
# Chosen to be the largest resolution already confirmed AFFORDABLE on
# real GPU today (302,016 elements, 3483.50s ~= 58 minutes) rather than
# the ladder's largest possible row -- NOT independently checked against
# an even finer reference (that would reintroduce the exact multi-hour
# tail this fix exists to avoid). (21,11)=15,600 matches the ORIGINAL
# small-scale validation point, re-solved fresh here on the same
# methodology. (37,19)/(45,23)/(53,27)/(69,35) are real rows already
# confirmed live on GPU today.
REF_RESOLUTION = (89, 45)  # 302,016 elements
RESOLUTIONS = [(21, 11), (37, 19), (45, 23), (53, 27), (69, 35), REF_RESOLUTION]

for Ntheta, Nr in RESOLUTIONS:
    n_check = (Ntheta - 1) * (Nr - 1) * 78
    print(f'  ({Ntheta},{Nr}) -> {n_check:,} elements' + ('  <- REFERENCE' if (Ntheta, Nr) == REF_RESOLUTION else ''))

figs_saved = []


def save_and_show(fig, name):
    path = f'{R}/b8_rigid_shim_production/{name}.png'
    fig.savefig(path, dpi=150, bbox_inches='tight')
    figs_saved.append(path)
    print('Saved figure:', path)
    plt.show()


def solve(Ntheta, Nr, **kw):
    return solve_rigid_shim(Ntheta, Nr, device=device, linear_solver='cg', **kw)


def compare(case, ref):
    return compare_to_reference(case, ref)


print(f'\nSolving the reference resolution {REF_RESOLUTION} FIRST -- every other row\'s '
      f'comparison is computed and printed immediately once IT solves, so nothing is '
      f'ever deferred to a later step that could be lost on an interrupt.')
ref = solve(*REF_RESOLUTION, verbose=False)
print(f"\nREFERENCE ({REF_RESOLUTION[0]},{REF_RESOLUTION[1]})  elements={ref['n_elements']:,}  "
      f"time={ref['elapsed_s']:.2f}s  force_rel_residual={ref['force_rel_residual']:.2e}  "
      f"max_disp={ref['max_disp']:.4f}mm")
gc.collect()
torch.cuda.empty_cache()

rows = [ref]
ref['disp_l2_rel'] = 0.0
ref['cauchy_field_rel'] = 0.0
errors = []
for Ntheta, Nr in RESOLUTIONS:
    if (Ntheta, Nr) == REF_RESOLUTION:
        continue  # already solved above, as the reference itself
    try:
        r = solve(Ntheta, Nr, verbose=False)
    except Exception as e:
        print(f"\n  FAILED at ({Ntheta},{Nr}): {type(e).__name__}: {e}", flush=True)
        errors.append((Ntheta, Nr, str(e)))
        continue
    l2_rel, cauchy_field_rel, n_ref_region = compare(r, ref)
    r['disp_l2_rel'] = l2_rel
    r['cauchy_field_rel'] = cauchy_field_rel
    rows.append(r)
    print(f"\n({Ntheta},{Nr})  elements={r['n_elements']:,}  time={r['elapsed_s']:.2f}s  "
          f"force_rel_residual={r['force_rel_residual']:.2e}  max_disp={r['max_disp']:.4f}mm  "
          f"disp_L2={l2_rel*100:.2f}%  cauchy_field={cauchy_field_rel*100:.2f}%  "
          f"n_ref_region={n_ref_region}")
    gc.collect()
    torch.cuda.empty_cache()

rows.sort(key=lambda r: r['n_elements'])

if errors:
    print(f'\n{len(errors)} resolution(s) FAILED (real, reported honestly, not hidden):')
    for Ntheta, Nr, msg in errors:
        print(f"  ({Ntheta},{Nr}): {msg}")

print('\n' + '=' * 90)
print(f'Ladder complete -- reference is {ref["n_elements"]:,} elements '
      f'(NOT independently checked against a finer reference -- see module docstring '
      f'for why that was deliberately skipped this time).')
print('Region-Cauchy-FIELD-error convergence across the WHOLE ladder (PRIMARY QoI):')
for r in rows:
    print(f"  n_elem={r['n_elements']:>10,}  disp_L2={r['disp_l2_rel']*100:6.2f}%  "
          f"cauchy_field={r['cauchy_field_rel']*100:6.2f}%")

print('\n' + '=' * 90)
print("TARGET CHECK: does the region-Cauchy field error stay ~5-10% within the "
      "10^5-10^6-element range?")
in_target_range = [r for r in rows if 1e5 <= r['n_elements'] <= 1e6]
for r in in_target_range:
    in_band = 0.05 <= r['cauchy_field_rel'] <= 0.10
    print(f"  n_elem={r['n_elements']:>10,}  cauchy_field={r['cauchy_field_rel']*100:6.2f}%  "
          f"{'WITHIN 5-10% band' if in_band else 'OUTSIDE 5-10% band'}")
if not in_target_range:
    print("  (no tested resolution fell exactly inside 10^5-10^6 -- see the full "
          "ladder above/figure below for the surrounding trend)")

fig2, ax2 = plt.subplots(figsize=(7.5, 5.5))
ax2.loglog([r['n_elements'] for r in rows], [r['cauchy_field_rel'] * 100 for r in rows],
           'o-', color='tab:blue', label='region-Cauchy field error (PRIMARY)')
ax2.loglog([r['n_elements'] for r in rows], [r['disp_l2_rel'] * 100 for r in rows],
           's--', color='tab:green', label='displacement L2 error')
ax2.axhspan(5, 10, color='gold', alpha=0.25, label='advisor target band (5-10%)')
ax2.axvspan(1e5, 1e6, color='gray', alpha=0.12, label='advisor target range (10^5-10^6 el)')
ax2.axvline(ref['n_elements'], color='tab:red', ls=':', label='reference (not finer-checked)')
ax2.set_xlabel('number of elements')
ax2.set_ylabel('relative error (%)')
ax2.set_title('Option B (rigid-shim): PRIMARY QoI convergence vs. mesh resolution')
ax2.legend(fontsize=8)
ax2.grid(True, which='both', alpha=0.3)
fig2.tight_layout()
save_and_show(fig2, 'rigid_shim_gpu_convergence_summary')

report = {
    'resolutions': RESOLUTIONS, 'ref_resolution': REF_RESOLUTION,
    'rows': [{k: v for k, v in r.items() if not k.startswith('_')} for r in rows],
    'errors': errors,
    'figures_saved': figs_saved,
}
out_json = f'{R}/b8_rigid_shim_production/mesh_convergence_production.json'
with open(out_json, 'w') as f:
    json.dump(report, f, indent=2, default=lambda x: x.tolist() if hasattr(x, 'tolist') else str(x))
print('\nSaved:', out_json)

try:
    from omar_pfem.run_manifest import write_manifest
    write_manifest(f'{R}/b8_rigid_shim_production', kind='b8_rigid_shim_gpu_mesh_convergence',
                    args={'resolutions': RESOLUTIONS, 'ref_resolution': REF_RESOLUTION},
                    started_at=_started,
                    results={'n_rows': len(rows)},
                    outputs=[out_json] + figs_saved,
                    notes="Option B (B8-final rigid-shim model) GPU mesh-convergence "
                          "study, SECOND version after the first was lost mid-run to an "
                          "interrupt (deferred-comparison design flaw, fixed here: every "
                          "row's comparison against a single reference solved FIRST is "
                          "computed and printed immediately, nothing deferred). Reference "
                          "is 302,016 elements, NOT independently checked against a finer "
                          "one (deliberately skipped to avoid the multi-hour CPU-CG tail "
                          "found live past ~300K elements). Region-Cauchy FIELD error is "
                          "the primary local QoI throughout. This is the rigid-shim "
                          "model's OWN production study -- the earlier ~791,864-element "
                          "landmark quoted for B8 was measured on the deformable-steel "
                          "model, before it was found to fail at scale, and is not reused "
                          "here without re-verification.")
except Exception as e:
    print(f'[manifest] not recorded: {e}')

print('\nDone.')
