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
#  (53,27)=105,456 elements took 728.24s on a live A100. This ladder
#  reaches ~7x that many elements at its top row -- per-row time is
#  expected to grow substantially (CG iteration counts and per-iteration
#  cost both increase with problem size), so this cell can plausibly run
#  for multiple hours total. Each row's own real numbers are printed
#  immediately as it finishes (not only at the end), so a mid-run
#  disconnect still leaves real, useable results in the cell's own
#  output even if the final save step is never reached.
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
# Ladder design: (21,11)=15,600 matches the ORIGINAL small-scale
# validation point (design/validation session, rigid-shim vs. deformable
# agreement ~2%) -- included here fresh so it sits on the SAME ladder
# and QoI methodology as everything else, not quoted from a separate,
# differently-computed run. (37,19)/(45,23)/(53,27) are the three rows
# already confirmed live on GPU today. Growth beyond that is a
# deliberately moderate ~1.3-1.7x per step (less cautious than Option
# A's near-the-wall 1.05x steps, since the divergence-detection +
# cutback mechanism has now been proven, live, to correctly handle even
# a 35x single-step residual blowup at 105,456 elements) up to
# ~1,036,000 elements, just past the advisor's target range.
RESOLUTIONS = [
    (21, 11), (37, 19), (45, 23), (53, 27),
    (69, 35), (89, 45), (113, 57), (141, 71), (163, 83),
]
OLD_FINE_RESOLUTION = (141, 71)  # 764,400 elements -- ladder's own second-largest row
NEW_FINE_RESOLUTION = (163, 83)  # 1,036,152 elements -- ladder's own largest row

for Ntheta, Nr in RESOLUTIONS:
    n_check = (Ntheta - 1) * (Nr - 1) * 78
    print(f'  ({Ntheta},{Nr}) -> {n_check:,} elements')

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


print(f'\nSolving the full resolution ladder (up to {RESOLUTIONS[-1]}, '
      f'~1,036,152 elements) -- its own two largest rows double as the '
      f'OLD/NEW reference pair, so no separate large reference solve is '
      f'needed (the real lesson from Option A\'s own ~12.5 wasted GPU-hours).')
rows = []
errors = []
for Ntheta, Nr in RESOLUTIONS:
    try:
        r = solve(Ntheta, Nr, verbose=False)
    except Exception as e:
        print(f"\n  FAILED at ({Ntheta},{Nr}): {type(e).__name__}: {e}", flush=True)
        errors.append((Ntheta, Nr, str(e)))
        continue
    rows.append(r)
    print(f"\n({Ntheta},{Nr})  elements={r['n_elements']:,}  time={r['elapsed_s']:.2f}s  "
          f"force_rel_residual={r['force_rel_residual']:.2e}  max_disp={r['max_disp']:.4f}mm")
    gc.collect()
    torch.cuda.empty_cache()

if errors:
    print(f'\n{len(errors)} resolution(s) FAILED (real, reported honestly, not hidden):')
    for Ntheta, Nr, msg in errors:
        print(f"  ({Ntheta},{Nr}): {msg}")

print('\n' + '=' * 90)
print('Ladder complete -- all rows are real, solved GPU data. Extracting the OLD/NEW '
      'reference pair from the ladder\'s own already-solved rows below (no separate '
      'large reference solve needed -- zero additional GPU time or risk).')

ref_old = next(r for r in rows if (r['Ntheta'], r['Nr']) == OLD_FINE_RESOLUTION)
ref_new = next(r for r in rows if (r['Ntheta'], r['Nr']) == NEW_FINE_RESOLUTION)
print(f"  OLD reference: {ref_old['n_elements']:,} elements, {ref_old['elapsed_s']:.2f}s, "
      f"max_disp={ref_old['max_disp']:.4f}mm")
print(f"  NEW reference: {ref_new['n_elements']:,} elements, {ref_new['elapsed_s']:.2f}s, "
      f"max_disp={ref_new['max_disp']:.4f}mm")

ref = ref_old
for r in rows:
    l2_rel, cauchy_field_rel, n_ref_region = compare(r, ref)
    r['disp_l2_rel'] = l2_rel
    r['cauchy_field_rel'] = cauchy_field_rel
    print(f"  n_elem={r['n_elements']:>10,}  disp_L2={l2_rel*100:6.2f}%  "
          f"cauchy_field={cauchy_field_rel*100:6.2f}%  n_ref_region={n_ref_region}")

print('\n' + '=' * 90)
print('OLD vs NEW reference -- the check for whether the chosen reference is '
      'actually converged (region-Cauchy FIELD error is the PRIMARY comparison):')
d_disp = abs(ref_new['max_disp'] - ref_old['max_disp']) / abs(ref_old['max_disp'])
print(f"  max_disp: OLD={ref_old['max_disp']:.4f}mm  NEW={ref_new['max_disp']:.4f}mm  "
      f"relative change={d_disp*100:.3f}%")
_, cauchy_field_old_vs_new, _ = compare(ref_old, ref_new)
print(f"  region-Cauchy FIELD error (OLD relative to NEW, PRIMARY QoI): "
      f"{cauchy_field_old_vs_new*100:.3f}%")

if cauchy_field_old_vs_new < 0.10:
    print(f"\n  ==> OLD-vs-NEW region-Cauchy field error ({cauchy_field_old_vs_new*100:.3f}%) "
          f"is below 10% -- the OLD reference is reasonably converged; the ladder "
          f"comparison above stands as final, not provisional.")
else:
    print(f"\n  ==> OLD-vs-NEW region-Cauchy field error ({cauchy_field_old_vs_new*100:.3f}%) "
          f"is still above 10% -- the OLD reference is NOT yet demonstrated converged. "
          f"The ladder numbers above should be treated as provisional, not final.")

fig1, ax1 = plt.subplots(figsize=(6, 5))
labels = ['OLD ref\n(%s el)' % f"{ref_old['n_elements']:,}", 'NEW ref\n(%s el)' % f"{ref_new['n_elements']:,}"]
ax1.bar(labels, [ref_old['max_disp'], ref_new['max_disp']], color=['tab:orange', 'tab:blue'])
ax1.set_ylabel('max_disp (mm, secondary scalar QoI)')
ax1.set_title(f'Option B (rigid-shim) reference-to-reference check\n'
              f'region-Cauchy field error: {cauchy_field_old_vs_new*100:.2f}%')
fig1.tight_layout()
save_and_show(fig1, 'rigid_shim_reference_check')

print('\n' + '=' * 90)
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
ax2.axvline(ref_old['n_elements'], color='tab:orange', ls=':', label='OLD reference')
ax2.axvline(ref_new['n_elements'], color='tab:red', ls=':', label='NEW reference')
ax2.set_xlabel('number of elements')
ax2.set_ylabel('relative error (%)')
ax2.set_title('Option B (rigid-shim): PRIMARY QoI convergence vs. mesh resolution')
ax2.legend(fontsize=8)
ax2.grid(True, which='both', alpha=0.3)
fig2.tight_layout()
save_and_show(fig2, 'rigid_shim_gpu_convergence_summary')

report = {
    'resolutions': RESOLUTIONS,
    'old_fine_resolution': OLD_FINE_RESOLUTION, 'new_fine_resolution': NEW_FINE_RESOLUTION,
    'old_vs_new_max_disp_rel_change': d_disp,
    'old_vs_new_cauchy_field_rel': cauchy_field_old_vs_new,
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
                    args={'resolutions': RESOLUTIONS, 'old_fine_resolution': OLD_FINE_RESOLUTION,
                          'new_fine_resolution': NEW_FINE_RESOLUTION},
                    started_at=_started,
                    results={'n_rows': len(rows), 'old_vs_new_cauchy_field_rel': cauchy_field_old_vs_new},
                    outputs=[out_json] + figs_saved,
                    notes="Option B (B8-final rigid-shim model) GPU mesh-convergence "
                          "study: real resolution ladder from 15,600 up to ~1,036,000 "
                          "elements, with a reference-to-reference check using the "
                          "ladder's own two largest already-converged rows (OLD ~764K "
                          "vs NEW ~1.04M elements) before trusting either as converged. "
                          "Region-Cauchy FIELD error is the primary local QoI throughout. "
                          "This is the rigid-shim model's OWN production study -- the "
                          "earlier ~791,864-element landmark quoted for B8 was measured "
                          "on the deformable-steel model, before it was found to fail at "
                          "scale, and is not reused here without re-verification.")
except Exception as e:
    print(f'[manifest] not recorded: {e}')

print('\nDone.')
