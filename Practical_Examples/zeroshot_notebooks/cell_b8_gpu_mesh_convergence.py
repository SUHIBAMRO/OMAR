# =====================================================================
#  CELL -- B8-FINAL (3D laminated annular elastomeric seismic bearing,
#  Option B) mesh convergence, extended to GPU resolutions into and
#  above the 10^5-10^6-element range the advisor asked about.
#
#  This REPLACES the B8-prototype notebook (frozen, unmodified, at
#  cell_b8_prototype_gpu_mesh_convergence.py -- its real GPU-confirmed
#  result, 7.53% region-Cauchy field error at 791,864 elements, stays
#  valid and is kept, not discarded). B8-final differs in three real
#  ways, per a 2026-09-23 technical review of the prototype:
#  (1) real geometry/materials from a published source (Kalantari &
#  Rofooei, 10th Canadian Conference on Earthquake Engineering, 2010):
#  R_IN=15mm, R_OUT=76mm, 20 rubber layers x 3mm, 19 steel shims x 3mm;
#  rubber G=0.68 MPa/K=2000 MPa; steel E=200 GPa/nu=0.3 (real linear-
#  elastic steel via a St. Venant-Kirchhoff psi branch, replacing the
#  prototype's SHIM_STIFFNESS_RATIO=100 Neo-Hookean proxy entirely);
#  (2) the stress-QoI region is now scoped to rubber-layer-1 elements
#  ONLY, near the rubber/shim-1 interface, with a runtime assertion
#  guarding against any shim element ever entering it (the prototype's
#  region spanned both materials with interpolation crossing that
#  discontinuity); (3) a CPU sanity check (mesh validity, det(F)>0,
#  equilibrium, shim-vs-rubber strain self-consistency, and a small
#  3-point convergence trend at 2,496/5,616/9,984 elements against a
#  15,600-element reference: cauchy_field = 4.85%/2.19%/0.67%, clean
#  and monotonic) was run BEFORE this GPU cell was built, per the
#  reviewer's own explicit sequencing.
#
#  Real, geometry-dependent conditioning check done before committing
#  GPU time: the real steel/rubber stiffness ratio here (~294,000:1,
#  E_steel/G_rubber = 200,000/0.68) is far more extreme than the
#  prototype's 100:1 proxy, and this project separately found (same
#  day, in the sibling Option-A/B3-groove-sharpness notebook) that a
#  bare Jacobi-preconditioned CG solver CAN struggle badly at large
#  scale for a geometry with a sharp, localized feature. B8's own
#  annular-laminate geometry has no such feature (the material
#  discontinuity is a full, regular layer, not a local singularity),
#  and a direct CPU test at 15,600 elements WITH the real steel/rubber
#  contrast converged with the same healthy pattern as every other
#  successful case in this project (5 iterations then a flat 2 per
#  increment, 23 total over 10 increments) -- real evidence, not an
#  assumption, that this specific risk does not carry over to B8.
#
#  Real lesson applied from the SAME day's Option-A incident: that
#  notebook attempted its large OLD/NEW references FIRST, so a stuck
#  large solve blocked every smaller, safer ladder row behind it,
#  wasting GPU time with nothing to show for it. This cell solves the
#  ladder FIRST (smallest to largest, up to and including the main
#  reference) and only THEN attempts the separate, finer reference
#  ladder-CHECK resolution -- so if the very largest size is ever slow
#  or fails, the full ladder's real numbers are already printed, saved
#  to Drive, and safe.
#
#  Per the 2026-09-21 standing rule: generates figures during the
#  analysis AND a final summary figure, saves them to Drive, and
#  displays them inline in this notebook's own output.
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

import numpy as np
import matplotlib.pyplot as plt
import torch
assert torch.cuda.is_available(), 'this cell needs a real GPU'
print('GPU:', torch.cuda.get_device_name(0))

# Real fix, not defensive boilerplate: confirmed directly (on this
# notebook's own B8-prototype/B3-groove siblings) that re-running this
# cell in the SAME Colab kernel after an earlier crash does NOT free
# that crash's GPU memory, even with gc.collect()/empty_cache()
# elsewhere in this cell -- because Jupyter/IPython automatically
# stores the last exception's full traceback (sys.last_traceback), and
# every local variable in every frame of that traceback (including the
# large GPU tensors alive at the moment of the crash) stays reachable
# -- and therefore un-collectable -- until that stored traceback itself
# is cleared. A true Runtime > Restart session clears it as a side
# effect of killing the process; simply re-running this cell does not.
# Clearing it explicitly here means a plain cell re-run recovers on its
# own.
import gc
import sys
for _attr in ('last_traceback', 'last_value', 'last_type'):
    if hasattr(sys, _attr):
        delattr(sys, _attr)
gc.collect()
torch.cuda.empty_cache()
print(f'GPU memory at start: {torch.cuda.memory_allocated()/1e9:.2f} GB allocated, '
      f'{torch.cuda.memory_reserved()/1e9:.2f} GB reserved (should be ~0 either way -- '
      f'if not, the runtime was not actually restarted and still holds an earlier '
      f'crash alive; use Runtime > Restart session, not just re-running this cell)')

from omar_pfem.data.mesh_convergence_B8 import solve_case, compare_to_reference

device = torch.device('cuda')

R = '/content/drive/MyDrive/pfem_run'
os.makedirs(f'{R}/b8_final', exist_ok=True)

# Real element counts, computed directly from generate_grid_hex8_
# laminated_bearing (not guessed) -- B8-final's real geometry (20
# rubber layers x 3mm, 19 shims x 3mm = 78 total z-element-layers at
# nz_per_rubber=nz_per_shim=2) gives a different (Ntheta,Nr) -> n_elem
# mapping than the prototype's (4 rubber layers, 18 z-element-layers),
# so this ladder was recomputed from scratch, not reused.
CPU_SCALE_RESOLUTIONS = [(9, 5), (13, 7), (17, 9), (21, 11)]
# GPU-scale ladder into the 10^5-10^6-element range the advisor asked
# about, spaced roughly log-uniformly: 105k / 225k / 390k / 639k / 901k.
GPU_RESOLUTIONS = [(53, 27), (77, 39), (101, 51), (129, 65), (153, 77)]
RESOLUTIONS = CPU_SCALE_RESOLUTIONS + GPU_RESOLUTIONS
# MAIN reference: (165,83) -> 1,048,944 elements, ~1.05M -- matches the
# reviewer's own explicit suggestion for where to put the reference
# ("reference around ~1.05M"), computed directly, not rounded to it by
# coincidence. Memory ceiling check (same real, computed-not-guessed
# formula established this project for B3/B8-prototype: one Newton
# iteration's local-stiffness tensor scales as n_elements*3.6864e-5 GB
# in float64): 1,048,944 el -> ~38.7GB for that tensor alone, safely
# under the ~71-72GB total ceiling found from B8-prototype's own real
# OOM data.
MAIN_REFERENCE = (165, 83)   # ~1,048,944 elements
# CHECK reference: (177,89) -> 1,208,064 elements, ~44.5GB for the same
# tensor -- right at the computed-safe ceiling, same margin used for
# B8-prototype's own NEW reference (1,254,528 el) and Option A's NEW
# reference (1,201,824 el). This is the reference-to-reference check
# that shows MAIN_REFERENCE is itself sufficiently converged, per the
# advisor's explicit request -- not skipped.
CHECK_REFERENCE = (177, 89)  # ~1,208,064 elements

figs_saved = []


def save_and_show(fig, name):
    path = f'{R}/b8_final/{name}.png'
    fig.savefig(path, dpi=150, bbox_inches='tight')
    figs_saved.append(path)
    print('Saved figure:', path)
    plt.show()


# Real lesson from the same day's Option-A incident (see module
# docstring above): solve the WHOLE ladder, including the main
# reference, BEFORE attempting the separate finer check-reference --
# so a slow/stuck check-reference solve can never erase or block the
# ladder's own real results.
print(f'\nSolving the resolution ladder, smallest to the main reference {MAIN_REFERENCE} '
      f'(~1,048,944 elements)...')
rows = []
for Ntheta, Nr in RESOLUTIONS:
    r = solve_case(Ntheta, Nr, device=device, verbose=True)
    rows.append(r)
    print(f"  ({Ntheta},{Nr})  elements={r['n_elements']:,}  time={r['elapsed_s']:.2f}s  "
          f"force_rel_residual={r['force_rel_residual']:.2e}")
    gc.collect()
    torch.cuda.empty_cache()

print(f'\nSolving the MAIN reference {MAIN_REFERENCE} (~1,048,944 elements)...')
ref_main = solve_case(*MAIN_REFERENCE, device=device, verbose=True)
print(f"  MAIN reference: {ref_main['n_elements']} elements, {ref_main['elapsed_s']:.2f}s, "
      f"n_region={ref_main['n_region']}, region_avg_sigma_xx={ref_main['region_avg_sigma_xx']:.4f} "
      f"(true_max={ref_main['region_true_max_sigma_xx']:.4f}, diagnostic only)")
gc.collect()
torch.cuda.empty_cache()

# Compare every ladder row to the MAIN reference now, before touching
# the check-reference, so these numbers exist on disk even if the
# check-reference below is ever slow or fails.
for r in rows:
    l2_rel, cauchy_field_rel, n_ref_region = compare_to_reference(r, ref_main)
    r['disp_l2_rel'] = l2_rel
    r['cauchy_field_rel'] = cauchy_field_rel
    r['n_ref_region'] = n_ref_region
    print(f"\n({r['Ntheta']},{r['Nr']})  elements={r['n_elements']:,}")
    print(f"  disp_L2_rel={l2_rel*100:.3f}%  "
          f"region-Cauchy FIELD error (PRIMARY)={cauchy_field_rel*100:.3f}%  "
          f"n_ref_region={n_ref_region}")

print('\n' + '=' * 90)
print(f'Solving the CHECK reference {CHECK_REFERENCE} (~1,208,064 elements) -- '
      'this is the check for whether MAIN_REFERENCE is actually converged...')
ref_check = solve_case(*CHECK_REFERENCE, device=device, verbose=True)
print(f"  CHECK reference: {ref_check['n_elements']} elements, {ref_check['elapsed_s']:.2f}s, "
      f"n_region={ref_check['n_region']}, region_avg_sigma_xx={ref_check['region_avg_sigma_xx']:.4f} "
      f"(true_max={ref_check['region_true_max_sigma_xx']:.4f}, diagnostic only)")
gc.collect()
torch.cuda.empty_cache()

print('\n' + '=' * 90)
print('MAIN vs CHECK reference -- the check for whether the chosen reference is '
      'actually converged (region-Cauchy field error is the PRIMARY comparison; '
      'region_avg relative change is a secondary scalar cross-check):')
d_avg = abs(ref_check['region_avg_sigma_xx'] - ref_main['region_avg_sigma_xx']) / abs(ref_main['region_avg_sigma_xx'])
print(f"  region_avg_sigma_xx: MAIN={ref_main['region_avg_sigma_xx']:.4f}  "
      f"CHECK={ref_check['region_avg_sigma_xx']:.4f}  relative change={d_avg*100:.3f}%")
_, cauchy_field_main_vs_check, _ = compare_to_reference(ref_main, ref_check)
print(f"  region-Cauchy FIELD error (MAIN relative to CHECK, PRIMARY QoI): "
      f"{cauchy_field_main_vs_check*100:.3f}%")
print(f"  (diagnostic only, NOT evidence either way) true_max_sigma_xx: "
      f"MAIN={ref_main['region_true_max_sigma_xx']:.4f}  CHECK={ref_check['region_true_max_sigma_xx']:.4f}")

if cauchy_field_main_vs_check < 0.10:
    print(f"\n  ==> MAIN-vs-CHECK region-Cauchy field error ({cauchy_field_main_vs_check*100:.3f}%) "
          f"is below 10% -- MAIN_REFERENCE is reasonably converged; the ladder comparison "
          f"above (already computed against MAIN_REFERENCE) stands.")
else:
    print(f"\n  ==> MAIN-vs-CHECK region-Cauchy field error ({cauchy_field_main_vs_check*100:.3f}%) "
          f"is still above 10% -- MAIN_REFERENCE is NOT yet demonstrated converged. "
          f"The ladder numbers above should be treated as provisional, not final.")

fig1, ax1 = plt.subplots(figsize=(6, 5))
labels = ['MAIN ref\n(%s el)' % f"{ref_main['n_elements']:,}", 'CHECK ref\n(%s el)' % f"{ref_check['n_elements']:,}"]
ax1.bar(labels, [ref_main['region_avg_sigma_xx'], ref_check['region_avg_sigma_xx']], color=['tab:orange', 'tab:blue'])
ax1.set_ylabel('region_avg_sigma_xx (secondary scalar QoI)')
ax1.set_title(f'B8-final reference-to-reference check\n'
              f'region-Cauchy field error: {cauchy_field_main_vs_check*100:.2f}%')
fig1.tight_layout()
save_and_show(fig1, 'B8_final_reference_check')

print('\n' + '=' * 90)
print('Region-Cauchy-FIELD-error convergence across the WHOLE ladder (PRIMARY QoI):')
for r in rows:
    print(f"  n_elem={r['n_elements']:>9,}  disp_L2={r['disp_l2_rel']*100:6.2f}%  "
          f"cauchy_field={r['cauchy_field_rel']*100:6.2f}%  "
          f"true_max_sxx={r['region_true_max_sigma_xx']:>10.4f} (diagnostic)")

print('\n' + '=' * 90)
print("TARGET CHECK: does the region-Cauchy field error stay ~5-10% within the "
      "10^5-10^6-element range?")
in_target_range = [r for r in rows if 1e5 <= r['n_elements'] <= 1e6]
for r in in_target_range:
    in_band = 0.05 <= r['cauchy_field_rel'] <= 0.10
    print(f"  n_elem={r['n_elements']:>9,}  cauchy_field={r['cauchy_field_rel']*100:6.2f}%  "
          f"{'WITHIN 5-10% band' if in_band else 'OUTSIDE 5-10% band'}")
if not in_target_range:
    print("  (no tested resolution fell exactly inside 10^5-10^6 -- see the full "
          "ladder above/figure below for the surrounding trend)")

fig2, (ax2a, ax2b) = plt.subplots(1, 2, figsize=(13, 5.5))
ax2a.loglog([r['n_elements'] for r in rows], [r['cauchy_field_rel'] * 100 for r in rows],
            'o-', color='tab:blue', label='region-Cauchy field error (PRIMARY)')
ax2a.loglog([r['n_elements'] for r in rows], [r['disp_l2_rel'] * 100 for r in rows],
            's--', color='tab:green', label='displacement L2 error')
ax2a.axhspan(5, 10, color='gold', alpha=0.25, label='advisor target band (5-10%)')
ax2a.axvspan(1e5, 1e6, color='gray', alpha=0.12, label='advisor target range (10^5-10^6 el)')
ax2a.axvline(ref_main['n_elements'], color='tab:orange', ls=':', label='MAIN reference')
ax2a.axvline(ref_check['n_elements'], color='tab:red', ls=':', label='CHECK reference')
ax2a.set_xlabel('number of elements')
ax2a.set_ylabel('relative error (%)')
ax2a.set_title('B8-final: PRIMARY QoI convergence vs. mesh resolution')
ax2a.legend(fontsize=8)
ax2a.grid(True, which='both', alpha=0.3)

ax2b.semilogx([r['n_elements'] for r in rows], [r['region_true_max_sigma_xx'] for r in rows],
              '^-', color='tab:purple')
ax2b.axvspan(1e5, 1e6, color='gray', alpha=0.12)
ax2b.set_xlabel('number of elements')
ax2b.set_ylabel('true_max_sigma_xx (raw peak)')
ax2b.set_title('DIAGNOSTIC ONLY -- true peak stress\n(NOT used to judge convergence or difficulty)')
ax2b.grid(True, which='both', alpha=0.3)

fig2.suptitle('B8-final (real published-source laminated seismic bearing) -- '
              'GPU mesh-convergence summary', fontsize=13)
fig2.tight_layout()
save_and_show(fig2, 'B8_final_gpu_convergence_summary')

report = {
    'resolutions': RESOLUTIONS,
    'main_reference': MAIN_REFERENCE, 'check_reference': CHECK_REFERENCE,
    'main_vs_check_region_avg_rel_change': d_avg,
    'main_vs_check_cauchy_field_rel': cauchy_field_main_vs_check,
    'rows': [{k: v for k, v in r.items() if not k.startswith('_')} for r in rows],
    'main_reference_result': {k: v for k, v in ref_main.items() if not k.startswith('_')},
    'check_reference_result': {k: v for k, v in ref_check.items() if not k.startswith('_')},
    'figures_saved': figs_saved,
}
out_json = f'{R}/b8_final/mesh_convergence_extended.json'
with open(out_json, 'w') as f:
    json.dump(report, f, indent=2, default=lambda x: x.tolist() if hasattr(x, 'tolist') else str(x))
print('\nSaved:', out_json)

try:
    from omar_pfem.run_manifest import write_manifest
    write_manifest(f'{R}/b8_final', kind='b8_final_gpu_mesh_convergence',
                    args={'resolutions': RESOLUTIONS, 'main_reference': MAIN_REFERENCE,
                          'check_reference': CHECK_REFERENCE},
                    started_at=_started,
                    results={'n_rows': len(rows), 'main_vs_check_cauchy_field_rel': cauchy_field_main_vs_check},
                    outputs=[out_json] + figs_saved,
                    notes="B8-final (real published-source laminated seismic bearing, real "
                          "geometry/materials, rubber-only QoI region) GPU mesh-convergence "
                          "study: real resolution ladder into the 10^5-10^6-element range, "
                          "with a reference-to-reference check (MAIN ~1.05M vs CHECK ~1.21M "
                          "elements) before trusting MAIN as converged. Region-Cauchy FIELD "
                          "error is the primary local QoI throughout; true_max is diagnostic "
                          "only. Ladder solved and compared BEFORE the check-reference, so a "
                          "slow/stuck check-reference can never erase the ladder's own "
                          "results (lesson from the same day's Option-A GPU incident).")
except Exception as e:
    print(f'[manifest] not recorded: {e}')

print('\nDone.')
