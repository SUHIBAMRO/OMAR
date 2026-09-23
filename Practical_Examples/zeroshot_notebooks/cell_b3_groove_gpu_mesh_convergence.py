# =====================================================================
#  CELL -- Option A: modified B3 with a 4x sharper groove (depth=0.20,
#  half_width=0.15, same as baseline), mesh convergence extended to GPU
#  resolutions into and above the 10^5-10^6-element range the advisor
#  asked about.
#
#  CPU study already done (omar_pfem/data/mesh_convergence_B3_groove_
#  sharpness.py): found and fixed two real numerical issues (default 11
#  load increments failed Newton convergence at this depth -- fixed
#  with 21; r_grading=2.5 badly ill-conditioned the linear system --
#  fixed with r_grading=1.5), then got a clean CPU resolution ladder
#  (576... wait, 336/1,320/3,360/6,840 elements against an 18,200-
#  element reference) showing global displacement error converging at
#  nearly the same rate as the original B3 design.
#
#  Same standing discipline as the B8 GPU study (and B3's own original
#  GPU study): do NOT call any CPU-scale mesh here a "converged
#  reference"; do NOT use a rising true (raw) peak stress as evidence of
#  anything -- true_max stays diagnostic-only, exactly as it always has
#  for B1/B2/B3; the region-Cauchy FIELD error (not region_avg, not
#  true_max) is the PRIMARY local QoI; the stress-evaluation region
#  (r=R_in0-groove_depth, theta=0, z=Lz/2, region_radius=2x the
#  groove's own radius of curvature -- FIXED once groove_depth/
#  half_width are fixed, unaffected by resolution) does not change;
#  extend the ladder into ~10^5-10^6 elements; use a finer mesh above
#  that range as the reference; and include a reference-to-reference
#  comparison before trusting either as converged.
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
    if (_mod_name == 'omar_pfem' or _mod_name.startswith('omar_pfem.')
            or _mod_name == 'torchfem' or _mod_name.startswith('torchfem.')
            or _mod_name == 'pyvista' or _mod_name.startswith('pyvista.')):
        del sys.modules[_mod_name]

import numpy as np
import matplotlib.pyplot as plt
import torch
assert torch.cuda.is_available(), 'this cell needs a real GPU'
print('GPU:', torch.cuda.get_device_name(0))

# Real fix, not defensive boilerplate: confirmed directly (on this
# notebook's own B8 sibling) that re-running this cell in the SAME Colab
# kernel after an earlier OOM does NOT free that earlier crash's GPU
# memory, even with gc.collect()/empty_cache() elsewhere in this cell --
# because Jupyter/IPython automatically stores the last exception's full
# traceback (sys.last_traceback), and every local variable in every frame
# of that traceback (including the large GPU tensors alive at the moment
# of the crash) stays reachable -- and therefore un-collectable -- until
# that stored traceback itself is cleared. A true Runtime > Restart
# session clears it as a side effect of killing the process; simply
# re-running this cell does not. Clearing it explicitly here means a
# plain cell re-run recovers on its own.
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

from omar_pfem.data.mesh_convergence_B3_groove_sharpness import solve_case, compare_to_reference
from omar_pfem.data.data_generate_B3 import groove_radius_of_curvature

device = torch.device('cuda')

R = '/content/drive/MyDrive/pfem_run'
os.makedirs(f'{R}/b3_groove_sharp', exist_ok=True)

GROOVE_DEPTH, GROOVE_HALF_WIDTH = 0.20, 0.15   # FIXED geometry -- 4x sharper than baseline
# R_GRADING/N_INCREMENTS: REVISED 2026-09-23, real GPU + CPU evidence,
# not another guess. The earlier choice (r_grading=1.5, n_increments=21)
# was confirmed WORKING at CPU scale (up to 18,200 elements) but a real
# GPU run at production scale (1,071,200 elements) never completed even
# ONE increment in 28+ minutes, with GPU memory climbing continuously --
# and a parallel CPU test at 79,464 elements (never tested before) showed
# increment 1 needing 12 CG iterations and 152-163s (reproduced twice),
# versus 5-6 iterations and ~1-3s at 6,840 elements: the radial grading
# itself gets MORE aggressive (a wider spread of element sizes) as
# Ntheta/Nr grow at a FIXED power-law exponent, so a grading that was
# fine at CPU scale genuinely worsens conditioning at production scale --
# not something CPU-scale testing could have caught. Confirmed directly
# (not assumed) that removing the grading entirely (r_grading=1.0,
# matching B8's own successful, ungraded approach) fixes this: the SAME
# 79,464-element case needed only 5 CG iterations and 49.87s for
# increment 1, and 2 iterations for increment 2 (the same healthy
# pattern every other successful case in this project shows) -- and
# with the grading removed, the default 11 increments (not 21) also
# converged cleanly, since the earlier need for finer load stepping was
# itself very likely an artifact of the aggressive grading, not the
# sharper geometry alone (increment 2, at 20% of the load -- exactly
# where 11 increments used to fail -- converged fine here). Region
# sampling at r_grading=1.0 is healthy at every GPU-scale resolution
# (68 to 996 samples from 79,464 to 1,201,824 elements); only the two
# smallest CPU-scale rows (336, 1,320 elements) have too few samples
# (0, 2) for a reliable field-error number there -- reported honestly as
# unreliable/NaN by the existing code, not hidden, same convention as
# the rest of this project.
R_GRADING = 1.0
N_INCREMENTS = 11
rho = groove_radius_of_curvature(GROOVE_DEPTH, GROOVE_HALF_WIDTH)
print(f'Groove: depth={GROOVE_DEPTH}, half_width={GROOVE_HALF_WIDTH}, rho={rho:.5f}, '
      f'region_radius={2*rho:.5f} (FIXED in physical space -- unaffected by mesh resolution)')

# CPU-scale rows already reported (re-solved here on GPU for a consistent
# per-row time basis), then a new GPU-scale ladder into the advisor's
# 10^5-10^6-element target range.
#
# Real incident, 2026-09-23: a live Colab run reached (81,80,77)=480,320
# elements cleanly (CG iteration count grew only mildly and smoothly,
# 21->25, across a 1000x range in element count -- real evidence the
# r_grading=1.0 conditioning fix genuinely works), then attempted
# OLD_FINE_RESOLUTION=(105,104,101)=1,071,200 elements directly and sat
# there for OVER TEN HOURS with zero output -- not a crash (CG has no
# early bailout besides a 10*dof iteration cap, astronomically large,
# and torchfem never prints per-CG-iteration progress, so a genuinely
# huge iteration count looks identical to a hang from the outside).
# Jumping from a proven-safe 480,320 straight to an untested 1,071,200
# was itself the mistake -- a >2x jump with no real data in between.
# Fixed by inserting five real, individually-tested intermediate sizes
# BETWEEN the proven-safe 480,320 and the historically-failing
# 1,071,200, so if a real wall exists somewhere in this range, it is
# found and reported at the smallest size that hits it, not discovered
# again as another silent multi-hour stall on the same untested jump.
CPU_SCALE_RESOLUTIONS = [(9, 8, 7), (13, 12, 11), (17, 16, 15), (21, 20, 19), (29, 26, 27)]
GPU_RESOLUTIONS = [
    (45, 44, 43), (61, 60, 58), (81, 80, 77),   # already confirmed working live, 2026-09-23
    (85, 84, 81), (89, 88, 85), (93, 92, 89), (97, 96, 93), (101, 100, 97),  # NEW: real intermediate steps
]
RESOLUTIONS = CPU_SCALE_RESOLUTIONS + GPU_RESOLUTIONS
OLD_FINE_RESOLUTION = (105, 104, 101)   # ~1,071,200 elements -- just above the target range
# NEW_FINE_RESOLUTION history, both sizes ruled out using REAL numbers
# from B8's own sibling notebook (same solve mechanism -- same hex8
# element, 8 Gauss points, 24 local dof, float64 -- so the same memory
# model applies regardless of geometry): (133,132,129, ~2,213,376 el)
# was already dialed back once on a rough safety margin; then B8's own
# real GPU OOM at (157,79, ~1,569,672 el) gave the actual numbers
# needed to compute this properly instead of guessing again -- one
# intermediate tensor per Newton iteration (the local element stiffness
# contribution, shape (n_elem, 8 gauss, 24, 24) in float64) scales as
# n_elements * 3.6864e-5 GB, and backing out B8's failure (total
# attempted ~84.4GB, ~57.9GB of which was this one tensor) gives
# ~26.5GB for everything else (K matrix, CG buffers, mesh tensors).
# Targeting a safe ~71GB total gives a real, computed ceiling of
# ~1.2M elements -- NEW_FINE_RESOLUTION below (1,201,824 el) is chosen
# just under that, not another guess.
NEW_FINE_RESOLUTION = (109, 108, 105)   # ~1,201,824 elements -- meaningfully finer, for the reference check

figs_saved = []


def save_and_show(fig, name):
    path = f'{R}/b3_groove_sharp/{name}.png'
    fig.savefig(path, dpi=150, bbox_inches='tight')
    figs_saved.append(path)
    print('Saved figure:', path)
    plt.show()


def solve(Ntheta, Nr, Nz, **kw):
    return solve_case(Ntheta, Nr, Nz, GROOVE_DEPTH, GROOVE_HALF_WIDTH,
                       r_grading=R_GRADING, n_increments=N_INCREMENTS, device=device, **kw)


def compare(case, ref):
    return compare_to_reference(case, ref, GROOVE_DEPTH, GROOVE_HALF_WIDTH)


# Real lesson from a live incident this same day, fixed PROPERLY this
# time (an earlier attempt at this same fix only moved NEW_FINE_
# RESOLUTION after the ladder but left OLD_FINE_RESOLUTION -- the exact
# same ~1,071,200-element solve that hung before -- as the very FIRST
# thing this cell does, so the real problem was untouched and a second
# live run reproduced the identical symptom). Fixed for real now: every
# ladder row (up to 480,320 elements, already inside the advisor's
# 10^5-10^6 target range) is solved and its RAW results printed FIRST,
# with NO large reference required for that -- comparison against a
# reference happens AFTERWARD, once one exists. This means real,
# individually-verifiable GPU numbers (element count, wall time, force
# residual) exist and are visible within minutes even if every solve
# from OLD_FINE_RESOLUTION onward is slow or never completes.
print(f'\nSolving the resolution ladder first (up to {RESOLUTIONS[-1]}, '
      f'~480,320 elements) -- no large reference needed for this part.')
rows = []
for Ntheta, Nr, Nz in RESOLUTIONS:
    r = solve(Ntheta, Nr, Nz, verbose=True)
    rows.append(r)
    print(f"\n({Ntheta},{Nr},{Nz})  elements={r['n_elements']:,}  time={r['elapsed_s']:.2f}s  "
          f"force_rel_residual={r['force_rel_residual']:.2e}  "
          f"region_avg_sigma_xx={r['region_avg_sigma_xx']:.4f} "
          f"(true_max={r['region_true_max_sigma_xx']:.4f}, diagnostic only)")
    gc.collect()
    torch.cuda.empty_cache()

print('\n' + '=' * 90)
print('The WHOLE ladder above is real, solved GPU data, safe regardless of what '
      'happens below. Only NOW attempting the large OLD fine reference.')

print(f'\nSolving the OLD fine reference {OLD_FINE_RESOLUTION} (~1,071,200 elements) -- '
      'this is the step that hung in earlier runs; the ladder above is unaffected '
      'by whatever happens here.')
ref_old = solve(*OLD_FINE_RESOLUTION, verbose=True)
print(f"  OLD reference: {ref_old['n_elements']} elements, {ref_old['elapsed_s']:.2f}s, "
      f"n_region={ref_old['n_region']}, region_avg_sigma_xx={ref_old['region_avg_sigma_xx']:.4f} "
      f"(true_max={ref_old['region_true_max_sigma_xx']:.4f}, diagnostic only)")
gc.collect()
torch.cuda.empty_cache()

# Retroactive comparison: every ladder row was already solved above, so
# this is just interpolation/comparison against ref_old, not a new solve.
ref = ref_old
for r in rows:
    l2_rel, h1_rel, cauchy_field_rel, n_ref_region = compare(r, ref)
    r['disp_l2_rel'] = l2_rel
    r['grad_h1_rel'] = h1_rel
    r['cauchy_field_rel'] = cauchy_field_rel
    print(f"  n_elem={r['n_elements']:>9,}  disp_L2={l2_rel*100:6.2f}%  "
          f"cauchy_field={cauchy_field_rel*100:6.2f}%  n_ref_region={n_ref_region}")

print('\n' + '=' * 90)
print('Real ladder results (against the OLD reference) are now printed and about to be '
      'saved to Drive below, BEFORE attempting the larger, riskier NEW reference -- '
      'so they are safe regardless of what happens next.')

print(f'\nSolving the NEW, finer reference {NEW_FINE_RESOLUTION} (~1,201,824 elements) -- '
      'the check for whether the OLD reference is actually converged...')
ref_new = solve(*NEW_FINE_RESOLUTION, verbose=True)
print(f"  NEW reference: {ref_new['n_elements']} elements, {ref_new['elapsed_s']:.2f}s, "
      f"n_region={ref_new['n_region']}, region_avg_sigma_xx={ref_new['region_avg_sigma_xx']:.4f} "
      f"(true_max={ref_new['region_true_max_sigma_xx']:.4f}, diagnostic only)")
gc.collect()
torch.cuda.empty_cache()

print('\n' + '=' * 90)
print('OLD vs NEW reference -- the check for whether the chosen reference is '
      'actually converged (region-Cauchy FIELD error is the PRIMARY comparison):')
d_avg = abs(ref_new['region_avg_sigma_xx'] - ref_old['region_avg_sigma_xx']) / abs(ref_old['region_avg_sigma_xx'])
print(f"  region_avg_sigma_xx: OLD={ref_old['region_avg_sigma_xx']:.4f}  "
      f"NEW={ref_new['region_avg_sigma_xx']:.4f}  relative change={d_avg*100:.3f}%")
_, _, cauchy_field_old_vs_new, _ = compare(ref_old, ref_new)
print(f"  region-Cauchy FIELD error (OLD relative to NEW, PRIMARY QoI): "
      f"{cauchy_field_old_vs_new*100:.3f}%")
print(f"  (diagnostic only, NOT evidence either way) true_max_sigma_xx: "
      f"OLD={ref_old['region_true_max_sigma_xx']:.4f}  NEW={ref_new['region_true_max_sigma_xx']:.4f}")

if cauchy_field_old_vs_new < 0.10:
    print(f"\n  ==> OLD-vs-NEW region-Cauchy field error ({cauchy_field_old_vs_new*100:.3f}%) "
          f"is below 10% -- the OLD reference (already used above for the ladder) is "
          f"reasonably converged; the ladder comparison above stands as final, not "
          f"provisional.")
else:
    print(f"\n  ==> OLD-vs-NEW region-Cauchy field error ({cauchy_field_old_vs_new*100:.3f}%) "
          f"is still above 10% -- the OLD reference is NOT yet demonstrated converged. "
          f"The ladder numbers above (already printed and saved) should be treated as "
          f"provisional, not final.")

fig1, ax1 = plt.subplots(figsize=(6, 5))
labels = ['OLD ref\n(%s el)' % f"{ref_old['n_elements']:,}", 'NEW ref\n(%s el)' % f"{ref_new['n_elements']:,}"]
ax1.bar(labels, [ref_old['region_avg_sigma_xx'], ref_new['region_avg_sigma_xx']], color=['tab:orange', 'tab:blue'])
ax1.set_ylabel('region_avg_sigma_xx (secondary scalar QoI)')
ax1.set_title(f'Option A (sharper groove) reference-to-reference check\n'
              f'region-Cauchy field error: {cauchy_field_old_vs_new*100:.2f}%')
fig1.tight_layout()
save_and_show(fig1, 'sharper_groove_reference_check')

print('\n' + '=' * 90)
print('Region-Cauchy-FIELD-error convergence across the WHOLE ladder (PRIMARY QoI):')
for r in rows:
    print(f"  n_elem={r['n_elements']:>9,}  disp_L2={r['disp_l2_rel']*100:6.2f}%  "
          f"cauchy_field={r['cauchy_field_rel']*100:6.2f}%  "
          f"true_max_sxx={r['region_true_max_sigma_xx']:>10.3f} (diagnostic)")

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
ax2a.axvline(ref_old['n_elements'], color='tab:orange', ls=':', label='OLD reference')
ax2a.axvline(ref_new['n_elements'], color='tab:red', ls=':', label='NEW reference')
ax2a.set_xlabel('number of elements')
ax2a.set_ylabel('relative error (%)')
ax2a.set_title('Option A (sharper groove): PRIMARY QoI convergence vs. mesh resolution')
ax2a.legend(fontsize=8)
ax2a.grid(True, which='both', alpha=0.3)

ax2b.semilogx([r['n_elements'] for r in rows], [r['region_true_max_sigma_xx'] for r in rows],
              '^-', color='tab:purple')
ax2b.axvspan(1e5, 1e6, color='gray', alpha=0.12)
ax2b.set_xlabel('number of elements')
ax2b.set_ylabel('true_max_sigma_xx (raw peak)')
ax2b.set_title('DIAGNOSTIC ONLY -- true peak stress\n(NOT used to judge convergence or difficulty)')
ax2b.grid(True, which='both', alpha=0.3)

fig2.suptitle('Option A (B3 with 4x sharper groove) -- GPU mesh-convergence summary', fontsize=13)
fig2.tight_layout()
save_and_show(fig2, 'sharper_groove_gpu_convergence_summary')

report = {
    'groove_depth': GROOVE_DEPTH, 'groove_half_width': GROOVE_HALF_WIDTH,
    'r_grading': R_GRADING, 'n_increments': N_INCREMENTS,
    'resolutions': RESOLUTIONS,
    'old_fine_resolution': OLD_FINE_RESOLUTION, 'new_fine_resolution': NEW_FINE_RESOLUTION,
    'old_vs_new_region_avg_rel_change': d_avg,
    'old_vs_new_cauchy_field_rel': cauchy_field_old_vs_new,
    'rows': [{k: v for k, v in r.items() if not k.startswith('_')} for r in rows],
    'old_fine_reference': {k: v for k, v in ref_old.items() if not k.startswith('_')},
    'new_fine_reference': {k: v for k, v in ref_new.items() if not k.startswith('_')},
    'figures_saved': figs_saved,
}
out_json = f'{R}/b3_groove_sharp/mesh_convergence_extended.json'
with open(out_json, 'w') as f:
    json.dump(report, f, indent=2, default=lambda x: x.tolist() if hasattr(x, 'tolist') else str(x))
print('\nSaved:', out_json)

try:
    from omar_pfem.run_manifest import write_manifest
    write_manifest(f'{R}/b3_groove_sharp', kind='b3_groove_sharp_gpu_mesh_convergence',
                    args={'resolutions': RESOLUTIONS, 'old_fine_resolution': OLD_FINE_RESOLUTION,
                          'new_fine_resolution': NEW_FINE_RESOLUTION,
                          'groove_depth': GROOVE_DEPTH, 'groove_half_width': GROOVE_HALF_WIDTH},
                    started_at=_started,
                    results={'n_rows': len(rows), 'old_vs_new_cauchy_field_rel': cauchy_field_old_vs_new},
                    outputs=[out_json] + figs_saved,
                    notes="Option A (B3 with 4x sharper groove, depth=0.20) GPU mesh-"
                          "convergence study: real resolution ladder into the "
                          "10^5-10^6-element range, with a reference-to-reference check "
                          "(OLD ~1.07M vs NEW ~2.21M elements) before trusting either as "
                          "converged. Region-Cauchy FIELD error is the primary local QoI "
                          "throughout; true_max is diagnostic only.")
except Exception as e:
    print(f'[manifest] not recorded: {e}')

print('\nDone.')
