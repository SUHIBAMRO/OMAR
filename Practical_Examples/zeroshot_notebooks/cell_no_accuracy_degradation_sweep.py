# =====================================================================
#  CELL -- full NO accuracy sweep (real ground truth, all QoIs) across
#  every resolution that matters for Timon round-10 item 1: the exact
#  same N torch-fem was just measured at (13-49), PLUS enough
#  intermediate/large points (101-1401) to see how the NO's own accuracy
#  degrades on the way to the already-known catastrophic result at
#  N=1401 (640% displacement error).
#
#  WHY THIS EXISTS (two purposes merged into one sweep, both raised
#  2026-09-12): (1) Omar asked, reasonably, whether the N=1401 result
#  might itself be wrong -- there is real history in this project of
#  results that looked like genuine findings turning out to be bugs (the
#  float32/float64 mesh-mismatch bug that produced a false
#  "non-convergence" alarm across six consecutive runs, found and fixed
#  the same day). Running the SAME already-debugged pipeline at several
#  resolutions in between N=49 and N=1401 shows whether the error rises
#  smoothly (expected if the operator is genuinely being pushed past its
#  trained range) or jumps suspiciously (which would instead point at a
#  bug specific to N=1401). (2) The real point of Timon's item 1 needs
#  the NO's own accuracy, in the SAME QoI set FEM was already checked in
#  (L2, H1, energy, PK1 stress, reaction), at the SAME resolutions
#  torch-fem was just measured at (torchfem_convergence_vs_fine_
#  reference.json, N=13..49) -- the existing zero-shot study
#  (point7a_results/zeroshot_B1_neo_hookean.json) only ever measured
#  plain L2 against a fixed N=101 reference, never the full QoI set
#  against each N's own real converged ground truth. This sweep is what
#  makes the direct "at this N, NO gets X% error; what's the coarsest
#  FEM N that also gets <=X%?" table possible.
#
#  COST: cheap. The ground-truth solver (assembled+direct) is fast even
#  at N=1401 (see assembled_direct_convergence_production_N401_1401.json:
#  58.54s wall-clock for a SINGLE-shot solve there; this cell's nsteps=10
#  load-stepping costs some multiple of that, still on the order of a
#  few minutes at the largest N tested here). All the small/intermediate
#  N below are far cheaper than N=1401. Expect the whole sweep (16
#  resolutions) well under an hour on a real GPU, likely much less.
#
#  RESUMABLE: run_accuracy_degradation_sweep skips any N already present
#  in its output JSON.
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
sys.path.insert(0, f'{WORK}/report_builders')

for _mod_name in list(sys.modules):
    if _mod_name == 'omar_pfem' or _mod_name.startswith('omar_pfem.'):
        del sys.modules[_mod_name]

import torch
assert torch.cuda.is_available(), 'this cell needs a real GPU'
print('GPU:', torch.cuda.get_device_name(0))

# Confirm the direct solver is ACTUALLY available before spending any real
# GPU time -- same discipline as every other assembled+direct notebook.
from torch_sla.backends import is_cudss_available
if not is_cudss_available():
    raise RuntimeError(
        "cuDSS is NOT available after installing nvmath-python[cu12] -- the ground-truth "
        "solve below would silently fall back to an iterative solver, not the direct one "
        "this notebook is meant to use. Check the pip install output above for the real error.")
print('cuDSS (real direct solver on CUDA) is available.')

R = '/content/drive/MyDrive/pfem_run'
# BUG FOUND 2026-09-12: a hardcoded path here ('results/checkpoints/
# B1_neo_hookean/model_best.pt', which never existed on Drive) silently
# fell back to 'data_driven/B1_neo_hookean/model_best.pt' -- a COMPLETELY
# DIFFERENT model (train_data_driven.py's own data-driven-loss baseline
# from the round-5/6 comparison study). Found via a real GPU run of this
# exact sweep: disp_rel_L2 was ~620-640% at EVERY N from 13 to 1401 --
# flat regardless of resolution, the signature of evaluating the wrong
# model everywhere, not a real resolution-dependent accuracy failure.
# Fixed properly this time: resolve by CONTENT (sha256), verified against
# the zero-shot study's own already-trusted checkpoint fingerprint, not
# by guessing a path -- see resolve_b1_checkpoint.py's own docstring.
from omar_pfem.resolve_b1_checkpoint import resolve_b1_neo_hookean_checkpoint
CKPT, _ckpt_fp = resolve_b1_neo_hookean_checkpoint(R)
print(f'Resolved checkpoint (verified by fingerprint): {CKPT}')
print('Using checkpoint:', CKPT)

device = torch.device('cuda')

from omar_pfem.measure_inference_latency import build_model
from omar_pfem.no_accuracy_at_n1401 import run_accuracy_degradation_sweep
import argparse

args = argparse.Namespace(
    model='Transolver_Irregular_Mesh', n_hidden=256, n_layers=4, n_heads=8,
    mlp_ratio=2, dropout=0.1, unified_pos=0, ref=16, slice_num=128, fun_dim=4,
    use_soft_dirichlet=1, Lx=1.0, Ly=1.0, R_out=2.0,
)
model = build_model(args, device).to(torch.float32)
model.load_state_dict(torch.load(CKPT, map_location=device))
print('Checkpoint loaded, cast to float32.')

# Widened 2026-09-12 (Omar's own instruction, matching a second-opinion
# review's plan): every resolution the NO's own zero-shot study tested
# (13-49, so this sweep's numbers are directly comparable to the SAME
# resolutions torch-fem was just measured at in
# torchfem_convergence_vs_fine_reference.json -- needed to actually find
# the coarsest FEM N matching the NO's accuracy, not just confirm the
# N=1401 finding), PLUS the intermediate/large points already used to
# check the N=1401 result isn't an isolated fluke. Every one of these
# runs through the EXACT SAME already-debugged pipeline as N=1401 (real
# converged ground truth, same QoI set: L2, H1, energy, PK1 stress,
# reaction) -- not the older zero-shot study's own L2-only numbers
# (which used a fixed N=101 reference rather than each N's own exact
# solution, and never measured H1/energy/stress/reaction at all).
RESOLUTIONS = [13, 17, 21, 25, 29, 33, 37, 41, 45, 49, 101, 201, 401, 701, 1001, 1401]

OUT_JSON = f'{R}/no_accuracy_degradation_sweep.json'
# checkpoint_fingerprint pinned to the resolved checkpoint's own hash: an
# earlier run of this exact sweep silently mixed rows from the WRONG
# checkpoint (see the checkpoint-resolution comment above) with a plain
# resume-by-N. Passing the fingerprint here makes run_accuracy_degradation_
# sweep detect that mismatch itself and discard every stale row instead of
# keeping them -- no manual Drive cleanup needed.
rows = run_accuracy_degradation_sweep(model, args, RESOLUTIONS, OUT_JSON, device,
                                       checkpoint_fingerprint=_ckpt_fp)

print('\n' + '=' * 70)
print('RESULT -- NO accuracy vs. N, N=13..1401 (real ground truth at every point)')
print('=' * 70)
print(f"{'N':<8}{'converged_likely':<20}{'disp_rel_L2':<16}{'L2_rel':<14}{'H1_semi_rel':<14}")
for r in rows:
    gt = r['ground_truth_convergence']
    print(f"{r['N']:<8}{str(gt['converged_likely']):<20}{r['fp32']['disp_rel_L2']:<16.4e}"
          f"{r['fp32']['L2_rel']:<14.4e}{r['fp32']['H1_semi_rel']:<14.4e}")

not_converged = [r['N'] for r in rows if not r['ground_truth_convergence']['converged_likely']]
if not_converged:
    print(f"\n*** WARNING: ground truth did NOT converge at N={not_converged} -- "
          f"those rows compare the NO against a WRONG reference, exclude them "
          f"before drawing any conclusion. ***")
else:
    print('\nGround truth converged at every N tested -- every row above is a genuine '
          'NO-vs-real-FEM comparison.')

print(json.dumps(rows, indent=2))

# ---- Direct crossover against the already-measured torch-fem low-N sweep ----
# L2_rel here (from compute_l2_h1_errors_cross_order, same function torch-fem's
# own convergence study uses) is the SAME metric definition as torch-fem's
# l2_rel -- unlike disp_rel_L2 (a simpler RMS metric), so this is the field to
# use for a like-for-like crossover, not disp_rel_L2.
FEM_JSON = f'{REPO}/Practical_Examples/omar_pfem/torchfem_convergence_vs_fine_reference.json'
if os.path.exists(FEM_JSON):
    with open(FEM_JSON) as f:
        fem_rows = sorted([r for r in json.load(f)['rows'] if r.get('l2_rel') is not None],
                           key=lambda r: r['N'])
    print('\n' + '=' * 70)
    print('CROSSOVER -- for each NO resolution, the coarsest torch-fem N tested here')
    print('that already matches or beats the NO\'s own L2_rel error at that N')
    print('=' * 70)
    for r in rows:
        no_l2 = r['fp32']['L2_rel']
        match = next((fr for fr in fem_rows if fr['l2_rel'] <= no_l2), None)
        if match:
            print(f"  NO at N={r['N']:<6} L2_rel={no_l2:.4e}  ->  torch-fem already matches "
                  f"at N={match['N']} (l2_rel={match['l2_rel']:.4e}, "
                  f"wall_clock={match['torchfem_wall_clock_s']:.2f}s)")
        else:
            print(f"  NO at N={r['N']:<6} L2_rel={no_l2:.4e}  ->  no torch-fem point in this "
                  f"sweep is coarse enough to match (need N<{fem_rows[0]['N']} -- extend the "
                  f"torch-fem sweep coarser if this matters)")
else:
    print(f"\n(torch-fem comparison file not found at {FEM_JSON} -- run the low-N torch-fem "
          f"sweep notebook first for the crossover table)")

# ---- Figure: error vs. N, log-log, with the training resolutions marked ----
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from plot_style import PRIMARY, SECONDARY

Ns = [r['N'] for r in rows]
disp = [r['fp32']['disp_rel_L2'] for r in rows]

fig, ax = plt.subplots(figsize=(7, 4.5), dpi=200)
ax.loglog(Ns, disp, 'o-', color=PRIMARY, label='fp32 disp_rel_L2')
for train_n in (21, 33):
    ax.axvline(train_n, color='gray', linestyle=':', alpha=0.6)
ax.axvline(49, color=SECONDARY, linestyle='--', alpha=0.8,
           label='top of zero-shot validated range (N=49)')
ax.set_xlabel('N')
ax.set_ylabel('Relative displacement error (disp_rel_L2)')
ax.set_title('NO accuracy degradation from N=49 to N=1401')
ax.grid(True, which='both', alpha=0.25)
ax.legend()
fig.tight_layout()
FIG_PATH = f'{R}/fig_no_accuracy_degradation_sweep.png'
fig.savefig(FIG_PATH)
print('\nSaved figure:', FIG_PATH)
print('Saved data:', OUT_JSON)
