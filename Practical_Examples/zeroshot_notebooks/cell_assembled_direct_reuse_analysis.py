# =====================================================================
#  CELL -- real GPU test of the cuDSS analysis-reuse Newton loop
#  (`reuse_analysis=True` on solve_assembled_direct, 2026-09-11).
#
#  BACKGROUND, per Omar's own request to keep improving the assembled+
#  direct solver further: profile_cudss_analysis_reuse.py already found
#  (real A100 run) that ANALYSIS was 95.7% of one full cuDSS solve's own
#  time at N=401, and that reusing it (instead of redoing it every
#  Newton iteration, which is what torch_sla's own generic nonlinear_
#  solve does) gives the SAME answer. That was measured on 3 isolated
#  matrices, not a real end-to-end Newton solve.
#
#  THIS CELL tests the REAL thing: a full custom Newton loop
#  (_newton_cudss_reuse_analysis in omar_pfem/assembled_direct_solver.py)
#  that computes ANALYSIS once per solve and reuses it for every Newton
#  iteration via an in-place value-buffer update, bypassing torch_sla's
#  own nonlinear_solve entirely for the linear-solve step.
#
#  WHAT THIS TESTS, IN ORDER:
#  1. Correctness at N=11: reuse_analysis=True vs. reuse_analysis=False
#     (the already CPU-verified default) -- must match to ~1e-6 relative
#     or everything below is untrustworthy and should be ignored.
#  2. Real end-to-end speed at production N=401/701/1001/1401, both
#     reuse_analysis settings, so the REAL total-solve speedup can be
#     measured (not just the isolated linear-solve number already
#     found) -- since assembly (vmap+hessian per Newton iteration) and
#     residual/line-search evaluations are NOT sped up by this change,
#     only the actual end-to-end number says how much this matters for
#     the whole solve, not just its linear-algebra phase.
#
#  STILL EXPERIMENTAL: this is a NEW code path (bypasses torch_sla's own
#  nonlinear_solve for the first time), not yet run for real. Per the
#  standing PROJECT_STATUS.md reminder (applies equally to this, same
#  category of change), verify here first, then this is Omar's own call
#  on whether/when to bring it to Timon -- not something to finalize
#  unprompted even if the numbers look great.
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
print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available()
      else 'NONE -- this needs CUDA (cuDSS has no CPU path)')
if not torch.cuda.is_available():
    raise RuntimeError('No CUDA device -- this test cannot run on CPU.')

from torch_sla.backends import is_cudss_available
if not is_cudss_available():
    raise RuntimeError("cuDSS is NOT available after installing nvmath-python[cu12] -- "
                        "check the pip install output above.")
print('cuDSS is available.')

# ---- Step 1: correctness re-check on-device ----------------------------
print('\n' + '=' * 70)
print('STEP 1: correctness re-check (N=11, reuse_analysis=True vs False)')
print('=' * 70)
run([sys.executable, '-m', 'omar_pfem.assembled_direct_solver', 'reuse_check', '11'])

# ---- Step 2: real end-to-end speed, both settings, production N -------
print('\n' + '=' * 70)
print('STEP 2: production-scale END-TO-END speed, reuse_analysis True vs False')
print('=' * 70)

R = '/content/drive/MyDrive/pfem_run'
os.makedirs(R, exist_ok=True)
RESOLUTIONS = [401, 701, 1001, 1401]

OUT_BASELINE = f'{R}/assembled_direct_convergence_production_N401_1401.json'
OUT_REUSE = f'{R}/assembled_direct_reuse_analysis_production_N401_1401.json'

print('\n-- reuse_analysis=False (already-committed numbers reused if present) --')
run([
    sys.executable, '-u', '-m', 'omar_pfem.assembled_direct_solver', 'convergence',
    ','.join(str(n) for n in RESOLUTIONS), OUT_BASELINE, '/content/drive/MyDrive/pfem_ckpt', '2236',
])

print('\n-- reuse_analysis=True (the new code path -- genuinely new numbers) --')
run([
    sys.executable, '-u', '-m', 'omar_pfem.assembled_direct_solver', 'convergence',
    ','.join(str(n) for n in RESOLUTIONS), OUT_REUSE, '/content/drive/MyDrive/pfem_ckpt', '2236',
    'reuse',
])

with open(OUT_BASELINE) as f:
    base_rows = {r['N']: r for r in json.load(f)['rows']}
with open(OUT_REUSE) as f:
    reuse_rows = {r['N']: r for r in json.load(f)['rows']}

print('\n' + '=' * 70)
print('END-TO-END COMPARISON: reuse_analysis=False vs. True (real, full Newton solves)')
print('=' * 70)
print(f'{"N":<6} {"baseline s":<12} {"reuse s":<12} {"speedup":<10} {"l2_rel match?":<14}')
for N in RESOLUTIONS:
    b = base_rows.get(N, {})
    r = reuse_rows.get(N, {})
    bs = b.get('assembled_direct_wall_clock_s')
    rs = r.get('assembled_direct_wall_clock_s')
    b_l2 = b.get('l2_rel')
    r_l2 = r.get('l2_rel')
    l2_match = 'yes' if (b_l2 is not None and r_l2 is not None
                          and abs(b_l2 - r_l2) / (abs(b_l2) + 1e-30) < 1e-3) else '(check)'
    if bs is not None and rs is not None:
        print(f'{N:<6} {bs:<12.2f} {rs:<12.2f} {bs/rs:<10.2f}x {l2_match:<14}')
    else:
        print(f'{N:<6} {"(n/a)":<12} {"(n/a)":<12} {"(n/a)":<10} {"(n/a)":<14}')

print('\n' + '=' * 70)
print('ANALYSIS (Steps 1-2)')
print('=' * 70)
print('l2_rel should match between the two settings (same physics, same tolerance) -- if it')
print('does not, something in the new code path is wrong and the speedup number should not be')
print('trusted. If it matches AND reuse_analysis=True is meaningfully faster end-to-end (not just')
print('in the isolated linear-solve measurement), that is real evidence this optimization is')
print('worth keeping as the new opt-in default for solve_assembled_direct -- still Omar\'s own call,')
print('and per the standing reminder, not something to present to Timon without that conversation')
print('first.')

# ---- Step 3: symmetric-storage on top of analysis-reuse ----------------
# Per Omar's own request to keep pushing further ("جرب الطريقه هاي هات
# نجربها"): the assembled Jacobian is currently GENERAL (non-symmetric) in
# cuDSS's own terms even though the underlying tangent (a Hessian) IS
# symmetric, because fixed-DOF rows are zeroed without the matching
# columns. build_sparse_jac_fn(symmetric_bc=True) fixes this -- confirmed
# EXACT (not approximate) on CPU already, before any GPU time: this
# project's own Dirichlet BCs hold u_fixed=0 identically at every Newton
# iterate, so the eliminated columns contribute exactly zero to the
# free-DOF equations regardless. A full CPU Newton solve with
# symmetric_bc=True gave a BIT-FOR-BIT IDENTICAL answer to the default at
# N=11 and N=21 (0.000e+00 relative difference) -- not just close.
# matrix_type='symmetric' then lets cuDSS skip storing/factorizing the
# redundant triangle, typically cheaper than general LU for the same
# matrix. NEVER RUN ON GPU before this cell -- Step 3's own correctness
# check is what actually verifies the lower-triangle-filtering logic in
# _newton_cudss_reuse_analysis is right, not an assumption.
print('\n' + '=' * 70)
print('STEP 3: symmetric-storage cuDSS test (on top of analysis-reuse)')
print('=' * 70)
print('CPU already confirmed (before this notebook ran): symmetric_bc=True gives a BIT-FOR-BIT')
print('identical full Newton solve to the default at N=11/21 (0.000e+00 relative difference).')
run([sys.executable, '-m', 'omar_pfem.assembled_direct_solver', 'reuse_check', '11', 'symmetric'])

OUT_SYMMETRIC = f'{R}/assembled_direct_reuse_symmetric_production_N401_1401.json'
print('\n-- reuse_analysis=True, matrix_type=symmetric (genuinely new numbers) --')
run([
    sys.executable, '-u', '-m', 'omar_pfem.assembled_direct_solver', 'convergence',
    ','.join(str(n) for n in RESOLUTIONS), OUT_SYMMETRIC, '/content/drive/MyDrive/pfem_ckpt', '2236',
    'reuse_symmetric',
])

with open(OUT_SYMMETRIC) as f:
    sym_rows = {r['N']: r for r in json.load(f)['rows']}

print('\n' + '=' * 70)
print('THREE-WAY END-TO-END COMPARISON: baseline vs. reuse vs. reuse+symmetric')
print('=' * 70)
print(f'{"N":<6} {"baseline s":<12} {"reuse s":<12} {"reuse+sym s":<14} {"l2_rel match?":<14}')
for N in RESOLUTIONS:
    b = base_rows.get(N, {})
    r = reuse_rows.get(N, {})
    s = sym_rows.get(N, {})
    bs = b.get('assembled_direct_wall_clock_s')
    rs = r.get('assembled_direct_wall_clock_s')
    ss = s.get('assembled_direct_wall_clock_s')
    b_l2, s_l2 = b.get('l2_rel'), s.get('l2_rel')
    l2_match = 'yes' if (b_l2 is not None and s_l2 is not None
                          and abs(b_l2 - s_l2) / (abs(b_l2) + 1e-30) < 1e-3) else '(check)'
    bs_s = f'{bs:.2f}' if bs is not None else '(n/a)'
    rs_s = f'{rs:.2f}' if rs is not None else '(n/a)'
    ss_s = f'{ss:.2f}' if ss is not None else '(n/a)'
    print(f'{N:<6} {bs_s:<12} {rs_s:<12} {ss_s:<14} {l2_match:<14}')

print('\n' + '=' * 70)
print('ANALYSIS (Step 3)')
print('=' * 70)
print('l2_rel must match the baseline for the symmetric-storage numbers to be trusted at all --')
print('if the lower-triangle filtering in _newton_cudss_reuse_analysis were wrong, this is where')
print('it would show up as a real, wrong displacement field, not just a timing anomaly. If it')
print('matches AND reuse+symmetric is faster than reuse alone, that is a further real win on top')
print('of the analysis-reuse speedup -- still experimental, still Omar\'s own call on next steps.')
