# =====================================================================
#  CELL -- Real-GPU efficiency comparison: this project's own GPU-native
#  matrix-free Newton-CG solver vs. torch-fem, an established PyTorch
#  FEM library (item #13). Read this before running.
#
#  WHY THIS EXISTS: Timon's round-7 email named this comparison
#  directly -- "an efficient GPU implementation ... necessary for a
#  fair comparison to a NO" -- with no preference between torch-fem and
#  TensorMesh (the library he mentioned first). Deliberately done AFTER
#  item #4 (the geometric multigrid preconditioner fix): comparing
#  before fixing our own solver's preconditioner would have measured a
#  configuration Timon had already flagged as suboptimal.
#
#  SCOPE (Omar's choice): the large-scale matrix-free solver (behind
#  Table 20/20a/20b/20c in the report, reaching millions of DOF) at the
#  SAME resolutions those tables use (401/701/1001/1401) -- not the
#  small-scale batched dense solver compared against the neural
#  operator elsewhere in the report.
#
#  WHY torch-fem IS NOT MATRIX-FREE, AND WHY THAT MATTERS: torch-fem
#  always explicitly assembles a sparse tangent stiffness matrix (its
#  own base.py: assemble_matrix / self.K), even in its "cg" iterative
#  mode -- confirmed by reading its source. Our own solver never forms
#  K at all (its entire reason for existing -- see
#  matrix_free_solver.py's own docstring), so this comparison is not
#  just "whose CG is faster," it is a genuine architectural difference
#  that should show up most clearly in peak GPU MEMORY at the largest
#  resolutions, which this cell measures for both solvers via
#  torch.cuda.max_memory_allocated -- not only wall-clock.
#
#  A REAL BUG FOUND IN TORCH-FEM DURING THIS ITEM'S OWN CPU VALIDATION:
#  writing our Neo-Hookean energy as psi = mu/2*(I1-3-2*ln(det(F))) +
#  lam/2*ln(det(F))^2, using torch.log(torch.linalg.det(F)) for ln(J)
#  (a completely standard way to write it), makes the SECOND derivative
#  (the tangent stiffness torch-fem needs, via double backprop) come
#  out as all-NaN AT F=IDENTITY -- exactly the point every solve starts
#  from -- a known sharp edge in torch.linalg.det's double-backward,
#  not a physics bug. Confirmed directly: vmap(jacrev(jacrev(psi)))(F,
#  params) at F=I gives an all-NaN Hessian with torch.log(torch.linalg.
#  det(F)) and a correct, finite one with torch.linalg.slogdet(F)[1]
#  instead (mathematically identical value, numerically stable
#  gradient). torchfem_comparison.py's own neo_hookean_psi_3d uses
#  slogdet for exactly this reason -- worth keeping in mind if this
#  comparison is ever extended to a different material.
#
#  VALIDATED ON CPU BEFORE THIS CELL WAS EVER WRITTEN (`python -m
#  omar_pfem.torchfem_comparison <N>`, small N): both solvers, given
#  the SAME mesh, material field, boundary conditions, and load (this
#  project's own build_mesh_and_bcs is reused directly for both, and
#  generate_grid_Q4's element node order was confirmed identical to
#  torch-fem's Quad1 convention, so nodes/elements pass through with no
#  reordering), converge to the SAME displacement field -- relative
#  difference ~2-5e-6 at N=11 and N=21, well inside torch-fem's own
#  float32 working precision (see torchfem_comparison.py's own
#  docstring for why float32: its near_null_space() hardcodes
#  torch.eye(3) at float32 and errors on a float64 model).
#
#  RESULT, N=401/701/1001 (real A100 run, 2026-09-08, after five real
#  environment/library bugs found and fixed along the way -- see
#  PROJECT_STATUS.md item #13 for the full story of each one):
#  torch-fem is dramatically FASTER in wall-clock than our own
#  matrix-free mgv solver -- 6.26s/11.00s/14.46s vs. 2615.8s/7205.4s/
#  17314.8s, i.e. ~418x/655x/1197x, an advantage that GROWS with N (the
#  opposite direction the memory argument above predicted). Report this
#  honestly, but with two caveats stated alongside it, not hidden: (1)
#  the two solvers are NOT run at matched precision/tolerance --
#  torch-fem's own float32 working precision forces stol=1e-4, Newton
#  rtol=atol=1e-3, while "ours" runs float64 with cg_tol=1e-8,
#  newton_tol=1e-8 (several orders tighter), which plausibly explains a
#  large share of the gap on its own; (2) "ours" own peak_mem_mb is
#  `null` at all three rows because "ours" resumed from item #4's own
#  checkpoint (avoiding ~15h of redundant GPU time) rather than solving
#  fresh -- so torch-fem's own memory (3292.6/9669.8/19498.9 MB,
#  committed) currently has nothing real to compare against. Omar's own
#  call (2026-09-08), asked directly once this gap was confirmed and
#  the real cost of fixing it was made explicit (~44min/~2h/~5h of NEW
#  GPU time, since a real, non-resumed "ours" solve is exactly item #4's
#  own already-known cost, not free like this comparison's resumed
#  numbers): accept the wall-clock-only comparison as sufficient for
#  Timon's stated question, note the memory gap honestly as a
#  deliberately-left-open item rather than spend more GPU hours closing
#  it. Extending to N=1401 (below) IS worth doing given how the sweep
#  turned out -- torch-fem's own solve time stayed in the single/low
#  double digits of seconds even at 2M DOF, and "ours" resumes for free
#  regardless, so real new cost for N=1401 is well under a minute.
#
#  WHAT TO CHECK when this finishes (now covers N=1401 too):
#    1. wall_clock_s, both solvers, at N=401/701/1001/1401 -- does the
#       ~400x-1200x gap (growing with N) continue at the largest size.
#    2. peak_mem_mb, torch-fem only (see the "ours" caveat above) --
#       does its own memory keep scaling ~linearly with n_dof.
#    3. ours_cg_failures should be 0 at every N (item #4's own
#       already-confirmed result) -- if not, something regressed and
#       should be investigated before trusting the rest of that row.
#  Report the real numbers either way -- a result unfavorable to our
#  own solver is still the honest answer to Timon's question.
# =====================================================================
import os, subprocess, sys, time

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
    run(['git', '-C', REPO, 'fetch', 'origin',
                    'claude/claude-code-question-d307wp'])
    run(['git', '-C', REPO, 'checkout',
                    'claude/claude-code-question-d307wp'])
    run(['git', '-C', REPO, 'reset', '--hard',
                    'origin/claude/claude-code-question-d307wp'])

run([sys.executable, '-m', 'pip', 'install', '-q', 'torch-fem'])

WORK = f'{REPO}/Practical_Examples'
os.chdir(WORK)
sys.path.insert(0, WORK)

import torch
print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available()
      else 'NONE - Runtime > Change runtime type > GPU required for this comparison')

# Cheap CPU-scale sanity check before trusting the real GPU numbers --
# same discipline as every other GPU run in this project: confirm the
# two solvers still agree before spending real time on a comparison
# between them.
run([sys.executable, '-m', 'omar_pfem.torchfem_comparison', '11'])

OUT_JSON = '/content/drive/MyDrive/pfem_run/torchfem_comparison_B1_neo_hookean.json'
os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)

# item #4's own checkpoints already hold the fully-converged mgv solve
# at all four of these resolutions (Round6_MGV_Recheck_N701_1001_1401.
# ipynb's real ~14.86h run) -- reusing them here means "ours" resumes
# in seconds and this cell's own real cost is torch-fem's solve time
# alone, not ~15h spent reproducing numbers already committed in
# highdof_stress_qoi_results/*.json.
CHECKPOINT_DIR = '/content/drive/MyDrive/pfem_ckpt'

# N=401/701/1001 finished 2026-09-08 (real result: torch-fem ~418x-
# 1197x faster in wall-clock, see this file's header comment).
# Omar's call once that result was in: extend to N=1401 too, since the
# real cost turned out to be tiny (torch-fem's own solve time stayed
# under 15s even at N=1001, and "ours" resumes for free regardless of
# how many N are listed here) -- this run will skip 401/701/1001
# (already in OUT_JSON) and solve only the new N=1401.
RESOLUTIONS = [401, 701, 1001, 1401]

# This cell has been re-run several times in this SAME Colab kernel while
# this comparison's own code was still being fixed (pyvista/IPython, then
# the torch-fem device-mismatch bug below) -- `git reset --hard` above
# only updates the FILES on disk, it does not touch Python's own
# `sys.modules` cache. The CPU correctness check just above runs as a
# fresh subprocess every time (always sees the latest code), but a plain
# `import omar_pfem...` in THIS process reuses whatever was already
# imported earlier in this same kernel session, silently ignoring any
# fix committed since -- exactly what happened when the device-mismatch
# fix below was pushed but a stale in-kernel import kept reproducing the
# pre-fix error. Force a clean re-import of this project's own package
# every time this cell runs, regardless of what an earlier cell run in
# this kernel already imported.
for _mod_name in list(sys.modules):
    if _mod_name == 'omar_pfem' or _mod_name.startswith('omar_pfem.'):
        del sys.modules[_mod_name]

t0 = time.time()
from omar_pfem.torchfem_comparison import run_sweep
rows = run_sweep(RESOLUTIONS, OUT_JSON, checkpoint_dir=CHECKPOINT_DIR)
elapsed = time.time() - t0

print(f'\nDone in {elapsed/3600:.2f} h. Results: {OUT_JSON}')
print('\nWHAT TO CHECK:')
for r in rows:
    speedup = r['torchfem_wall_clock_s'] / r['ours_wall_clock_s']
    mem_note = (f"mem {r['ours_peak_mem_mb']:.0f}MB vs {r['torchfem_peak_mem_mb']:.0f}MB"
                if r['ours_peak_mem_mb'] and r['torchfem_peak_mem_mb'] else "mem: not recorded (CPU run?)")
    print(f"  N={r['N']}: ours {r['ours_wall_clock_s']:.1f}s (cg_failures="
          f"{r['ours_cg_failures']}) vs torch-fem {r['torchfem_wall_clock_s']:.1f}s "
          f"-- ours is {speedup:.2f}x {'faster' if speedup > 1 else 'slower'}; {mem_note}")
print('  Report every number honestly, including any where torch-fem wins --')
print('  that is a legitimate answer to Timon\'s question, not a failure to fix.')
