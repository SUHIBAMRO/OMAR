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
#  WHAT THIS CELL DOES: runs the real N=401/701/1001/1401 sweep on GPU,
#  solving with BOTH solvers at each resolution (ours: multigrid
#  preconditioner, item #4's best available; torch-fem: its own
#  Jacobi-preconditioned CG, since AMG would need an extra pyamg/amgx
#  dependency this comparison does not assume is installed -- both
#  sides use their best readily-available option, not necessarily their
#  theoretical best). Resumable across resolutions: a re-run of this
#  cell skips any N already written to OUT_JSON.
#
#  WHAT TO CHECK when this finishes:
#    1. wall_clock_s, both solvers, at each N -- which is faster, and
#       by how much, at each of the four resolutions.
#    2. peak_mem_mb, both solvers -- this is where the matrix-free vs.
#       assembled-sparse architectural difference should show up most
#       clearly, especially at N=1401 (millions of DOF).
#    3. ours_cg_failures should be 0 at every N (item #4's own result,
#       already confirmed separately) -- if not, something about this
#       specific run regressed and should be investigated before
#       trusting the rest of the row.
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

t0 = time.time()
from omar_pfem.torchfem_comparison import run_sweep
rows = run_sweep([401, 701, 1001, 1401], OUT_JSON, checkpoint_dir=CHECKPOINT_DIR)
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
