# Point 8 results — GPU-native FEM at millions of DOF

B1 × Neo-Hookean on an A100-SXM4-80GB, matrix-free Newton-CG, FP64, ten load
steps. Three resolutions completed; a fourth was still running when the log
ends.

| N | DOF | mesh build | solve | µs/DOF | peak GPU |
|---|---|---|---|---|---|
| 501 | 502,002 | 7.4 s | 26.9 min | 3,219 | 1,123 MB |
| 701 | 982,802 | 14.3 s | 74.8 min | 4,566 | 1,568 MB |
| 1001 | 2,004,002 | 28.7 s | 202.8 min | 6,073 | 2,035 MB |
| 1401 | 3,925,602 | 56.8 s | *interrupted at load step 10/10* | — | — |

## The result: cost is not linear in problem size

µs/DOF is the figure that shows whether cost grows linearly, and it does not
stay flat — it rises from 3,219 to 6,073, a factor of **1.89 across a 4×
increase in degrees of freedom**. Fitting the three points gives

> **cost ~ DOF^1.46**

with pairwise exponents of 1.52 and 1.40, so the trend is consistent rather
than an artefact of one point.

This is worth stating plainly because it is the opposite of what a
matrix-free solver is usually assumed to give. Each CG iteration costs O(DOF),
but the *number* of CG iterations needed grows as the mesh is refined — the
condition number of the tangent scales with 1/h², and Jacobi preconditioning
only partly offsets it. The solver is O(DOF) in memory but not in time.

## Memory behaves exactly as claimed

Peak GPU memory fits **818 MB fixed + 607 MB per million DOF** across all
three rows to within 153 MB. That is the O(DOF) scaling the matrix-free design
was chosen for, and it means the 3.93M DOF case needs roughly **3.2 GB of the
A100's 80 GB**. Memory is nowhere near the constraint; time is.

For contrast, `gpu_fem_solver.py` forms the tangent densely: at 2M DOF that
matrix alone would be about 32 TB in FP64. The matrix-free solver reaches
these sizes at all, which was the point.

## Two things this run does NOT contain

**No cost breakdown.** The cell reported `code at: 5d648d9`, which predates
commit `5edc8b7` where the residual / preconditioner / CG timing buckets were
added. Timon's round-6 request — *"provide some breakdown of the cost. The key
cost should be the solver while the assembly should be minimal"* — is
therefore **still open**, and cannot be answered from this run.

**No small resolutions.** The same old commit still defaulted to starting at
N=501, so N=101, 201, 301 and 401 were skipped. Timon asked for "smaller
intermediate numbers"; those four cost minutes between them and would give the
µs/DOF curve its lower half, which matters because the whole point is the
shape of the trend.

Re-running the current notebook fixes both. N=1401's solver checkpoint is on
Drive, so it resumes rather than restarting; the three completed rows are
already in the output JSON and are skipped.

## When the breakdown does arrive, read it carefully

Timon expects the solver to dominate and assembly to be minimal, and the CG
percentage will very likely look like it confirms that. It does not, cleanly:
a matrix-free solver never forms K, so **every CG iteration is a
Hessian-vector product**, which is itself an assembly-like pass over all
elements. CG time is solver time that contains assembly work by construction.
The clean split he has in mind exists only for a solver that assembles K once
and factorises it — which `gpu_fem_solver.py` does, and which cannot reach
these sizes. That caveat belongs in the reply, not just in the code comment.
