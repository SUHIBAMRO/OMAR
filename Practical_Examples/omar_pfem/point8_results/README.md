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

## µs/DOF is U-shaped, and that changes the reading

The second run added N=101–401 with the cost breakdown. Putting both runs
together on the same A100:

| N | DOF | solve | µs/DOF | residual / precond / CG |
|---|---|---|---|---|
| 101 | 20,402 | 6.6 min | 19,410 | 0.3% / 0.3% / 99.4% |
| 201 | 80,802 | 13.3 min | 9,876 | 0.1% / 0.1% / 99.8% |
| 301 | 181,202 | 20.4 min | 6,755 | 0.0% / 0.1% / 99.9% |
| 401 | 321,602 | 27.7 min | 5,168 | 0.0% / 0.1% / 99.9% |
| **501** | **502,002** | **26.9 min** | **3,215** | — |
| 701 | 982,802 | 74.8 min | 4,566 | — |
| 1001 | 2,004,002 | 202.8 min | 6,072 | — |

The cost per degree of freedom **falls sixfold** from N=101 to N=501, then
**doubles again** by N=1001. The solver has a sweet spot near half a million
DOF, and the two branches have different causes:

* **Below it**, the problem is too small to fill the GPU. Each CG iteration is
  a tiny amount of arithmetic behind a fixed kernel-launch cost, so most of
  the time is overhead — the same effect Section 8.5 measures at batch size 1.
* **Above it**, the CG iteration count grows with refinement: the tangent's
  condition number scales with 1/h² and Jacobi preconditioning only partly
  offsets it. That is the DOF^1.46 branch.

Reporting only the large end would have made this look like a solver that
simply degrades. It does not; it has an operating range, and 0.5M DOF is the
middle of it.

## The cost breakdown, and how to read it honestly

Timon's expectation was *"The key cost should be the solver while the assembly
should be minimal."* Taken at face value the numbers confirm it emphatically:
explicit assembly — residual plus Jacobi preconditioner — is **0.1% to 0.6%**
of the solve, and CG is **99.4% to 99.9%**.

That reading is misleading, and the caveat matters more than the number. A
matrix-free solver never forms K, so **every CG iteration is a Hessian-vector
product, which is itself a pass over all elements doing assembly-like work**.
The 99.9% is solver time that contains the assembly by construction; it is not
evidence that assembly is cheap, only that it has been moved inside the CG
loop where this instrumentation cannot separate it.

The clean split Timon has in mind exists for a solver that assembles K once
and factorises it. `gpu_fem_solver.py` is that solver and does report separate
`t_assembly_s` and `t_solve_s` — but it forms the tangent densely and cannot
reach these sizes (a 2M×2M FP64 tangent is about 32 TB). So the honest answer
is that at these problem sizes the question does not have the clean answer the
expectation assumes, and the reply should say so rather than quote 99.9% and
let it stand.

## Two things this run does NOT contain

**No breakdown for N=501–1001.** Those three were completed by the earlier
commit `5d648d9`, before the timing buckets existed, and resume skips them.
Re-deriving their breakdown would mean re-solving three hours of work for a
number the four smaller resolutions already establish, so it is not worth
doing unless Timon asks.

**N=1401 unfinished.** 3.93M DOF; the run was inside load step 10/10 when the
log ended. Its solver checkpoint is on Drive, so re-running resumes rather
than restarting.
