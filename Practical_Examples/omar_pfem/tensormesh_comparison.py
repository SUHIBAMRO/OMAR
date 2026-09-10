"""Item TBD: GPU-FEM (this project's own matrix-free Newton-CG solver)
vs. TensorMesh (camlab-ethz/TensorMesh, pip name `tensormesh-fem`) --
Timon's round-9 reply named this comparison directly, with an explicit
requirement NOT to use TensorMesh's own L-BFGS energy-minimization
approach (their own hyperelastic_beam.py example uses exactly that,
"inadequate and strongly mesh dependent under refinement" per Timon's
own group's torsion-problem experience) and instead "a Newton-type
solve with a direct solver."

THE REAL, INSTALLED PACKAGE, CONFIRMED BY DIRECT INTROSPECTION, NOT
ASSUMED: `pip show tensormesh-fem` -> depends on `torch-sla` (pip name
`torch-sla`, home page `walkerchi/torch-sla` -- a DIFFERENT repo than
one candidate source cited during this investigation, `sparsexlab/
torch-sla`; the two disagree on whether nonlinear_solve's default
Jacobian path is dense-then-sparsified or matrix-free, and only the
actually-installed walkerchi/torch-sla's own source was trusted here).

TWO REAL FACTS CONFIRMED BY READING THE INSTALLED SOURCE DIRECTLY
(torch_sla/sparse_tensor/autograd.py, torch_sla/backends/__init__.py),
NOT FROM DOCS OR A CITATION:
1. `SparseTensor.nonlinear_solve`'s default (`jac_fn=None`) path is
   genuinely a DENSE `torch.autograd.functional.jacobian(...,
   vectorize=True)` call, reshaped to (n,n) and only THEN sparsified
   via `torch.nonzero` -- literally commented "# Dense autograd
   Jacobian, then sparsify (robust default)" in the source. This does
   NOT scale to this project's real problem sizes (hundreds of
   thousands to millions of DOF) -- an explicit sparse `jac_fn` would
   be needed for production scale, not attempted yet.
2. `CUDA_ITERATIVE_THRESHOLD = 2_000_000` (torch_sla/backends/
   __init__.py): direct solvers (cuDSS) are only used below ~2M DOF on
   CUDA; above that, `choose_backend` silently falls back to an
   iterative (Jacobi-preconditioned) method. This project's own N=1001
   (2,004,002 DOF) sits right at this line and N=1401 (3,925,602 DOF)
   is well past it -- TensorMesh would NOT actually honor "direct
   solver" at our own largest benchmark resolutions without a smaller
   fine_N or explicit override, a real constraint on this comparison's
   honest scope, not a bug to work around silently.

WHY torch.func.grad, NOT torch.autograd.grad, FOR THE RESIDUAL:
nonlinear_solve's own first residual evaluation (before any Newton
step, just to check ||F(u0)||) calls residual_fn with a plain tensor
that does not require grad -- torch.autograd.grad errors on this
("element 0 of tensors does not require grad"). torch.func.grad is a
purely functional transform that works regardless of the input's own
requires_grad state and composes correctly with nonlinear_solve's own
outer torch.autograd.functional.jacobian call.

THE SAME NaN-HESSIAN-AT-F=I BUG THIS PROJECT ALREADY FOUND FOR
TORCH-FEM, FOUND AGAIN HERE INDEPENDENTLY: writing lnJ as
torch.log(torch.det(F)) makes the SECOND derivative (the tangent
Newton needs) NaN at F=I (deformation gradient = identity, i.e. zero
displacement, exactly where every solve starts) -- the same
torch.linalg.det double-backward sharp edge documented in
torchfem_comparison.py's own neo_hookean_psi_3d docstring. Fixed the
same way: torch.linalg.slogdet(F)[1] instead.

STATUS (2026-09-10): small-scale proof of concept WORKS -- a tiny
2x2-element quad mesh, B1-style Neo-Hookean, Newton + 'lu' direct
solve, converges cleanly (||F|| 5774 -> 57.9 -> 0.0032 -> 9.1e-11 in 3
iterations) to a physically plausible displacement. NOT yet
cross-validated numerically against "ours" or torch-fem's own solution
at this same tiny problem (only checked for physical plausibility so
far) -- that is the immediate next step, matching this project's own
standing discipline (mms_study.py, torchfem_comparison.py) of
confirming two solvers agree before trusting either one's timing.
Scaling to real B1 resolutions (N=51+) and Q9 not yet attempted;
requires deciding whether to accept the dense-Jacobian cost at small/
medium N or write an explicit sparse jac_fn first.
"""
import torch

from omar_pfem.high_dof_convergence_study import build_mesh_and_bcs


class NeoHookean2DAssembler:
    """Placeholder for the real tensormesh.ElementAssembler subclass --
    see the module docstring and PROJECT_STATUS.md for the actual
    working smoke-test code (run ad hoc, not yet wired into a
    resumable sweep like torchfem_comparison.py's run_sweep). Kept as
    a stub here so this module imports cleanly without a hard
    dependency on tensormesh-fem being installed everywhere this
    project's code is read, matching torchfem_comparison.py's own
    defensive-import style for its own optional dependency."""

    def __init__(self, *args, **kwargs):
        raise NotImplementedError(
            "See PROJECT_STATUS.md's 2026-09-10 entry for the real, "
            "working small-scale proof-of-concept code (residual via "
            "torch.func.grad on an ElementAssembler.energy(), NaN-Hessian "
            "fix via torch.linalg.slogdet, K.nonlinear_solve(..., "
            "method='newton', linear_method='lu')). Not yet promoted to "
            "a resumable sweep module -- do that once the small-scale "
            "result is cross-validated against 'ours' own solution."
        )
