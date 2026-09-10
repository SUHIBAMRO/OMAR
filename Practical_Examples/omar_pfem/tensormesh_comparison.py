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
`torch-sla`, home page `walkerchi/torch-sla`).

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
   is well past it.

A REAL BUG WAS SUSPECTED IN TENSORMESH'S OWN QUADRATURE, THEN
RETRACTED: a trivial "constant-integrand energy should equal the unit
square's area" probe returned 1/sqrt(3) instead of 1.0 when the mesh
was hand-built via a raw `meshio.Mesh(nodes, [('quad', elements)])`
using THIS PROJECT'S OWN element node order (perimeter/counter-
clockwise: bottom-left, bottom-right, top-right, top-left -- the same
convention meshio/VTK and torch-fem both use). Re-running the SAME
trivial probe using TensorMesh's own official `gen_rectangle`
generator instead gave the CORRECT area (1.0) immediately, proving the
fault was in this project's own mesh construction, not the library.
Inspecting `gen_rectangle`'s own generated `cells['quad']` connectivity
directly (e.g. element `[0, 4, 7, 8]`, where node 7 is physically
top-left and node 8 top-right) showed TensorMesh's `Quadrilateral`
element expects nodes in TENSOR-PRODUCT order (bottom-left, bottom-
right, TOP-LEFT, top-right) -- the last two indices swapped relative
to this project's own convention. Fix: `elements[:, [0, 1, 3, 2]]`
before building the meshio.Mesh. Verified on the trivial area probe
first (gives exactly 1.0 after the fix), then on the full real
B1 x Neo-Hookean x N=3 case below.

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

STATUS (2026-09-10): CORRECT, verified against "ours" own solver, not
just plausible. B1 x Neo-Hookean x N=3, solved via
K.nonlinear_solve(residual, u0, f_ext, method='newton',
linear_method='lu') (a real direct factorization, not CG), converges
cleanly in 3 Newton iterations and matches "ours" own displacement
field to ~10 significant digits (max displacement 0.00569220941... on
both sides). Not yet scaled to production N (matching torch-fem's own
N=51...1401 sweep) or Q9 -- needs deciding whether to accept the
dense-Jacobian cost at small/medium N or write an explicit sparse
jac_fn first, given nonlinear_solve's own default path is dense (see
above).
"""
import numpy as np
import torch

from omar_pfem.high_dof_convergence_study import build_mesh_and_bcs


def to_tensormesh_element_order(elements):
    """This project's own Q4 generators (generate_grid_Q4 etc.) order
    each element's nodes perimeter/counter-clockwise: bottom-left,
    bottom-right, top-right, top-left -- the same convention meshio/VTK
    and torch-fem's own Quad1 both use (already confirmed identical to
    torch-fem in torchfem_comparison.py). TensorMesh's own
    `Quadrilateral` element instead expects TENSOR-PRODUCT order:
    bottom-left, bottom-right, TOP-LEFT, top-right -- swap the last two
    columns. See this module's own docstring for how this was found
    (comparing against TensorMesh's own `gen_rectangle` output) rather
    than assumed."""
    return elements[:, [0, 1, 3, 2]]


def _make_assembler_class():
    """The same 2D compressible Neo-Hookean energy density
    materials_torch.py's own neo_hookean_energy_density_vectorized
    implements (psi = mu/2*(I1-2-2*lnJ) + lam/2*lnJ^2), written for
    TensorMesh's own ElementAssembler.element_energy(gradu, mu, lam)
    calling convention (one quadrature point, per-element mu/lam
    supplied via `element_data`). Uses torch.linalg.slogdet, not
    torch.log(torch.det(F)) -- see this module's own docstring for why
    (a NaN-second-derivative bug at F=I, the same one already found and
    fixed for torch-fem).

    ElementAssembler is imported lazily here, not at module level, so
    this file still imports cleanly wherever tensormesh-fem isn't
    installed -- matching torchfem_comparison.py's own defensive-import
    style for its own optional dependency."""
    from tensormesh import ElementAssembler

    class NeoHookean2D(ElementAssembler):
        def element_energy(self, gradu, mu, lam):
            I = torch.eye(2, dtype=gradu.dtype, device=gradu.device)
            F = I + gradu
            _sign, lnJ = torch.linalg.slogdet(F)
            I1 = (F ** 2).sum()
            return (mu / 2.0) * (I1 - 2.0 - 2.0 * lnJ) + (lam / 2.0) * (lnJ ** 2)

    return NeoHookean2D


def build_tensormesh_model(nodes, elements, mu, lam, dtype=torch.float64):
    """nodes/elements come directly from build_mesh_and_bcs -- the SAME
    arrays this project's own solver and torchfem_comparison.py both
    use, not a re-derivation. `elements` is reordered to TensorMesh's
    own tensor-product convention (see to_tensormesh_element_order)
    before building the mesh."""
    import meshio
    from tensormesh import Mesh

    elements_tm = to_tensormesh_element_order(elements)
    mio_mesh = meshio.Mesh(nodes.astype(np.float64), [("quad", elements_tm)])
    tm_mesh = Mesh(mio_mesh)
    NeoHookean2D = _make_assembler_class()
    model = NeoHookean2D.from_mesh(tm_mesh)
    mu_t = torch.tensor(mu, dtype=dtype)
    lam_t = torch.tensor(lam, dtype=dtype)
    return tm_mesh, model, mu_t, lam_t


def solve_tensormesh(nodes, elements, free_dofs, fext_full, mu, lam, dtype=torch.float64,
                      tol=1e-8, method="newton", linear_method="lu"):
    """Newton + direct solver (per Timon's own request), NOT the
    L-BFGS energy-minimization approach TensorMesh's own hyperelastic_
    beam.py example uses. Returns the full nodal displacement field,
    shape (n_nodes, 2), matching solve_ours'/solve_theirs' own return
    convention in torchfem_comparison.py."""
    from tensormesh import LinearElasticityElementAssembler

    torch.set_default_dtype(dtype)
    n_nodes = nodes.shape[0]
    tm_mesh, model, mu_t, lam_t = build_tensormesh_model(nodes, elements, mu, lam, dtype=dtype)

    fixed_set = set(np.setdiff1d(np.arange(n_nodes * 2), free_dofs).tolist())
    free_mask_dof = torch.tensor([i not in fixed_set for i in range(n_nodes * 2)])
    f_ext_flat = torch.tensor(fext_full, dtype=dtype)

    def energy_fn(u_flat):
        return model.energy(point_data={"u": u_flat.reshape(n_nodes, 2)},
                             element_data={"mu": mu_t, "lam": lam_t})

    def residual(u_flat, _A, f_ext):
        grad_full = torch.func.grad(energy_fn)(u_flat)
        res = grad_full - f_ext
        return torch.where(free_mask_dof, res, u_flat)

    # LinearElasticityElementAssembler used only for its correctly-sized/
    # patterned SparseMatrix object to call .nonlinear_solve on -- its
    # own linear physics is irrelevant here (nonlinear_solve's default
    # Jacobian path is its own dense-then-sparsify autograd Jacobian,
    # not this K's values; see this module's own docstring).
    K = LinearElasticityElementAssembler.from_mesh(tm_mesh, E=1.0, nu=0.3)(tm_mesh.points)

    u0 = torch.zeros(n_nodes * 2, dtype=dtype)
    u = K.nonlinear_solve(residual, u0, f_ext_flat, method=method, verbose=False,
                           max_iter=30, tol=tol, linear_method=linear_method)
    return u.reshape(n_nodes, 2).detach().numpy()


def _correctness_check(N=3):
    print(f"=== correctness check: our solver vs. TensorMesh (Newton+{'-'}), "
          f"B1 x Neo-Hookean, N={N} ===")
    nodes, elements, free_dofs, fext_full, elem_params = build_mesh_and_bcs(
        "B1", "Q4", N, "neo_hookean", torch.device("cpu"), torch.float64)
    mu, lam = elem_params

    from omar_pfem.torchfem_comparison import solve_ours
    u_ours, t_ours, stats, _peak = solve_ours(nodes, elements, free_dofs, elem_params, fext_full)
    print(f"  ours:       wall_clock={t_ours:.2f}s, cg_iters_total={stats['cg_iters_total']}")

    import time
    t0 = time.time()
    u_tm = solve_tensormesh(nodes, elements, free_dofs, fext_full, mu, lam)
    t_tm = time.time() - t0
    print(f"  TensorMesh: wall_clock={t_tm:.2f}s")

    diff = np.linalg.norm(u_ours.reshape(-1) - u_tm.reshape(-1))
    ref = np.linalg.norm(u_ours) + 1e-30
    rel_diff = diff / ref
    print(f"  relative displacement-field difference: {rel_diff:.3e}")
    ok = rel_diff < 1e-6
    print("  " + ("PASS" if ok else "FAIL"))
    return ok, rel_diff


if __name__ == "__main__":
    import sys
    N = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    ok, _rel_diff = _correctness_check(N)
    sys.exit(0 if ok else 1)
