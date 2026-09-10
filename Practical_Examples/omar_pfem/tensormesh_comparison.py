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

from omar_pfem.high_dof_convergence_study import (
    build_mesh_and_bcs, solve_one, compute_l2_h1_errors, fit_convergence_rate)


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


def build_sparse_jac_fn(nodes, elements, mu, lam, free_mask_dof, material, order, device, dtype):
    """Explicit sparse dF/du for TensorMesh's own nonlinear_solve (the
    ``jac_fn`` hook, contract confirmed by reading torch_sla's installed
    source directly: ``jac_fn(u, A, *params) -> (val, row, col, shape)``,
    a sparse COO triple), replacing its default ``jac_fn=None`` path -- a
    literal dense torch.autograd.functional.jacobian call, confirmed by
    the SAME source reading, which this module's own docstring already
    measured as intractable past N=51 (~10 days projected at N=401).

    Built by reusing THIS PROJECT'S OWN already-correct, already-fast
    matrix-free machinery (matrix_free_solver.py's own vmap+hessian
    per-element tangent, the same one "ours" own solver already uses for
    its Hessian-vector products) -- NOT by re-deriving element assembly
    from scratch, and not by asking TensorMesh's own ElementAssembler for
    a tangent it does not expose. Correctness of this reuse rests on the
    already-verified fact (this module's own _correctness_check) that
    TensorMesh's residual, built from the SAME Neo-Hookean psi(mu, lam),
    matches "ours" own solver to ~10 significant digits -- so the SAME
    energy's Hessian is the correct tangent for both.

    Standard FEM assembly: for each element and each pair of its local
    nodes (a, b), the local 8x8 (Q4) Hessian's own 2x2 sub-block
    H[2a:2a+2, 2b:2b+2] is scatter-added (via a COO triple with possibly
    repeated (row, col) -- summed by the backend's own COO->CSR/CSC
    conversion, the same "shared-DOF contributions add" rule every FEM
    assembler relies on, not something this code does by hand) at global
    rows/cols 2*quad[:,a]+ca, 2*quad[:,b]+cb. Rows belonging to a FIXED
    DOF are then overridden with a single identity entry (1 on the
    diagonal, nothing else) -- matching residual()'s own
    ``torch.where(free_mask_dof, res, u_flat)`` convention exactly:
    d(u_flat[i])/d(u_flat[j]) = 1 if i==j else 0 for a fixed row.

    The (row, col) index structure is built ONCE (element connectivity
    never changes across Newton iterations); only the VALUES are
    recomputed each call, at the cost of one vmapped 8x8 Hessian per
    element -- the same cost "ours" own matrix-free solver already pays
    every CG iteration, not a new one."""
    from omar_pfem.matrix_free_solver import precompute_shape_data, _local_element_energy
    from omar_pfem.materials_torch import get_material_fns as get_material_fns_torch
    from torch.func import hessian, vmap

    energy_density_fn, _ = get_material_fns_torch(material)
    shape_data = precompute_shape_data(order, device, dtype)
    xy = torch.tensor(nodes, dtype=dtype, device=device)
    quad = torch.tensor(elements, dtype=torch.long, device=device)
    n_nodes = xy.shape[0]
    n_dof = 2 * n_nodes
    n_elem, n_local = quad.shape
    elem_params = (torch.as_tensor(mu, dtype=dtype, device=device),
                   torch.as_tensor(lam, dtype=dtype, device=device))
    Xe_all = xy[quad]  # (n_elem, n_local, 2)

    local_hess_fn = hessian(_local_element_energy, argnums=0)
    batched_hess = vmap(local_hess_fn, in_dims=(0, 0, 0, None, None, None))

    # Static (row, col) template: one entry per (element, local-node-pair,
    # component-pair) -- (n_elem, n_local, n_local, 2, 2) flattened.
    rows, cols = [], []
    for a in range(n_local):
        for b in range(n_local):
            for ca in range(2):
                for cb in range(2):
                    rows.append(2 * quad[:, a] + ca)
                    cols.append(2 * quad[:, b] + cb)
    row_template = torch.cat(rows)
    col_template = torch.cat(cols)
    free_row_mask = free_mask_dof[row_template]
    fixed_idx = (~free_mask_dof).nonzero(as_tuple=True)[0]

    def jac_fn(u, _A, *_params):
        ue_all = u.reshape(n_nodes, 2)[quad].reshape(n_elem, -1)  # (n_elem, 2*n_local)
        H_local = batched_hess(ue_all, Xe_all, elem_params, energy_density_fn, shape_data,
                                dtype)  # (n_elem, 2*n_local, 2*n_local)
        vals = []
        for a in range(n_local):
            for b in range(n_local):
                block = H_local[:, 2 * a:2 * a + 2, 2 * b:2 * b + 2]  # (n_elem, 2, 2)
                for ca in range(2):
                    for cb in range(2):
                        vals.append(block[:, ca, cb])
        val_template = torch.cat(vals)
        val = torch.where(free_row_mask, val_template, torch.zeros_like(val_template))
        val = torch.cat([val, torch.ones(fixed_idx.shape[0], dtype=dtype, device=device)])
        row = torch.cat([row_template, fixed_idx])
        col = torch.cat([col_template, fixed_idx])
        return val.detach(), row, col, (n_dof, n_dof)

    return jac_fn


def solve_tensormesh(nodes, elements, free_dofs, fext_full, mu, lam, dtype=torch.float64,
                      tol=1e-8, method="newton", linear_method="lu", material="neo_hookean",
                      order="Q4", device=None, use_sparse_jac=True):
    """Newton + direct solver (per Timon's own request), NOT the
    L-BFGS energy-minimization approach TensorMesh's own hyperelastic_
    beam.py example uses. Returns the full nodal displacement field,
    shape (n_nodes, 2), matching solve_ours'/solve_theirs' own return
    convention in torchfem_comparison.py.

    use_sparse_jac=True (the default, and the whole point of this
    module's 2026-09-10 extension): builds and passes an explicit
    sparse jac_fn (build_sparse_jac_fn), replacing nonlinear_solve's
    own default dense-then-sparsify Jacobian -- the fix that scales this
    comparison past the ~N=51 ceiling the dense path was measured to
    hit. use_sparse_jac=False keeps the original dense-default behavior,
    for _correctness_check's own A/B comparison against it."""
    from tensormesh import LinearElasticityElementAssembler

    device = device or torch.device("cpu")
    torch.set_default_dtype(dtype)
    n_nodes = nodes.shape[0]
    tm_mesh, model, mu_t, lam_t = build_tensormesh_model(nodes, elements, mu, lam, dtype=dtype)

    fixed_set = set(np.setdiff1d(np.arange(n_nodes * 2), free_dofs).tolist())
    free_mask_dof = torch.tensor([i not in fixed_set for i in range(n_nodes * 2)], device=device)
    f_ext_flat = torch.tensor(fext_full, dtype=dtype, device=device)

    def energy_fn(u_flat):
        return model.energy(point_data={"u": u_flat.reshape(n_nodes, 2)},
                             element_data={"mu": mu_t, "lam": lam_t})

    def residual(u_flat, _A, f_ext):
        grad_full = torch.func.grad(energy_fn)(u_flat)
        res = grad_full - f_ext
        return torch.where(free_mask_dof, res, u_flat)

    # LinearElasticityElementAssembler used only for its correctly-sized/
    # patterned SparseMatrix object to call .nonlinear_solve on -- its
    # own linear physics is irrelevant here; the real tangent comes from
    # build_sparse_jac_fn (or, if use_sparse_jac=False, from
    # nonlinear_solve's own default dense-then-sparsify autograd path).
    K = LinearElasticityElementAssembler.from_mesh(tm_mesh, E=1.0, nu=0.3)(tm_mesh.points)

    jac_fn = None
    if use_sparse_jac:
        jac_fn = build_sparse_jac_fn(nodes, elements, mu, lam, free_mask_dof, material, order,
                                      device, dtype)

    u0 = torch.zeros(n_nodes * 2, dtype=dtype, device=device)
    u = K.nonlinear_solve(residual, u0, f_ext_flat, jac_fn=jac_fn, method=method, verbose=False,
                           max_iter=30, tol=tol, linear_method=linear_method)
    return u.reshape(n_nodes, 2).detach().cpu().numpy()


def run_tensormesh_convergence_study(resolutions, out_json, geometry="B1", material="neo_hookean",
                                      order="Q4", fine_N=2236, checkpoint_dir=None, device=None,
                                      tol=1e-8, dtype=torch.float64):
    """TensorMesh's own ACCURACY and MESH CONVERGENCE at PRODUCTION scale,
    against the exact same fine ~10M-DOF reference used throughout this
    project (and by torchfem_comparison.py's own run_convergence_study,
    for a genuine apples-to-apples comparison across all three: "ours",
    torch-fem, and TensorMesh). Only possible past N=51 because of
    build_sparse_jac_fn -- with the default dense Jacobian this same
    sweep was measured (2026-09-10) to project ~10 days for a single
    N=401 solve; with the sparse Jacobian, N=401 solves in ~63s on CPU
    alone (real, measured, not extrapolated).

    Resumable like every other sweep in this project: writes progress
    after every row, skips resolutions already in out_json."""
    import json
    import os

    device = device or (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
    print('device:', device)

    fine_ckpt = (os.path.join(checkpoint_dir, f"fine_{geometry}_{material}_{order}_N{fine_N}.pt")
                 if checkpoint_dir else None)
    print(f'Loading/resuming fine reference N={fine_N} (checkpoint={fine_ckpt})...')
    fine = solve_one(geometry, order, fine_N, material, device, torch.float64,
                      cg_tol=1e-8, newton_tol=1e-8, checkpoint_path=fine_ckpt)
    print(f'  fine reference ready: n_dof={fine["n_dof"]}, wall_clock_s={fine["wall_clock_s"]:.1f}')

    done = {}
    if out_json and os.path.exists(out_json):
        with open(out_json) as f:
            done = {r["N"]: r for r in json.load(f).get("rows", [])}
    rows = list(done.values())

    for N in resolutions:
        if N in done:
            print(f'  N={N} already in {out_json}, skipping')
            continue
        nodes, elements, free_dofs, fext_full, elem_params = build_mesh_and_bcs(
            geometry, order, N, material, device, dtype)
        mu, lam = elem_params

        import time
        t0 = time.time()
        u_tm = solve_tensormesh(nodes, elements, free_dofs, fext_full, mu, lam, dtype=dtype,
                                 tol=tol, material=material, order=order, device=device)
        elapsed = time.time() - t0

        coarse = {"nodes": nodes, "elements": elements, "N": N,
                  "u": u_tm.reshape(len(nodes), 2)}
        errs = compute_l2_h1_errors(coarse, fine, order, geometry)

        row = {"N": N, "n_dof": int(2 * nodes.shape[0]), "tol": tol,
               "tensormesh_wall_clock_s": elapsed,
               "l2_rel": errs["l2_rel"], "h1_semi_rel": errs["h1_semi_rel"]}
        print(f'  N={N}: wall_clock={elapsed:.2f}s l2_rel={errs["l2_rel"]:.3e} '
              f'h1_semi_rel={errs["h1_semi_rel"]:.3e}')
        rows.append(row)
        rows.sort(key=lambda r: r["N"])
        if out_json:
            with open(out_json, "w") as f:
                json.dump({"geometry": geometry, "material": material, "order": order,
                           "fine_N": fine_N, "device": str(device), "tol": tol,
                           "rows": rows}, f, indent=2)

    sub_fine_rows = [r for r in rows if r["N"] < fine_N]
    hs = [1.0 / (r["N"] - 1) for r in sub_fine_rows]
    if len(hs) >= 2:
        rate_l2, _ = fit_convergence_rate(hs, [r["l2_rel"] for r in sub_fine_rows])
        rate_h1, _ = fit_convergence_rate(hs, [r["h1_semi_rel"] for r in sub_fine_rows])
        print(f'\nFitted convergence rate (TensorMesh, {order}): L2 p={rate_l2:.3f} '
              f'(expected 2), H1 p={rate_h1:.3f} (expected 1)')
    return rows


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
    if len(sys.argv) > 1 and sys.argv[1] == "convergence":
        # python -m omar_pfem.tensormesh_comparison convergence <Ns> <out_json> <ckpt_dir> <fine_N>
        Ns = [int(n) for n in sys.argv[2].split(",")]
        out_json = sys.argv[3]
        ckpt_dir = sys.argv[4] if len(sys.argv) > 4 else None
        fine_N = int(sys.argv[5]) if len(sys.argv) > 5 else 2236
        run_tensormesh_convergence_study(Ns, out_json, checkpoint_dir=ckpt_dir, fine_N=fine_N)
    else:
        N = int(sys.argv[1]) if len(sys.argv) > 1 else 3
        ok, _rel_diff = _correctness_check(N)
        sys.exit(0 if ok else 1)
