"""B8-final, RIGID-SHIM formulation: each of the 19 internal steel shims
is represented as an exact finite-rotation rigid body, bonded to the
adjacent rubber surfaces through node-sharing, with its rigid motion
determined by equilibrium (a multi-point constraint / kinematic
condensation, NOT a prescribed motion and NOT a stiffness-ratio proxy).

Each shim's freedom is 3 unknowns (t_x, t_z, theta_y), not the general
6 (t, theta) rigid_shim_kinematics.py otherwise supports: this mesh is
a half-cylinder (theta in [0, pi]) with a genuine mirror-symmetry plane
at y=0 (the SAME reason data_generate_B8.rigid_top_plate_displacement
restricts the top plate's own rigid BC to compression + shear_x + a
single rotation phi_y about y, with uy=0 identically). A shim's own
node group spans that y=0 plane, so a GENERAL 6-dof rigid motion would
only satisfy uy=0 there by coincidence at isolated nodes -- tearing the
shim's own rigidity apart at the symmetry boundary instead of keeping
the whole shim exactly rigid. Restricting a priori to
(t_x, t_z, theta_y) (see rigid_shim_kinematics.
rigid_body_displacement_and_jacobian_symmetric) keeps uy=0 exactly for
every point on the plane, for any value of the 3 free parameters --
confirmed directly (a first attempt at the general 6-dof version
diverged; this symmetric restriction was the real, structural fix, not
a numerical tuning fix).

Real motivation (2026-09-23): B8-final's deformable-steel model (real
E=200 GPa steel via St. Venant-Kirchhoff, mesh_convergence_B8.py) failed
to converge at 105,456 elements with BOTH a Jacobi-preconditioned CG
solver AND a real GPU AmgX/AMG solver (torch-fem's strongest available
option) -- CG did not reach tolerance within its iteration limit either
way, and AmgX raised its own explicit NOT_CONVERGED error after 1000
iterations. Since the SAME problem failed with the weakest and the
strongest available linear solver, the failure is not a solver-choice
problem: it is the extreme real steel/rubber stiffness ratio itself
(~294,000:1) making the assembled system intrinsically ill-conditioned
at that scale.

Real, data-driven justification for treating the shims as rigid rather
than searching for yet another linear solver: the deformable model's
own real diagnostic showed the shims barely deform at all
(max_strain_shim=3.36e-04 vs max_strain_rubber=4.50e-01 at 15,600
elements) -- so replacing their deformable St. Venant-Kirchhoff response
with an EXACT rigid-body constraint removes the extreme stiffness
contrast from the assembled system entirely (after kinematic
condensation, a shim's contribution to the reduced system is a
compact, well-scaled 6x6-ish block, regardless of how stiff its
"real" material would have been), rather than approximating a physical
quantity that is already known to be negligible.

Implementation: torch-fem has no built-in rigid multi-point-constraint
mechanism (confirmed directly against its API), so this module drives
torch-fem's own low-level element physics (Solid.integrate_material,
assemble_matrix, assemble_rhs -- the same calls torch-fem's own
`solve()` uses internally) through a CUSTOM Newton-Raphson loop with a
kinematic reduction: the Solid model contains ONLY rubber elements (no
shim elements at all -- their own material choice would be numerically
irrelevant post-condensation, so they are dropped rather than kept with
an arbitrary value); the reduced unknown vector is [free rubber DOFs,
shim_1 (t,theta), ..., shim_19 (t,theta)]; at each Newton iteration the
full nodal displacement increment is reconstructed from the reduced
unknowns (identity for free rubber DOFs, the exact rigid map for each
shim's own nodes -- which include the interface nodes SHARED with the
adjacent rubber layers, so bonding is automatic, not a separate
constraint), and the full residual/tangent are reduced via the
constraint Jacobian J (built once per shim via automatic
differentiation of the exact rotation map): R_reduced = J^T @ R_full,
K_reduced = J^T @ K_full @ J -- the standard consistent linearization
for a holonomic multi-point constraint (this literally computes the
net force AND moment the rubber exerts on each rigid shim about its own
reference point, i.e., "rigid translation and rotation determined by
equilibrium", not prescribed).

The reduced linear solve supports two methods via `linear_solver`
(2026-09-23 addition, real evidence-driven, not speculative): 'direct'
(SciPy sparse LU, `spsolve`, the original and still the default) and
'cg' (Jacobi-preconditioned CG via SciPy, WITH per-iteration progress
printing -- unlike torch-fem's own CG, which has no early bailout and
never prints progress, making a genuinely slow solve indistinguishable
from a silent hang; this project already paid for that lesson once with
Option A's ~12.5 wasted GPU-hours, so this wrapper does not repeat it).
Real motivation for adding 'cg': a live Colab run at 75,504 elements
showed the direct solve taking a highly disproportionate amount of time
relative to its own cost at 50,544 elements (confirmed via `top`: the
python process was genuinely at 100% CPU, not hung, just very slow) --
a known, structural property of sparse DIRECT solvers on 3D FEM systems
(fill-in scales much worse than linearly with problem size in 3D, which
is exactly why the REST of this project always uses an iterative CG
solver for large 3D meshes instead of a direct solve). There is also a
real, principled reason to expect CG to behave WELL here specifically:
the shim stiffness contrast that made the DEFORMABLE model's own CG
fail at 105,456 elements is structurally absent from this reduced
system (shims are exact rigid constraints, not elements in the
assembled K at all), so the reduced system's conditioning is not
expected to inherit that problem. 'cg' is validated against 'direct' at
an already fully solved, real resolution (50,544 elements) before being
trusted at any larger, previously-untested size -- same discipline as
every other change in this project (see rigid_shim_solver_verify_cg.py).
If that changes at production scale regardless of solver choice, that
is itself a real, reportable finding (see mesh_convergence_B8.py's own
docstring for the standing caution: rubber is also near-incompressible,
and displacement-based HEX8 elements can suffer volumetric-locking-
related conditioning problems independent of the shim material question
-- do not assume the shim fix is the only possible source of a future
conditioning problem at scale).
"""
import time

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import torch

from omar_pfem.data.data_generate_B8 import (
    MU_RUBBER, LAM_RUBBER, N_RUBBER_LAYERS, R_IN, R_OUT, T_RUBBER, T_SHIM,
    boundary_node_sets, build_z_axis, generate_grid_hex8_laminated_bearing,
    layer_bands, rigid_top_plate_displacement,
)
from omar_pfem.data.mesh_convergence_B8 import COMPRESSION_FRAC, PHI_Y, SHEAR_X
from omar_pfem.data.rigid_shim_kinematics import (
    rigid_body_displacement_and_jacobian_symmetric, shim_node_groups,
)
from omar_pfem.torchfem_comparison import neo_hookean_psi_3d


def _csr_torch_to_scipy(K_torch):
    K = K_torch.coalesce() if K_torch.layout == torch.sparse_coo else K_torch
    crow = K.crow_indices().cpu().numpy()
    col = K.col_indices().cpu().numpy()
    val = K.values().cpu().numpy()
    n = K.shape[0]
    return sp.csr_matrix((val, col, crow), shape=(n, n))


def _solve_reduced(K_reduced, R_reduced, method="direct", cg_rtol=1e-10, cg_maxiter=None,
                    verbose=False, tag=""):
    """Solves K_reduced @ dq = -R_reduced. method='direct': SciPy sparse
    LU. method='cg': Jacobi-preconditioned CG -- WITH progress printing
    every iteration (small systems) or every 20th (large), so a slow
    solve is visibly distinguishable from a hang, unlike torch-fem's own
    CG (see module docstring)."""
    if method == "direct":
        return spla.spsolve(K_reduced, -R_reduced)
    elif method == "cg":
        # Real assembled tangents are symmetric only up to floating-point
        # roundoff (conservative hyperelastic internal forces); CG
        # assumes exact symmetry, so enforce it explicitly rather than
        # trust roundoff to be negligible.
        K_sym = 0.5 * (K_reduced + K_reduced.T)
        diag = K_sym.diagonal()
        assert np.all(diag > 0), "non-positive diagonal in K_reduced -- Jacobi preconditioner invalid"
        M = spla.LinearOperator(K_sym.shape, matvec=lambda x: x / diag)
        n = K_sym.shape[0]
        maxiter = cg_maxiter or (10 * n)
        it_counter = [0]

        def callback(xk):
            it_counter[0] += 1
            if verbose and (it_counter[0] <= 10 or it_counter[0] % 20 == 0):
                r = -R_reduced - K_sym @ xk
                print(f"    {tag}CG iter {it_counter[0]}: |r_cg|={np.linalg.norm(r):.3e}", flush=True)

        dq, info = spla.cg(K_sym, -R_reduced, rtol=cg_rtol, maxiter=maxiter, M=M, callback=callback)
        if info != 0:
            raise RuntimeError(f"CG did not converge for the reduced system "
                                f"(info={info}, {it_counter[0]}/{maxiter} iterations attempted)")
        if verbose:
            print(f"    {tag}CG converged in {it_counter[0]} iterations", flush=True)
        return dq
    else:
        raise ValueError(f"unknown linear_solver {method!r}")


def solve_case(Ntheta, Nr, n_rubber_layers=N_RUBBER_LAYERS, nz_per_rubber=2, nz_per_shim=2,
               r_grading=1.0, n_increments=11, dtype=torch.float64, device=None,
               verbose=False, max_iter=30, rtol=1e-8, atol=1e-8,
               linear_solver="direct", cg_rtol=1e-10, cg_maxiter=None, max_cutbacks=10):
    """Same signature/return-dict conventions as
    mesh_convergence_B8.solve_case (the deformable-steel model), for a
    direct, apples-to-apples comparison via that module's own
    compare_to_reference. `linear_solver`: 'direct' (SciPy sparse LU,
    default, matches every result already reported for this model) or
    'cg' (Jacobi-preconditioned CG -- see module docstring for why this
    was added and how it is validated before being trusted at scale).

    `max_cutbacks`: automatic load-step halving on Newton failure, same
    mechanism the deformable-steel model gets for free from torch-fem's
    own `model.solve()` (its own log messages say "did not converge...
    after N cutbacks" -- that IS this mechanism). This custom Newton
    loop reimplements Newton by hand (torch-fem has no rigid-MPC
    support), so it never inherited that robustness feature -- a real,
    confirmed root cause (2026-09-23): at 75,504 elements, BOTH
    linear_solver='direct' (exact) and 'cg' produced the IDENTICAL
    diverging residual trajectory (1.187e5 -> 2.274e6 -> 1.227e9),
    proving the failure was never about linear-solver accuracy at all --
    a plain, single-shot Newton step per increment is simply too fragile
    at this scale, exactly the kind of failure load-step cutback exists
    to fix."""
    device = device or torch.device("cpu")
    nodes, elements, is_shim, Lz = generate_grid_hex8_laminated_bearing(
        R_IN, R_OUT, Ntheta, Nr, n_rubber_layers=n_rubber_layers,
        nz_per_rubber=nz_per_rubber, nz_per_shim=nz_per_shim, r_grading=r_grading)
    bottom, top, outer, inner, sym = boundary_node_sets(nodes, R_IN, R_OUT, Lz)

    def hex_signed_volume(pts):
        tets = [(0, 1, 3, 4), (1, 2, 3, 6), (1, 3, 4, 6), (3, 4, 6, 7), (1, 4, 5, 6)]
        return sum(np.linalg.det(np.array([pts[b] - pts[a], pts[c] - pts[a], pts[d] - pts[a]])) / 6.0
                   for a, b, c, d in tets)
    n_inverted = sum(1 for el in elements if hex_signed_volume(nodes[el]) <= 0)
    assert n_inverted == 0, f"{n_inverted} inverted elements -- mesh invalid"

    z_boundaries, band_is_shim, band_nz = layer_bands(
        n_rubber_layers, nz_per_rubber, nz_per_shim, T_RUBBER, T_SHIM)
    groups = shim_node_groups(nodes, None, band_is_shim, z_boundaries)
    n_shims = len(groups)
    assert n_shims == n_rubber_layers - 1

    rubber_elements = elements[~is_shim]
    rubber_idx = np.nonzero(~is_shim)[0]  # maps rubber-only-model element index -> full-mesh element index
    zs, _layer_is_shim = build_z_axis(n_rubber_layers, nz_per_rubber, nz_per_shim, T_RUBBER, T_SHIM)

    old_default_dtype = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    try:
        # Real bug, confirmed directly on a live Colab run: torchfem's
        # `char_lengths` is a LAZILY-CACHED property, first computed
        # whichever moment integrate_material first calls it (inside
        # this custom Newton loop, not at Solid construction) -- so it
        # is only correctly placed on `device` if the WHOLE loop runs
        # inside `with torch.device(device):`, not just the `Solid(...)`
        # construction line. The deformable-steel model's own solve_case
        # wraps its entire `model.solve(...)` call in exactly this same
        # context for the same reason; this loop needs the same
        # wrapping, just spelled out manually since there is no single
        # `model.solve()` call here to wrap.
        with torch.device(device), torch.no_grad():
            from torchfem import Solid
            from torchfem.materials import Hyperelastic3D

            n_nodes = nodes.shape[0]
            nodes_t = torch.tensor(nodes, dtype=dtype, device=device)
            rubber_elements_t = torch.tensor(rubber_elements, dtype=torch.long, device=device)
            material = Hyperelastic3D(psi=neo_hookean_psi_3d,
                                       params=torch.tensor([MU_RUBBER, LAM_RUBBER], dtype=dtype, device=device))
            with torch.device(device):
                model = Solid(nodes_t, rubber_elements_t, material)

            # Dirichlet BCs -- identical to the deformable-steel model.
            constraints = torch.zeros(n_nodes, 3, dtype=torch.bool, device=device)
            displacements = torch.zeros(n_nodes, 3, dtype=dtype, device=device)
            constraints[bottom, :] = True
            ux, uy, uz = rigid_top_plate_displacement(
                nodes[top], Lz, compression=COMPRESSION_FRAC * Lz, shear_x=SHEAR_X, phi_y=PHI_Y)
            constraints[top, :] = True
            displacements[top, 0] = torch.tensor(ux, dtype=dtype, device=device)
            displacements[top, 1] = torch.tensor(uy, dtype=dtype, device=device)
            displacements[top, 2] = torch.tensor(uz, dtype=dtype, device=device)
            constraints[sym, 1] = True

            con_mask = constraints.ravel()
            con = torch.nonzero(con_mask, as_tuple=False).ravel()
            con_np = con.cpu().numpy()
            DU_target_full = displacements.ravel()  # full target at load factor 1

            shim_owned_node_mask = np.zeros(n_nodes, dtype=bool)
            for idx, _, _ in groups:
                shim_owned_node_mask[idx] = True
            shim_owned_dof_mask = torch.tensor(np.repeat(shim_owned_node_mask, 3), device=device)

            free_mask = (~con_mask) & (~shim_owned_dof_mask)
            free_idx = torch.nonzero(free_mask, as_tuple=False).ravel()
            free_idx_np = free_idx.cpu().numpy()
            n_free = free_idx.numel()
            n_reduced = n_free + 3 * n_shims  # 3 dof/shim: t_x, t_z, theta_y (symmetric subspace)

            shim_data = []
            for idx, z0, z1 in groups:
                idx_t = torch.tensor(idx, dtype=torch.long, device=device)
                X = nodes_t[idx_t]
                X_ref = X.mean(dim=0)
                X_rel = X - X_ref
                dof_idx_np = np.stack([idx * 3, idx * 3 + 1, idx * 3 + 2], axis=1).ravel()
                shim_data.append((idx_t, X_ref, X_rel, dof_idx_np))

            def q_to_full_du_and_J(q):
                """q: (n_reduced,) torch tensor -> (du_full (n_dofs,) torch tensor,
                J scipy.sparse.csr_matrix of shape (n_dofs, n_reduced))."""
                du_full = torch.zeros(n_nodes * 3, dtype=dtype, device=device)
                du_full[free_idx] = q[:n_free]

                J_rows, J_cols, J_vals = [], [], []
                # Free-rubber identity block.
                J_rows.append(free_idx_np)
                J_cols.append(np.arange(n_free))
                J_vals.append(np.ones(n_free))

                offset = n_free
                for k, (idx_t, X_ref, X_rel, dof_idx_np) in enumerate(shim_data):
                    qk = q[offset:offset + 3]
                    u_k, J_k = rigid_body_displacement_and_jacobian_symmetric(qk, X_rel)
                    du_full[dof_idx_np] = u_k.reshape(-1)
                    n_k = idx_t.numel()
                    J_k_np = J_k.detach().cpu().numpy().reshape(3 * n_k, 3)
                    # dof_idx_np interleaves x,y,z per node in node order, matching
                    # J_k's own (n_k, 3, 3) -> (3*n_k, 3) row-major reshape.
                    rows_k = np.repeat(dof_idx_np, 3)
                    cols_k = np.tile(np.arange(offset, offset + 3), 3 * n_k)
                    J_rows.append(rows_k)
                    J_cols.append(cols_k)
                    J_vals.append(J_k_np.ravel())
                    offset += 3
                rows = np.concatenate(J_rows)
                cols = np.concatenate(J_cols)
                vals = np.concatenate(J_vals)
                J = sp.csr_matrix((vals, (rows, cols)), shape=(n_nodes * 3, n_reduced))
                return du_full, J

            n_int, n_elem = model.n_int, model.n_elem
            u_cur = torch.zeros(n_nodes, 3, dtype=dtype, device=device)
            grad_cur = model.initial_grad.to(dtype=dtype, device=device).expand(n_int, n_elem, 3, 3).clone()
            flux_cur = torch.zeros(n_int, n_elem, 3, 3, dtype=dtype, device=device)
            state_cur = torch.zeros(n_int, n_elem, model.n_state, dtype=dtype, device=device)
            de0 = model.ext_strain.to(dtype=dtype, device=device)
            model.K = torch.empty(0)  # torch-fem's own solve() does this before its loop too

            def newton_attempt(lam0, target, q0, label):
                """ONE Newton solve for the load step [lam0, target],
                starting from the converged reduced state q0. Does NOT
                touch u_cur/grad_cur/flux_cur/state_cur (those stay the
                last COMMITTED state until the caller commits a
                converged attempt). Returns (converged, q_new, du_full,
                it_final, res_norm)."""
                step = target - lam0
                DU_step = step * DU_target_full
                q_local = q0.clone()
                res_norm = None
                for it in range(max_iter):
                    du_full, J = q_to_full_du_and_J(q_local)
                    du_full = du_full.clone()
                    du_full[con] = DU_step[con]

                    k, f_i, grad_new, flux_new, state_new = model.integrate_material(
                        u_cur, grad_cur, flux_cur, state_cur, du_full, de0, it, True)
                    K_full = model.assemble_matrix(k, con)
                    F_int_full = model.assemble_rhs(f_i)
                    R_full = F_int_full.clone()
                    R_full[con] = 0.0

                    R_full_np = R_full.detach().cpu().numpy()
                    R_reduced = J.T @ R_full_np
                    res_norm = np.linalg.norm(R_reduced)
                    if it == 0:
                        res0 = max(res_norm, 1e-30)
                    if verbose:
                        print(f"  {label} iter {it}: |R_reduced|={res_norm:.3e}")
                    if res_norm < atol or res_norm < rtol * res0:
                        return True, q_local, du_full, it, res_norm

                    K_full_sp = _csr_torch_to_scipy(K_full)
                    K_reduced = (J.T @ K_full_sp @ J).tocsc()
                    # Real bug found and fixed (2026-09-24): a bad linear
                    # solve (CG failing to converge, or the reduced
                    # tangent becoming genuinely invalid after an
                    # overshooting Newton step -- e.g. `_solve_reduced`'s
                    # own "non-positive diagonal" assertion) used to
                    # propagate straight out of this function, past the
                    # cutback loop entirely, crashing the whole solve
                    # instead of triggering a cutback. Confirmed live:
                    # cutback never fired even though the exact failure
                    # mode it exists for occurred. Treating any such
                    # failure as "this attempt did not converge" lets the
                    # caller's cutback loop retry with a smaller step,
                    # which is what should have happened all along.
                    try:
                        dq_np = _solve_reduced(K_reduced, R_reduced, method=linear_solver,
                                                cg_rtol=cg_rtol, cg_maxiter=cg_maxiter, verbose=verbose,
                                                tag=f"{label} iter {it} ")
                    except (AssertionError, RuntimeError) as e:
                        if verbose:
                            print(f"  {label} iter {it}: linear solve failed "
                                  f"({type(e).__name__}: {e}) -- triggering cutback", flush=True)
                        return False, q_local, None, None, res_norm
                    q_local = q_local + torch.tensor(dq_np, dtype=dtype, device=device)
                return False, q_local, None, None, res_norm

            increments = torch.linspace(0.0, 1.0, n_increments, dtype=dtype, device=device)
            q = torch.zeros(n_reduced, dtype=dtype, device=device)
            lam = 0.0
            shim_state_history = []
            t0 = time.time()
            for n in range(1, n_increments):
                target = float(increments[n])
                sub_target = target
                n_cutbacks = 0
                while True:
                    label = f"increment {n} (lam={lam:.4f}->{sub_target:.4f})"
                    converged, q_new, du_full, it, res_norm = newton_attempt(lam, sub_target, q, label)
                    if converged:
                        _, f_i, grad_cur, flux_cur, state_cur = model.integrate_material(
                            u_cur, grad_cur, flux_cur, state_cur, du_full, de0, it, True,
                            compute_stiffness=False)
                        u_cur = u_cur + du_full.view(-1, 3)
                        q = q_new
                        lam = sub_target
                        if sub_target >= target - 1e-12:
                            break
                        sub_target = target  # try to leap back to the full remaining step
                        continue
                    n_cutbacks += 1
                    if n_cutbacks > max_cutbacks:
                        raise RuntimeError(f"rigid-shim Newton did not converge in increment {n} "
                                            f"after {max_cutbacks} cutbacks (final |R_reduced|={res_norm:.3e})")
                    sub_target = lam + (sub_target - lam) / 2.0
                    if verbose:
                        print(f"  [cutback {n_cutbacks}] increment {n}: Newton failed, "
                              f"retrying with a smaller step up to lam={sub_target:.4f}")

                shim_q_this = []
                offset = n_free
                for k in range(n_shims):
                    shim_q_this.append(q[offset:offset + 3].detach().cpu().numpy().copy())
                    offset += 3
                shim_state_history.append(shim_q_this)

            elapsed = time.time() - t0

            u_np = u_cur.cpu().numpy()
            assert np.isfinite(u_np).all(), "NaN/Inf in solution"

            # Everything below matches mesh_convergence_B8.py's own ELEMENT-MAJOR
            # convention (n_elem, n_int, ...), i.e. the same layout compare_to_
            # reference expects -- grad_cur/flux_cur are (n_int, n_elem, ...)
            # internally (integrate_material's own convention), so transpose once
            # here before any of the region/energy bookkeeping below.
            grad_elem_major = grad_cur.transpose(0, 1)
            flux_elem_major = flux_cur.transpose(0, 1)

            F_elem = grad_elem_major.mean(dim=1)
            detF_elem = torch.linalg.det(F_elem)
            assert (detF_elem > 0).all(), "non-positive det(F) somewhere -- element inversion"
            detF_gauss = torch.linalg.det(grad_elem_major)
            assert (detF_gauss > 0).all(), "non-positive det(F) at a Gauss point"

            f_np = model.assemble_rhs(
                model.integrate_material(u_cur, grad_cur, flux_cur, state_cur,
                                          torch.zeros_like(du_full), de0, max_iter, True,
                                          compute_stiffness=False)[1]
            ).view(-1, 3).cpu().numpy()
            reaction_force_bottom = f_np[bottom].sum(axis=0)
            reaction_force_top = f_np[top].sum(axis=0)
            total_force = f_np.sum(axis=0)
            force_scale = max(np.linalg.norm(reaction_force_bottom), 1e-8)
            force_rel_residual = float(np.linalg.norm(total_force) / force_scale)

            iweights_np = model.etype.iweights.cpu().numpy()
            _, _, detJ_gauss = model.eval_shape_functions(model.etype.ipoints)
            detJ_gauss_np = detJ_gauss.transpose(0, 1).cpu().numpy()  # -> (n_elem, n_int)
            vol_weight = iweights_np[None, :] * np.abs(detJ_gauss_np)

            P_gauss_np = flux_elem_major.cpu().numpy()
            F_gauss_np = grad_elem_major.cpu().numpy()
            detF_gauss_np = detF_gauss.cpu().numpy()  # already (n_elem, n_int): grad_elem_major is already elem-major
            sigma_gauss = (np.einsum("egij,egjk->egik", P_gauss_np, F_gauss_np.transpose(0, 1, 3, 2))
                           / detF_gauss_np[..., None, None])
            sigma_elem_rubber = sigma_gauss.mean(axis=1)

            # compare_to_reference indexes region masks against the FULL mesh's
            # element ordering (rubber+shim interleaved, from generate_grid_
            # hex8_laminated_bearing), but this model only has rubber elements
            # (shim elements were dropped -- see module docstring). Scatter the
            # rubber-only per-element arrays back into full-length arrays at
            # their original mesh positions (rubber_idx), leaving shim-element
            # entries as NaN -- always excluded by construction from the
            # rubber-only-layer-1 region mask compare_to_reference builds, but
            # present so array shapes/indexing match the full element count.
            n_elem_full = len(elements)
            F_elem_full = np.full((n_elem_full, 3, 3), np.nan)
            F_elem_full[rubber_idx] = F_elem.cpu().numpy()
            sigma_elem_full = np.full((n_elem_full, 3, 3), np.nan)
            sigma_elem_full[rubber_idx] = sigma_elem_rubber
            vol_weight_full = np.full((n_elem_full, vol_weight.shape[1]), np.nan)
            vol_weight_full[rubber_idx] = vol_weight

            # Total strain energy, for Omar's requested comparison: integrate a
            # per-element strain-energy density from the rubber material's own
            # psi (Neo-Hookean), evaluated at each Gauss point's F, weighted by
            # the real volume measure -- the same vol_weight already used for
            # region averaging elsewhere in this project.
            F_gauss_t = torch.tensor(F_gauss_np, dtype=dtype)
            mu_t, lam_t = torch.tensor(MU_RUBBER, dtype=dtype), torch.tensor(LAM_RUBBER, dtype=dtype)
            I1 = torch.einsum("...ij,...ij->...", F_gauss_t, F_gauss_t)
            J_det = torch.linalg.det(F_gauss_t)
            lnJ = torch.log(J_det)
            psi_gauss = 0.5 * mu_t * (I1 - 3.0 - 2.0 * lnJ) + 0.5 * lam_t * lnJ ** 2
            total_strain_energy = float((psi_gauss.cpu().numpy() * vol_weight).sum())

            return {
                "Ntheta": Ntheta, "Nr": Nr, "n_elements": len(elements), "n_nodes": n_nodes,
                "n_shim_elements": int(is_shim.sum()), "Lz": Lz,
                "reaction_force_bottom": reaction_force_bottom.tolist(),
                "reaction_force_top": reaction_force_top.tolist(),
                "force_rel_residual": force_rel_residual,
                "total_strain_energy": total_strain_energy,
                "max_disp": float(np.linalg.norm(u_np, axis=1).max()),
                "elapsed_s": elapsed,
                "shim_state_history": shim_state_history,
                "_u": u_np, "_F_elem": F_elem_full, "_sigma_elem": sigma_elem_full,
                "_vol_weight": vol_weight_full, "_is_shim": is_shim, "_zs": zs,
                "_Ntheta": Ntheta, "_Nr": Nr, "_r_grading": r_grading,
                "_n_rubber_layers": n_rubber_layers, "_nz_per_rubber": nz_per_rubber,
                "_nz_per_shim": nz_per_shim,
            }
    finally:
        torch.set_default_dtype(old_default_dtype)
