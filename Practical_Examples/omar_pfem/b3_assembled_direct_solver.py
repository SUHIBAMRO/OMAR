"""EXPERIMENTAL, NOT YET RUN ON GPU, NOT A FINALIZED RESULT.

Port of assembled_direct_solver.py's own GPU-native assembled+direct
approach (explicit sparse tangent, cuDSS direct solve, with the two
optimizations already measured to help in 2D -- analysis-phase reuse
across Newton iterations, and symmetric-matrix storage) to B3's 3D hex
mesh, per the advisor's follow-up review: "The optimized assembled/
direct solver MUST be included ... For B3, also benchmark this solver
at relevant resolutions if possible."

B3's own existing FEM reference (data/mesh_convergence_B3.py's
solve_case) uses torch-fem's own model.solve(..., method="cg",
preconditioner="jacobi") -- an ITERATIVE solve, not assembled+direct.
This module does not replace that reference; it adds the SAME kind of
assembled+direct alternative this project already built and measured
for B1's 2D Q4 mesh, generalized to B3's 3D hex8 elements, so the two
can be compared the same way the 2D baselines already are.

WHAT IS REUSED, NOT REIMPLEMENTED:
- The reference-configuration shape-function machinery (geom["B"],
  geom["detJ"], geom["iweights"]) and the Neo-Hookean energy density
  (neo_hookean_energy_density_batched) are train_B3.py's own
  build_fixed_geometry/total_potential_energy_B3 -- the exact functions
  B3's real Deep-Energy-Method training already depends on being
  correct, not new numerics written for this module.
- The Newton-with-cuDSS-analysis-reuse loop (_newton_cudss_reuse_
  analysis) is assembled_direct_solver.py's own, completely unchanged --
  it only needs a residual_fn/jac_fn pair with the right call signature,
  and does not know or care what element type produced them.
- The sparse-tangent assembly pattern (per-element local energy, vmap +
  torch.func.hessian, static row/col COO template, fixed-row identity
  override) is tensormesh_comparison.build_sparse_jac_fn's own pattern,
  generalized from Q4's (4 local nodes x 2 components = 8 local dof) to
  hex8's (8 local nodes x 3 components = 24 local dof) -- the loop
  structure is identical, only the arities differ.

WHAT IS GENUINELY NEW (and therefore the part that needs real GPU
verification before any number from this module is trusted):
- _local_element_energy_hex: a single-hex-element version of
  total_potential_energy_B3's own per-Gauss-point energy sum.
- The residual's Dirichlet convention. B1/B2's existing convention
  (torch.where(free_mask_dof, res, u_flat)) assumes every fixed DOF's
  target is exactly 0, which is true for B1/B2 but NOT for B3: B3's
  inner surface is fixed to a NONZERO rigid-rotation displacement. The
  generalization used here -- torch.where(free_mask_dof, res, u_flat -
  u_prescribed) -- reduces to the original when u_prescribed=0 and is
  the standard way to impose a nonzero essential BC by row replacement;
  it does not change build_sparse_jac_fn_b3's identity-row Jacobian
  entries at all, since d(u_i - u_prescribed_i)/du = identity regardless
  of the (constant) target value.

STANDING CONSTRAINT (same as assembled_direct_solver.py's own, and
applies equally here): this must be verified on a GPU test notebook
first -- specifically, run_correctness_check() below, comparing this
solver's displacement field against data/mesh_convergence_B3.py's own
already-trusted solve_case() at a small, cheap resolution -- and then
run past the advisor, before any number from this module is finalized,
quoted in the paper, or presented as an official project result. The
benchmark driver (run_b3_benchmark) refuses to run unless the
correctness check for that exact resolution has already passed in the
same process, so a timing number can never be produced without the
check that justifies trusting it.
"""
import argparse
import json
import os
import time

import numpy as np
import torch

from omar_pfem.assembled_direct_solver import _newton_cudss_reuse_analysis
from omar_pfem.train_B3 import (
    build_fixed_geometry, rigid_rotation_displacement_torch, total_potential_energy_B3)
from omar_pfem.data.data_generate_B3_dataset import DEFAULT_RESOLUTION, sample_material_and_load


def _det3x3(F):
    """Explicit cofactor-expansion determinant of a single (3,3) matrix.
    NOT torch.linalg.slogdet: a real, GPU-verified-on-CPU bug was found
    here (2026-10-10) where vmap(torch.func.hessian(...)) over this
    module's per-element energy, when that energy used slogdet, silently
    returned garbage (~1e9-1e15x too large) for SOME elements in a batch
    while being exactly correct for others and exactly correct when the
    same element was differentiated alone (outside vmap) -- a vmap+
    nested-hessian+slogdet composition issue, not a physics or indexing
    bug (confirmed by comparing against torch.autograd.functional.hessian
    computed one element at a time, which was correct for every element
    including the ones vmap got wrong). train_B3.py's own
    neo_hookean_energy_density_batched (which DOES use slogdet) is left
    untouched since it is only ever differentiated once, via torch.func.
    grad with no vmap or nested hessian, where this issue does not arise
    -- confirmed by this module's own residual() working correctly with
    it. This module's 2D sibling (materials_torch.py's neo-Hookean) never
    hit this because it already uses an explicit 2x2 determinant formula,
    not slogdet -- the same fix, generalized to 3x3, applied here."""
    return (F[0, 0] * (F[1, 1] * F[2, 2] - F[1, 2] * F[2, 1])
            - F[0, 1] * (F[1, 0] * F[2, 2] - F[1, 2] * F[2, 0])
            + F[0, 2] * (F[1, 0] * F[2, 1] - F[1, 1] * F[2, 0]))


def _neo_hookean_psi_explicit_det(F, mu, lam):
    """Single-matrix Neo-Hookean density, same formula as train_B3.py's
    own neo_hookean_energy_density_batched (verified to match it to
    machine precision at non-degenerate F), via _det3x3 instead of
    slogdet -- see _det3x3's own docstring for why. F: (3,3); mu, lam:
    scalars (0-dim tensors)."""
    J = torch.clamp(_det3x3(F), min=1e-8)
    I1 = torch.sum(F ** 2)
    lnJ = torch.log(J)
    return (mu / 2.0) * (I1 - 3.0 - 2.0 * lnJ) + (lam / 2.0) * (lnJ ** 2)

_CORRECTNESS_VERIFIED = False
"""Set to True only by a passing call to run_correctness_check() in THIS
process. run_b3_benchmark() checks this instead of trusting a caller-
supplied flag, so a benchmark call can't accidentally skip the gate by
passing the wrong boolean -- the only way to set this True is to have
just run and passed the real check."""


def _local_element_energy_hex(ue_flat, B_e, detJ_e, iweights, mu_e, lam_e):
    """Single-element energy, vmappable over elements. ue_flat: (24,) =
    local (8_node, 3_dof) flattened NODE-MAJOR (index = node*3+component)
    -- the same convention matrix_free_solver._local_element_energy uses
    for Q4 (ue_flat.reshape(n_local, 2)), so build_sparse_jac_fn_b3's own
    block-slicing of the resulting local Hessian by (node_a, node_b) is
    valid the same way build_sparse_jac_fn's is. B_e: (8_gauss, 3, 8) --
    geom["B"] sliced/permuted to this one element, in (dof-component,
    node) order; H[d,c] = sum_q ue[q,d] * B_e[g][c,q] reproduces total_
    potential_energy_B3's own per-element contribution to grad_u at Gauss
    point g, just with ue's two axes in the opposite (node-major) order.
    detJ_e: (8_gauss,). iweights: (8_gauss,), shared across elements."""
    ue = ue_flat.reshape(8, 3)
    eye = torch.eye(3, dtype=ue.dtype, device=ue.device)
    U = torch.zeros((), dtype=ue.dtype, device=ue.device)
    for g in range(B_e.shape[0]):
        H = torch.einsum("qd,cq->dc", ue, B_e[g])
        F = eye + H
        psi = _neo_hookean_psi_explicit_det(F, mu_e, lam_e)
        U = U + psi * detJ_e[g] * iweights[g]
    return U


def build_sparse_jac_fn_b3(geom, E_node, nu_node, free_mask_dof, device, dtype):
    """Hex8 analog of tensormesh_comparison.build_sparse_jac_fn: 8 local
    nodes x 3 components = 24 local dof per element (vs. Q4's 4x2=8),
    otherwise the identical static-template + vmap+hessian + fixed-row-
    identity pattern. E_node, nu_node: (n_nodes,) -- ONE sample's material
    field (not batched over many samples; an FEM solve is for one sample
    at a time)."""
    from torch.func import hessian, vmap

    elements = geom["elements"]  # (n_elem, 8)
    n_elem, n_local = elements.shape
    n_nodes = geom["n_nodes"]
    n_dof = 3 * n_nodes

    B_perm = geom["B"].permute(1, 0, 2, 3).contiguous()   # (n_elem, 8_gauss, 3, 8)
    detJ_perm = geom["detJ"].permute(1, 0).contiguous()   # (n_elem, 8_gauss)
    iweights = geom["iweights"]                           # (8_gauss,)

    E_elem = E_node[elements].mean(dim=1)   # (n_elem,)
    nu_elem = nu_node[elements].mean(dim=1)
    mu_elem = E_elem / (2 * (1 + nu_elem))
    lam_elem = E_elem * nu_elem / ((1 + nu_elem) * (1 - 2 * nu_elem))

    local_hess_fn = hessian(_local_element_energy_hex, argnums=0)
    batched_hess = vmap(local_hess_fn, in_dims=(0, 0, 0, None, 0, 0))

    # Static (row, col) template -- one entry per (element, local-node-pair,
    # component-pair): (n_elem, 8, 8, 3, 3) flattened, d-major/q-minor to
    # match ue_flat's own (3,8)-flatten order used above.
    rows, cols = [], []
    for a in range(n_local):
        for b in range(n_local):
            for ca in range(3):
                for cb in range(3):
                    rows.append(3 * elements[:, a] + ca)
                    cols.append(3 * elements[:, b] + cb)
    row_template = torch.cat(rows)
    col_template = torch.cat(cols)
    free_row_mask = free_mask_dof[row_template]
    fixed_idx = (~free_mask_dof).nonzero(as_tuple=True)[0]

    def jac_fn(u, _A):
        ue_all = u.reshape(n_nodes, 3)[elements].reshape(n_elem, -1)  # (n_elem, 24), node-major
        H_local = batched_hess(ue_all, B_perm, detJ_perm, iweights, mu_elem, lam_elem)
        vals = []
        for a in range(n_local):
            for b in range(n_local):
                block = H_local[:, 3 * a:3 * a + 3, 3 * b:3 * b + 3]  # (n_elem,3,3)
                for ca in range(3):
                    for cb in range(3):
                        vals.append(block[:, ca, cb])
        val_template = torch.cat(vals)
        val = torch.where(free_row_mask, val_template, torch.zeros_like(val_template))
        val = torch.cat([val, torch.ones(fixed_idx.shape[0], dtype=dtype, device=device)])
        row = torch.cat([row_template, fixed_idx])
        col = torch.cat([col_template, fixed_idx])
        return val.detach(), row, col, (n_dof, n_dof)

    return jac_fn


def build_fixed_geometry_with_groove(Ntheta, Nr, Nz, device, dtype, groove_depth,
                                      groove_half_width, r_grading=1.0):
    """train_B3.py's own build_fixed_geometry, with the groove geometry
    and R_GRADING passed in explicitly instead of hardcoded from data_
    generate_B3_dataset's production constants -- needed because run_
    correctness_check must match data/mesh_convergence_B3.py's own
    (different, older) groove constants exactly, not the production
    geometry, to compare like with like (see run_correctness_check's own
    docstring for the real mismatch this caught). Otherwise identical to
    build_fixed_geometry line for line; not a new implementation."""
    from omar_pfem.data.data_generate_B3 import generate_grid_hex8_bushing, boundary_node_sets
    from omar_pfem.data.mesh_convergence_B3 import R_IN0, R_OUT, LZ

    nodes, elements = generate_grid_hex8_bushing(
        R_IN0, R_OUT, LZ, Ntheta, Nr, Nz, groove_depth, groove_half_width, r_grading=r_grading)
    inner, outer, sym = boundary_node_sets(nodes, R_IN0, R_OUT, LZ, groove_depth, groove_half_width)

    thetas = np.linspace(0.0, np.pi, Ntheta)
    ts = np.linspace(0.0, 1.0, Nr) ** r_grading
    j_idx = (np.arange(Ntheta * Nr * Nz) // Nr) % Ntheta
    i_idx = np.arange(Ntheta * Nr * Nz) % Nr
    theta_node = thetas[j_idx]
    t_node = ts[i_idx]

    old_default_dtype = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    try:
        from torchfem import Solid
        from torchfem.materials import Hyperelastic3D
        from omar_pfem.torchfem_comparison import neo_hookean_psi_3d

        nodes_t = torch.tensor(nodes, dtype=dtype, device=device)
        elements_t = torch.tensor(elements, dtype=torch.long, device=device)
        dummy_params = torch.tensor([1.0, 1.0], dtype=dtype, device=device)
        material = Hyperelastic3D(psi=neo_hookean_psi_3d, params=dummy_params)
        with torch.device(device):
            model = Solid(nodes_t, elements_t, material)
        N, B, detJ = model.eval_shape_functions(model.etype.ipoints)
        iweights = model.etype.iweights.to(device=device, dtype=dtype)
    finally:
        torch.set_default_dtype(old_default_dtype)

    return {
        "nodes": nodes_t, "elements": elements_t,
        "inner": torch.tensor(inner, device=device), "outer": torch.tensor(outer, device=device),
        "sym": torch.tensor(sym, device=device),
        "theta_node": torch.tensor(theta_node, dtype=dtype, device=device),
        "t_node": torch.tensor(t_node, dtype=dtype, device=device),
        "B": B, "detJ": detJ, "iweights": iweights,
        "n_nodes": nodes.shape[0], "n_elements": elements.shape[0],
    }


def _b3_dirichlet(geom, phi, device, dtype):
    """Discrete, node-wise Dirichlet data for a direct FEM solve (NOT
    train_B3.py's own continuous-ramp network ansatz, which has no
    meaning here -- there is no network): outer fully fixed at 0, inner
    fixed to the true rigid-rotation displacement, symmetry nodes fixed
    in u_y only. Mirrors data/mesh_convergence_B3.py's solve_case own
    constraints/displacements construction exactly, just assembled into
    a flat free_mask_dof / prescribed_flat pair for the assembled-direct
    residual below instead of torch-fem's own Mesh.constraints API."""
    n_nodes = geom["n_nodes"]
    free_mask = torch.ones(3 * n_nodes, dtype=torch.bool, device=device)
    prescribed = torch.zeros(3 * n_nodes, dtype=dtype, device=device)

    # geom["outer"]/["inner"]/["sym"] are BOOLEAN masks over nodes (shape
    # (n_nodes,), boundary_node_sets' own convention -- confirmed directly,
    # not assumed: `outer.dtype` is torch.bool), not integer index lists.
    # Converting to index tensors here is required: `3 * outer + c` on a
    # bool tensor silently does elementwise int arithmetic over ALL
    # n_nodes entries (producing values c or c+3) instead of selecting the
    # True positions -- a real bug caught by run_correctness_check before
    # this fix (shape mismatch at assignment, (n_nodes,) vs the expected
    # per-boundary count, since this file originally treated them as index
    # arrays by mistake).
    outer = geom["outer"].nonzero(as_tuple=True)[0]
    inner = geom["inner"].nonzero(as_tuple=True)[0]
    sym = geom["sym"].nonzero(as_tuple=True)[0]
    for c in range(3):
        free_mask[3 * outer + c] = False
    x0, z0 = geom["nodes"][inner, 0], geom["nodes"][inner, 2]
    phi_t = torch.as_tensor([phi], dtype=dtype, device=device)
    ux, uy, uz = rigid_rotation_displacement_torch(x0, z0, phi_t)
    ux, uy, uz = ux[0], uy[0], uz[0]
    for c in range(3):
        free_mask[3 * inner + c] = False
    prescribed[3 * inner + 0] = ux
    prescribed[3 * inner + 1] = uy
    prescribed[3 * inner + 2] = uz
    free_mask[3 * sym + 1] = False  # u_y = 0 on the symmetry plane
    return free_mask, prescribed


def solve_b3_assembled_direct(geom, E_node, nu_node, phi, device, dtype=torch.float64,
                               tol=1e-8, max_iter=30, reuse_analysis=True,
                               matrix_type="general", return_stats=False):
    """One B3 Newton solve via the assembled+direct (cuDSS) path. Returns
    (u_np (n_nodes,3), stats) if return_stats else u_np. CUDA-only when
    reuse_analysis=True (cuDSS has no CPU path -- same constraint as
    assembled_direct_solver.solve_assembled_direct)."""
    n_nodes = geom["n_nodes"]
    n_dof = 3 * n_nodes
    free_mask_dof, prescribed = _b3_dirichlet(geom, phi, device, dtype)

    jac_fn_full = build_sparse_jac_fn_b3(geom, E_node, nu_node, free_mask_dof, device, dtype)

    E_node_b = E_node.unsqueeze(0)   # (1, n_nodes) -- total_potential_energy_B3's own batch convention
    nu_node_b = nu_node.unsqueeze(0)

    def total_energy(u_flat):
        # Reuses train_B3.py's own already-trusted, already-vectorized
        # (einsum-over-all-elements, no Python per-element loop) energy
        # function directly, at batch size 1 -- NOT a second, hand-rolled
        # implementation. Bit-identical physics to what B3's real DEM
        # training already depends on being correct.
        return total_potential_energy_B3(u_flat.reshape(1, n_nodes, 3), E_node_b, nu_node_b, geom)[0]

    def residual(u_flat, _A, f_ext):
        # B3 has no external-force term (prescribed-displacement loading
        # only, train_B3.py's own Pi=U convention) -- f_ext is zero and
        # kept only so this matches _newton_cudss_reuse_analysis's own
        # residual_fn(u, A, f_ext) call signature.
        grad_full = torch.func.grad(total_energy)(u_flat) - f_ext
        return torch.where(free_mask_dof, grad_full, u_flat - prescribed)

    u0 = prescribed.clone()  # start from the Dirichlet data, not zero -- B3's inner
                              # boundary is far from u=0, and zero-init plus a large
                              # rigid rotation risks element inversion on the first
                              # Newton step (same reasoning as OUTPUT_SCALE in train_B3.py)
    f_ext = torch.zeros(n_dof, dtype=dtype, device=device)

    if reuse_analysis:
        if device.type != "cuda":
            raise RuntimeError("reuse_analysis=True requires CUDA (cuDSS has no CPU path).")
        u, stats = _newton_cudss_reuse_analysis(
            residual, jac_fn_full, u0, f_ext, tol=tol, atol=1e-10, max_iter=max_iter,
            line_search=True, verbose=False, matrix_type=matrix_type)
    else:
        from torch_sla import SparseTensor
        ident = torch.arange(n_dof, device=device)
        A = SparseTensor(torch.ones(n_dof, dtype=dtype, device=device), ident, ident, (n_dof, n_dof))
        u = A.nonlinear_solve(residual, u0, f_ext, jac_fn=jac_fn_full, method="newton",
                              verbose=False, max_iter=max_iter, tol=tol, linear_method="lu",
                              linear_solver=("cudss" if device.type == "cuda" else "auto"))
        stats = None

    u_np = u.reshape(n_nodes, 3).detach().cpu().numpy()
    if return_stats:
        return u_np, stats
    return u_np


def run_correctness_check(Ntheta=7, Nr=6, Nz=6, device=None, dtype=torch.float64, tol=1e-3):
    """Mandatory gate (see module docstring): solves the SAME small B3
    case two ways -- this module's assembled+direct solver, and data/
    mesh_convergence_B3.py's own already-trusted solve_case (torch-fem,
    CG+Jacobi) -- and reports the agreement. Deliberately tiny/cheap
    (Ntheta=7,Nr=6,Nz=6 -- well below the production 21x20x19) so this
    runs in seconds and can gate every real benchmark run below it.

    solve_case is NOT parameterized by an (E, nu, phi) sample -- it
    always uses data/mesh_convergence_B3.py's own fixed module-level E,
    NU, PHI constants (E=1000.0, NU=0.45, PHI=0.05). This check uses the
    SAME constants for the assembled+direct side, so both solves see
    the identical material/loading, not two different problems that
    happen to use the same mesh.

    IMPORTANT, found and fixed 2026-10-10: data/mesh_convergence_B3.py's
    own GROOVE_DEPTH (0.05) is NOT the production groove depth (0.20,
    data/data_generate_B3_dataset.py's GROOVE_DEPTH -- a shallower groove
    left over from an earlier mesh-convergence round, before the groove
    was deepened to its final production value). train_B3.py's own
    build_fixed_geometry hardcodes the PRODUCTION constants, so calling
    it directly here would silently compare two solvers on two DIFFERENT
    geometries and misreport a ~3-8% field disagreement as a solver bug
    (confirmed directly: that is exactly what happened on the first run
    of this check, traced to this exact constant mismatch, not a solver
    defect). This check therefore builds its own geometry with mesh_
    convergence_B3.py's OWN groove constants, matching solve_case's
    geometry exactly -- run_b3_benchmark below still uses the real
    production geometry via build_fixed_geometry, which is correct for
    what it reports.

    Returns (max_abs_diff, max_rel_diff, passed: bool); raises if
    max_rel_diff > tol rather than returning a silently-unusable result."""
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    from omar_pfem.data.mesh_convergence_B3 import (
        E as E_CONST, NU as NU_CONST, PHI as PHI_CONST,
        GROOVE_DEPTH as CHECK_GROOVE_DEPTH, GROOVE_HALF_WIDTH as CHECK_GROOVE_HALF_WIDTH,
        R_IN0, R_OUT, LZ, solve_case)

    print(f"[correctness check] Ntheta={Ntheta}, Nr={Nr}, Nz={Nz}, device={device}, "
          f"E={E_CONST}, NU={NU_CONST}, PHI={PHI_CONST}, "
          f"groove_depth={CHECK_GROOVE_DEPTH} (solve_case's own, NOT the production 0.20)")

    t0 = time.time()
    ref = solve_case(Ntheta, Nr, Nz, dtype=dtype, device=device, verbose=False)
    u_ref = ref["_u"]  # (n_nodes, 3) numpy, torch-fem's own already-trusted solve
    print(f"  reference (torch-fem, CG+Jacobi): {time.time() - t0:.2f}s, "
          f"max_disp={ref['max_disp']:.4f}")

    geom = build_fixed_geometry_with_groove(
        Ntheta, Nr, Nz, device, dtype, CHECK_GROOVE_DEPTH, CHECK_GROOVE_HALF_WIDTH,
        r_grading=1.0)
    E_node = torch.full((geom["n_nodes"],), E_CONST, dtype=dtype, device=device)
    nu_node = torch.full((geom["n_nodes"],), NU_CONST, dtype=dtype, device=device)

    t0 = time.time()
    u_new = solve_b3_assembled_direct(
        geom, E_node, nu_node, PHI_CONST, device, dtype=dtype,
        reuse_analysis=(device.type == "cuda"), matrix_type="general")
    print(f"  new (assembled+direct{'+ cuDSS' if device.type == 'cuda' else ' (CPU)'}): "
          f"{time.time() - t0:.2f}s, max_disp={np.linalg.norm(u_new, axis=1).max():.4f}")

    abs_diff = np.abs(u_new - u_ref)
    scale = max(np.abs(u_ref).max(), 1e-12)
    max_abs = float(abs_diff.max())
    max_rel = float(max_abs / scale)
    passed = max_rel <= tol
    print(f"  max abs diff: {max_abs:.3e}, max rel diff (vs. field magnitude): {max_rel:.3e} "
          f"-- {'PASS' if passed else 'FAIL'} (tol={tol:.0e})")
    if not passed:
        raise RuntimeError(
            f"B3 assembled+direct solver disagrees with the trusted torch-fem "
            f"reference by {max_rel:.3e} (tol={tol:.0e}) at Ntheta={Ntheta}, Nr={Nr}, "
            f"Nz={Nz} -- DO NOT trust or report any timing from this module until "
            f"this is root-caused and fixed. Likely places to check first: the "
            f"Dirichlet residual convention in solve_b3_assembled_direct (B3's "
            f"nonzero inner-rotation BC, generalized here from B1/B2's own "
            f"u_fixed=0 convention), or the local-element dof ordering in "
            f"_local_element_energy_hex / build_sparse_jac_fn_b3.")
    global _CORRECTNESS_VERIFIED
    _CORRECTNESS_VERIFIED = True
    return max_abs, max_rel, passed


def run_b3_benchmark(resolutions, out_json, device=None, dtype=torch.float64,
                      reuse_analysis=True, matrix_type="general", seed=0):
    """Benchmarks solve_b3_assembled_direct at the given (Ntheta,Nr,Nz)
    resolutions -- pass DEFAULT_RESOLUTION (the 6,840-element production
    mesh) and/or triples from this project's own mesh-convergence ladder
    for an apples-to-apples comparison with already-published B3 numbers.
    REFUSES to run unless run_correctness_check() has already passed in
    THIS process (see module docstring and _CORRECTNESS_VERIFIED) -- the
    gate cannot be bypassed by a caller-supplied flag, only by actually
    running and passing the check first."""
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not _CORRECTNESS_VERIFIED:
        raise RuntimeError(
            "run_b3_benchmark refuses to run before run_correctness_check() "
            "has passed in this process -- see this module's own standing-"
            "constraint docstring. This gate is deliberate, not a bug: call "
            "run_correctness_check() first (it is fast -- a small mesh) and "
            "only then call this function again in the same process/notebook.")

    rows = []
    for (Ntheta, Nr, Nz) in resolutions:
        geom = build_fixed_geometry(Ntheta, Nr, Nz, device, dtype=dtype)
        E_node_np, nu_node_np, phi = sample_material_and_load(Ntheta, Nr, Nz, seed=seed)
        E_node = torch.tensor(E_node_np, dtype=dtype, device=device)
        nu_node = torch.tensor(nu_node_np, dtype=dtype, device=device)

        torch.cuda.synchronize() if device.type == "cuda" else None
        t0 = time.time()
        u, stats = solve_b3_assembled_direct(
            geom, E_node, nu_node, phi, device, dtype=dtype,
            reuse_analysis=reuse_analysis, matrix_type=matrix_type, return_stats=True)
        torch.cuda.synchronize() if device.type == "cuda" else None
        elapsed = time.time() - t0
        peak_mem = torch.cuda.max_memory_allocated(device) / 1e6 if device.type == "cuda" else None

        row = {"Ntheta": Ntheta, "Nr": Nr, "Nz": Nz,
               "n_elements": geom["n_elements"], "n_nodes": geom["n_nodes"],
               "elapsed_s": elapsed, "peak_mem_mb": peak_mem,
               "reuse_analysis": reuse_analysis, "matrix_type": matrix_type,
               "newton_stats": stats}
        rows.append(row)
        print(f"[Ntheta={Ntheta},Nr={Nr},Nz={Nz}] {geom['n_elements']:,} elements: "
              f"{elapsed:.2f}s, peak_mem={peak_mem}")
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)

    with open(out_json, "w") as f:
        json.dump(rows, f, indent=2)
    print(f"Written to {out_json}")
    return rows


if __name__ == "__main__":
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--run_correctness_check", action="store_true")
    p.add_argument("--run_benchmark", action="store_true")
    p.add_argument("--out_json", default="b3_assembled_direct_benchmark.json")
    p.add_argument("--resolutions", default="21,20,19",
                   help="semicolon-separated Ntheta,Nr,Nz triples, e.g. "
                        "'21,20,19;27,26,25'")
    args = p.parse_args()

    if args.run_correctness_check:
        run_correctness_check()
    if args.run_benchmark:
        if not args.run_correctness_check:
            run_correctness_check()  # the gate requires it to have run in THIS process
        res_list = [tuple(int(x) for x in trip.split(",")) for trip in args.resolutions.split(";")]
        run_b3_benchmark(res_list, args.out_json)
