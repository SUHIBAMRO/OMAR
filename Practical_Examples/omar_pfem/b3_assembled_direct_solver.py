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

_CORRECTNESS_VERIFIED = set()
"""(matrix_type, symmetric_bc) pairs added ONLY by a passing call to
run_correctness_check() with that same pair, in THIS process.
run_b3_benchmark() checks ITS OWN (matrix_type, symmetric_bc) is a member
instead of trusting a caller-supplied flag or a check of a DIFFERENT
configuration -- added 2026-10-11 when symmetric_bc/matrix_type="symmetric"
support was added, so passing the default ("general", False) check can
never be silently treated as clearance for the newer symmetric path too;
each combination actually benchmarked must have been separately verified."""


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


def build_sparse_jac_fn_b3(geom, E_node, nu_node, free_mask_dof, device, dtype,
                            symmetric_bc=False, vmap_chunk_size=None):
    """Hex8 analog of tensormesh_comparison.build_sparse_jac_fn: 8 local
    nodes x 3 components = 24 local dof per element (vs. Q4's 4x2=8),
    otherwise the identical static-template + vmap+hessian + fixed-row-
    identity pattern. E_node, nu_node: (n_nodes,) -- ONE sample's material
    field (not batched over many samples; an FEM solve is for one sample
    at a time).

    symmetric_bc=False (default, unchanged): masks entries by ROW only,
    same as tensormesh_comparison.build_sparse_jac_fn's own default --
    the stored matrix is non-symmetric (general) even though the
    underlying continuous tangent is a symmetric Hessian, because a
    free row's entry in a FIXED column survives while the mirror (a
    fixed row's entry in a free column) is zeroed by the row mask.

    symmetric_bc=True (opt-in, 2026-10-11, added chasing a real cuDSS
    ALLOC_FAILED on B3's larger ladder resolutions -- see PROJECT_STATUS.
    md): additionally masks by column, producing a genuinely symmetric
    matrix, mirroring tensormesh_comparison's own symmetric_bc=True.
    UNLIKE that 2D version, this is only an EXACT equivalent reformulation
    if the CALLER also applies the Dirichlet lift to the Newton solve's
    own initial guess u0 (u0[fixed] := prescribed) BEFORE the first
    iteration -- B3's prescribed values are NOT identically zero like the
    2D project's homogeneous BCs, so du_fixed is only guaranteed to stay
    exactly 0 at every iteration (making the dropped column's
    contribution, K[free,fixed] @ du_fixed, exactly 0 too) if u already
    satisfies the full prescribed BC from the very first iterate, not
    just at convergence. solve_b3_assembled_direct does this; do not use
    symmetric_bc=True from any other caller without the same lift.

    vmap_chunk_size=None (default, unchanged): the outer vmap over
    elements processes all n_elem elements' local Hessians in one single
    batched call. Real GPU evidence (2026-10-11, B3's 480,320-element
    ladder row, see PROJECT_STATUS.md): this costs ~6.6 GB of PyTorch-
    tracked memory for that one mesh, EVERY Newton iteration (jac_fn
    recomputes H_local fresh each call, nothing is cached across
    iterations) -- on top of whatever cuDSS's own ANALYSIS/FACTORIZATION
    holds for the whole solve, this was enough to OOM on iteration 2 even
    after symmetric_bc=True's storage savings. Passing an integer here
    forwards it to torch.func.vmap's own chunk_size (processes that many
    elements at a time instead of all of them at once) -- mathematically
    identical output (vmap with chunking is required to give the exact
    same result, not an approximation), trading some wall-clock time for
    much lower peak memory. Not yet verified end-to-end on GPU; verified
    on CPU to give bit-identical output to vmap_chunk_size=None."""
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
    batched_hess = vmap(local_hess_fn, in_dims=(0, 0, 0, None, 0, 0),
                         chunk_size=vmap_chunk_size)

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
    # rows/cols are 576 separate (n_elem,) int64 buffers (genuine storage,
    # not views -- each came from an arithmetic op `3*elements[:,a]+ca`, not
    # plain indexing), together the SAME total size as row_template/
    # col_template. torch.cat materializes a new buffer, so without this,
    # both the pre-cat pieces and the concatenated result stay alive at
    # once -- a real, measured contributor to this function's ~24.7 GB
    # one-time setup cost at B3's 839,040-element ladder row (see
    # PROJECT_STATUS.md), found chasing the still-open memory ceiling
    # blocking that resolution. Safe: row_template/col_template (what the
    # closure below actually needs) are untouched.
    del rows, cols
    free_row_mask = free_mask_dof[row_template]
    keep_mask = (free_row_mask & free_mask_dof[col_template]) if symmetric_bc else free_row_mask
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
        # vals' 576 entries are VIEWS into H_local's own storage (confirmed
        # directly: pure basic indexing -- slices + Python ints -- always
        # returns a view in PyTorch, never a copy), so H_local's storage
        # stays alive as long as ANY tensor still references it; deleting
        # H_local alone (without also dropping every view into it) would
        # free nothing. torch.cat(vals) below allocates a genuinely NEW,
        # independent buffer the same total size as H_local -- so without
        # this del, H_local's storage and val_template are both alive at
        # once, real redundant memory, not a hypothetical one: this
        # function is called every Newton iteration (not just once per
        # increment), so this cost recurs on every call, and is a direct,
        # measured contributor to the OOM/ALLOC_FAILED ceiling blocking
        # B3's 839,040-element ladder row (PROJECT_STATUS.md).
        #
        # `block` MUST also be deleted here, not just H_local/vals -- a
        # REAL bug in this fix's first version, found from a real GPU run
        # that showed zero measured memory improvement despite this same
        # del being present: `block` is the Python for-loop variable, and
        # Python does NOT scope loop variables to the loop body -- after
        # the nested loop above exits, `block` still names the LAST
        # iteration's view (a=7,b=7 slice), independently holding its own
        # reference to H_local's storage, regardless of `del H_local,
        # vals`. A storage is freed only once EVERY tensor view into it is
        # gone; one leftover reference (`block`) is enough to keep the
        # whole thing alive. Confirmed directly with a weakref-based CPU
        # test: deleting H_local+vals alone left the storage alive;
        # additionally deleting `block` collected it. Bit-identical output
        # either way regardless of this bug -- val_template only ever
        # reads from these views, never writes to them, so the computed
        # VALUES were always correct; only the memory was never actually
        # freed.
        val_template = torch.cat(vals)
        del H_local, vals, block
        val = torch.where(keep_mask, val_template, torch.zeros_like(val_template))
        del val_template
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
                               matrix_type="general", return_stats=False,
                               n_increments=11, symmetric_bc=False, vmap_chunk_size=None):
    """B3 Newton solve via the assembled+direct (cuDSS) path, using
    INCREMENTAL loading (phi ramped 0 -> target over n_increments steps,
    each warm-started from the previous increment's converged solution)
    -- NOT a single full-load Newton solve from the Dirichlet data.

    This matters: a first real-GPU run of this function (single-shot,
    no incremental loading) PASSED the small-mesh correctness check
    (Ntheta=9,Nr=7,Nz=6, phi=0.05, constant material) but FAILED TO
    CONVERGE at the production mesh with a real sampled material field
    and phi (Ntheta=21,Nr=20,Nz=19, ||F||=4.1e4 after the full 30-
    iteration budget, silently producing a 'successful' timing number
    for a physically meaningless field). data/mesh_convergence_B3.py's
    own already-trusted solve_case -- the reference this module is
    checked against -- never attempts a single-shot full-load solve
    either: it already uses 11 load increments
    (`torch.linspace(0.0, 1.0, 11)`) for exactly this reason (standard
    nonlinear-FEM practice for a large prescribed-displacement BC with
    real material heterogeneity, not something this module invented).
    n_increments=11 here matches that convention so the two solvers are
    compared on equal footing, not a harder-for-no-reason single shot
    vs. an easier incremental reference.

    Returns (u_np (n_nodes,3), stats) if return_stats else u_np, where
    stats now also carries "converged" (bool, True only if every
    increment converged) and "increment_stats" (list, one entry per
    increment) -- return_stats=True always needed to check this before
    trusting any timing number. CUDA-only when reuse_analysis=True
    (cuDSS has no CPU path -- same constraint as assembled_direct_
    solver.solve_assembled_direct).

    symmetric_bc=False (default, unchanged -- every already-reported
    number, including the real GPU timings in PROJECT_STATUS.md, used
    this): build_sparse_jac_fn_b3's own non-symmetric (general) matrix,
    matrix_type should stay "general" to match. symmetric_bc=True
    (opt-in, 2026-10-11, added chasing a real cuDSS ALLOC_FAILED at
    B3's larger ladder resolutions -- the assembled matrix is a
    continuous Hessian and IS mathematically symmetric, but was never
    being stored/factorized that way, costing cuDSS extra ANALYSIS/
    FACTORIZATION memory for no reason): also explicitly lifts each
    increment's initial guess u0 to satisfy the Dirichlet data EXACTLY
    at every fixed DOF before Newton starts (u0 := where(free, u0,
    prescribed)) -- WITHOUT this, du_fixed can be nonzero on an
    increment's first iteration (B3's prescribed values are not
    identically zero like the 2D project's homogeneous BCs), and
    dropping the fixed-DOF columns (what symmetric_bc=True does) would
    then silently discard a real, nonzero coupling term instead of a
    provably-zero one. With the lift, du_fixed=0 is exact from the
    first iterate onward, matching build_sparse_jac_fn_b3's own
    symmetric_bc=True docstring exactly. Pair with matrix_type=
    "symmetric" to actually get cuDSS's cheaper symmetric storage/
    factorization; passing symmetric_bc=True with matrix_type="general"
    would needlessly also still work (the dropped entries are zero) but
    buys nothing. Verified on GPU 2026-10-11 (B3's production mesh,
    6,840 elements): converged, 10.33s, matching the general path's own
    result -- but at the 480,320-element ladder resolution, symmetric_bc=
    True cut cuDSS's ANALYSIS memory from ~34 GB to ~28 GB and this
    module's own CSR/coalesce setup from ~5.85 GB to ~1.1 GB (getting
    PAST the earlier ALLOC_FAILED, all the way through FACTORIZATION),
    but still hit a real CUDA OutOfMemoryError on the SECOND Newton
    iteration's jac_fn(u,None) call -- the vmap(hessian(...)) computation
    itself costs ~6.6 GB of PyTorch-tracked memory per call, recomputed
    fresh (not cached) every iteration, which on top of cuDSS's own ~28 GB
    held for the whole solve was enough to exceed 80 GB. See
    vmap_chunk_size below for the next real attempt at this.

    vmap_chunk_size=None (default, unchanged): forwarded to
    build_sparse_jac_fn_b3 -- see its own docstring. Caps the outer
    vmap's own internal batch size over elements, trading wall-clock for
    peak memory; mathematically identical output."""
    import warnings

    n_nodes = geom["n_nodes"]
    n_dof = 3 * n_nodes
    E_node_b = E_node.unsqueeze(0)   # (1, n_nodes) -- total_potential_energy_B3's own batch convention
    nu_node_b = nu_node.unsqueeze(0)

    def total_energy(u_flat):
        # Reuses train_B3.py's own already-trusted, already-vectorized
        # (einsum-over-all-elements, no Python per-element loop) energy
        # function directly, at batch size 1 -- NOT a second, hand-rolled
        # implementation. Bit-identical physics to what B3's real DEM
        # training already depends on being correct.
        return total_potential_energy_B3(u_flat.reshape(1, n_nodes, 3), E_node_b, nu_node_b, geom)[0]

    f_ext = torch.zeros(n_dof, dtype=dtype, device=device)
    u = torch.zeros(n_dof, dtype=dtype, device=device)  # increment 0: phi=0, u=0 is exact
    increment_stats = []
    converged_all = True

    # Narrowing down the real 2026-10-11 finding (mem_allocated flat at
    # 37,916 MB after EVERY increment, including the very first -- not a
    # growing per-increment leak, something is allocated once and held).
    # These checkpoints localize WHICH step inside increment 1 first jumps
    # to that level, instead of guessing further.
    if device.type == "cuda":
        print(f"    [solve_b3_assembled_direct] mem_allocated at entry (before any "
              f"increment): {torch.cuda.memory_allocated(device) / 1e6:.1f} MB")

    alphas = torch.linspace(0.0, 1.0, n_increments, dtype=dtype, device=device)[1:]
    for i_alpha, alpha in enumerate(alphas):
        if device.type == "cuda" and i_alpha == 0:
            print(f"    [increment 0, before _b3_dirichlet/jac_fn_full] mem_allocated: "
                  f"{torch.cuda.memory_allocated(device) / 1e6:.1f} MB")
        free_mask_dof, prescribed = _b3_dirichlet(geom, float(alpha) * phi, device, dtype)
        jac_fn_full = build_sparse_jac_fn_b3(geom, E_node, nu_node, free_mask_dof, device, dtype,
                                              symmetric_bc=symmetric_bc,
                                              vmap_chunk_size=vmap_chunk_size)
        if device.type == "cuda" and i_alpha == 0:
            print(f"    [increment 0, after build_sparse_jac_fn_b3, before Newton] "
                  f"mem_allocated: {torch.cuda.memory_allocated(device) / 1e6:.1f} MB")

        def residual(u_flat, _A, f_ext, free_mask_dof=free_mask_dof, prescribed=prescribed):
            # B3 has no external-force term (prescribed-displacement loading
            # only, train_B3.py's own Pi=U convention) -- f_ext is zero and
            # kept only so this matches _newton_cudss_reuse_analysis's own
            # residual_fn(u, A, f_ext) call signature.
            grad_full = torch.func.grad(total_energy)(u_flat) - f_ext
            return torch.where(free_mask_dof, grad_full, u_flat - prescribed)

        u0 = u.clone()  # warm-start from the previous (converged) increment
        if symmetric_bc:
            # Required for symmetric_bc=True's column-drop to be exact
            # (see build_sparse_jac_fn_b3's own docstring): lift u0 to
            # satisfy THIS increment's full Dirichlet data exactly before
            # Newton's first iterate, so du_fixed=0 holds from the start
            # instead of only after one (asymmetric) Newton step. u0 was
            # already correct at free DOFs (warm-started); this only
            # overwrites the fixed ones, which otherwise still held the
            # PREVIOUS increment's (smaller) prescribed values.
            u0 = torch.where(free_mask_dof, u0, prescribed)

        if reuse_analysis:
            if device.type != "cuda":
                raise RuntimeError("reuse_analysis=True requires CUDA (cuDSS has no CPU path).")
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                u, stats = _newton_cudss_reuse_analysis(
                    residual, jac_fn_full, u0, f_ext, tol=tol, atol=1e-10, max_iter=max_iter,
                    line_search=True, verbose=False, matrix_type=matrix_type,
                    mem_debug=(i_alpha == 0))
            inc_converged = not any("did not converge" in str(w.message) for w in caught)
            if not inc_converged:
                # Real non-convergence seen at B3's 480,320-element ladder
                # row (2026-10-11) -- the generic "DID NOT CONVERGE" line
                # run_b3_benchmark prints doesn't say WHICH increment or
                # the final ||F||, both needed before guessing a fix
                # (e.g. more load increments) rather than finding the
                # real cause. _newton_cudss_reuse_analysis's own warning
                # message already carries ||F|| -- print it immediately,
                # tagged with this increment's alpha, instead of letting
                # it get lost among the per-iteration cuDSS timing prints.
                for w in caught:
                    if "did not converge" in str(w.message):
                        print(f"    [increment alpha={float(alpha):.2f}] NON-CONVERGENCE: "
                              f"{w.message}")
        else:
            from torch_sla import SparseTensor
            ident = torch.arange(n_dof, device=device)
            A = SparseTensor(torch.ones(n_dof, dtype=dtype, device=device), ident, ident, (n_dof, n_dof))
            u = A.nonlinear_solve(residual, u0, f_ext, jac_fn=jac_fn_full, method="newton",
                                  verbose=False, max_iter=max_iter, tol=tol, linear_method="lu",
                                  linear_solver=("cudss" if device.type == "cuda" else "auto"))
            stats, inc_converged = None, None  # torch_sla's own nonlinear_solve raises internally
                                                # on non-convergence rather than warning -- no flag to check

        if inc_converged is False:
            converged_all = False
        mem_after_mb = torch.cuda.memory_allocated(device) / 1e6 if device.type == "cuda" else None
        increment_stats.append({"alpha": float(alpha), "stats": stats, "converged": inc_converged,
                                 "mem_allocated_mb_after": mem_after_mb})
        # Diagnostic only (added after a real 39.3 GB peak_mem_mb was seen for
        # this same 6,840-element mesh on Omar's A100, see run_b3_benchmark's
        # own comment) -- prints CURRENT (not peak) allocation after every
        # increment so a genuine per-increment leak (monotonically growing
        # mem_after_mb across the loop) can be told apart from one expensive
        # increment's transient peak that is correctly freed again right
        # after. Cheap (one sync-free query), left on unconditionally rather
        # than gated behind a flag no caller has any reason to look for.
        if device.type == "cuda":
            print(f"    [increment alpha={float(alpha):.2f}] mem_allocated after solve: "
                  f"{mem_after_mb:.1f} MB")

    u_np = u.reshape(n_nodes, 3).detach().cpu().numpy()
    agg_stats = {"converged": converged_all, "increment_stats": increment_stats,
                 "n_newton_iters": sum((s["stats"] or {}).get("n_newton_iters", 0) for s in increment_stats)}
    if not converged_all:
        warnings.warn(
            f"B3 assembled+direct solve did NOT converge on at least one load increment "
            f"(see stats['increment_stats']) -- the returned field and any timing built on "
            f"it are not physically meaningful and must not be trusted or reported.")
    if return_stats:
        return u_np, agg_stats
    return u_np


def run_correctness_check(Ntheta=7, Nr=6, Nz=6, device=None, dtype=torch.float64, tol=1e-3,
                           matrix_type="general", symmetric_bc=False):
    """Mandatory gate (see module docstring): solves the SAME small B3
    case two ways -- this module's assembled+direct solver, and data/
    mesh_convergence_B3.py's own already-trusted solve_case (torch-fem,
    CG+Jacobi) -- and reports the agreement. Deliberately tiny/cheap
    (Ntheta=7,Nr=6,Nz=6 -- well below the production 21x20x19) so this
    runs in seconds and can gate every real benchmark run below it.

    matrix_type/symmetric_bc=("general", False) (default, unchanged):
    gates the original path every already-published B3 number used. Pass
    matrix_type="symmetric", symmetric_bc=True to separately gate that
    (2026-10-11) alternative before run_b3_benchmark is ever called with
    it -- run_b3_benchmark's own gate checks for the EXACT (matrix_type,
    symmetric_bc) pair it is about to run, not just "some" pair, so
    passing this with the default config does not clear the symmetric one
    or vice versa.

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
    u_new, new_stats = solve_b3_assembled_direct(
        geom, E_node, nu_node, PHI_CONST, device, dtype=dtype,
        reuse_analysis=(device.type == "cuda"), matrix_type=matrix_type, return_stats=True,
        symmetric_bc=symmetric_bc)
    print(f"  new (assembled+direct{'+ cuDSS' if device.type == 'cuda' else ' (CPU)'}): "
          f"{time.time() - t0:.2f}s, max_disp={np.linalg.norm(u_new, axis=1).max():.4f}, "
          f"converged={new_stats.get('converged')}")
    if new_stats.get("converged") is False:
        raise RuntimeError(
            "solve_b3_assembled_direct did not converge even on the small correctness-"
            "check mesh -- the field-agreement comparison below would be meaningless; "
            "fix convergence before trusting anything from this module.")

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
    _CORRECTNESS_VERIFIED.add((matrix_type, symmetric_bc))
    if device.type == "cuda":
        # Same reasoning as run_b3_benchmark's per-row cleanup: give back
        # whatever PyTorch's caching allocator reserved for this (tiny)
        # mesh before run_b3_benchmark starts its own, much bigger
        # resolutions in the same process -- cheap insurance, not a
        # response to a measured problem at this mesh size specifically.
        torch.cuda.empty_cache()
    return max_abs, max_rel, passed


def run_b3_benchmark(resolutions, out_json, device=None, dtype=torch.float64,
                      reuse_analysis=True, matrix_type="general", seed=0,
                      symmetric_bc=False, vmap_chunk_size=None, n_increments=11):
    """Benchmarks solve_b3_assembled_direct at the given (Ntheta,Nr,Nz)
    resolutions -- pass DEFAULT_RESOLUTION (the 6,840-element production
    mesh) and/or triples from this project's own mesh-convergence ladder
    for an apples-to-apples comparison with already-published B3 numbers.
    REFUSES to run unless run_correctness_check() has already passed in
    THIS process (see module docstring and _CORRECTNESS_VERIFIED) -- the
    gate cannot be bypassed by a caller-supplied flag, only by actually
    running and passing the check first.

    symmetric_bc=False (default, matches run_correctness_check's own gate,
    which always uses the general/non-symmetric matrix): pass True together
    with matrix_type="symmetric" to use cuDSS's cheaper symmetric storage/
    factorization -- see solve_b3_assembled_direct's own docstring for why
    this requires the paired Dirichlet-lift it applies internally, and
    PROJECT_STATUS.md (2026-10-11 entry) for the CPU verification (dense-
    Newton, Ntheta=9/Nr=7/Nz=6, real sampled material/phi) that this gives
    the identical solution to symmetric_bc=False (max rel diff 6.5e-11)
    while producing a genuinely symmetric matrix (max |J-J^T| ~4.5e-13,
    vs. ~2,271 for symmetric_bc=False on the same problem) -- added
    specifically to reduce cuDSS's ANALYSIS/FACTORIZATION memory use after
    a real ALLOC_FAILED at B3's larger ladder resolutions; this got the
    480,320-element row through FACTORIZATION but it still OOM'd on the
    second Newton iteration's own Hessian computation -- see
    vmap_chunk_size.

    vmap_chunk_size=None (default, unchanged): forwarded to
    solve_b3_assembled_direct / build_sparse_jac_fn_b3 -- see their own
    docstrings. Caps peak memory of the per-iteration vmap(hessian(...))
    computation at the cost of some wall-clock time; mathematically
    identical output.

    n_increments=11 (default, unchanged -- matches the already-trusted
    reference solver's own convention): with vmap_chunk_size fixing the
    earlier per-iteration OOM, Omar's run of the 480,320-element row got
    all the way through the 10-increment loop (no crash) but genuinely
    FAILED TO CONVERGE on at least one increment, after an alarming 30/30
    max_iter Newton budget (3,093s total -- vs. the ~3 iterations/
    increment every smaller case needed). This is a real, separate
    problem from memory -- the SAME failure mode this module already
    found and fixed once before (single-shot loading failing on a hard
    real sampled material/phi at the production mesh scale, see the
    "production mesh silently did NOT converge" entry in PROJECT_STATUS.
    md), so more load increments is the well-motivated first thing to
    try, not a blind guess. Pass a larger value (e.g. 21) to test this
    hypothesis before assuming something is actually broken in
    symmetric_bc/vmap_chunk_size themselves."""
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if (matrix_type, symmetric_bc) not in _CORRECTNESS_VERIFIED:
        raise RuntimeError(
            f"run_b3_benchmark refuses to run matrix_type={matrix_type!r}, "
            f"symmetric_bc={symmetric_bc} before run_correctness_check(matrix_type="
            f"{matrix_type!r}, symmetric_bc={symmetric_bc}) has passed in this "
            f"process -- see this module's own standing-constraint docstring. This "
            f"gate is deliberate, not a bug, and checks this EXACT configuration, "
            f"not just any passing check: call run_correctness_check() with the "
            f"SAME matrix_type/symmetric_bc first (it is fast -- a small mesh) and "
            f"only then call this function again in the same process/notebook.")

    rows = []
    for (Ntheta, Nr, Nz) in resolutions:
        geom = build_fixed_geometry(Ntheta, Nr, Nz, device, dtype=dtype)
        # sample_material_and_load returns 5 values (E_node, nu_node, phi,
        # E_field, nu_field) -- the last two are the pre-ravel (Ntheta,Nr,Nz)
        # fields, kept by its own caller in data_generate_B3_dataset.py for
        # dataset bookkeeping; not needed here, so discarded positionally
        # rather than guessed away as a 3-tuple.
        E_node_np, nu_node_np, phi, _E_field, _nu_field = sample_material_and_load(
            Ntheta, Nr, Nz, seed=seed)
        E_node = torch.tensor(E_node_np, dtype=dtype, device=device)
        nu_node = torch.tensor(nu_node_np, dtype=dtype, device=device)

        torch.cuda.synchronize() if device.type == "cuda" else None
        t0 = time.time()
        u, stats = solve_b3_assembled_direct(
            geom, E_node, nu_node, phi, device, dtype=dtype,
            reuse_analysis=reuse_analysis, matrix_type=matrix_type, return_stats=True,
            symmetric_bc=symmetric_bc, vmap_chunk_size=vmap_chunk_size,
            n_increments=n_increments)
        torch.cuda.synchronize() if device.type == "cuda" else None
        elapsed = time.time() - t0
        peak_mem = torch.cuda.max_memory_allocated(device) / 1e6 if device.type == "cuda" else None

        converged = stats.get("converged") if stats else None
        row = {"Ntheta": Ntheta, "Nr": Nr, "Nz": Nz,
               "n_elements": geom["n_elements"], "n_nodes": geom["n_nodes"],
               "elapsed_s": elapsed, "peak_mem_mb": peak_mem,
               "reuse_analysis": reuse_analysis, "matrix_type": matrix_type,
               "symmetric_bc": symmetric_bc, "vmap_chunk_size": vmap_chunk_size,
               "n_increments": n_increments,
               "converged": converged, "newton_stats": stats}
        rows.append(row)
        if converged is False:
            print(f"[Ntheta={Ntheta},Nr={Nr},Nz={Nz}] {geom['n_elements']:,} elements: "
                  f"DID NOT CONVERGE -- elapsed_s={elapsed:.2f}s is NOT a valid timing number, "
                  f"the returned field is not physically meaningful. DO NOT report this row.")
        else:
            print(f"[Ntheta={Ntheta},Nr={Nr},Nz={Nz}] {geom['n_elements']:,} elements: "
                  f"{elapsed:.2f}s, peak_mem={peak_mem} (converged={converged})")
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
            # Real bug found on Omar's A100 (2026-10-11): the FIRST row
            # (6,840 el., the tiny production mesh) reported peak_mem_mb=
            # 39,300 (39.3 GB!) -- wildly more than the 667 MB measured for
            # the identical mesh in an earlier run with no other resolution
            # queued after it -- and the VERY NEXT row (480,320 el.) then
            # failed immediately with cuDSS ALLOC_FAILED during its first
            # FACTORIZATION call. reset_peak_memory_stats above only resets
            # the TRACKING counter; it does not hand memory back to the
            # driver. PyTorch's caching allocator keeps whatever it reserved
            # at that peak (plausible here: vmap(hessian(...))'s forward-
            # over-reverse-mode double-backward constructs a large number of
            # parallel per-element x per-direction intermediate buffers) for
            # its OWN future reuse, but never releases it to the OS -- so it
            # is invisible to cuDSS's own separate, raw cudaMalloc calls.
            # One expensive row can therefore permanently starve cuDSS on
            # every later, larger row in the SAME process, independent of
            # whether that row's own matrix would have fit in the memory
            # that's actually free. del + empty_cache() here actually
            # returns PyTorch's reserved-but-idle memory to the driver
            # before the next (typically much bigger) resolution starts.
            del u
            torch.cuda.empty_cache()

        # Written after EVERY row, not just at the end -- a larger
        # resolution later in the list can OOM or hang (direct solvers'
        # fill-in grows worse than iterative CG's in 3D, untested at the
        # bigger ladder sizes), and losing already-computed smaller-mesh
        # timings to a crash on a later row would silently discard real,
        # hard-won GPU results for no reason.
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
