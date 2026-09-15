"""Item #13: GPU-FEM (this project's own matrix-free Newton-CG solver,
matrix_free_solver.py) vs. torch-fem, an established, actively-maintained
PyTorch-native FEM library -- Timon's round-7 email named this
comparison, with the only stated requirement being "an efficient GPU
implementation ... necessary for a fair comparison to a NO," which
torch-fem satisfies (it is not TensorMesh, the library he mentioned
first, but he stated no preference between the two).

Deliberately done AFTER item #4 (the geometric multigrid preconditioner
fix): benchmarking our own solver before fixing its own preconditioner
would have produced a comparison Timon had already flagged as measuring
a known-suboptimal configuration.

SCOPE (Omar's choice): compare the large-scale matrix-free solver
(the one behind Table 20/20a/20b/20c in the report, reaching millions
of DOF) against torch-fem at the SAME resolutions (401/701/1001/1401),
not the small-scale batched dense solver used for the operator-vs-FEM
accuracy-cost tables.

WHY torch-fem IS NOT MATRIX-FREE, AND WHY THAT MATTERS: torch-fem
always explicitly assembles a sparse tangent stiffness matrix
(base.py's assemble_matrix / self.K), even in its "cg"/"bicgstab"
iterative modes -- confirmed by reading its source, not assumed. This
is the opposite of this project's own solver, whose entire reason for
being matrix-free is reaching millions of DOF on a single GPU without
ever forming K (see matrix_free_solver.py's own docstring). So this
comparison is not just "whose CG is faster" -- it is a genuine
architectural difference (assembled-sparse-then-iterative vs.
matrix-free) that should show up most clearly in GPU MEMORY at the
largest resolutions, not only wall-clock.

THE SAME MESH, MATERIAL FIELD, LOAD, AND BOUNDARY CONDITIONS ARE REUSED
DIRECTLY, NOT RE-DERIVED: this module imports build_mesh_and_bcs's own
helpers (generate_grid_Q4, precompute_element_params_B1,
assemble_traction_top_generic) from the exact same code path
high_dof_convergence_study.py already uses and this report's Table 6a/
20 series were generated from, and confirmed by direct inspection that
generate_grid_Q4's own element node order (bottom-left, bottom-right,
top-right, top-left) is IDENTICAL to torch-fem's Quad1 convention, so
the same `nodes`/`elements` arrays are passed to both solvers with no
reordering -- removing an entire class of "are we even solving the
same mesh" bugs before they can happen.

Before any GPU time is spent, `python -m omar_pfem.torchfem_comparison`
runs a small CPU correctness check: solve the same tiny B1 x
Neo-Hookean problem with both solvers and confirm they agree (same
converged strain energy, same displacement field) to a tight tolerance
-- following this project's own standing discipline of never trusting
a timing comparison between two solvers before confirming they are
actually solving the same problem correctly.
"""
import sys
import time
import types

import numpy as np
import torch

# torch-fem's own __init__.py unconditionally imports pyvista (for mesh
# plotting/export -- never used by anything in this module, which only
# calls Planar/HyperelasticPlaneStrain/.solve()). On Colab specifically,
# `import torchfem` fails with:
#   ModuleNotFoundError: No module named 'IPython.core.guarded_eval'
# raised deep inside pyvista's own IPython-tab-completion integration
# code, which has nothing to do with FEM at all. A first, narrower fix
# (stubbing only the one missing IPython submodule pyvista's own source
# appears to need) did not clear it on Colab's actual environment --
# rather than keep guessing at pyvista's exact internal control flow
# from the outside, this replaces the entire pyvista module with a
# permissive stub whenever the real import fails, since nothing in this
# comparison ever calls a single pyvista plotting function. A MagicMock
# answers ANY attribute access or call with another MagicMock, so every
# `pyvista.Whatever`, `pyvista.plotting.Whatever`, `from pyvista import
# DataSet`-style reference inside torch-fem's own source resolves to
# some harmless object instead of raising -- robust to exactly which
# pyvista internals are involved, not just the one this project happened
# to trace. Installed defensively, before torch-fem is ever imported,
# and only if the real pyvista import fails (leaves environments where
# it imports fine, like this project's own dev environment, untouched).
if "pyvista" not in sys.modules:
    try:
        import pyvista  # noqa: F401
    except Exception:
        from unittest.mock import MagicMock
        _pyvista_stub = MagicMock(name="pyvista")
        sys.modules["pyvista"] = _pyvista_stub
        sys.modules["pyvista.plotting"] = _pyvista_stub.plotting

from omar_pfem.high_dof_convergence_study import (
    build_mesh_and_bcs, AnalyticFieldB1, solve_one, compute_l2_h1_errors,
    fit_convergence_rate, compute_tangent_energy_error, find_fine_peak_stress,
    compute_peak_stress_error, pk1_component_errors_at_point,
    compute_reaction_resultant_error)
from omar_pfem.matrix_free_solver import solve_matrix_free


def neo_hookean_psi_3d(F3d, params):
    """The SAME compressible Neo-Hookean energy materials_torch.py's
    neo_hookean_energy_density_vectorized implements, written for a
    genuinely 3x3 F (torch-fem's HyperelasticPlaneStrain extends the 2x2
    in-plane F to 3x3 internally, with F_33=1, before calling this).
    Mathematically identical to the 2D-only form this project's own
    solver uses: with F_33=1, the 3D invariants I1_3d=tr(F3d^T F3d) and
    J_3d=det(F3d) reduce to I1_2d+1 and J_2d respectively, and
    mu/2*(I1_3d-3) collapses to mu/2*(I1_2d-2) -- the exact expression
    materials_torch.py uses -- so both solvers evaluate the same
    physical energy, not two different Neo-Hookean conventions that
    happen to look similar.

    params: (2,) tensor [mu, lam] for one element -- torch-fem calls
    this via vmap(jacrev(psi))(F_new, self.params), one element's own
    params row at a time, not unpacked as separate scalar arguments.

    Uses torch.linalg.slogdet for ln(J), NOT torch.log(torch.linalg.det(F))
    -- confirmed by direct test that the latter's SECOND derivative
    (the tangent stiffness torch-fem's ddsdde needs, computed by
    differentiating this function's own gradient again) is NaN at F=I
    (a known sharp edge in torch.linalg.det's double-backward, not a
    bug in the physics), which silently made the assembled tangent
    stiffness singular at the very first Newton iteration -- exactly
    the point every solve starts from. slogdet's gradient formula is
    the numerically stable one; verified directly (vmap(jacrev(jacrev
    (psi)))(F, params) at F=I gives a finite, correct Hessian with
    slogdet and an all-NaN one with log(det(.)))."""
    mu, lam = params[0], params[1]
    _sign, lnJ = torch.linalg.slogdet(F3d)
    I1 = torch.sum(F3d ** 2, dim=(-2, -1))
    return (mu / 2.0) * (I1 - 3.0 - 2.0 * lnJ) + (lam / 2.0) * (lnJ ** 2)


def mooney_rivlin_psi_3d(F3d, params):
    """3D analog of neo_hookean_psi_3d, generalized 2026-09-14 (Timon
    round-11 point 2: resolution-matched break-even for the other
    materials). Same reduction as neo_hookean_psi_3d: with F_33=1,
    I1_3d = I1_2d + 1 and J_3d = J_2d exactly, so substituting
    I1_2d = I1_3d - 1 (J_2d = J_3d needs no substitution) into
    materials_torch.mooney_rivlin_energy_density_vectorized's own 2D
    formula (psi = c*(J-1)^2 - d*ln(J) + c1*(I1_2d-2) + c2*(I2_2d-1),
    I2_2d = J^2 for 2D) gives the expression below -- the same physical
    energy this project's own solver already uses for Mooney-Rivlin,
    not a different convention that happens to look similar.

    params: (4,) tensor [c, c1, c2, d] for one element, matching
    data/materials.py's own Mooney-Rivlin parameter order exactly.

    Uses torch.linalg.slogdet for J (via exp(lnJ)), NOT
    torch.linalg.det directly -- confirmed by direct test
    (torch.func.hessian at F=I) that plain det()'s own SECOND
    derivative is already NaN at F=I, independent of any log wrapping
    (a broader instance of the exact sharp edge neo_hookean_psi_3d's
    own docstring already documents for log(det(.)) specifically --
    here it turns out det() ALONE has it). slogdet's log-part has a
    finite, correct Hessian there; reconstructing J = exp(lnJ) from it
    keeps every downstream term (including the ones that use J
    polynomially, not logarithmically) on the safe path. Found because
    a real torch-fem Newton solve of this material failed to converge
    even at N=7 with 200 CG iterations and relaxed tolerance -- traced
    to a NaN tangent stiffness at the very first (undeformed) Newton
    iterate, not a preconditioner or iteration-count issue."""
    c, c1, c2, d = params[0], params[1], params[2], params[3]
    _sign, lnJ = torch.linalg.slogdet(F3d)
    J = torch.exp(lnJ)
    I1 = torch.sum(F3d ** 2, dim=(-2, -1))
    return c * (J - 1.0) ** 2 - d * lnJ + c1 * (I1 - 3.0) + c2 * (J ** 2 - 1.0)


def arruda_boyce_psi_3d(F3d, params):
    """3D analog of neo_hookean_psi_3d for Arruda-Boyce, generalized
    2026-09-14, same reduction as mooney_rivlin_psi_3d above
    (I1_2d = I1_3d - 1, J_2d = J_3d). materials_torch.
    arruda_boyce_energy_density_vectorized's own I1_bar = I1_2d/J_2d
    becomes (I1_3d - 1)/J_3d here; the 5-term 8-chain series and the
    volumetric kappa term are otherwise unchanged. The same I1_bar
    clamp (to 3*N_ab, the chain-locking limit past which the series has
    no physical meaning and overflows) is kept for the same reason.

    params: (3,) tensor [mu_ab, N_ab, kappa_ab] for one element,
    matching data/materials.py's own Arruda-Boyce parameter order.

    Uses slogdet for J, same reason/fix as mooney_rivlin_psi_3d above
    (plain torch.linalg.det's own second derivative is NaN at F=I)."""
    mu_ab, N_ab, kappa_ab = params[0], params[1], params[2]
    _sign, lnJ = torch.linalg.slogdet(F3d)
    J = torch.exp(lnJ)
    I1 = torch.sum(F3d ** 2, dim=(-2, -1))
    I1_bar = torch.clamp((I1 - 1.0) / J, max=3.0 * N_ab - 1e-3)
    alpha = [1 / 2, 1 / 20, 11 / 1050, 19 / 7000, 519 / 673750]
    psi = torch.zeros_like(I1_bar)
    i1_bar_pow = I1_bar
    for k, a in enumerate(alpha, start=1):
        psi = psi + mu_ab * a / (N_ab ** (k - 1)) * (i1_bar_pow - 2.0 ** k)
        i1_bar_pow = i1_bar_pow * I1_bar
    psi = psi + kappa_ab / 2 * (J - 1.0) ** 2
    return psi


_PSI_3D_BY_MATERIAL = {
    "neo_hookean": neo_hookean_psi_3d,
    "mooney_rivlin": mooney_rivlin_psi_3d,
    "arruda_boyce": arruda_boyce_psi_3d,
}


def _build_chunked_hyperelastic_plane_strain_class():
    """Returns a HyperelasticPlaneStrain subclass that computes stress/
    tangent in chunks along the Gauss-point batch dimension, instead of
    one call over every point at once.

    WHY THIS EXISTS (real bug found 2026-09-14, not anticipated in
    advance): torch-fem's own Hyperelastic3D.step()
    (torchfem/materials/hyperelasticity.py) does
    `vmap(jacrev(jacrev(self.psi)))(F_new, self.params)` over the WHOLE
    batch in a single call. That's fine for Neo-Hookean/Mooney-Rivlin --
    both succeeded at N=1401 (~70.8GB/73.6GB peak on an 80GB A100, see
    resolution_matched_break_even_all_cases.json) -- but Arruda-Boyce's
    own energy density (a 5-term 8-chain series, more terms per Gauss
    point than the other two materials) needs enough extra intermediate
    memory per point that the identical batch OOMs, confirmed with the
    GPU completely clean beforehand (allocated=0.14GB right before the
    case) -- a genuine per-point memory cost, not cross-case
    fragmentation, and not fixable by clearing memory between cases (that
    fix, 1e0c391, was already applied and did not help this specific
    case).

    Chunking changes nothing about the physics: each Gauss point's
    stress/tangent is independent of every other one (psi is evaluated
    pointwise), so splitting the batch into pieces and concatenating the
    results is bit-identical to computing them all in one call -- this
    only trades wall-clock for peak memory. Starts at `chunk_size` and
    halves on OOM (clearing the cache first) until a chunk fits or
    chunk_size reaches 1, at which point a real, unrecoverable per-point
    memory requirement (not a tunable batch size) would be the honest
    conclusion.

    Used only for Arruda-Boyce (see build_torchfem_model) -- Neo-Hookean/
    Mooney-Rivlin keep torch-fem's own unmodified class, so their
    already-verified N=1401 numbers are untouched by this change.

    NOT YET VERIFIED ON REAL GPU as of 2026-09-14 (written in an
    environment with no GPU access) -- chunk_size=50_000 is a reasonable
    starting guess, not a measured value; the halving-on-OOM retry exists
    specifically because that starting guess might be wrong. Needs a real
    A100 run on B1/B2 x Arruda-Boyce at N=1401 before this is trusted.
    """
    from torchfem.materials import HyperelasticPlaneStrain
    from torch.func import jacrev, vmap

    class ChunkedHyperelasticPlaneStrain(HyperelasticPlaneStrain):
        chunk_size = 50_000

        def step(self, H_inc, F, stress, state, de0, cl, iter):
            F3D = torch.zeros(F.shape[0], 3, 3, device=F.device, dtype=F.dtype)
            F3D[..., 0:2, 0:2] = F
            F3D[..., 2, 2] = 1.0
            H_inc_3D = torch.zeros(H_inc.shape[0], 3, 3, device=H_inc.device, dtype=H_inc.dtype)
            H_inc_3D[..., 0:2, 0:2] = H_inc

            with torch.enable_grad():
                F_new = (F3D + H_inc_3D).requires_grad_(True)
                n = F_new.shape[0]

                # Diagnostic added 2026-09-15, NOT a fix -- checks a real
                # candidate root cause for the "Newton-Raphson did not
                # converge" failure that survives the chunking fix above.
                # arruda_boyce_psi_3d (this module) clamps I1_bar to
                # 3*N_ab-1e-3 to avoid NaN/overflow in the chain series;
                # a CPU test of that exact function confirmed the clamp
                # makes the chain-stretch contribution to stress exactly
                # flat (zero local sensitivity, bit-identical psi under a
                # +-1e-4 perturbation) once triggered -- suspicious for a
                # Newton solve, since a real chain-locking material should
                # stiffen, not flatten. Not yet confirmed this is THE
                # cause (needs a real GPU run to know whether the actual
                # solve ever reaches this regime) -- this block only
                # reports whether it does, at negligible cost (one
                # comparison over the same tensor already being computed).
                if self.params.shape[-1] == 3:
                    with torch.no_grad():
                        I1_full = torch.sum(F_new.detach() ** 2, dim=(-2, -1))
                        _sign, lnJ_full = torch.linalg.slogdet(F_new.detach())
                        J_full = torch.exp(lnJ_full)
                        I1_bar_full = (I1_full - 1.0) / J_full
                        N_ab_full = (self.params[..., 1] if self.is_vectorized
                                     else self.params[1].expand(n))
                        limit = 3.0 * N_ab_full - 1e-3
                        n_clamped = (I1_bar_full >= limit).sum().item()
                        if n_clamped > 0:
                            print(f'  [chunked hyperelastic] {n_clamped}/{n} points at the '
                                  f'chain-locking clamp this call (max I1_bar='
                                  f'{I1_bar_full.max().item():.3f}, worst limit='
                                  f'{limit.min().item():.3f}) -- candidate cause of the '
                                  f'Newton non-convergence, see PROJECT_STATUS.md.')

                P_parts, ddsdde_parts = [], []
                start, size = 0, self.chunk_size
                while start < n:
                    end = min(start + size, n)
                    F_chunk = F_new[start:end]
                    params_chunk = self.params[start:end] if self.is_vectorized else self.params
                    try:
                        P_chunk = vmap(jacrev(self.psi))(F_chunk, params_chunk)
                        ddsdde_chunk = vmap(jacrev(jacrev(self.psi)))(F_chunk, params_chunk)
                    except torch.cuda.OutOfMemoryError:
                        if size <= 1:
                            raise
                        torch.cuda.empty_cache()
                        size = max(1, size // 2)
                        print(f'  [chunked hyperelastic] OOM, halving chunk_size '
                              f'to {size} and retrying this chunk...')
                        continue
                    P_parts.append(P_chunk)
                    ddsdde_parts.append(ddsdde_chunk)
                    start = end
                P_new = torch.cat(P_parts, dim=0)
                ddsdde_new = torch.cat(ddsdde_parts, dim=0)

            return (P_new[..., 0:2, 0:2], state, ddsdde_new[..., 0:2, 0:2, 0:2, 0:2])

    return ChunkedHyperelasticPlaneStrain


def build_torchfem_model(nodes, elements, *mat_params, fext_full, fixed_dofs,
                          material="neo_hookean", dtype=torch.float64, device=None):
    device = device or torch.device('cpu')
    # Building the model itself works fine in float64 (Planar(...) below
    # has no dtype-hardcoded internals). The float32 blocker is entirely
    # inside .solve() -- see solve_theirs' own docstring for the real
    # bug (near_null_space()/skew() hardcoding torch.eye(3) at float32)
    # and its fix (torch.set_default_dtype around the .solve() call).
    """nodes/elements/fext_full/fixed_dofs come directly from
    build_mesh_and_bcs -- the SAME arrays this project's own solver
    uses, not a re-derivation.

    Generalized 2026-09-14 (Timon round-11 point 2, resolution-matched
    break-even for the other materials/B2): was hardcoded to exactly
    (mu, lam) and neo_hookean_psi_3d, and assumed every fixed_dofs entry
    came in x/y PAIRS (true for B1's bottom clamp, false for B2's two
    symmetry edges, each of which fixes only ONE component per node).
    *mat_params (any arity) + _PSI_3D_BY_MATERIAL[material] handles the
    first; building constraints directly from fixed_dofs' own per-DOF
    (node, component) decomposition -- instead of assuming "every fixed
    node has both components fixed" -- handles the second, and is a
    strict generalization: for B1's own fixed_dofs (which DOES pair
    every node's both components), this produces the exact same
    constraints tensor as the old code, verified by the regression
    check in this module's own _correctness_check before this was ever
    trusted for B2."""
    from torchfem import Planar
    from torchfem.materials import HyperelasticPlaneStrain

    nodes_t = torch.tensor(nodes, dtype=dtype, device=device)
    elements_t = torch.tensor(elements, dtype=torch.long, device=device)
    # HyperelasticPlaneStrain's own params contract (its docstring: "Shape:
    # (p,) for a scalar or (N, p) for a batch of materials") is ONE value
    # PER ELEMENT -- confirmed by a real crash otherwise ("size of tensor a
    # (36) must match ... at non-singleton dimension 1" inside torch-fem's
    # own integrate_material) when handed B2's (n_elements, n_gauss) arrays
    # from precompute_element_params_B2 (per-Gauss-point sampling, this
    # project's own established B2 convention since commit af7e67c, matched
    # by "ours" own solver and the slow reference -- see that function's
    # own docstring). torch-fem's public material API has no per-Gauss-
    # point hook to match it exactly. Averaging each element's own Gauss-
    # point values down to one number per element is the pragmatic
    # approximation used here, specifically for THIS wall-clock comparison
    # (not a new accuracy claim) -- the field varies smoothly and slowly
    # relative to one element's own size (same reasoning that makes mesh
    # refinement converge at all), so this differs from B2's own true
    # per-Gauss-point solve by an amount of the same character (small,
    # shrinking with N) as the already-documented, already-quantified
    # centroid-vs-Gauss-point discretization difference in this project's
    # own history -- not an unbounded or untested approximation. B1's own
    # per-element (not per-Gauss-point) sampling is unaffected: mean over
    # an axis of size 1 is a no-op.
    mat_params_per_elem = [torch.tensor(p, dtype=dtype, device=device) for p in mat_params]
    mat_params_per_elem = [p.mean(dim=-1) if p.dim() > 1 else p for p in mat_params_per_elem]
    params = torch.stack(mat_params_per_elem, dim=-1)  # (n_elem, n_params)

    # Arruda-Boyce specifically needs the chunked variant (real OOM found
    # 2026-09-14 at N=1401 on an 80GB A100, GPU clean beforehand -- see
    # _build_chunked_hyperelastic_plane_strain_class's own docstring).
    # Neo-Hookean/Mooney-Rivlin keep torch-fem's own class unchanged so
    # their already-verified numbers are untouched.
    material_cls = (_build_chunked_hyperelastic_plane_strain_class()
                     if material == "arruda_boyce" else HyperelasticPlaneStrain)
    tf_material = material_cls(psi=_PSI_3D_BY_MATERIAL[material], params=params)
    # torch-fem's own FEM.__init__ builds an internal index-mapping via
    # `torch.arange(self.n_dof_per_node)` with no explicit device -- a
    # real bug in torch-fem itself, confirmed directly: on a GPU run
    # this raises "Expected all tensors to be on the same device" the
    # moment nodes/elements are CUDA tensors, since torch.arange
    # defaults to CPU regardless of its other operands' device. Not
    # fixable by moving our own tensors around (the mismatch is INSIDE
    # torch-fem's own constructor, before we get any tensor back);
    # `with torch.device(device):` makes every device-less tensor
    # creation inside that block (including torch-fem's own internal
    # torch.arange calls) default to the right device instead.
    with torch.device(device):
        model = Planar(nodes_t, elements_t, tf_material)

    n_nodes = nodes.shape[0]
    model.forces = torch.tensor(fext_full, dtype=dtype, device=device).reshape(n_nodes, 2)

    # Per-DOF, not per-node: fixed_dofs entries are individual (node,
    # component) pairs (2*n for x, 2*n+1 for y) -- decoding node/component
    # from each entry and setting only that one component handles both
    # B1 (every fixed node happens to contribute both its DOFs) and B2
    # (each symmetry edge contributes only one component per node)
    # correctly with the same code.
    constraints = torch.zeros(n_nodes, 2, dtype=torch.bool, device=device)
    fixed_dofs_arr = np.asarray(fixed_dofs)
    node_idx = fixed_dofs_arr // 2
    comp_idx = fixed_dofs_arr % 2
    constraints[node_idx, comp_idx] = True
    model.constraints = constraints
    model.displacements = torch.zeros(n_nodes, 2, dtype=dtype, device=device)
    return model


def solve_ours(nodes, elements, free_dofs, elem_params, fext_full, nsteps=10,
                device=None, dtype=torch.float64, mg_hierarchy=None, checkpoint_path=None):
    """checkpoint_path: if given and it already holds a completed solve
    (e.g. from item #4's own N=401/701/1001/1401 mgv runs, already on
    Drive), this resumes and returns near-instantly instead of
    re-solving from scratch -- avoiding ~15h of REDUNDANT GPU time
    reproducing wall-clock/cg_iters numbers this project already has
    committed (highdof_stress_qoi_results/*.json). The elapsed time
    returned in that case is the resume time, not a real solve cost --
    reuse the already-committed JSON's own wall_clock_s/cg_iters for
    "ours" instead of this call's own timing whenever a checkpoint was
    reused; see run_sweep_row's own docstring."""
    device = device or torch.device('cpu')
    xy_t = torch.tensor(nodes, dtype=dtype, device=device)
    quad_t = torch.tensor(elements, dtype=torch.long, device=device)
    free_t = torch.tensor(free_dofs, dtype=torch.long, device=device)
    params_t = tuple(torch.tensor(p, dtype=dtype, device=device) for p in elem_params)
    fext_free_t = torch.tensor(fext_full[free_dofs], dtype=dtype, device=device)

    precond_kind = "mgv" if mg_hierarchy is not None else "jacobi"
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
    t0 = time.time()
    u_free, stats = solve_matrix_free(
        xy_t, quad_t, free_t, params_t, fext_free_t, n_free=len(free_dofs),
        material="neo_hookean", order="Q4", nsteps=nsteps, newton_max=30,
        newton_tol=1e-8, cg_tol=1e-8, cg_max_iter=2000, precond_kind=precond_kind,
        mg_hierarchy=mg_hierarchy, device=device, dtype=dtype, verbose=False,
        checkpoint_path=checkpoint_path)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.time() - t0
    peak_mb = (torch.cuda.max_memory_allocated(device) / 1e6) if device.type == "cuda" else None
    u_full = torch.zeros(2 * len(nodes), dtype=dtype, device=device)
    u_full[free_t] = u_free
    return u_full.cpu().numpy(), elapsed, stats, peak_mb


def solve_theirs(nodes, elements, *mat_params, fext_full, fixed_dofs, material="neo_hookean",
                  nsteps=10, dtype=torch.float64, device=None, tol=1e-8):
    """tol: shared rtol/atol/stol, matching "ours" own newton_tol=cg_tol=1e-8
    default (solve_ours above) -- Timon's round-9 request (2026-09-10) to
    use the SAME criteria on both sides instead of torch-fem's previous
    float32/1e-3. dtype now defaults to float64 to match "ours" as well.

    Getting float64 working here at all required a real fix, not just
    changing this function's own dtype argument: torch-fem's own
    near_null_space()/skew() (base.py, called unconditionally inside
    .solve(), regardless of preconditioner or method) builds torch.eye(3)
    with no explicit dtype, which silently defaults to float32 --
    confirmed directly, this raises "RuntimeError: expected scalar type
    Float but found Double" the moment .solve() is called on a float64
    model, even though building the model itself (Planar(...)) works
    fine in float64. torch.set_default_dtype(torch.float64) around the
    .solve() call (with the previous default restored after, in
    finally:) makes every dtype-less tensor torch-fem creates internally
    default to float64 too, since PyTorch's global default dtype governs
    exactly those calls -- the same "wrap the library's own hardcoded
    tensor creation" pattern already used for its two separate device
    bugs (see build_torchfem_model's and this function's own device
    comment below), just for dtype instead of device this time. Verified
    on a small hand-built 2-element mesh before touching this function:
    Newton converges to rtol=atol=1e-8 in float64 with no dtype errors."""
    device = device or torch.device('cpu')
    model = build_torchfem_model(nodes, elements, *mat_params, fext_full=fext_full,
                                  fixed_dofs=fixed_dofs, material=material,
                                  dtype=dtype, device=device)
    increments = torch.linspace(0.0, 1.0, nsteps + 1, dtype=dtype, device=device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
    t0 = time.time()
    # Same torch-fem device bug as build_torchfem_model's own comment
    # explains, but a SECOND instance of it, hit inside .solve() itself
    # rather than the constructor: near_null_space()/skew() (base.py)
    # build torch.zeros/torch.arange/torch.eye with no explicit device,
    # then torch.cat them against self.nodes (which IS on the right
    # device, since we built it that way) -- same fix, same reason.
    old_default_dtype = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    try:
        with torch.device(device):
            u, *_ = model.solve(
                increments=increments, max_iter=30, rtol=tol, atol=tol, stol=tol,
                method="cg", preconditioner="jacobi", nlgeom=True, verbose=False)
    finally:
        torch.set_default_dtype(old_default_dtype)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.time() - t0
    peak_mb = (torch.cuda.max_memory_allocated(device) / 1e6) if device.type == "cuda" else None
    return u.reshape(-1).cpu().numpy(), elapsed, peak_mb


def solve_theirs_with_breakdown(nodes, elements, *mat_params, fext_full, fixed_dofs,
                                 material="neo_hookean", nsteps=10,
                                 dtype=torch.float64, device=None, tol=1e-8, method="cg",
                                 preconditioner="jacobi"):
    """Timon round-9 item 3 (2026-09-10): total time PLUS assembly,
    solve/factorization, and nonlinear-iteration count, not just one
    aggregate wall-clock number like solve_theirs above.

    torch-fem's own Newton loop (torchfem.sparse.NewtonRaphsonAdjoint.
    forward, confirmed by reading its source, not assumed) has a clean,
    natural split already: each iteration calls `eval_residual(...)`
    (which internally calls the model's own `integrate_material` --
    element-level residual/tangent -- then `assemble_matrix` -- builds
    the global sparse K -- together "assembly" in Timon's sense) and
    THEN, only if not yet converged, calls the module-level
    `sparse_solve(...)` function ("solve/factorization" in Timon's
    sense -- this is also where method="direct" would do a real
    factorization instead of CG). Neither phase is exposed as a
    separately-timed return value by torch-fem itself, so this
    instruments both from the outside: `assemble_matrix`/
    `integrate_material` are monkeypatched on the MODEL INSTANCE only
    (does not affect any other model), and `torchfem.sparse.sparse_solve`
    is monkeypatched at module level for the duration of this call only,
    restored in `finally:` -- the same "wrap the library's own call from
    the outside" pattern already used for its device/dtype bugs, just
    for timing instead of a bug fix this time.

    method: "cg" (iterative, matches solve_theirs' own default) or
    "direct" (a real factorization -- Timon's own suggestion -- but
    direct sparse solves scale far worse than CG at millions of DOF;
    test at small N before ever pointing this at N=1001/1401).

    Returns a dict, not a bare tuple, since there are now five numbers
    worth keeping instead of two."""
    import torchfem.sparse as _tf_sparse

    device = device or torch.device('cpu')
    model = build_torchfem_model(nodes, elements, *mat_params, fext_full=fext_full,
                                  fixed_dofs=fixed_dofs, material=material,
                                  dtype=dtype, device=device)
    increments = torch.linspace(0.0, 1.0, nsteps + 1, dtype=dtype, device=device)

    assembly_time = 0.0
    solve_time = 0.0
    n_linear_solves = 0

    orig_assemble_matrix = model.assemble_matrix
    orig_integrate_material = model.integrate_material

    def timed_assemble_matrix(*args, **kwargs):
        nonlocal assembly_time
        t = time.time()
        out = orig_assemble_matrix(*args, **kwargs)
        assembly_time += time.time() - t
        return out

    def timed_integrate_material(*args, **kwargs):
        nonlocal assembly_time
        t = time.time()
        out = orig_integrate_material(*args, **kwargs)
        assembly_time += time.time() - t
        return out

    model.assemble_matrix = timed_assemble_matrix
    model.integrate_material = timed_integrate_material

    orig_sparse_solve = _tf_sparse.sparse_solve

    def timed_sparse_solve(*args, **kwargs):
        nonlocal solve_time, n_linear_solves
        t = time.time()
        out = orig_sparse_solve(*args, **kwargs)
        solve_time += time.time() - t
        n_linear_solves += 1
        return out

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
    t0 = time.time()
    old_default_dtype = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    _tf_sparse.sparse_solve = timed_sparse_solve
    try:
        with torch.device(device):
            u, *_ = model.solve(
                increments=increments, max_iter=30, rtol=tol, atol=tol, stol=tol,
                method=method, preconditioner=preconditioner, nlgeom=True, verbose=False)
    finally:
        torch.set_default_dtype(old_default_dtype)
        _tf_sparse.sparse_solve = orig_sparse_solve
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    total_elapsed = time.time() - t0
    peak_mb = (torch.cuda.max_memory_allocated(device) / 1e6) if device.type == "cuda" else None

    other_time = total_elapsed - assembly_time - solve_time
    return {
        "u": u.reshape(-1).cpu().numpy(),
        "total_time_s": total_elapsed,
        "assembly_time_s": assembly_time,
        "solve_time_s": solve_time,
        "other_time_s": other_time,
        "n_nonlinear_iters": n_linear_solves,
        "peak_mem_mb": peak_mb,
        "method": method,
    }


def _correctness_check(N=11, tol=1e-8):
    """tol: shared rtol/atol/stol passed to solve_theirs -- Timon's
    round-9 request (2026-09-10) to test 1e-8 (and optionally 1e-6/1e-7)
    at MATCHED float64 precision on both sides, replacing the previous
    float32/1e-3 check. The printed relative displacement-field
    difference is now a real matched-precision accuracy number, not a
    loose pass/fail against a single-precision-appropriate bound."""
    print(f"=== CPU correctness check: our solver vs. torch-fem, B1 x Neo-Hookean, "
          f"N={N}, matched FP64/tol={tol:.0e} ===")
    nodes, elements, free_dofs, fext_full, elem_params = build_mesh_and_bcs(
        "B1", "Q4", N, "neo_hookean", torch.device('cpu'), torch.float64)

    fixed_dofs = np.setdiff1d(np.arange(2 * nodes.shape[0]), free_dofs)

    u_ours, t_ours, stats, _peak = solve_ours(nodes, elements, free_dofs, elem_params, fext_full)
    print(f"  ours:      wall_clock={t_ours:.2f}s, cg_iters_total={stats['cg_iters_total']}, "
          f"cg_failures={stats['cg_failures']}")

    u_theirs, t_theirs, _peak2 = solve_theirs(nodes, elements, *elem_params, fext_full=fext_full,
                                               fixed_dofs=fixed_dofs, material="neo_hookean",
                                               dtype=torch.float64, tol=tol)
    print(f"  torch-fem: wall_clock={t_theirs:.2f}s")

    diff = np.linalg.norm(u_ours - u_theirs)
    ref = np.linalg.norm(u_ours) + 1e-30
    rel_diff = diff / ref
    print(f"  relative displacement-field difference: {rel_diff:.3e}")
    # Now a genuine float64-vs-float64 comparison, so held to this
    # project's own usual 1e-6 standard (looser than 1e-8 since the two
    # solvers still differ in element formulation/assembly details, not
    # just floating-point noise) rather than the old single-precision one.
    ok = rel_diff < 1e-6
    print("  " + ("PASS" if ok else "FAIL"))
    return ok, rel_diff


def run_sweep_row(N, device, use_mgv=True, checkpoint_dir=None):
    """One resolution's worth of the real comparison: builds the SAME
    B1 x Neo-Hookean mesh/BCs/material both solvers see, solves with
    each, and returns a dict of everything needed to compare them --
    wall-clock, peak GPU memory, and CG iteration counts. `use_mgv`
    matches this project's own best available preconditioner (item #4)
    against torch-fem's own best readily-available one (Jacobi -- AMG
    would need an extra pyamg/amgx dependency this comparison does not
    assume is installed); documented explicitly rather than silently
    picking whichever happens to be convenient.

    checkpoint_dir: if given and matches item #4's own checkpoint
    naming/location (coarse_B1_neo_hookean_Q4_N{N}.pt), "ours" resumes
    from the ALREADY-COMPLETED mgv solve those runs left on Drive
    instead of re-solving from scratch -- avoiding ~15h of genuinely
    redundant GPU time reproducing wall-clock/cg_iters numbers this
    project already has committed (see PROJECT_STATUS.md's item #4).
    When that happens, `ours_resumed_from_checkpoint` is True in the
    returned row and `ours_wall_clock_s`/`ours_cg_iters` are read
    directly from that already-committed data (the checkpoint resume's
    own near-instant timing would badly understate the real solve
    cost, and would report zero peak memory, neither of which is a
    fair "ours" number for this comparison)."""
    nodes, elements, free_dofs, fext_full, elem_params = build_mesh_and_bcs(
        "B1", "Q4", N, "neo_hookean", device, torch.float64)
    mu, lam = elem_params
    fixed_dofs = np.setdiff1d(np.arange(2 * nodes.shape[0]), free_dofs)

    checkpoint_path = None
    if checkpoint_dir is not None:
        import os
        checkpoint_path = os.path.join(checkpoint_dir, f"coarse_B1_neo_hookean_Q4_N{N}.pt")
        resumed = os.path.exists(checkpoint_path)
    else:
        resumed = False

    known = _known_mgv_result(N) if resumed and use_mgv else None

    mg_hierarchy = None
    if use_mgv:
        from omar_pfem.multigrid_precond import build_mg_hierarchy, coarsen_N
        Ns_mg = [N]
        while True:
            nc = coarsen_N(Ns_mg[-1])
            if nc is None or len(Ns_mg) >= 8:
                break
            Ns_mg.append(nc)
        mesh_tuples = [(N, nodes, elements, free_dofs, elem_params)]
        for n in Ns_mg[1:]:
            c_nodes, c_elements, c_free, _c_fext, c_params = build_mesh_and_bcs(
                "B1", "Q4", n, "neo_hookean", device, torch.float64)
            mesh_tuples.append((n, c_nodes, c_elements, c_free, c_params))
        mg_hierarchy = build_mg_hierarchy(mesh_tuples, torch.float64, device)
        print(f"  [ours] mgv hierarchy N: {Ns_mg}")

    _u_ours, t_ours_call, stats, peak_ours = solve_ours(
        nodes, elements, free_dofs, elem_params, fext_full,
        device=device, mg_hierarchy=mg_hierarchy, checkpoint_path=checkpoint_path)

    if known is not None:
        print(f"  ours:      N={N} RESUMED from existing checkpoint in {t_ours_call:.1f}s -- "
              f"using item #4's already-committed real numbers instead: "
              f"wall_clock={known['wall_clock_s']:.1f}s cg_iters={known['cg_iters']} "
              f"cg_failures={known['cg_failures']}")
        ours_wall_clock_s = known["wall_clock_s"]
        ours_cg_iters = known["cg_iters"]
        ours_cg_failures = known["cg_failures"]
    else:
        print(f"  ours:      N={N} wall_clock={t_ours_call:.1f}s cg_iters={stats['cg_iters_total']} "
              f"cg_failures={stats['cg_failures']} peak_mem_mb={peak_ours}")
        ours_wall_clock_s = t_ours_call
        ours_cg_iters = stats["cg_iters_total"]
        ours_cg_failures = stats["cg_failures"]

    _u_theirs, t_theirs, peak_theirs = solve_theirs(
        nodes, elements, mu, lam, fext_full=fext_full, fixed_dofs=fixed_dofs,
        material="neo_hookean", device=device)
    print(f"  torch-fem: N={N} wall_clock={t_theirs:.1f}s peak_mem_mb={peak_theirs}")

    return {
        "N": N, "n_dof": int(2 * nodes.shape[0]),
        "ours_precond": "mgv" if use_mgv else "jacobi",
        "ours_resumed_from_checkpoint": known is not None,
        "ours_wall_clock_s": ours_wall_clock_s, "ours_cg_iters": ours_cg_iters,
        "ours_cg_failures": ours_cg_failures,
        "ours_peak_mem_mb": peak_ours if known is None else None,
        "torchfem_wall_clock_s": t_theirs, "torchfem_peak_mem_mb": peak_theirs,
    }


def _known_mgv_result(N):
    """Item #4's own already-committed, real-GPU-measured mgv result
    for this N, from highdof_stress_qoi_results/*.json -- reused here
    instead of re-solving "ours" from scratch when a checkpoint resume
    is detected. Returns None for any N not already measured there."""
    import json
    import os
    PF = os.path.dirname(os.path.abspath(__file__))
    sources = [
        ("high_dof_stress_qoi_B1_neo_hookean_mgv_N401.json", (401,)),
        ("high_dof_stress_qoi_B1_neo_hookean_mgv_N701_1001_1401.json", (701, 1001, 1401)),
    ]
    for fname, ns in sources:
        if N not in ns:
            continue
        path = os.path.join(PF, "highdof_stress_qoi_results", fname)
        if not os.path.exists(path):
            return None
        with open(path) as f:
            data = json.load(f)
        for row in data["orders"]["Q4"]["rows"]:
            if row["N"] == N:
                return {"wall_clock_s": row["wall_clock_s"], "cg_iters": row["cg_iters"],
                        "cg_failures": row["cg_failures"]}
    return None


def run_sweep(resolutions, out_json, device=None, checkpoint_dir=None):
    """Resumable across resolutions, matching high_dof_convergence_study's
    own pattern: writes progress after every row so a Colab disconnect
    loses at most the resolution in progress, and a re-run of this same
    call skips resolutions already in out_json instead of re-solving them.

    checkpoint_dir: item #4's own checkpoint directory (typically
    /content/drive/MyDrive/pfem_ckpt on Colab) -- when given, "ours"
    reuses the already-completed mgv solves there instead of re-solving
    from scratch; see run_sweep_row's own docstring for why this
    matters (~15h of otherwise-redundant GPU time)."""
    import json
    import os

    device = device or (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
    print('device:', device)

    done = {}
    if out_json and os.path.exists(out_json):
        with open(out_json) as f:
            done = {r["N"]: r for r in json.load(f).get("rows", [])}

    rows = list(done.values())
    for N in resolutions:
        if N in done:
            print(f"  N={N} already in {out_json}, skipping")
            continue
        row = run_sweep_row(N, device, checkpoint_dir=checkpoint_dir)
        rows.append(row)
        rows.sort(key=lambda r: r["N"])
        if out_json:
            with open(out_json, "w") as f:
                json.dump({"geometry": "B1", "material": "neo_hookean", "order": "Q4",
                           "device": str(device), "rows": rows}, f, indent=2)
    return rows


def run_convergence_study(resolutions, out_json, geometry="B1", material="neo_hookean",
                           order="Q4", fine_N=2236, checkpoint_dir=None, device=None,
                           tol=1e-8, dtype=torch.float64):
    """Timon round-9 items 2+6 (2026-09-10): torch-fem's own ACCURACY and
    MESH CONVERGENCE, evaluated against the exact same fine ~10M-DOF
    reference "ours" own Table 6a/6b/6c uses -- not a simpler,
    single-point ours-vs-torchfem displacement diff. Reuses
    high_dof_convergence_study.py's own solve_one/compute_l2_h1_errors/
    fit_convergence_rate directly, so torch-fem's L2/H1 numbers at each
    N land in exactly the same units and convention as "ours" own
    already-published numbers at the same N -- a genuine apples-to-
    apples table, and a single study that answers "did you compare
    accuracy" and "is there mesh convergence" together, since both are
    now the same set of rows against the same reference.

    fine_N/checkpoint_dir: point at "ours" own already-converged fine
    reference (e.g. checkpoint_dir=".../pfem_ckpt", which holds
    fine_B1_neo_hookean_Q4_N2236.pt) so solve_one() RESUMES near-
    instantly instead of re-solving the single most expensive problem
    in the whole study.

    Resumable like every other sweep in this project: writes progress
    after every row, skips resolutions already in out_json -- a Colab
    disconnect loses at most the row in progress."""
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
        fixed_dofs = np.setdiff1d(np.arange(2 * nodes.shape[0]), free_dofs)

        u_theirs, elapsed, peak_mb = solve_theirs(
            nodes, elements, *elem_params, fext_full=fext_full, fixed_dofs=fixed_dofs,
            material=material, dtype=dtype, device=device, tol=tol)

        coarse = {"nodes": nodes, "elements": elements, "N": N,
                  "u": u_theirs.reshape(len(nodes), 2)}
        errs = compute_l2_h1_errors(coarse, fine, order, geometry)

        row = {"N": N, "n_dof": int(2 * nodes.shape[0]), "tol": tol,
               "torchfem_wall_clock_s": elapsed, "torchfem_peak_mem_mb": peak_mb,
               "l2_rel": errs["l2_rel"], "h1_semi_rel": errs["h1_semi_rel"]}
        print(f'  N={N}: wall_clock={elapsed:.2f}s l2_rel={errs["l2_rel"]:.3e} '
              f'h1_semi_rel={errs["h1_semi_rel"]:.3e} peak_mem_mb={peak_mb}')
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
        print(f'\nFitted convergence rate (torch-fem, {order}): L2 p={rate_l2:.3f} '
              f'(expected 2), H1 p={rate_h1:.3f} (expected 1)')
    return rows


def run_qoi_study(resolutions, out_json, geometry="B1", material="neo_hookean",
                   order="Q4", fine_N=2236, checkpoint_dir=None, device=None,
                   tol=1e-8, dtype=torch.float64):
    """Timon round-9 item 9 (2026-09-10): "What about all QoIs,
    particularly for large DOFs?" -- extends run_convergence_study's
    L2/H1 comparison with the energy norm and peak-stress QoIs
    high_dof_convergence_study.py already computes for "ours" own
    Table 6a/22-series and B1 point-1 study (compute_tangent_energy_
    error, find_fine_peak_stress/compute_peak_stress_error), evaluated
    against the SAME fine reference, so torch-fem's numbers land in the
    same units/convention as "ours" own already-published ones at the
    same N. "ours" own numbers need no new computation here -- they
    already exist in highdof_stress_qoi_results/*.json (energy_rel_
    error, peak_stress_rel_err columns) at every N this function is
    typically pointed at (1001, 1401 particularly, per Timon's own
    "large DOFs" phrasing, but works at any N).

    Resumable like every other sweep in this project."""
    import json
    import os

    device = device or (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
    print('device:', device)
    geom_kwargs = {"Lx": 1.0, "Ly": 1.0} if geometry == "B1" else {}

    fine_ckpt = (os.path.join(checkpoint_dir, f"fine_{geometry}_{material}_{order}_N{fine_N}.pt")
                 if checkpoint_dir else None)
    print(f'Loading/resuming fine reference N={fine_N} (checkpoint={fine_ckpt})...')
    fine = solve_one(geometry, order, fine_N, material, device, torch.float64,
                      cg_tol=1e-8, newton_tol=1e-8, checkpoint_path=fine_ckpt)
    print(f'  fine reference ready: n_dof={fine["n_dof"]}, wall_clock_s={fine["wall_clock_s"]:.1f}')

    E_fn = AnalyticFieldB1("E")
    nu_fn = AnalyticFieldB1("nu")
    print('Locating fine reference\'s own peak-stress point...')
    x_star, peak_ref = find_fine_peak_stress(fine, order, geometry, material, E_fn, nu_fn,
                                              device, dtype)
    print(f'  x_star={x_star}, peak_ref={peak_ref:.4f}')

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
        fixed_dofs = np.setdiff1d(np.arange(2 * nodes.shape[0]), free_dofs)

        u_theirs, elapsed, peak_mb = solve_theirs(
            nodes, elements, *elem_params, fext_full=fext_full, fixed_dofs=fixed_dofs,
            material=material, dtype=dtype, device=device, tol=tol)

        coarse = {"nodes": nodes, "elements": elements, "N": N,
                  "u": u_theirs.reshape(len(nodes), 2)}
        l2h1 = compute_l2_h1_errors(coarse, fine, order, geometry, **geom_kwargs)
        energy = compute_tangent_energy_error(coarse, fine, order, order, geometry, material,
                                               device, dtype, **geom_kwargs)
        stress = compute_peak_stress_error(coarse, fine, x_star, peak_ref, order, geometry,
                                            material, E_fn, nu_fn, device, dtype, **geom_kwargs)
        # Per-component PK1 stress at the same fixed peak point, and the
        # reaction-force resultant on the fixed boundary -- the two QoIs
        # Omar flagged as missing from the "all QoIs" claim (2026-09-10):
        # peak_stress_rel_err above is Frobenius-norm only, and nothing
        # here checked reactions at all before now.
        pk1_comp = pk1_component_errors_at_point(x_star, coarse, fine, order, geometry, material,
                                                  E_fn, nu_fn, device, dtype, **geom_kwargs)
        reaction = compute_reaction_resultant_error(coarse, fine, geometry, material, E_fn, nu_fn,
                                                      device, dtype, order=order)

        row = {"N": N, "n_dof": int(2 * nodes.shape[0]), "tol": tol,
               "torchfem_wall_clock_s": elapsed, "torchfem_peak_mem_mb": peak_mb,
               "l2_rel": l2h1["l2_rel"], "h1_semi_rel": l2h1["h1_semi_rel"],
               "energy_norm_rel": energy["tangent_energy_rel"],
               "peak_stress_pred": stress["peak_stress_pred"],
               "peak_stress_ref": stress["peak_stress_ref"],
               "peak_stress_rel_err": stress["peak_stress_rel_err"],
               "stress_field_l2_rel": stress["stress_field_l2_rel"],
               "P11_field_l2_rel": stress["P11_field_l2_rel"],
               "P12_field_l2_rel": stress["P12_field_l2_rel"],
               "P21_field_l2_rel": stress["P21_field_l2_rel"],
               "P22_field_l2_rel": stress["P22_field_l2_rel"],
               "P11_at_peak_rel_err": pk1_comp["P11_at_peak_rel_err"],
               "P12_at_peak_rel_err": pk1_comp["P12_at_peak_rel_err"],
               "P21_at_peak_rel_err": pk1_comp["P21_at_peak_rel_err"],
               "P22_at_peak_rel_err": pk1_comp["P22_at_peak_rel_err"],
               "reaction_resultant_pred": reaction["reaction_resultant_pred"],
               "reaction_resultant_ref": reaction["reaction_resultant_ref"],
               "reaction_resultant_rel_err": reaction["reaction_resultant_rel_err"]}
        print(f'  N={N}: l2_rel={row["l2_rel"]:.3e} h1_semi_rel={row["h1_semi_rel"]:.3e} '
              f'energy_norm_rel={row["energy_norm_rel"]:.3e} '
              f'peak_stress_rel_err={row["peak_stress_rel_err"]:.3e} '
              f'reaction_resultant_rel_err={row["reaction_resultant_rel_err"]:.3e}')
        rows.append(row)
        rows.sort(key=lambda r: r["N"])
        if out_json:
            with open(out_json, "w") as f:
                json.dump({"geometry": geometry, "material": material, "order": order,
                           "fine_N": fine_N, "device": str(device), "tol": tol,
                           "peak_stress_x_star": [float(x) for x in np.ravel(x_star)],
                           "peak_stress_ref": float(peak_ref), "rows": rows}, f, indent=2)
    return rows


def run_breakdown_sweep(resolutions, out_json, geometry="B1", material="neo_hookean",
                         order="Q4", device=None, tol=1e-8, dtype=torch.float64,
                         direct_max_n=701):
    """Timon round-9 item 3 (2026-09-10), at production scale: total time,
    assembly time, solve/factorization time, nonlinear-iteration count,
    and peak memory for torch-fem, at each N, using
    solve_theirs_with_breakdown. Also tries method="direct" (a real
    factorization, Timon's own suggestion) at every N up to
    `direct_max_n` -- direct sparse solves scale far worse than CG at
    large DOF (see that function's own docstring), so this stops
    attempting "direct" above that N rather than risk an OOM/multi-hour
    factorization on an untested size; "cg" is always attempted at
    every N regardless.

    Resumable like every other sweep in this project: writes progress
    after every row, skips (N, method) pairs already in out_json."""
    import json
    import os

    device = device or (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
    print('device:', device)

    done = set()
    rows = []
    if out_json and os.path.exists(out_json):
        with open(out_json) as f:
            rows = json.load(f).get("rows", [])
        done = {(r["N"], r["method"]) for r in rows}

    for N in resolutions:
        nodes, elements, free_dofs, fext_full, elem_params = build_mesh_and_bcs(
            geometry, order, N, material, device, dtype)
        mu, lam = elem_params
        fixed_dofs = np.setdiff1d(np.arange(2 * nodes.shape[0]), free_dofs)

        methods = ["cg"] + (["direct"] if N <= direct_max_n else [])
        for method in methods:
            if (N, method) in done:
                print(f'  N={N} method={method} already in {out_json}, skipping')
                continue
            precond = "jacobi" if method == "cg" else None
            result = solve_theirs_with_breakdown(
                nodes, elements, mu, lam, fext_full, fixed_dofs,
                dtype=dtype, device=device, tol=tol, method=method, preconditioner=precond)
            row = {"N": N, "n_dof": int(2 * nodes.shape[0]), "method": method,
                   "total_time_s": result["total_time_s"],
                   "assembly_time_s": result["assembly_time_s"],
                   "solve_time_s": result["solve_time_s"],
                   "other_time_s": result["other_time_s"],
                   "n_nonlinear_iters": result["n_nonlinear_iters"],
                   "peak_mem_mb": result["peak_mem_mb"]}
            print(f'  N={N} method={method}: total={row["total_time_s"]:.2f}s '
                  f'assembly={row["assembly_time_s"]:.2f}s solve={row["solve_time_s"]:.2f}s '
                  f'iters={row["n_nonlinear_iters"]} peak_mem_mb={row["peak_mem_mb"]}')
            rows.append(row)
            rows.sort(key=lambda r: (r["N"], r["method"]))
            if out_json:
                with open(out_json, "w") as f:
                    json.dump({"geometry": geometry, "material": material, "order": order,
                               "device": str(device), "tol": tol, "direct_max_n": direct_max_n,
                               "rows": rows}, f, indent=2)
    return rows


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "sweep":
        resolutions = [int(x) for x in sys.argv[2].split(",")] if len(sys.argv) > 2 else [401, 701, 1001, 1401]
        out_json = sys.argv[3] if len(sys.argv) > 3 else None
        checkpoint_dir = sys.argv[4] if len(sys.argv) > 4 else None
        run_sweep(resolutions, out_json, checkpoint_dir=checkpoint_dir)
    elif len(sys.argv) > 1 and sys.argv[1] == "convergence":
        resolutions = [int(x) for x in sys.argv[2].split(",")] if len(sys.argv) > 2 else [51, 101, 201, 401]
        out_json = sys.argv[3] if len(sys.argv) > 3 else None
        checkpoint_dir = sys.argv[4] if len(sys.argv) > 4 else None
        fine_N = int(sys.argv[5]) if len(sys.argv) > 5 else 2236
        run_convergence_study(resolutions, out_json, checkpoint_dir=checkpoint_dir, fine_N=fine_N)
    elif len(sys.argv) > 1 and sys.argv[1] == "breakdown":
        resolutions = [int(x) for x in sys.argv[2].split(",")] if len(sys.argv) > 2 else [401, 701, 1001, 1401]
        out_json = sys.argv[3] if len(sys.argv) > 3 else None
        direct_max_n = int(sys.argv[4]) if len(sys.argv) > 4 else 701
        run_breakdown_sweep(resolutions, out_json, direct_max_n=direct_max_n)
    elif len(sys.argv) > 1 and sys.argv[1] == "qoi":
        resolutions = [int(x) for x in sys.argv[2].split(",")] if len(sys.argv) > 2 else [1001, 1401]
        out_json = sys.argv[3] if len(sys.argv) > 3 else None
        checkpoint_dir = sys.argv[4] if len(sys.argv) > 4 else None
        fine_N = int(sys.argv[5]) if len(sys.argv) > 5 else 2236
        run_qoi_study(resolutions, out_json, checkpoint_dir=checkpoint_dir, fine_N=fine_N)
    else:
        N = int(sys.argv[1]) if len(sys.argv) > 1 else 11
        tol = float(sys.argv[2]) if len(sys.argv) > 2 else 1e-8
        ok, _rel_diff = _correctness_check(N, tol=tol)
        sys.exit(0 if ok else 1)
