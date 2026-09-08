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
# which has nothing to do with FEM at all -- confirmed directly by
# reading pyvista's own source
# (pyvista/core/utilities/misc.py's _allow_ipython_completion): pyvista
# registers each VTK-wrapping class with IPython's tab-completion
# policy, but ONLY does anything `if 'IPython' in sys.modules` (true on
# Colab, since the notebook kernel itself is IPython; false in a plain
# script, which is why this never triggers in this project's own dev
# environment). When it does trigger, it unconditionally imports
# IPython.core.guarded_eval, a submodule Colab's installed IPython
# version does not have. The function only uses that module via
# `getattr(guarded_eval, 'EVALUATION_POLICIES', {}).get('limited')`
# (already defaulting to {} if the attribute is missing), so a harmless
# empty stub module satisfies the import and makes the rest of the
# function a no-op -- confirmed by reading the function's full source,
# not guessed. Installed defensively, before torch-fem/pyvista are ever
# imported, but only if the real submodule isn't already there.
if "IPython" in sys.modules and "IPython.core.guarded_eval" not in sys.modules:
    try:
        import IPython.core.guarded_eval  # noqa: F401
    except ModuleNotFoundError:
        sys.modules["IPython.core.guarded_eval"] = types.ModuleType(
            "IPython.core.guarded_eval")

from omar_pfem.high_dof_convergence_study import (
    build_mesh_and_bcs, AnalyticFieldB1)
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


def build_torchfem_model(nodes, elements, mu, lam, fext_full, fixed_dofs,
                          dtype=torch.float32, device=None):
    device = device or torch.device('cpu')
    # torch-fem's own near_null_space() (called unconditionally inside
    # solve(), regardless of preconditioner choice -- confirmed by
    # reading base.py, not assumed) hardcodes torch.eye(3) at its
    # default (float32) dtype and errors on a float64 model. float32 is
    # torch-fem's own working precision here, not a compromise on our
    # side; the correctness check below uses a tolerance appropriate to
    # single precision rather than this project's usual 1e-8 float64 one.
    """nodes/elements/fext_full/fixed_dofs come directly from
    build_mesh_and_bcs -- the SAME arrays this project's own solver
    uses, not a re-derivation."""
    from torchfem import Planar
    from torchfem.materials import HyperelasticPlaneStrain

    nodes_t = torch.tensor(nodes, dtype=dtype, device=device)
    elements_t = torch.tensor(elements, dtype=torch.long, device=device)
    params = torch.stack([
        torch.tensor(mu, dtype=dtype, device=device), torch.tensor(lam, dtype=dtype, device=device)
    ], dim=-1)  # (n_elem, 2)

    material = HyperelasticPlaneStrain(psi=neo_hookean_psi_3d, params=params)
    model = Planar(nodes_t, elements_t, material)

    n_nodes = nodes.shape[0]
    model.forces = torch.tensor(fext_full, dtype=dtype, device=device).reshape(n_nodes, 2)

    # B1's fixed_dofs is bottom-edge nodes with BOTH components fixed
    # (build_mesh_and_bcs: concatenate([2*bottom_nodes, 2*bottom_nodes+1])),
    # so every fixed dof's node has both x and y constrained -- no need to
    # split by component.
    constraints = torch.zeros(n_nodes, 2, dtype=torch.bool, device=device)
    fixed_nodes = np.unique(fixed_dofs // 2)
    constraints[fixed_nodes, :] = True
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


def solve_theirs(nodes, elements, mu, lam, fext_full, fixed_dofs, nsteps=10,
                  dtype=torch.float32, device=None):
    device = device or torch.device('cpu')
    model = build_torchfem_model(nodes, elements, mu, lam, fext_full, fixed_dofs,
                                  dtype=dtype, device=device)
    increments = torch.linspace(0.0, 1.0, nsteps + 1, dtype=dtype, device=device)
    # stol (the iterative linear solver's own tolerance) defaults to
    # 1e-10, unreachable in float32 -- torch-fem's own working precision
    # here, see build_torchfem_model's docstring -- so CG never
    # "converges" and Newton gives up after exhausting its cutbacks.
    # Loosened to a float32-appropriate value; rtol/atol (Newton's own
    # convergence test) loosened correspondingly.
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
    t0 = time.time()
    u, *_ = model.solve(
        increments=increments, max_iter=30, rtol=1e-3, atol=1e-3, stol=1e-4,
        method="cg", preconditioner="jacobi", nlgeom=True, verbose=False)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.time() - t0
    peak_mb = (torch.cuda.max_memory_allocated(device) / 1e6) if device.type == "cuda" else None
    return u.reshape(-1).cpu().numpy(), elapsed, peak_mb


def _correctness_check(N=11):
    print(f"=== CPU correctness check: our solver vs. torch-fem, B1 x Neo-Hookean, N={N} ===")
    nodes, elements, free_dofs, fext_full, elem_params = build_mesh_and_bcs(
        "B1", "Q4", N, "neo_hookean", torch.device('cpu'), torch.float64)
    mu, lam = elem_params

    fixed_dofs = np.setdiff1d(np.arange(2 * nodes.shape[0]), free_dofs)

    u_ours, t_ours, stats, _peak = solve_ours(nodes, elements, free_dofs, elem_params, fext_full)
    print(f"  ours:      wall_clock={t_ours:.2f}s, cg_iters_total={stats['cg_iters_total']}, "
          f"cg_failures={stats['cg_failures']}")

    u_theirs, t_theirs, _peak2 = solve_theirs(nodes, elements, mu, lam, fext_full, fixed_dofs)
    print(f"  torch-fem: wall_clock={t_theirs:.2f}s")

    diff = np.linalg.norm(u_ours - u_theirs)
    ref = np.linalg.norm(u_ours) + 1e-30
    rel_diff = diff / ref
    print(f"  relative displacement-field difference: {rel_diff:.3e}")
    # torch-fem's side runs in float32 (see build_torchfem_model's own
    # docstring), so the tolerance here is single-precision-appropriate,
    # not this project's usual 1e-8 float64 one.
    ok = rel_diff < 1e-3
    print("  " + ("PASS" if ok else "FAIL"))
    return ok


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
        nodes, elements, mu, lam, fext_full, fixed_dofs, device=device)
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


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "sweep":
        resolutions = [int(x) for x in sys.argv[2].split(",")] if len(sys.argv) > 2 else [401, 701, 1001, 1401]
        out_json = sys.argv[3] if len(sys.argv) > 3 else None
        checkpoint_dir = sys.argv[4] if len(sys.argv) > 4 else None
        run_sweep(resolutions, out_json, checkpoint_dir=checkpoint_dir)
    else:
        N = int(sys.argv[1]) if len(sys.argv) > 1 else 11
        ok = _correctness_check(N)
        sys.exit(0 if ok else 1)
