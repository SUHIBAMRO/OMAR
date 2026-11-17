"""B1 physical-problem mesh ladder for the verification agent.

Solves B1 (unit square, clamped bottom edge, top-edge traction) with the
closed-form fields of mesh_convergence.AnalyticFieldB1, so the same
continuous problem is discretized at every N, and records global,
pointwise, peak and region-averaged QoIs. The clamped bottom edge meets
the traction-free vertical edges at two corners where linear elasticity
predicts a stress singularity, so the peak stress is expected to fail the
convergence gate while the energy and displacements pass.
"""
import json
import os

import numpy as np
import torch
from scipy.interpolate import LinearNDInterpolator

from omar_pfem.gpu_fem_solver import precompute_element_params_B1, shape_Q4_torch
from omar_pfem.materials_torch import get_material_fns as get_material_fns_torch
from omar_pfem.mesh_convergence import AnalyticFieldB1, _compute_strain_energy

QUERY_POINTS = {"top_off_center": (0.7, 1.0), "mid_domain": (0.3, 0.6)}
INTERIOR_REGION = ((0.5, 0.5), 0.15)


def gauss_point_stress(nodes, elements, u, elem_params, material, dtype=torch.float64):
    """First Piola-Kirchhoff stress at every 2x2 Gauss point, by autograd of
    the same energy density the solver uses. Returns (P, x_gp, w_gp)."""
    energy_density_fn, _ = get_material_fns_torch(material)
    xy = torch.tensor(nodes, dtype=dtype)
    quad = torch.tensor(elements, dtype=torch.long)
    uv = torch.tensor(u, dtype=dtype)
    params = tuple(torch.tensor(p, dtype=dtype) for p in elem_params)
    Xe, ue = xy[quad], uv[quad]
    g = 1.0 / np.sqrt(3.0)
    Ps, xs, ws, Js = [], [], [], []
    for xi, eta in [(-g, -g), (g, -g), (g, g), (-g, g)]:
        N_, dN_dxi = shape_Q4_torch(torch.tensor(xi, dtype=dtype), torch.tensor(eta, dtype=dtype),
                                    xy.device, dtype)
        J0 = torch.einsum("qai,aj->qij", Xe, dN_dxi)
        detJ0 = torch.linalg.det(J0)
        dN_dX = torch.einsum("aj,qjk->qak", dN_dxi, torch.linalg.inv(J0))
        F = (torch.eye(2, dtype=dtype) + torch.einsum("qai,qaj->qij", ue, dN_dX)).requires_grad_(True)
        psi = energy_density_fn(F, *params, dtype=dtype)
        (P,) = torch.autograd.grad(psi.sum(), F)
        Ps.append(P.detach())
        Js.append(torch.linalg.det(F.detach()))
        xs.append(torch.einsum("a,qai->qi", N_, Xe))
        ws.append(detJ0)
    gauss_point_stress.last_detF = torch.cat(Js)
    return torch.cat(Ps), torch.cat(xs), torch.cat(ws)


def grade_nodes(nodes, a):
    """Cluster nodes toward the two clamped/free corners (0,0) and (1,0):
    x -> s - a sin(2 pi s)/(2 pi) refines both vertical edges, y -> t - a
    sin(pi t)/pi refines the clamped bottom edge. Element size at the
    corners shrinks by (1-a); edges stay exactly at 0 and 1, topology and
    node count are unchanged. a=0 is the uniform mesh."""
    if a == 0.0:
        return nodes
    s, t = nodes[:, 0], nodes[:, 1]
    x = s - a * np.sin(2 * np.pi * s) / (2 * np.pi)
    y = t - a * np.sin(np.pi * t) / np.pi
    for v, ref in ((x, s), (y, t)):
        v[np.isclose(ref, 0.0)] = 0.0
        v[np.isclose(ref, 1.0)] = 1.0
    return np.column_stack([x, y])


def solve_level(N, material, grading=0.0, tol=1e-7, dump_path=None, dump_radius=0.4):
    """dump_path: save |P|_F, positions and weights of the Gauss points within
    dump_radius of the interior-region centre, for post-hoc (smooth) region
    definitions."""
    from omar_pfem.data.data_generate_B1 import generate_grid_Q4, solve_hyperelastic_TL_spatial
    E_fn, nu_fn, ty_fn = AnalyticFieldB1("E"), AnalyticFieldB1("nu"), AnalyticFieldB1("ty")
    nodes, elements = generate_grid_Q4(1.0, 1.0, N, N)
    nodes = grade_nodes(nodes, grading)
    import time
    t0 = time.time()
    u = solve_hyperelastic_TL_spatial(nodes, elements, E_fn, nu_fn, ty_fn, 1.0,
                                      nsteps=10, newton_max=30, tol=tol, material=material)
    elem_params = precompute_element_params_B1(nodes, elements, E_fn, nu_fn, material)
    P, x_gp, w_gp = gauss_point_stress(nodes, elements, u, elem_params, material)
    P_norm = torch.linalg.norm(P.reshape(-1, 4), dim=1)
    (cx, cy), r = INTERIOR_REGION
    mask = ((x_gp[:, 0] - cx) ** 2 + (x_gp[:, 1] - cy) ** 2) < r ** 2
    if dump_path is not None:
        d2 = (x_gp[:, 0] - cx) ** 2 + (x_gp[:, 1] - cy) ** 2
        near = (d2 < dump_radius ** 2).numpy()
        np.savez_compressed(dump_path, pos=x_gp.numpy()[near], P_norm=P_norm.numpy()[near],
                            w=w_gp.numpy()[near], centre=np.array([cx, cy]), radius=r,
                            N=N, grading=grading)
    row = {
        "N": N, "grading": grading, "newton_tol": tol, "h": 1.0 / (N - 1),
        "n_dof": 2 * len(nodes),
        "min_detF": float(gauss_point_stress.last_detF.min()),
        "min_elem_size": float(min(np.diff(np.unique(nodes[:, 0])).min(),
                                   np.diff(np.unique(nodes[:, 1])).min())),
        "wall_clock_s": time.time() - t0,
        "max_disp_mag": float(np.sqrt((u ** 2).sum(1)).max()),
        "strain_energy": _compute_strain_energy(nodes, elements, u, E_fn, nu_fn, material,
                                                lambda n, e, Ef, nf, m: elem_params),
        "peak_PK1_frob": float(P_norm.max()),
        "interior_avg_PK1_frob": float((P_norm[mask] * w_gp[mask]).sum() / w_gp[mask].sum()),
        "interior_region_gauss_points": int(mask.sum()),
    }
    iu, iv = LinearNDInterpolator(nodes, u[:, 0]), LinearNDInterpolator(nodes, u[:, 1])
    for name, (qx, qy) in QUERY_POINTS.items():
        row[f"u_{name}"] = float(iu(qx, qy))
        row[f"v_{name}"] = float(iv(qx, qy))
    return row


def run_ladder(Ns, material, cache, grading=0.0, tol=1e-7):
    """Resumable: rows are cached by (N, grading, newton_tol); rows from
    before those keys existed are uniform (grading 0) with tol 1e-7."""
    rows = json.load(open(cache)) if os.path.exists(cache) else []
    for r in rows:
        r.setdefault("grading", 0.0)
        r.setdefault("newton_tol", 1e-7)
    done = {(r["N"], r["grading"], r["newton_tol"]) for r in rows}
    for N in Ns:
        if (N, grading, tol) in done:
            continue
        row = solve_level(N, material, grading, tol)
        rows.append(row)
        rows.sort(key=lambda r: (r["newton_tol"], r["grading"], r["N"]))
        json.dump(rows, open(cache, "w"), indent=1)
        print(f"N={N:<4} grading={grading} " + "  ".join(
            f"{k}={v:.6g}" for k, v in row.items() if k not in ("N", "h", "n_dof", "grading")),
            flush=True)
    return [r for r in rows if r["N"] in Ns and r["grading"] == grading
            and r["newton_tol"] == tol]


def dump_levels(material="neo_hookean", cases=((11, 0.0), (21, 0.0), (41, 0.0), (81, 0.0),
                                                (161, 0.0), (81, 0.5), (81, 0.8))):
    """Re-solve the given (N, grading) cases and save Gauss-point |P|_F near the
    interior region (b1_gauss_interior/); skips existing files."""
    here = os.path.dirname(os.path.abspath(__file__))
    ddir = os.path.join(here, "b1_gauss_interior")
    os.makedirs(ddir, exist_ok=True)
    for N, a in cases:
        path = os.path.join(ddir, f"N{N}_a{a}.npz")
        if os.path.exists(path):
            continue
        row = solve_level(N, material, a, dump_path=path)
        print(f"dumped {path}: interior_avg_PK1_frob={row['interior_avg_PK1_frob']:.9g} "
              f"({row['wall_clock_s']:.0f}s)", flush=True)


if __name__ == "__main__":
    dump_levels()
