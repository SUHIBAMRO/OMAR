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
    Ps, xs, ws = [], [], []
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
        xs.append(torch.einsum("a,qai->qi", N_, Xe))
        ws.append(detJ0)
    return torch.cat(Ps), torch.cat(xs), torch.cat(ws)


def solve_level(N, material):
    from omar_pfem.data.data_generate_B1 import generate_grid_Q4, solve_hyperelastic_TL_spatial
    E_fn, nu_fn, ty_fn = AnalyticFieldB1("E"), AnalyticFieldB1("nu"), AnalyticFieldB1("ty")
    nodes, elements = generate_grid_Q4(1.0, 1.0, N, N)
    u = solve_hyperelastic_TL_spatial(nodes, elements, E_fn, nu_fn, ty_fn, 1.0,
                                      nsteps=10, newton_max=30, tol=1e-7, material=material)
    elem_params = precompute_element_params_B1(nodes, elements, E_fn, nu_fn, material)
    P, x_gp, w_gp = gauss_point_stress(nodes, elements, u, elem_params, material)
    P_norm = torch.linalg.norm(P.reshape(-1, 4), dim=1)
    (cx, cy), r = INTERIOR_REGION
    mask = ((x_gp[:, 0] - cx) ** 2 + (x_gp[:, 1] - cy) ** 2) < r ** 2
    row = {
        "N": N, "h": 1.0 / (N - 1), "n_dof": 2 * len(nodes),
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


def run_ladder(Ns, material, cache):
    rows = json.load(open(cache)) if os.path.exists(cache) else []
    done = {r["N"] for r in rows}
    for N in Ns:
        if N in done:
            continue
        row = solve_level(N, material)
        rows.append(row)
        rows.sort(key=lambda r: r["N"])
        json.dump(rows, open(cache, "w"), indent=1)
        print(f"N={N:<4} " + "  ".join(f"{k}={v:.6g}" for k, v in row.items()
                                       if k not in ("N", "h", "n_dof")), flush=True)
    return [r for r in rows if r["N"] in Ns]
