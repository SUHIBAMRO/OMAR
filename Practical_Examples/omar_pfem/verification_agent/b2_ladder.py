"""B2 refinement ladder for the verification agent: a HELD-OUT problem.

B2 (quarter ring, R_in = 1, R_out = 2, pressure on the inner arc, symmetry
on both straight edges, outer arc free; Q4, plane strain, total Lagrangian,
the project's own CPU solver data_generate_B2.solve_hyperelastic_TL_ring)
with the FIXED smooth fields of the project's B2 mesh-convergence study
(mesh_convergence.AnalyticFieldB2): E = 1000 + 200 sin(2 theta), nu = 0.3,
p = 5 + 2 sin(2 theta).

The verification agent was designed on B1 and B3 only; this ladder lets it
run on B2 WITHOUT any change to its rules, tolerances or gates.

Levels: N = 7, 13, 25, 49, 97, 193 nodes per direction (N - 1 divisible by
6), so the query points (theta = pi/3; r = 1 and r = 1.5) are mesh NODES on
every level -- no interpolation error. Strategies (radial grading toward
the loaded inner arc, the generator's r_grading) at N = 49 for information;
the rules do not require them for a smooth problem.

    python -m omar_pfem.verification_agent.b2_ladder [--Ns 7,13,25,49,97,193]
"""
import argparse
import json
import os
import time

import numpy as np
import torch

from omar_pfem.gpu_fem_solver import precompute_element_params_B2, shape_Q4_torch
from omar_pfem.materials_torch import get_material_fns as get_material_fns_torch
from omar_pfem.mesh_convergence import AnalyticFieldB2, _compute_strain_energy

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "b2_ladder_neo_hookean.json")
R_IN, R_OUT = 1.0, 2.0
QUERY = {"inner": (np.pi / 3, 1.0), "mid": (np.pi / 3, 1.5)}       # (theta, r)
REGION = ((np.pi / 4, 1.5), 0.15)        # interior disc: centre (theta, r), radius
KERNEL_ELL = 0.1


def gauss_point_stress_b2(nodes, elements, u, elem_params, material, dtype=torch.float64):
    """PK1 at every 2x2 Gauss point with the per-Gauss-point material
    parameters B2's solver uses (precompute_element_params_B2: (n_el, 4)
    per parameter, fem_core's Gauss order)."""
    energy_density_fn, _ = get_material_fns_torch(material)
    xy = torch.tensor(nodes, dtype=dtype)
    quad = torch.tensor(elements, dtype=torch.long)
    uv = torch.tensor(u, dtype=dtype)
    Xe, ue = xy[quad], uv[quad]
    g = 1.0 / np.sqrt(3.0)
    Ps, xs, ws, Js = [], [], [], []
    for k, (xi, eta) in enumerate([(-g, -g), (g, -g), (g, g), (-g, g)]):
        params = tuple(torch.tensor(np.asarray(p)[:, k], dtype=dtype) for p in elem_params)
        N_, dN_dxi = shape_Q4_torch(torch.tensor(xi, dtype=dtype), torch.tensor(eta, dtype=dtype),
                                    xy.device, dtype)
        J0 = torch.einsum("qai,aj->qij", Xe, dN_dxi)
        dN_dX = torch.einsum("aj,qjk->qak", dN_dxi, torch.linalg.inv(J0))
        F = (torch.eye(2, dtype=dtype) + torch.einsum("qai,qaj->qij", ue, dN_dX)).requires_grad_(True)
        psi = energy_density_fn(F, *params, dtype=dtype)
        (P,) = torch.autograd.grad(psi.sum(), F)
        Ps.append(P.detach())
        Js.append(torch.linalg.det(F.detach()))
        xs.append(torch.einsum("a,qai->qi", N_, Xe))
        ws.append(torch.linalg.det(J0))
    return torch.cat(Ps), torch.cat(xs), torch.cat(ws), torch.cat(Js)


def _node_at(nodes, theta, r):
    p = np.array([r * np.cos(theta), r * np.sin(theta)])
    d = np.linalg.norm(nodes - p, axis=1)
    i = int(np.argmin(d))
    return i, float(d[i])


def solve_level(N, material="neo_hookean", r_grading=1.0, tol=1e-8):
    from omar_pfem.data.data_generate_B2 import generate_grid_Q4_ring, solve_hyperelastic_TL_ring
    E_fn, nu_fn, p_fn = AnalyticFieldB2("E"), AnalyticFieldB2("nu"), AnalyticFieldB2("p")
    nodes, elements = generate_grid_Q4_ring(R_IN, R_OUT, N, N, r_grading=r_grading)
    t0 = time.time()
    u = solve_hyperelastic_TL_ring(nodes, elements, E_fn, nu_fn, p_fn, R_IN, nsteps=10,
                                   newton_max=30, tol=tol, material=material)
    elapsed = time.time() - t0
    elem_params = precompute_element_params_B2(nodes, elements, E_fn, nu_fn, material)
    P, x_gp, w_gp, J = gauss_point_stress_b2(nodes, elements, u, elem_params, material)
    Pn = torch.linalg.norm(P.reshape(-1, 4), dim=1).numpy()
    x_gp, w_gp = x_gp.numpy(), w_gp.numpy()
    (tc, rc), rad = REGION
    c = np.array([rc * np.cos(tc), rc * np.sin(tc)])
    d = np.linalg.norm(x_gp - c, axis=1)
    sharp = d < rad
    k = w_gp * np.exp(-(d / KERNEL_ELL) ** 2)
    row = {"N": N, "r_grading": r_grading, "newton_tol": tol, "h": 1.0 / (N - 1),
           "n_elem": int(len(elements)), "n_dof": int(2 * len(nodes)), "wall_clock_s": elapsed,
           "min_detF": float(J.min()),
           "max_disp_mag": float(np.sqrt((u ** 2).sum(1)).max()),
           "strain_energy": _compute_strain_energy(nodes, elements, u, E_fn, nu_fn, material,
                                                   lambda n, e, Ef, nf, m: elem_params),
           "peak_PK1_frob": float(Pn.max()),
           "interior_avg_PK1_frob": float((Pn[sharp] * w_gp[sharp]).sum() / w_gp[sharp].sum()),
           "interior_gk_PK1_frob": float((Pn * k).sum() / k.sum()),
           "interior_region_gauss_points": int(sharp.sum())}
    for name, (th, r) in QUERY.items():
        i, dist = _node_at(nodes, th, r)
        row[f"u_{name}"], row[f"v_{name}"] = float(u[i, 0]), float(u[i, 1])
        row[f"query_{name}_node_distance"] = dist
    return row


def run_ladder(Ns, r_grading=1.0, tol=1e-8, material="neo_hookean"):
    """Resumable: rows cached by (N, r_grading, newton_tol)."""
    rows = json.load(open(CACHE)) if os.path.exists(CACHE) else []
    done = {(r["N"], r["r_grading"], r["newton_tol"]) for r in rows}
    for N in Ns:
        if (N, r_grading, tol) in done:
            continue
        row = solve_level(N, material, r_grading, tol)
        rows.append(row)
        rows.sort(key=lambda r: (r["newton_tol"], r["r_grading"], r["N"]))
        json.dump(rows, open(CACHE, "w"), indent=1)
        print(f"N={N:<4} r_grading={r_grading} ({row['wall_clock_s']:.0f}s) " + "  ".join(
            f"{k}={v:.6g}" for k, v in row.items()
            if k not in ("N", "h", "n_dof", "r_grading", "newton_tol", "wall_clock_s")), flush=True)
    return sorted((r for r in rows if r["N"] in Ns and r["r_grading"] == r_grading
                   and r["newton_tol"] == tol), key=lambda r: r["N"])


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--Ns", default="7,13,25,49,97,193")
    ap.add_argument("--strategies", default="1.5,2.0", help="r_grading values at N=49")
    args = ap.parse_args(argv)
    run_ladder([int(n) for n in args.Ns.split(",")])
    for rg in [float(v) for v in args.strategies.split(",") if v]:
        run_ladder([49], r_grading=rg)


if __name__ == "__main__":
    main()
