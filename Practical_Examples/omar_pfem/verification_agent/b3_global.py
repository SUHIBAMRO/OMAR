"""B3 Q20 reference with extended QoIs, for the verification agent.

Same mesh builders, material, boundary conditions, load stepping and
solver settings as solve_quadratic_hex_groove_stress(_z_refined); the
solve body is reproduced here (and checked to reproduce the stored
region averages) so it can additionally return what the LLM critic
asked for on 2026-11-16:

  global      reaction moment about the rotation axis (y through
              (0,0,Lz/2)) on the rotated inner surface, total strain energy
  admissibility  min det F over every Gauss point of the body
  invariants  region averages of von Mises stress and pressure
  frame       region averages of cylindrical components (r, theta, z about
              the bushing axis z) instead of global Cartesian ones
  solver      optional tighter tolerances, for an iterative-error check

The model is the y>=0 half (symmetry plane y=0); global quantities are for
that half.
"""
import json
import os
import sys
import time

import numpy as np
import torch

from omar_pfem.data.data_generate_B3 import rigid_rotation_displacement, groove_radius_of_curvature
from omar_pfem.data.data_generate_B3_dataset import GROOVE_DEPTH, GROOVE_HALF_WIDTH
from omar_pfem.data.solve_quadratic_hex_groove_stress import (
    build_quadratic_mesh_with_curved_surfaces, R_IN0, LZ, E, NU, PHI)
from omar_pfem.torchfem_comparison import neo_hookean_psi_3d

HERE = os.path.dirname(os.path.abspath(__file__))


def _family_resolutions():
    """Read the exact (Ntheta, Nr, Nz) of the stored global family."""
    P = os.path.abspath(os.path.join(HERE, "..", ".."))
    res = [tuple(r["resolution"]) for r in
           json.load(open(os.path.join(P, "item_1f_avg_for_original_four_points.json")))]
    for name in ("item_1c_extra_point_23520_percomponent_scf.json",
                 "item_1d_sixth_family_point_33120.json",
                 "item_1e_seventh_family_point_45000.json"):
        res.append(tuple(json.load(open(os.path.join(P, name)))["resolution"]))
    return sorted(res, key=lambda t: t[0] * t[1] * t[2])


def _full_integration_solid():
    """torchfem Solid whose 20-node hexahedron uses the full 3x3x3 Gauss
    rule (27 points) instead of torchfem's default 2x2x2 (8 points, the
    reduced rule for a 20-node serendipity element)."""
    from torchfem import Solid
    from torchfem.elements import Hexa1, Hexa2, Tetra1, Tetra2, classproperty

    g = (3.0 / 5.0) ** 0.5
    pts1, w1 = (-g, 0.0, g), (5.0 / 9.0, 8.0 / 9.0, 5.0 / 9.0)

    class Hexa2Full(Hexa2):
        @classproperty
        def ipoints(cls):
            return torch.tensor([[a, b, c] for c in pts1 for b in pts1 for a in pts1])

        @classproperty
        def iweights(cls):
            return torch.tensor([wa * wb * wc for wc in w1 for wb in w1 for wa in w1])

    class SolidFull(Solid):
        element_types = (Tetra1, Hexa1, Tetra2, Hexa2Full)

    return SolidFull


def solve_extended(Ntheta, Nr, Nz, z_beta=None, rtol=1e-8, atol=1e-8, dtype=torch.float64,
                   dump_path=None, dump_radius=0.25, quadrature="reduced"):
    """quadrature: "reduced" = torchfem's 2x2x2 rule for Q20 (every stored
    result), "full" = 3x3x3 (27 points)."""
    from torchfem import Solid
    from torchfem.materials import Hyperelastic3D
    if quadrature not in ("reduced", "full"):
        raise ValueError(quadrature)
    SolidCls = _full_integration_solid() if quadrature == "full" else Solid

    if z_beta is None:
        nodes, elements, inner, outer, sym = build_quadratic_mesh_with_curved_surfaces(
            Ntheta, Nr, Nz, r_grading=1.0, groove_depth=GROOVE_DEPTH)
    else:
        from omar_pfem.data.solve_quadratic_hex_groove_stress_z_refined import (
            build_quadratic_mesh_with_curved_surfaces_z_refined)
        nodes, elements, inner, outer, sym = build_quadratic_mesh_with_curved_surfaces_z_refined(
            Ntheta, Nr, Nz, z_beta, r_grading=1.0, groove_depth=GROOVE_DEPTH)

    nodes_t = torch.tensor(nodes, dtype=dtype)
    elements_t = torch.tensor(elements, dtype=torch.long)
    mu = E / (2 * (1 + NU))
    lam = E * NU / ((1 + NU) * (1 - 2 * NU))
    params = torch.tensor([mu, lam], dtype=dtype)
    model = SolidCls(nodes_t, elements_t, Hyperelastic3D(psi=neo_hookean_psi_3d, params=params))
    n_nodes = nodes.shape[0]
    model.forces = torch.zeros(n_nodes, 3, dtype=dtype)
    constraints = torch.zeros(n_nodes, 3, dtype=torch.bool)
    displacements = torch.zeros(n_nodes, 3, dtype=dtype)
    constraints[outer, :] = True
    ux, uy, uz = rigid_rotation_displacement(nodes[inner], LZ, PHI)
    constraints[inner, :] = True
    displacements[inner, 0] = torch.tensor(ux, dtype=dtype)
    displacements[inner, 1] = torch.tensor(uy, dtype=dtype)
    displacements[inner, 2] = torch.tensor(uz, dtype=dtype)
    constraints[sym, 1] = True
    model.constraints, model.displacements = constraints, displacements

    increments = torch.linspace(0.0, 1.0, 11, dtype=dtype)
    old = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    t0 = time.time()
    try:
        with torch.no_grad():
            u, f, P, F, state = model.solve(
                increments=increments, max_iter=30, rtol=rtol, atol=atol, stol=1e-8,
                method="cg", preconditioner="jacobi", verbose=False,
                aggregate_integration_points=False)
        N_gauss, _, detJ = model.eval_shape_functions(model.etype.ipoints)
    finally:
        torch.set_default_dtype(old)
    elapsed = time.time() - t0

    u_np, f_np = u.cpu().numpy(), f.cpu().numpy()
    P = P.transpose(0, 1).cpu().numpy()          # (n_elem, n_gauss, 3, 3)
    F = F.transpose(0, 1).cpu().numpy()
    detF = np.linalg.det(F)
    w = model.etype.iweights.cpu().numpy()[None, :] * np.abs(detJ.transpose(0, 1).cpu().numpy())
    sigma = np.einsum("egij,egkj->egik", P, F) / detF[..., None, None]
    pos = torch.einsum("gn,enD->egD", N_gauss, nodes_t[elements_t]).cpu().numpy()

    # --- standard region QoIs (identical definitions to the stored data) ---
    ref = np.array([R_IN0 - GROOVE_DEPTH, 0.0, LZ / 2.0])
    radius = 2.0 * groove_radius_of_curvature(GROOVE_DEPTH, GROOVE_HALF_WIDTH)
    dist = np.linalg.norm(pos - ref[None, None, :], axis=-1)
    mask = dist < radius
    if dump_path is not None:
        # Gauss-point data near the groove, for post-hoc region definitions
        near = dist < dump_radius
        np.savez_compressed(dump_path, pos=pos[near], sigma=sigma[near], w=w[near],
                            ref=ref, radius=radius, n_elem=elements.shape[0])
    s, wr = sigma[mask], w[mask]
    W = wr.sum()
    avg = lambda v: float((v * wr).sum() / W)
    comps = {"xx": (0, 0), "yy": (1, 1), "zz": (2, 2), "xy": (0, 1), "xz": (0, 2), "yz": (1, 2)}
    per_component = {k: {"avg": avg(s[:, i, j]), "max": float(np.abs(s[:, i, j]).max())}
                     for k, (i, j) in comps.items()}
    frob = np.sqrt(np.einsum("nij,nij->n", s, s))

    # --- invariants ---
    p = -np.trace(s, axis1=1, axis2=2) / 3.0
    dev = s + p[:, None, None] * np.eye(3)
    vm = np.sqrt(1.5 * np.einsum("nij,nij->n", dev, dev))

    # --- cylindrical components about the bushing axis z ---
    th = np.arctan2(pos[mask][:, 1], pos[mask][:, 0])
    c, sn = np.cos(th), np.sin(th)
    Q = np.zeros((len(th), 3, 3))
    Q[:, 0, 0], Q[:, 0, 1], Q[:, 1, 0], Q[:, 1, 1], Q[:, 2, 2] = c, sn, -sn, c, 1.0
    sc = np.einsum("nai,nij,nbj->nab", Q, s, Q)
    cyl = {"rr": (0, 0), "tt": (1, 1), "zz": (2, 2), "rt": (0, 1), "rz": (0, 2), "tz": (1, 2)}
    per_cyl = {k: {"avg": avg(sc[:, i, j]), "max": float(np.abs(sc[:, i, j]).max())}
               for k, (i, j) in cyl.items()}

    # --- global QoIs ---
    x_def = nodes + u_np
    r_arm = x_def[inner] - np.array([0.0, 0.0, LZ / 2.0])
    moment_y = float((r_arm[:, 2] * f_np[inner, 0] - r_arm[:, 0] * f_np[inner, 2]).sum())
    with torch.no_grad():
        psi = neo_hookean_psi_3d(torch.tensor(F.reshape(-1, 3, 3), dtype=dtype), params)
    strain_energy = float((psi.cpu().numpy().reshape(w.shape) * w).sum())

    return {
        "Ntheta": Ntheta, "Nr": Nr, "Nz": Nz, "z_beta": z_beta, "quadrature": quadrature,
        "n_gauss_per_elem": int(len(model.etype.iweights)),
        "n_elem": int(elements.shape[0]), "n_nodes": int(n_nodes),
        "rtol": rtol, "atol": atol, "elapsed_s": elapsed,
        "n_region_gauss": int(mask.sum()),
        "region_avg_sigma_frob": avg(frob), "per_component": per_component,
        "region_avg_von_mises": avg(vm), "region_max_von_mises": float(vm.max()),
        "region_avg_pressure": avg(p),
        "per_cylindrical": per_cyl,
        "reaction_moment_y": moment_y, "strain_energy": strain_energy,
        "min_detF": float(detF.min()), "max_detF": float(detF.max()),
    }


def run(cases, cache):
    rows = json.load(open(cache)) if os.path.exists(cache) else []
    key = lambda r: (r["Ntheta"], r["Nr"], r["Nz"], r["z_beta"], r["rtol"])
    done = {key(r) for r in rows}
    for (Nt, Nr, Nz, zb, rtol) in cases:
        if (Nt, Nr, Nz, zb, rtol) in done:
            continue
        r = solve_extended(Nt, Nr, Nz, z_beta=zb, rtol=rtol, atol=rtol)
        rows.append(r)
        json.dump(rows, open(cache, "w"), indent=1)
        print(f"{r['n_elem']:>6} z_beta={zb} rtol={rtol:g}: M_y={r['reaction_moment_y']:.6g} "
              f"W={r['strain_energy']:.6g} vm_avg={r['region_avg_von_mises']:.5g} "
              f"frob_avg={r['region_avg_sigma_frob']:.5g} minJ={r['min_detF']:.4f} "
              f"({r['elapsed_s']:.0f}s)", flush=True)
    return rows


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "family"
    cache = os.path.join(HERE, "b3_extended.json")
    fam = _family_resolutions()
    if which == "smoke":
        run([(*fam[0], None, 1e-8)], cache)
    elif which == "family":
        run([(*r, None, 1e-8) for r in fam], cache)
    elif which == "strategies":
        run([(21, 18, 31, zb, 1e-8) for zb in (1.0, 2.0, 3.0)], cache)
    elif which == "dump":
        # one solve per case, saving Gauss-point data within 0.25 of the groove root
        ddir = os.path.join(HERE, "b3_gauss_near_groove")
        os.makedirs(ddir, exist_ok=True)
        cases = [(*r, None) for r in fam] + [(21, 18, 31, zb) for zb in (1.0, 2.0, 3.0)]
        for Nt, Nr_, Nz, zb in cases:
            path = os.path.join(ddir, f"{Nt}_{Nr_}_{Nz}_zb{zb}.npz")
            if os.path.exists(path):
                continue
            r = solve_extended(Nt, Nr_, Nz, z_beta=zb, dump_path=path)
            print(f"dumped {path} ({r['n_elem']} el, {r['elapsed_s']:.0f}s, "
                  f"frob_avg={r['region_avg_sigma_frob']:.6g})", flush=True)
    elif which == "fullint":
        # reduced (8-point) vs full (27-point) integration of the Q20 elements,
        # on the coarsest level and on the 10,200-element level
        out = os.path.join(HERE, "b3_fullint.json")
        rows = json.load(open(out)) if os.path.exists(out) else []
        ddir = os.path.join(HERE, "b3_gauss_near_groove_fullint")
        os.makedirs(ddir, exist_ok=True)
        for Nt, Nr_, Nz in (fam[0], fam[2]):
            if any((r["Ntheta"], r["Nr"], r["Nz"]) == (Nt, Nr_, Nz) for r in rows):
                continue
            r = solve_extended(Nt, Nr_, Nz, quadrature="full",
                               dump_path=os.path.join(ddir, f"{Nt}_{Nr_}_{Nz}_full.npz"))
            rows.append(r)
            json.dump(rows, open(out, "w"), indent=1)
            print(f"full integration {r['n_elem']} el ({r['n_gauss_per_elem']} pts/el, "
                  f"{r['elapsed_s']:.0f}s): M_y={r['reaction_moment_y']:.6g} "
                  f"W={r['strain_energy']:.6g} frob_avg={r['region_avg_sigma_frob']:.5g} "
                  f"p_avg={r['region_avg_pressure']:.5g} minJ={r['min_detF']:.4f}", flush=True)
    elif which == "solver":
        run([(*fam[0], None, 1e-11), (*fam[2], None, 1e-11)], cache)
