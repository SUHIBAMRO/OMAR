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


def surface_nodal_stress(etype, nodes, elements, sigma, inner, ref, radius):
    """Cauchy stress at the corner nodes of the INNER (groove) surface within
    `radius` of `ref`: the 2x2x2 Gauss values of every adjacent element are
    extrapolated to its corners with the trilinear interpolant through the
    Gauss points (exact for fields linear in the parent coordinates), then
    averaged over the elements sharing the node. Reduced (8-point) rule only.
    Returns positions (n,3) and stresses (n,3,3)."""
    ip = etype.ipoints.numpy()
    if ip.shape[0] != 8:
        raise ValueError("surface extrapolation implemented for the 2x2x2 rule only")
    g = abs(float(ip[0, 0]))
    corners = etype.iso_coords[:8].numpy()
    # E[c, q] = prod_d (1 + s_qd * xi_cd / g) / 2, s = sign of the Gauss coordinate
    E = np.prod((1.0 + np.sign(ip)[None, :, :] * corners[:, None, :] / g) / 2.0, axis=2)
    on_inner = np.zeros(len(nodes), bool)
    on_inner[inner] = True
    near = on_inner & (np.linalg.norm(nodes - ref[None, :], axis=1) < radius)
    sel = np.where(near[elements[:, :8]].any(axis=1))[0]
    corner_sig = np.einsum("cq,eqij->ecij", E, sigma[sel])          # (n_sel, 8, 3, 3)
    acc = np.zeros((len(nodes), 3, 3))
    cnt = np.zeros(len(nodes))
    for k in range(8):
        idx = elements[sel, k]
        np.add.at(acc, idx, corner_sig[:, k])
        np.add.at(cnt, idx, 1.0)
    keep = near & (cnt > 0)
    return nodes[keep], acc[keep] / cnt[keep][:, None, None]


def solve_extended(Ntheta, Nr, Nz, z_beta=None, rtol=1e-8, atol=1e-8, dtype=torch.float64,
                   dump_path=None, dump_radius=0.25, quadrature="reduced", r_grading=1.0,
                   surface_path=None, linear_solver="cg"):
    """linear_solver: "cg" (Jacobi-preconditioned CG, every stored family
    result) or "direct" (pypardiso); both stop Newton at the same rtol/atol,
    the direct solver is for meshes whose element-size contrast makes CG slow.
    quadrature: "reduced" = torchfem's 2x2x2 rule for Q20 (every stored
    result), "full" = 3x3x3 (27 points). r_grading > 1 clusters the radial
    layers toward the inner (groove) surface (t -> t^r_grading).
    surface_path: save the nodal (extrapolated) stress on the groove surface
    within dump_radius of the groove root."""
    from torchfem import Solid
    from torchfem.materials import Hyperelastic3D
    if quadrature not in ("reduced", "full"):
        raise ValueError(quadrature)
    SolidCls = _full_integration_solid() if quadrature == "full" else Solid

    if z_beta is None:
        nodes, elements, inner, outer, sym = build_quadratic_mesh_with_curved_surfaces(
            Ntheta, Nr, Nz, r_grading=r_grading, groove_depth=GROOVE_DEPTH)
    else:
        from omar_pfem.data.solve_quadratic_hex_groove_stress_z_refined import (
            build_quadratic_mesh_with_curved_surfaces_z_refined)
        nodes, elements, inner, outer, sym = build_quadratic_mesh_with_curved_surfaces_z_refined(
            Ntheta, Nr, Nz, z_beta, r_grading=r_grading, groove_depth=GROOVE_DEPTH)

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
                method=linear_solver,
                preconditioner="jacobi" if linear_solver == "cg" else None, verbose=False,
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
    if surface_path is not None:
        spos, ssig = surface_nodal_stress(model.etype, nodes, elements, sigma, inner, ref,
                                          dump_radius)
        np.savez_compressed(surface_path, pos=spos, sigma=ssig, ref=ref, radius=radius,
                            n_elem=elements.shape[0])
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
        "r_grading": r_grading,
        "n_gauss_per_elem": int(len(model.etype.iweights)),
        "n_elem": int(elements.shape[0]), "n_nodes": int(n_nodes),
        "rtol": rtol, "atol": atol, "elapsed_s": elapsed, "linear_solver": linear_solver,
        "detJ_max_over_min": float(detJ.abs().max() / detJ.abs().min()),
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
    elif which == "radial":
        # radial strategy at 10,200 el: layers clustered toward the groove surface.
        # Valid gradings only: with the groove (curvature radius 0.0228) the
        # power-law grading collapses the first radial layer, so that the snapped
        # Q20 midside nodes bulge through it -- r_grading 1.75 or more gives
        # negative Jacobians, 1.5 a detJ contrast of 39-79 (uniform 3-4) on
        # which Jacobi-CG stalls; 1.1-1.3 keep it at 5-15.
        out = os.path.join(HERE, "b3_radial.json")
        rows = json.load(open(out)) if os.path.exists(out) else []
        ddir = os.path.join(HERE, "b3_gauss_radial")
        os.makedirs(ddir, exist_ok=True)
        for rg in (1.1, 1.2, 1.3):
            if any(r["r_grading"] == rg for r in rows):
                continue
            r = solve_extended(*fam[2], r_grading=rg,
                               dump_path=os.path.join(ddir, f"21_18_31_rg{rg}.npz"),
                               surface_path=os.path.join(ddir, f"21_18_31_rg{rg}_surface.npz"),
                               linear_solver="direct")
            rows.append(r)
            json.dump(rows, open(out, "w"), indent=1)
            print(f"radial r_grading={rg}: {r['n_elem']} el ({r['elapsed_s']:.0f}s) "
                  f"M_y={r['reaction_moment_y']:.6g} frob_avg={r['region_avg_sigma_frob']:.5g} "
                  f"minJ={r['min_detF']:.4f}", flush=True)
    elif which == "surface":
        # nodal groove-surface stress for every family level (and any finer level)
        ddir = os.path.join(HERE, "b3_surface")
        os.makedirs(ddir, exist_ok=True)
        extra = [tuple(r) for r in json.load(open(os.path.join(HERE, "b3_fine_levels.json")))] \
            if os.path.exists(os.path.join(HERE, "b3_fine_levels.json")) else []
        for Nt, Nr_, Nz in list(fam) + extra:
            path = os.path.join(ddir, f"{Nt}_{Nr_}_{Nz}.npz")
            if os.path.exists(path):
                continue
            r = solve_extended(Nt, Nr_, Nz, surface_path=path)
            print(f"surface {path} ({r['n_elem']} el, {r['elapsed_s']:.0f}s)", flush=True)
    elif which == "fine":
        # one finer family level: the family's own progression (Ntheta +4, Nr +2,
        # Nz +5 per level) continued, 41x28x56 = 59,400 el, ratio 1.097 in
        # (n_elem)^(1/3) over 45,000 el (the family's ratios fall 1.34 -> 1.11).
        # 48x34x66 = 100,815 el was tried first and killed for memory (14 GB RSS
        # at the 15 GB session limit).
        res = (41, 28, 56)
        json.dump([list(res)], open(os.path.join(HERE, "b3_fine_levels.json"), "w"))
        rows = json.load(open(cache)) if os.path.exists(cache) else []
        if not any((r["Ntheta"], r["Nr"], r["Nz"], r["z_beta"]) == (*res, None) for r in rows):
            ddir = os.path.join(HERE, "b3_gauss_near_groove")
            sdir = os.path.join(HERE, "b3_surface")
            os.makedirs(sdir, exist_ok=True)
            r = solve_extended(*res, dump_path=os.path.join(ddir, "{}_{}_{}_zbNone.npz".format(*res)),
                               surface_path=os.path.join(sdir, "{}_{}_{}.npz".format(*res)))
            rows.append(r)
            json.dump(rows, open(cache, "w"), indent=1)
            print(f"fine level {r['n_elem']} el ({r['elapsed_s']:.0f}s): "
                  f"M_y={r['reaction_moment_y']:.6g} W={r['strain_energy']:.6g} "
                  f"frob_avg={r['region_avg_sigma_frob']:.5g} minJ={r['min_detF']:.4f}",
                  flush=True)
    elif which == "solver":
        run([(*fam[0], None, 1e-11), (*fam[2], None, 1e-11)], cache)
