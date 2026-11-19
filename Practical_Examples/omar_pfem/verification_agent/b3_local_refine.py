"""Local refinement of B3 toward the groove root (the next investigation the
diagnosis engine names for the B3 maxima: h <= rho/4 = 0.0057 at the root).

The project's builder (solve_quadratic_hex_groove_stress_z_refined) grades
z (sinh, z_beta) and r (power, r_grading); this module adds a circumferential
grading toward theta = 0, where the region of interest sits:
    theta = pi (t - a sin(pi t) / pi),  t = theta_uniform / pi,
element size at theta = 0 shrinks by (1 - a), at theta = pi grows by (1 + a).
Element count and connectivity are unchanged; only node positions move
(applied to the hex8 nodes before the Q20 upgrade and curved-surface
snapping, which work on each node's own theta).

    python -m omar_pfem.verification_agent.b3_local_refine scan
    python -m omar_pfem.verification_agent.b3_local_refine solve
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "b3_local_refine.json")
RHO = 0.0228
EDGES = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4), (0, 4), (1, 5),
         (2, 6), (3, 7)]


def theta_map(nodes, a):
    if a == 0.0:
        return nodes
    r = np.hypot(nodes[:, 0], nodes[:, 1])
    th = np.clip(np.arctan2(nodes[:, 1], nodes[:, 0]), 0.0, np.pi)
    t = th / np.pi
    th2 = np.pi * (t - a * np.sin(np.pi * t) / np.pi)
    out = nodes.copy()
    out[:, 0], out[:, 1] = r * np.cos(th2), r * np.sin(th2)
    return out


def build(Nt, Nr, Nz, z_beta=0.0, r_grading=1.0, theta_a=0.0):
    import omar_pfem.data.solve_quadratic_hex_groove_stress_z_refined as Z
    from omar_pfem.data.data_generate_B3_dataset import GROOVE_DEPTH
    orig = Z.generate_grid_hex8_bushing_z_refined

    def graded(*args, **kw):
        n, e = orig(*args, **kw)
        return theta_map(n, theta_a), e
    Z.generate_grid_hex8_bushing_z_refined = graded
    try:
        return Z.build_quadratic_mesh_with_curved_surfaces_z_refined(
            Nt, Nr, Nz, z_beta, r_grading=r_grading, groove_depth=GROOVE_DEPTH)
    finally:
        Z.generate_grid_hex8_bushing_z_refined = orig


def build_exact(Nt, Nr, Nz, z_beta=0.0, r_grading=1.0, theta_a=0.0):
    """As build(), but EVERY mid-side node is placed by the exact transfinite
    map x = (r_in(z) + t (R_out - r_in(z))) (cos theta, sin theta), with
    (theta, t, z) averaged from its two corner nodes, not only those on the
    inner/outer surface. The project's builder snaps surface mid-side nodes
    to the curve but leaves interior ones on straight chords, so a thin
    first radial layer at the groove is crossed by the curved surface edge
    (negative Jacobians for r_grading >= 1.6 or more radial layers)."""
    import torch
    from torchfem.elements import Hexa1, linear_to_quadratic
    from omar_pfem.data.data_generate_B3 import groove_R_in
    from omar_pfem.data.data_generate_B3_dataset import GROOVE_DEPTH, GROOVE_HALF_WIDTH
    from omar_pfem.data.data_generate_B3_local_z_refine import (
        generate_grid_hex8_bushing_z_refined)
    from omar_pfem.data.solve_quadratic_hex_groove_stress import LZ, R_IN0, R_OUT
    from omar_pfem.data.data_generate_B3 import boundary_node_sets
    n8, e8 = generate_grid_hex8_bushing_z_refined(R_IN0, R_OUT, LZ, Nt, Nr, Nz, GROOVE_DEPTH,
                                                  GROOVE_HALF_WIDTH, r_grading=r_grading,
                                                  z_beta=z_beta)
    n8 = theta_map(n8, theta_a)
    rin = lambda z: groove_R_in(z, LZ, R_IN0, GROOVE_DEPTH, GROOVE_HALF_WIDTH)
    th8 = np.clip(np.arctan2(n8[:, 1], n8[:, 0]), 0.0, np.pi)
    z8 = n8[:, 2]
    t8 = (np.hypot(n8[:, 0], n8[:, 1]) - rin(z8)) / (R_OUT - rin(z8))
    n20, e20 = linear_to_quadratic(torch.tensor(n8), torch.tensor(e8))
    n20, e20 = n20.numpy().copy(), e20.numpy()
    edges = Hexa1.edges.numpy()
    mid = e20[:, 8:20].ravel()
    a, b = e8[:, edges[:, 0]].ravel(), e8[:, edges[:, 1]].ravel()
    th, t, z = 0.5 * (th8[a] + th8[b]), 0.5 * (t8[a] + t8[b]), 0.5 * (z8[a] + z8[b])
    r = rin(z) + t * (R_OUT - rin(z))
    n20[mid, 0], n20[mid, 1], n20[mid, 2] = r * np.cos(th), r * np.sin(th), z
    inner, outer, sym = boundary_node_sets(n20, R_IN0, R_OUT, LZ, GROOVE_DEPTH, GROOVE_HALF_WIDTH)
    return n20, e20, inner, outer, sym


def quality(Nt, Nr, Nz, z_beta=0.0, r_grading=1.0, theta_a=0.0, exact=False):
    """Root element size (largest/smallest corner edge of the elements at
    the root node) and the Jacobian contrast over the mesh."""
    import torch
    from torchfem.elements import Hexa2
    from omar_pfem.verification_agent.b3_global import GROOVE_DEPTH, LZ, R_IN0
    nodes, el, *_ = (build_exact if exact else build)(Nt, Nr, Nz, z_beta, r_grading, theta_a)
    ref = np.array([R_IN0 - GROOVE_DEPTH, 0.0, LZ / 2.0])
    root = int(np.argmin(np.linalg.norm(nodes - ref, axis=1)))
    adj = el[(el[:, :8] == root).any(axis=1)]
    if len(adj) == 0:
        adj = el[(el == root).any(axis=1)]
    lens = [np.linalg.norm(nodes[e[a]] - nodes[e[c]]) for e in adj for a, c in EDGES]
    X = torch.tensor(nodes)[torch.tensor(el)]
    dN = Hexa2().B(Hexa2.ipoints.double()).transpose(1, 2)
    detJ = torch.linalg.det(torch.einsum("eai,qaj->eqij", X, dN)).numpy()
    return {"Nt": Nt, "Nr": Nr, "Nz": Nz, "z_beta": z_beta, "r_grading": r_grading,
            "theta_a": theta_a, "n_elem": int(len(el)), "root_dist": float(
                np.linalg.norm(nodes[root] - ref)), "h_root_max": float(max(lens)),
            "h_root_min": float(min(lens)), "h_over_rho": float(max(lens) / RHO),
            "min_detJ": float(detJ.min()), "detJ_contrast": float(detJ.max() / detJ.min())}


def scan():
    rows = []
    for Nt, Nr, Nz in ((21, 18, 31), (29, 22, 41)):
        for zb, rg, ta in ((0, 1, 0), (3, 1.3, 0.8), (4, 1.3, 0.9), (5, 1.5, 0.9), (5, 2.0, 0.95)):
            try:
                q = quality(Nt, Nr, Nz, zb, rg, ta)
            except AssertionError as e:
                q = {"Nt": Nt, "Nr": Nr, "Nz": Nz, "z_beta": zb, "r_grading": rg,
                     "theta_a": ta, "error": str(e)[:80]}
            rows.append(q)
            print(q, flush=True)
    return rows


# local-refinement sequence toward the root (exact mid-side placement):
# h_root / rho = 2.50, 0.77, 0.44, 0.25, 0.15
SEQUENCE = [(21, 18, 31, 0.0, 1.0, 0.0), (21, 18, 31, 3.0, 1.3, 0.8), (21, 18, 31, 5.0, 1.5, 0.9),
            (21, 26, 31, 5.0, 1.5, 0.9), (21, 26, 31, 5.0, 1.7, 0.93)]


def solve_sequence(cases=SEQUENCE):
    from omar_pfem.verification_agent.mixed_b3 import Problem, qois
    rows = json.load(open(OUT)) if os.path.exists(OUT) else []
    done = {tuple(r["case"]) for r in rows}
    for c in cases:
        if tuple(c) in done:
            continue
        import time
        t0 = time.time()
        q = quality(*c, exact=True)
        pb = Problem(mesh=build_exact(*c))
        pb.solve()
        row = {"case": list(c), **q, "seconds": time.time() - t0, "newton": pb.hist, **qois(pb)}
        rows.append(row)
        json.dump(rows, open(OUT, "w"), indent=1)
        print(f"{c} h/rho={q['h_over_rho']:.3f} {row['seconds']:.0f}s M_y={row['reaction_moment_y']:.5f} "
              f"max_xx={row['max_sigma_xx_R0']:.4f} max_zz={row['max_sigma_zz_R0']:.4f} "
              f"max_vm={row['max_von_mises_R0']:.4f} gk2_xx={row['gk2_sigma_xx_avg']:.5f}",
              flush=True)
    return rows


if __name__ == "__main__":
    if sys.argv[1:] == ["solve"]:
        solve_sequence()
    if sys.argv[1:] == ["scan"]:
        json.dump(scan(), open(os.path.join(HERE, "b3_local_refine_scan.json"), "w"), indent=1)
