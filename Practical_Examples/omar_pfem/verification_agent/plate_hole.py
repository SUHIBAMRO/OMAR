"""Plate with a hole (the OpenPINN/PINO demonstration case): a HELD-OUT
benchmark for the verification agent.

2 m x 2 m plate, circular hole of radius 0.25 m at (1, 1), plane stress,
linear elasticity, E = 210 GPa, nu = 0.3. Left edge x = 0 fixed, right edge
x = 2 loaded by a uniform traction t_x = 1 MPa, top, bottom and hole free
(as in Wang et al., OpenPINN/PINO, Sec. 5.3, whose reference is one FEM mesh
with h = 0.025 and no convergence estimate).

Mesh: structured "polar" Q2 mesh around the hole, x = c + rho(s, theta)
(cos theta, sin theta) with rho = (1 - s^g) r_hole + s^g R(theta) and
R(theta) = 1 / max(|cos|, |sin|) the distance from the centre to the square
along the ray. N_theta = 8 m elements around (the square's corners are nodes),
N_r = 3 m radially; g = 1 uniform, g > 1 grades toward the hole. Q2
(9-node) elements; mid-side nodes on the exact circle.

Corners: the fixed edge meets the free top and bottom edges at (0,0) and
(0,2): 90-degree clamped/free corners (singular, plane stress). The hole is a
smooth, finite-radius concentration far from them.

    python -m omar_pfem.verification_agent.plate_hole [--ms 2,4,8,16,32,64]
"""
import argparse
import json
import os
import time

import numpy as np
import scipy.sparse.linalg as spla
import skfem as fem
from skfem.helpers import ddot, sym_grad, trace

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "plate_hole_ladder.json")
E, NU, T = 210e9, 0.3, 1e6
C, RH, L = (1.0, 1.0), 0.25, 2.0
MU = E / (2 * (1 + NU))
LAM_PS = E * NU / (1 - NU ** 2)               # plane-stress lambda
HOLE_TOP = (1.0, 1.25)                        # sigma_xx peak of the Kirsch-type field
R_REGION = 0.1                                # sharp region: within 0.1 of the hole top
KERNEL_ELL = 0.05


def corner_exponent():
    from omar_pfem.verification_agent.singularity import smallest_exponent
    return smallest_exponent(np.pi / 2, "clamped", "free", NU, "stress")


def mesh(m, g=1.0):
    """Periodic structured Q2 mesh: vertices (s_i, theta_j), j modulo N_theta;
    every mid-side and centre node placed by the exact map from its
    parametric coordinates (theta unwrapped across the seam)."""
    nt, nr = 8 * m, 3 * m
    s1 = np.linspace(0.0, 1.0, nr + 1)
    t1 = -np.pi / 4 + 2 * np.pi * np.arange(nt) / nt
    S, TH = np.meshgrid(s1, t1, indexing="ij")              # (nr+1, nt)
    S, TH = S.ravel(), TH.ravel()
    idx = lambda i, j: i * nt + (j % nt)
    t = np.array([[idx(i, j), idx(i + 1, j), idx(i + 1, j + 1), idx(i, j + 1)]
                  for i in range(nr) for j in range(nt)]).T

    def X(s, th):
        R = 1.0 / np.maximum(np.abs(np.cos(th)), np.abs(np.sin(th)))
        sg = s ** g
        rho = (1 - sg) * RH + sg * R
        return np.array([C[0] + rho * np.cos(th), C[1] + rho * np.sin(th)])

    m1 = fem.MeshQuad(X(S, TH), t)
    m2 = fem.MeshQuad2.from_mesh(m1)
    nv, nf = m1.p.shape[1], m1.facets.shape[1]

    def unwrap_mean(ths):
        ref = ths[0]
        d = (ths - ref + np.pi) % (2 * np.pi) - np.pi
        return ref + d.mean(axis=0)
    fa, fb = m1.facets
    s_f = 0.5 * (S[fa] + S[fb])
    th_f = unwrap_mean(np.stack([TH[fa], TH[fb]]))
    s_c = S[m1.t].mean(axis=0)
    th_c = unwrap_mean(TH[m1.t])
    dof = m2.doflocs.copy()
    # check the assumed ordering (vertices, facets, elements) on the linear midpoints
    assert np.allclose(dof[:, nv:nv + nf], 0.5 * (m1.p[:, fa] + m1.p[:, fb]))
    dof[:, nv:nv + nf] = X(s_f, th_f)
    dof[:, nv + nf:] = X(s_c, th_c)
    return fem.MeshQuad2(dof, m1.t)


def mesh_half(m, g=1.0):
    """The upper half (y >= 1) of mesh(m, g), element for element: the
    columns theta_j in [0, pi] (j = m .. 5m of the full mesh; theta = 0 and
    pi are vertex lines). Same exact placement of mid-side and centre nodes."""
    nt, nr = 8 * m, 3 * m
    s1 = np.linspace(0.0, 1.0, nr + 1)
    t1 = -np.pi / 4 + 2 * np.pi * np.arange(m, 5 * m + 1) / nt          # [0, pi]
    nc = len(t1)
    S, TH = np.meshgrid(s1, t1, indexing="ij")
    S, TH = S.ravel(), TH.ravel()
    idx = lambda i, j: i * nc + j
    t = np.array([[idx(i, j), idx(i + 1, j), idx(i + 1, j + 1), idx(i, j + 1)]
                  for i in range(nr) for j in range(nc - 1)]).T

    def X(s, th):
        R = 1.0 / np.maximum(np.abs(np.cos(th)), np.abs(np.sin(th)))
        rho = (1 - s ** g) * RH + s ** g * R
        return np.array([C[0] + rho * np.cos(th), C[1] + rho * np.sin(th)])
    m1 = fem.MeshQuad(X(S, TH), t)
    m2 = fem.MeshQuad2.from_mesh(m1)
    nv, nf = m1.p.shape[1], m1.facets.shape[1]
    fa, fb = m1.facets
    dof = m2.doflocs.copy()
    assert np.allclose(dof[:, nv:nv + nf], 0.5 * (m1.p[:, fa] + m1.p[:, fb]))
    dof[:, nv:nv + nf] = X(0.5 * (S[fa] + S[fb]), 0.5 * (TH[fa] + TH[fb]))
    dof[:, nv + nf:] = X(S[m1.t].mean(axis=0), TH[m1.t].mean(axis=0))
    return fem.MeshQuad2(dof, m1.t)


@fem.BilinearForm
def stiffness(u, v, w):
    eu, ev = sym_grad(u), sym_grad(v)
    return 2 * MU * ddot(eu, ev) + LAM_PS * trace(eu) * trace(ev)


def _vbasis(msh, elements=None):
    return fem.Basis(msh, fem.ElementVector(fem.ElementQuad2()), intorder=4, elements=elements)


def solve_level_lowmem(m, g=1.0, chunk=15000, half=None):
    """Same quantities as solve_level, with assembly and post-processing in
    element chunks (never all basis values at once): for m >= 128, where the
    one-shot version needs more than the 15 GB of this machine. Pardiso
    direct solve. Checked against solve_level (PLATE_LOWMEM=1 at m = 16)."""
    import pypardiso
    t0 = time.time()
    # half=True: upper half y >= 1 with v = 0 on the symmetry line (problem,
    # mesh and load are symmetric about y = 1, so the discrete solution is the
    # same); energy doubled; the hole QoIs lie in the upper half (the Gaussian
    # weight of lower-half points is below exp(-25))
    half = (os.environ.get("PLATE_HALF") == "1") if half is None else half
    msh = mesh_half(m, g) if half else mesh(m, g)
    nel = msh.t.shape[1]
    chunks = np.array_split(np.arange(nel), int(np.ceil(nel / chunk)))
    b0 = _vbasis(msh, chunks[0])
    K = None
    for ch in chunks:
        Kc = fem.asm(stiffness, _vbasis(msh, ch))
        K = Kc if K is None else K + Kc
    right = fem.FacetBasis(msh, fem.ElementVector(fem.ElementQuad2()),
                           facets=msh.facets_satisfying(lambda x: np.isclose(x[0], L)), intorder=4)
    f = fem.asm(fem.LinearForm(lambda v, w: T * v.value[0]), right)
    fixed = b0.get_dofs(lambda x: np.isclose(x[0], 0.0)).all()
    if half:
        sym = b0.get_dofs(lambda x: np.isclose(x[1], C[1]) & (np.abs(x[0] - C[0]) > RH / 2))
        fixed = np.union1d(fixed, sym.all("u^2"))
    u = fem.solve(*fem.condense(K, f, D=fixed), solver=lambda A, b: pypardiso.spsolve(
        A.tocsr(), b))
    energy = float(0.5 * u @ (K @ u)) * (2.0 if half else 1.0)
    del K
    acc = dict(h_w=0.0, h_s=0.0, h_n=0, k_w=0.0, k_s=0.0, h_pk=-np.inf, vm=-np.inf, vm_at=None,
               c_vm=-np.inf, c_at=None, dmin=np.inf, dmax=-np.inf)
    for ch in chunks:
        bc = _vbasis(msh, ch)
        gu = bc.interpolate(u).grad
        eps = 0.5 * (gu + np.einsum("ij...->ji...", gu))
        tr = eps[0, 0] + eps[1, 1]
        sxx = 2 * MU * eps[0, 0] + LAM_PS * tr
        syy = 2 * MU * eps[1, 1] + LAM_PS * tr
        sxy = 2 * MU * eps[0, 1]
        vm = np.sqrt(sxx ** 2 - sxx * syy + syy ** 2 + 3 * sxy ** 2)
        x, wq = bc.global_coordinates().value, bc.dx
        d = np.hypot(x[0] - HOLE_TOP[0], x[1] - HOLE_TOP[1])
        near = d < R_REGION
        k = wq * np.exp(-(d / KERNEL_ELL) ** 2)
        acc["h_w"] += (sxx * wq)[near].sum()
        acc["h_s"] += wq[near].sum()
        acc["h_n"] += int(near.sum())
        acc["k_w"] += (sxx * k).sum()
        acc["k_s"] += k.sum()
        if near.any():
            acc["h_pk"] = max(acc["h_pk"], float(sxx[near].max()))
        i = np.unravel_index(np.argmax(vm), vm.shape)
        if vm[i] > acc["vm"]:
            acc["vm"], acc["vm_at"] = float(vm[i]), [float(x[0][i]), float(x[1][i])]
        dc = np.minimum(np.hypot(x[0], x[1]), np.hypot(x[0], x[1] - L))
        cn = dc < 0.1
        if cn.any():
            i = np.unravel_index(np.argmax(np.where(cn, vm, -np.inf)), vm.shape)
            if vm[i] > acc["c_vm"]:
                acc["c_vm"], acc["c_at"] = float(vm[i]), [float(x[0][i]), float(x[1][i])]
        detF = (1 + gu[0, 0]) * (1 + gu[1, 1]) - gu[0, 1] * gu[1, 0]
        acc["dmin"], acc["dmax"] = min(acc["dmin"], detF.min()), max(acc["dmax"], detF.max())
    ux, uy = u[b0.get_dofs().all("u^1")], u[b0.get_dofs().all("u^2")]
    U = u[b0.nodal_dofs].T

    def node(px, py):
        j = int(np.argmin((msh.p[0] - px) ** 2 + (msh.p[1] - py) ** 2))
        assert np.hypot(msh.p[0, j] - px, msh.p[1, j] - py) < 1e-9, (px, py)
        return U[j]
    return {"posthoc_corner_peak_von_mises": acc["c_vm"],
            "posthoc_corner_peak_location": acc["c_at"],
            "posthoc_min_detF": float(acc["dmin"]), "posthoc_max_detF": float(acc["dmax"]),
            "m": m, "g": g, "h": 1.0 / m, "n_elem": int(nel) * (2 if half else 1),
            "n_dof": int(b0.N) if not half else int(2 * b0.N - (12 * m + 2) * 2),
            "wall_clock_s": time.time() - t0, "linear_solver": "pardiso (chunked, low memory)"
            + (", symmetric half model y >= 1" if half else ""),
            "u_right_mid": float(node(2.0, 1.0)[0]), "v_top_right": float(node(2.0, 2.0)[1]),
            "max_disp_mag": float(np.sqrt(ux ** 2 + uy ** 2).max()),
            "strain_energy": energy,
            "hole_avg_sigma_xx": float(acc["h_w"] / acc["h_s"]),
            "hole_gk_avg_sigma_xx": float(acc["k_w"] / acc["k_s"]),
            "hole_peak_sigma_xx": acc["h_pk"], "peak_von_mises": acc["vm"],
            "peak_location": acc["vm_at"], "hole_region_points": acc["h_n"]}


def solve_level(m, g=1.0):
    if m >= 128 or os.environ.get("PLATE_LOWMEM") == "1":
        return solve_level_lowmem(m, g)
    t0 = time.time()
    msh = mesh(m, g)
    ib = fem.Basis(msh, fem.ElementVector(fem.ElementQuad2()), intorder=4)
    nel = msh.t.shape[1]
    if nel <= 60000:
        K = fem.asm(stiffness, ib)
    else:
        # chunked assembly (same matrix up to summation order): the one-shot
        # assembly holds all local matrices at all Gauss points (~5 GB at m = 128)
        K = None
        for ch in np.array_split(np.arange(nel), int(np.ceil(nel / 20000))):
            Kc = fem.asm(stiffness, fem.Basis(msh, fem.ElementVector(fem.ElementQuad2()),
                                              intorder=4, elements=ch))
            K = Kc if K is None else K + Kc
    right = fem.FacetBasis(msh, fem.ElementVector(fem.ElementQuad2()),
                           facets=msh.facets_satisfying(lambda x: np.isclose(x[0], L)), intorder=4)
    f = fem.asm(fem.LinearForm(lambda v, w: T * v.value[0]), right)
    fixed = ib.get_dofs(lambda x: np.isclose(x[0], 0.0)).all()
    # direct solvers only: SuperLU (all pre-registered levels); Pardiso where
    # SuperLU runs out of memory (m = 128), PLATE_SOLVER=pardiso
    solver = os.environ.get("PLATE_SOLVER", "superlu")
    if solver == "pardiso":
        import pypardiso
        u = fem.solve(*fem.condense(K, f, D=fixed), solver=lambda A, b: pypardiso.spsolve(
            A.tocsr(), b))
    else:
        u = fem.solve(*fem.condense(K, f, D=fixed), solver=spla.spsolve)
    # stresses at the quadrature points
    gu = ib.interpolate(u).grad                               # (2, 2, nel, nq)
    eps = 0.5 * (gu + np.einsum("ij...->ji...", gu))
    tr = eps[0, 0] + eps[1, 1]
    sxx = 2 * MU * eps[0, 0] + LAM_PS * tr
    syy = 2 * MU * eps[1, 1] + LAM_PS * tr
    sxy = 2 * MU * eps[0, 1]
    vm = np.sqrt(sxx ** 2 - sxx * syy + syy ** 2 + 3 * sxy ** 2)
    x, wq = ib.global_coordinates().value, ib.dx
    d = np.hypot(x[0] - HOLE_TOP[0], x[1] - HOLE_TOP[1])
    near = d < R_REGION
    k = wq * np.exp(-(d / KERNEL_ELL) ** 2)
    ux, uy = u[ib.get_dofs().all("u^1")], u[ib.get_dofs().all("u^2")]
    U = u[ib.nodal_dofs].T
    def node(px, py):
        j = int(np.argmin((msh.p[0] - px) ** 2 + (msh.p[1] - py) ** 2))
        assert np.hypot(msh.p[0, j] - px, msh.p[1, j] - py) < 1e-9, (px, py)
        return U[j]
    u_rm = node(2.0, 1.0)
    u_tr = node(2.0, 2.0)
    ipk = np.unravel_index(np.argmax(vm), vm.shape)
    # POST HOC (added after the pre-registered ladder was solved): von Mises
    # near the clamped/free corners, and min det F
    dc = np.minimum(np.hypot(x[0], x[1]), np.hypot(x[0], x[1] - L))
    cn = dc < 0.1
    icn = np.unravel_index(np.argmax(np.where(cn, vm, -np.inf)), vm.shape)
    detF = (1 + gu[0, 0]) * (1 + gu[1, 1]) - gu[0, 1] * gu[1, 0]
    posthoc = {"posthoc_corner_peak_von_mises": float(vm[cn].max()),
               "posthoc_corner_peak_location": [float(x[0][icn]), float(x[1][icn])],
               "posthoc_min_detF": float(detF.min()), "posthoc_max_detF": float(detF.max())}
    return {**posthoc, "m": m, "g": g, "h": 1.0 / m, "n_elem": int(msh.t.shape[1]), "n_dof": int(ib.N),
            "wall_clock_s": time.time() - t0, "linear_solver": solver,
            "u_right_mid": float(u_rm[0]), "v_top_right": float(u_tr[1]),
            "max_disp_mag": float(np.sqrt(ux ** 2 + uy ** 2).max()),
            "strain_energy": float(0.5 * u @ (K @ u)),
            "hole_avg_sigma_xx": float((sxx * wq)[near].sum() / wq[near].sum()),
            "hole_gk_avg_sigma_xx": float((sxx * k).sum() / k.sum()),
            "hole_peak_sigma_xx": float(sxx[near].max()),
            "peak_von_mises": float(vm.max()),
            "peak_location": [float(x[0][ipk]), float(x[1][ipk])],
            "hole_region_points": int(near.sum())}


def run_ladder(ms, g=1.0):
    rows = json.load(open(CACHE)) if os.path.exists(CACHE) else []
    done = {(r["m"], r["g"]) for r in rows}
    for m in ms:
        if (m, g) in done:
            continue
        row = solve_level(m, g)
        rows.append(row)
        rows.sort(key=lambda r: (r["g"], r["m"]))
        json.dump(rows, open(CACHE, "w"), indent=1)
        print(f"m={m:<3} g={g} ({row['wall_clock_s']:.0f}s) " + "  ".join(
            f"{k}={v:.6g}" for k, v in row.items() if isinstance(v, float)
            and k not in ("h", "g", "wall_clock_s")), flush=True)
    return sorted((r for r in rows if r["m"] in ms and r["g"] == g), key=lambda r: r["m"])


def hole_element_size(m, g=1.0):
    """Largest corner-edge length of the elements touching the hole-top node."""
    msh = mesh(m, g)
    j = int(np.argmin((msh.p[0] - HOLE_TOP[0]) ** 2 + (msh.p[1] - HOLE_TOP[1]) ** 2))
    adj = msh.t[:, (msh.t == j).any(axis=0)]
    P = msh.p[:, adj]                                         # (2, 4, n_adj)
    return float(max(np.linalg.norm(P[:, a] - P[:, (a + 1) % 4], axis=0).max()
                     for a in range(4)))


PREREG_KEYS = ("u_right_mid", "v_top_right", "max_disp_mag", "strain_energy",
               "hole_avg_sigma_xx", "hole_gk_avg_sigma_xx", "hole_peak_sigma_xx",
               "peak_von_mises")


def posthoc_update():
    """Re-solve every stored level and ADD the post-hoc fields; the
    pre-registered QoIs must reproduce to 1e-10 (relative)."""
    rows = json.load(open(CACHE))
    for r in rows:
        if "posthoc_corner_peak_von_mises" in r:
            continue
        new = solve_level(r["m"], r["g"])
        for k in PREREG_KEYS:
            assert abs(new[k] - r[k]) <= 1e-10 * abs(r[k]), (r["m"], r["g"], k, new[k], r[k])
        r.update({k: v for k, v in new.items() if k.startswith("posthoc_")})
        json.dump(rows, open(CACHE, "w"), indent=1)
        print(f"m={r['m']} g={r['g']} corner vM {r['posthoc_corner_peak_von_mises']:.6g} at "
              f"{r['posthoc_corner_peak_location']}  det F in [{r['posthoc_min_detF']:.8f}, "
              f"{r['posthoc_max_detF']:.8f}]", flush=True)


REF_ERR_OUT = os.path.join(HERE, "plate_hole_reference_error.json")


def solve_q1(m):
    """Bilinear Q4 on the vertices of mesh(m) (straight edges: polygonal hole),
    a stand-in for a typical single linear FE reference."""
    q2 = mesh(m)
    msh = fem.MeshQuad(q2.p, q2.t)
    ib = fem.Basis(msh, fem.ElementVector(fem.ElementQuad1()), intorder=3)
    K = fem.asm(stiffness, ib)
    right = fem.FacetBasis(msh, fem.ElementVector(fem.ElementQuad1()),
                           facets=msh.facets_satisfying(lambda x: np.isclose(x[0], L)), intorder=3)
    f = fem.asm(fem.LinearForm(lambda v, w: T * v.value[0]), right)
    u = fem.solve(*fem.condense(K, f, D=ib.get_dofs(lambda x: np.isclose(x[0], 0.0)).all()),
                  solver=spla.spsolve)
    U = u[ib.nodal_dofs].T
    j = int(np.argmin((msh.p[0] - 2) ** 2 + (msh.p[1] - 1) ** 2))
    k = int(np.argmin((msh.p[0] - 2) ** 2 + (msh.p[1] - 2) ** 2))
    return {"m": m, "h_outer": 1.0 / m, "n_dof": int(ib.N), "strain_energy": float(0.5 * u @ (K @ u)),
            "u_right_mid": float(U[j, 0]), "v_top_right": float(U[k, 1])}


def reference_error_study(q1_ms=(8, 16, 32, 40, 64)):
    """POST HOC. Error of single-mesh references in the energy norm (the
    OpenPINN paper's reference is one FE mesh with h = 0.025 = 1/40).
    Linear elasticity, traction load, Galerkin: ||u - u_h||_E^2 = 2 (W - W_h),
    so the relative energy-norm error is sqrt((W - W_h) / W), W = strain energy.
    W is estimated by Richardson extrapolation of the Q2 family (order fitted
    from the finest three levels); the spread of W between that and the
    finest level is reported as its uncertainty."""
    rows = sorted((r for r in json.load(open(CACHE)) if r["g"] == 1.0), key=lambda r: r["m"])
    W = np.array([r["strain_energy"] for r in rows])
    p = np.log((W[-3] - W[-2]) / (W[-2] - W[-1])) / np.log(2)
    W_inf = W[-1] + (W[-1] - W[-2]) / (2 ** p - 1)
    ui = np.array([r["u_right_mid"] for r in rows])
    u_inf = ui[-1] + (ui[-1] - ui[-2]) / (2 ** p - 1)
    rel_E = lambda w: float(np.sqrt(max(W_inf - w, 0.0) / W_inf))
    q2 = [{"element": "Q9", "m": r["m"], "h_outer": 1.0 / r["m"], "n_dof": r["n_dof"],
           "energy_norm_rel_error": rel_E(r["strain_energy"]),
           "u_right_mid_rel_error": float(abs(r["u_right_mid"] - u_inf) / abs(u_inf))}
          for r in rows]
    q1 = []
    for m in q1_ms:
        r = solve_q1(m)
        q1.append({"element": "Q4 (polygonal hole)", **r,
                   "energy_norm_rel_error": rel_E(r["strain_energy"]),
                   "u_right_mid_rel_error": float(abs(r["u_right_mid"] - u_inf) / abs(u_inf))})
        print(q1[-1], flush=True)
    out = {"status": "post hoc", "W_extrapolated": float(W_inf), "W_order": float(p),
           "W_finest": float(W[-1]), "u_right_mid_extrapolated": float(u_inf),
           "note": "relative L2 displacement error (the paper's metric) is smaller than the "
                   "relative energy-norm error, typically by a factor of order h",
           "q2": q2, "q1": q1}
    json.dump(out, open(REF_ERR_OUT, "w"), indent=1)
    return out


MMS_OUT = os.path.join(HERE, "plate_hole_mms.json")


def mms_check(ns=(4, 8, 16, 32)):
    """Code check of the Q2 plane-stress element and assembly used here:
    manufactured u = (sin(pi x) sin(pi y), x y (1-x)(1-y)) on the unit square
    (u = 0 on the boundary), body force from the same bilinear form's strong
    operator; expected rates L2 3, H1-seminorm 2."""
    import sympy as sp
    X, Y = sp.symbols("x y")
    ue = [sp.sin(sp.pi * X) * sp.sin(sp.pi * Y), X * Y * (1 - X) * (1 - Y)]
    eps = [[sp.Rational(1, 2) * (sp.diff(ue[i], [X, Y][j]) + sp.diff(ue[j], [X, Y][i]))
            for j in range(2)] for i in range(2)]
    tr = eps[0][0] + eps[1][1]
    sig = [[2 * MU * eps[i][j] + (LAM_PS * tr if i == j else 0) for j in range(2)]
           for i in range(2)]
    fb = [sp.lambdify((X, Y), -(sp.diff(sig[i][0], X) + sp.diff(sig[i][1], Y))) for i in range(2)]
    uf = [sp.lambdify((X, Y), u) for u in ue]
    gf = [[sp.lambdify((X, Y), sp.diff(ue[i], v)) for v in (X, Y)] for i in range(2)]
    rows = []
    for n in ns:
        msh = fem.MeshQuad2.from_mesh(fem.MeshQuad.init_tensor(np.linspace(0, 1, n + 1),
                                                            np.linspace(0, 1, n + 1)))
        ib = fem.Basis(msh, fem.ElementVector(fem.ElementQuad2()), intorder=6)
        K = fem.asm(stiffness, ib)
        f = fem.asm(fem.LinearForm(lambda v, w: fb[0](*w.x) * v.value[0]
                                   + fb[1](*w.x) * v.value[1]), ib)
        u = fem.solve(*fem.condense(K, f, D=ib.get_dofs()), solver=spla.spsolve)
        ui = ib.interpolate(u)
        xq, dx = ib.global_coordinates().value, ib.dx
        e2 = sum(((ui.value[i] - uf[i](*xq)) ** 2 * dx).sum() for i in range(2))
        n2 = sum((uf[i](*xq) ** 2 * dx).sum() for i in range(2))
        g2 = sum(((ui.grad[i, j] - gf[i][j](*xq)) ** 2 * dx).sum()
                 for i in range(2) for j in range(2))
        gn = sum((gf[i][j](*xq) ** 2 * dx).sum() for i in range(2) for j in range(2))
        rows.append({"n": n, "L2_rel": float(np.sqrt(e2 / n2)), "H1_semi_rel": float(
            np.sqrt(g2 / gn))})
    rate = lambda k: float(np.log(rows[-2][k] / rows[-1][k]) / np.log(2))
    out = {"element": "Q9 (skfem ElementQuad2), plane stress", "rows": rows,
           "rates": {"L2": rate("L2_rel"), "H1_semi": rate("H1_semi_rel")},
           "expected": {"L2": 3, "H1_semi": 2}}
    out["as_expected"] = bool(abs(out["rates"]["L2"] - 3) < 0.2
                              and abs(out["rates"]["H1_semi"] - 2) < 0.2)
    json.dump(out, open(MMS_OUT, "w"), indent=1)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--ms", default="2,4,8,16,32,64")
    ap.add_argument("--strategies", default="1.5,2.0", help="radial grading g at m = 16")
    a = ap.parse_args(argv)
    run_ladder([int(v) for v in a.ms.split(",")])
    for g in [float(v) for v in a.strategies.split(",") if v]:
        run_ladder([16], g)


if __name__ == "__main__":
    import sys
    if sys.argv[1:] == ["posthoc"]:
        posthoc_update()
    elif sys.argv[1:] == ["referr"]:
        o = reference_error_study()
        for r in o["q2"] + o["q1"]:
            print(f"{r['element']:<20} m={r['m']:<3} dof={r['n_dof']:<8} energy-norm "
                  f"{100 * r['energy_norm_rel_error']:.4f}%  u_right_mid "
                  f"{100 * r['u_right_mid_rel_error']:.5f}%")
    elif sys.argv[1:] == ["mms"]:
        print(json.dumps(mms_check(), indent=1))
    else:
        main()
