"""U-shaped panel: a held-out problem for the intensity-aware remedy (round L, 2026-11-23).

Domain: the union of [0,1]x[0,2], [1,2]x[0,1], [2,3]x[0,2] (a U open at the top). Plane
strain, small-strain linear elasticity, E = 1000, nu = 0.3. Edge x = 0 clamped; edge x = 3
loaded by a uniform shear traction t_y = -1 (a cantilever in bending); all other edges free.
Corners: two 270-degree re-entrant free/free corners at (1, 1) and (2, 1) (same exponent,
lam = 0.5445, different intensities: the bending moment differs), two 90-degree
clamped/free corners at (0, 0) and (0, 2) (lam = 0.7112).

Mesh: Q2 (9-node) elements, n per unit length (h = 1/n), node lines at the integers. The
node coordinates come from MESH_PLAN: {"x": {line: g}, "y": {line: g}} - each half unit next
to a listed integer line is graded toward it as s ** g (remedy_intensity.coords). An empty
plan is the uniform mesh.

Quantities: strain_energy = 1/2 u.K u; v_tip = u_y at (3, 0); left_gk_von_mises and
right_gk_von_mises = von Mises averaged with exp(-(d/0.1)^2) about (1, 1) and (2, 1);
local_energy_<x>_<y> = strain energy within d < 0.25 of each rate-limiting corner (used by
the intensity estimate).

    python -m omar_pfem.verification_agent.ushape
"""
import time

import numpy as np
import skfem as fem
from skfem.helpers import ddot, sym_grad, trace

E, NU, T = 1000.0, 0.3, 1.0
MU = E / (2 * (1 + NU))
LAM = E * NU / ((1 + NU) * (1 - 2 * NU))
ELL, RHO = 0.1, 0.25
CORNERS = ((1, 1), (2, 1), (0, 0), (0, 2))          # the rate-limiting ones (corner analysis)
POLYGON = [(0, 0, "free"), (3, 0, "free"), (3, 2, "free"), (2, 2, "free"), (2, 1, "free"),
           (1, 1, "free"), (1, 2, "free"), (0, 2, "clamped")]
MESH_PLAN = {"x": {}, "y": {}}


def mesh(n, g=1.0):
    from omar_pfem.verification_agent.remedy_intensity import coords
    xs = coords(n, MESH_PLAN["x"], 0, 3)
    ys = coords(n, MESH_PLAN["y"], 0, 2)
    Nx, Ny = len(xs), len(ys)
    X, Y = np.meshgrid(xs, ys, indexing="ij")
    idx = lambda i, j: i * Ny + j
    t = [[idx(i, j), idx(i + 1, j), idx(i + 1, j + 1), idx(i, j + 1)]
         for i in range(Nx - 1) for j in range(Ny - 1)
         if not (n <= i < 2 * n and j >= n)]                 # the slot [1,2] x [1,2]
    t = np.array(t).T
    used = np.unique(t)
    remap = -np.ones(Nx * Ny, int)
    remap[used] = np.arange(len(used))
    p = np.array([X.ravel()[used], Y.ravel()[used]])
    return fem.MeshQuad2.from_mesh(fem.MeshQuad(p, remap[t]))


@fem.BilinearForm
def stiffness(u, v, w):
    eu, ev = sym_grad(u), sym_grad(v)
    return 2 * MU * ddot(eu, ev) + LAM * trace(eu) * trace(ev)


def solve_level(n, g=1.0):
    import pypardiso
    t0 = time.time()
    msh = mesh(n, g)
    b = fem.Basis(msh, fem.ElementVector(fem.ElementQuad2()), intorder=4)
    K = fem.asm(stiffness, b)
    right = fem.FacetBasis(msh, fem.ElementVector(fem.ElementQuad2()), intorder=4,
                           facets=msh.facets_satisfying(lambda x: np.abs(x[0] - 3.0) < 1e-12))
    f = fem.asm(fem.LinearForm(lambda v, w: -T * v.value[1]), right)
    assert abs(f.sum() + 2 * T) < 1e-9, f.sum()                   # total load -2
    fixed = b.get_dofs(lambda x: np.abs(x[0]) < 1e-12).all()
    u = fem.solve(*fem.condense(K, f, D=fixed),
                  solver=lambda A, rhs: pypardiso.spsolve(A.tocsr(), rhs))
    energy = float(0.5 * u @ (K @ u))
    gu = b.interpolate(u).grad
    eps = 0.5 * (gu + np.einsum("ij...->ji...", gu))
    tr = eps[0, 0] + eps[1, 1]
    sxx, syy = 2 * MU * eps[0, 0] + LAM * tr, 2 * MU * eps[1, 1] + LAM * tr
    szz, sxy = LAM * tr, 2 * MU * eps[0, 1]
    vm = np.sqrt(0.5 * ((sxx - syy) ** 2 + (syy - szz) ** 2 + (szz - sxx) ** 2) + 3 * sxy ** 2)
    dens = 0.5 * (sxx * eps[0, 0] + syy * eps[1, 1] + 2 * sxy * eps[0, 1])
    x, wq = b.global_coordinates().value, b.dx
    row = {"n": n, "g": g, "h": 1.0 / n, "n_elem": int(msh.t.shape[1]), "n_dof": int(u.size),
           "strain_energy": energy}
    for name, (cx, cy) in (("left", (1, 1)), ("right", (2, 1))):
        k = wq * np.exp(-(np.hypot(x[0] - cx, x[1] - cy) / ELL) ** 2)
        row[f"{name}_gk_von_mises"] = float((vm * k).sum() / k.sum())
    for cx, cy in CORNERS:
        inside = np.hypot(x[0] - cx, x[1] - cy) < RHO
        row[f"local_energy_{cx}_{cy}"] = float((dens * wq)[inside].sum())
    U = u[b.nodal_dofs].T
    j = int(np.argmin((msh.p[0] - 3.0) ** 2 + (msh.p[1] - 0.0) ** 2))
    assert np.hypot(msh.p[0, j] - 3.0, msh.p[1, j]) < 1e-12
    row["v_tip"] = float(U[j, 1])
    row["wall_clock_s"] = time.time() - t0
    return row
