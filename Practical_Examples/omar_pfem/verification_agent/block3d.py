"""Second unseen 3D benchmark (2026-11-23): a clamped block with singular edges.

Why: the first 3D benchmark (cylinder3d.py) is smooth. This one has what B1 has in
2D, in 3D and with quadratic elements: a clamped face meeting free faces along four
straight edges (90-degree clamped/free wedges, plane-strain exponent lambda = 0.7112
for nu = 0.3) and four vertices where those edges meet (3D vertex singularities whose
exponents are NOT computed here). It is the first 3D test of the post hoc
singular-order extension for degree-2 elements. No exact solution exists.

Geometry: block [0, 2] x [0, 1] x [0, 0.5]. x = 0 clamped; x = 2 loaded by a uniform
dead traction t = (0, 0, -1) (total force 0.5), applied as consistent nodal forces of
the 8-node faces (corner -A/12, midside A/3); all other faces free. Compressible
neo-Hookean (the project energy), E = 1000, nu = 0.3. 20-node hexahedra, torch-fem
Hexa2 with its default reduced rule (as B3 and cylinder3d). Levels (n_x, n_y, n_z) =
k (4, 2, 1), k = 1, 2, 3, 4, 6, 8 (cubic elements, h = 0.5/k). Strategies at k = 4:
x-grading toward the clamp x = 2 (s ** g), g = 1.5, 2.0 (same element count).

Quantities:
  strain_energy           sum over Gauss points of psi(F) |det J| w
  w_tip                   u_z at (2, 0.5, 0.5)
  mid_gk_avg_sxx          sigma_xx averaged with exp(-(d/0.15)^2) about (1, 0.5, 0.5)
  root_gk_avg_von_mises   von Mises averaged with exp(-(d/0.1)^2) about (0, 0.5, 0.5),
                          a point ON the clamped top edge (integrable singularity)
  peak_von_mises          max von Mises over all Gauss points (+ its location)

    python -m omar_pfem.verification_agent.block3d solve
"""
import json
import os
import sys
import time

import numpy as np
import torch

torch.set_default_dtype(torch.float64)
HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "block3d_ladder.json")

LX, LY, LZ = 2.0, 1.0, 0.5
E, NU = 1000.0, 0.3
MU = E / (2 * (1 + NU))
LAM = E * NU / ((1 + NU) * (1 - 2 * NU))
TAU = 1.0
LEVELS = (1, 2, 3, 4, 6, 8)
STRATEGY_LEVEL, GRADINGS = 4, (1.5, 2.0)
MID, ROOT = np.array([1.0, 0.5, 0.5]), np.array([0.0, 0.5, 0.5])
TIP = np.array([2.0, 0.5, 0.5])


def corner_exponent():
    from omar_pfem.verification_agent.singularity import smallest_exponent
    return smallest_exponent(np.pi / 2, "clamped", "free", NU, "strain")


def mesh(k, g=1.0):
    from torchfem.elements import linear_to_quadratic
    nx, ny, nz = 4 * k, 2 * k, k
    sx = np.linspace(0, 1, nx + 1)
    xx = LX * sx ** g                     # g > 1 clusters toward x = 0 (the clamp)
    yy, zz = np.linspace(0, LY, ny + 1), np.linspace(0, LZ, nz + 1)
    X, Y, Z = np.meshgrid(xx, yy, zz, indexing="ij")
    nodes = np.stack([X.ravel(), Y.ravel(), Z.ravel()], 1)
    idx = np.arange(len(nodes)).reshape(X.shape)
    el = [[idx[i, j, m], idx[i + 1, j, m], idx[i + 1, j + 1, m], idx[i, j + 1, m],
           idx[i, j, m + 1], idx[i + 1, j, m + 1], idx[i + 1, j + 1, m + 1], idx[i, j + 1, m + 1]]
          for i in range(nx) for j in range(ny) for m in range(nz)]
    n20, e20 = linear_to_quadratic(torch.tensor(nodes), torch.tensor(np.array(el)))
    return n20.numpy(), e20.numpy()


def end_face_forces(X, el):
    """Consistent nodal forces of the dead traction (0, 0, -TAU) on x = LX: for each
    8-node face, corner nodes -A/12, midside nodes A/3 (serendipity, uniform load)."""
    from torchfem.elements import Hexa2
    f = np.zeros((len(X), 3))
    face = Hexa2.facets.numpy()[3]        # local nodes of the face xi = +1: [1,2,6,5,9,18,13,17]
    for e in el:
        nn = e[face]
        if not np.allclose(X[nn, 0], LX):
            continue
        p = X[nn[:4]]
        A = np.linalg.norm(p[1] - p[0]) * np.linalg.norm(p[3] - p[0])
        f[nn[:4], 2] += -TAU * (-A / 12.0)
        f[nn[4:], 2] += -TAU * (A / 3.0)
    return f


def solve(k, g=1.0, nsteps=10, quadrature="reduced", tol=1e-10):
    if quadrature == "full":
        from omar_pfem.verification_agent.q20_full import SolidFull as Solid
    else:
        from torchfem import Solid
    from torchfem.materials import Hyperelastic3D
    from omar_pfem.torchfem_comparison import cauchy_from_torchfem_flux, neo_hookean_psi_3d
    X, el = mesh(k, g)
    model = Solid(torch.tensor(X), torch.tensor(el),
                  Hyperelastic3D(psi=neo_hookean_psi_3d, params=torch.tensor([MU, LAM])))
    n = len(X)
    con = torch.zeros(n, 3, dtype=torch.bool)
    con[np.abs(X[:, 0]) < 1e-12, :] = True
    model.constraints, model.displacements = con, torch.zeros(n, 3)
    fext = end_face_forces(X, el)
    assert abs(fext[:, 2].sum() + TAU * LY * LZ) < 1e-12
    model.forces = torch.tensor(fext)
    t0 = time.time()
    with torch.no_grad():
        u, f, S, F, _ = model.solve(increments=torch.linspace(0, 1, nsteps + 1), max_iter=30,
                                    rtol=tol, atol=tol, stol=tol, method="direct",
                                    aggregate_integration_points=False)
    sec = time.time() - t0
    S = S.transpose(0, 1).numpy()
    F = F.transpose(0, 1).numpy()
    sig = cauchy_from_torchfem_flux(S, F)
    Ng, _, detJ = model.eval_shape_functions(model.etype.ipoints)
    wq = model.etype.iweights.numpy()[None, :] * np.abs(detJ.transpose(0, 1).numpy())
    pos = torch.einsum("gn,enD->egD", Ng, torch.tensor(X)[torch.tensor(el)]).numpy()
    J = np.linalg.det(F)
    I1 = (F ** 2).sum((-2, -1))
    with np.errstate(invalid="ignore"):
        psi = MU / 2 * (I1 - 3 - 2 * np.log(J)) + LAM / 2 * np.log(J) ** 2
    dev = sig - np.trace(sig, axis1=-2, axis2=-1)[..., None, None] * np.eye(3) / 3
    vm = np.sqrt(1.5 * (dev ** 2).sum((-2, -1)))
    km = wq * np.exp(-(np.linalg.norm(pos - MID, axis=-1) / 0.15) ** 2)
    kr = wq * np.exp(-(np.linalg.norm(pos - ROOT, axis=-1) / 0.1) ** 2)
    U = u.numpy()
    j = int(np.argmin(np.linalg.norm(X - TIP, axis=1)))
    assert np.linalg.norm(X[j] - TIP) < 1e-12
    ip = np.unravel_index(np.argmax(vm), vm.shape)
    return {"k": k, "grading": g, "quadrature": quadrature, "h": 0.5 / k,
            "n_elem": int(len(el)), "n_dof": int(3 * n), "seconds": sec,
            "min_detF": float(np.nanmin(J)),
            "min_detF_location": pos.reshape(-1, 3)[int(np.nanargmin(J.ravel()))].tolist(),
            "strain_energy": float((psi * wq).sum()), "w_tip": float(U[j, 2]),
            "mid_gk_avg_sxx": float((sig[..., 0, 0] * km).sum() / km.sum()),
            "root_gk_avg_von_mises": float((vm * kr).sum() / kr.sum()),
            "peak_von_mises": float(vm.max()), "peak_location": pos[ip].tolist()}


def run(levels, g=1.0):
    rows = json.load(open(CACHE)) if os.path.exists(CACHE) else []
    done = {(r["k"], r["grading"]) for r in rows}
    for k in levels:
        if (k, g) in done:
            continue
        row = solve(k, g)
        rows.append(row)
        rows.sort(key=lambda r: (r["grading"], r["k"]))
        json.dump(rows, open(CACHE, "w"), indent=1)
        print(f"k={k} g={g} n_elem={row['n_elem']} dof={row['n_dof']} ({row['seconds']:.0f} s)",
              flush=True)
    return rows


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if argv and argv[0] == "solve":
        run(LEVELS)
        for g in GRADINGS:
            run([STRATEGY_LEVEL], g)


if __name__ == "__main__":
    main()
