"""Independent discretization of the L-shape, the edge crack and the plate with a hole
(round J, 2026-11-23). For the plate, the mid-edge nodes on the hole are projected onto
the exact circle, as in its Q2 mesh.

Their quantities certified with the post hoc singular-order extension had no
independent value (cross-check: single_family). This runs the SAME problem code
(lshape.solve_level / crack.solve_level: geometry, load, supports, quantities) with a
different element family: 6-node triangles P2, each Q2 element's quadrilateral split
into two triangles (MeshQuad.to_meshtri), so the vertices, the crack faces with their
duplicated nodes and the query points are unchanged. Same code (scikit-fem), other
element family: an independent discretization, not an independent code (as Cook Q2/Q4
and the block Q8/Q20). Stored apart: <module>_p2_ladder.json.

    python -m omar_pfem.verification_agent.independent_p2 [lshape|crack ...]
"""
import json
import os
import sys
import types

import numpy as np
import skfem as fem

HERE = os.path.dirname(os.path.abspath(__file__))
LEVELS = (2, 4, 8, 16, 32, 64, 128)


def _p2_fem():
    """The skfem namespace with ElementQuad2 replaced by ElementTriP2."""
    ns = types.SimpleNamespace(**{k: getattr(fem, k) for k in dir(fem) if not k.startswith("__")})
    ns.ElementQuad2 = fem.ElementTriP2
    return ns


def solve_p2(mod, n):
    q2_mesh, q2_fem = mod.mesh, mod.fem

    def tri_mesh(n_, g=1.0):
        m = q2_mesh(n_, g)
        tri = fem.MeshQuad(m.p[:, :m.t.max() + 1], m.t).to_meshtri()   # vertices only
        m2 = fem.MeshTri2.from_mesh(tri)
        if not hasattr(mod, "RH"):
            return m2
        # plate: mid-edge nodes on the hole projected onto the exact circle (the Q2 mesh
        # has them there too); interior mid-edge nodes stay straight
        dof = m2.doflocs.copy()
        nv, fa = tri.p.shape[1], tri.facets
        assert np.allclose(dof[:, nv:nv + fa.shape[1]], 0.5 * (tri.p[:, fa[0]] + tri.p[:, fa[1]]))
        c = np.array(mod.C)[:, None]
        ra = np.linalg.norm(tri.p[:, fa[0]] - c, axis=0)
        rb = np.linalg.norm(tri.p[:, fa[1]] - c, axis=0)
        on = np.isclose(ra, mod.RH) & np.isclose(rb, mod.RH)
        mid = dof[:, nv:nv + fa.shape[1]][:, on] - c
        dof[:, nv + np.where(on)[0]] = c + mod.RH * mid / np.linalg.norm(mid, axis=0)
        return fem.MeshTri2(dof, tri.t)
    mod.mesh, mod.fem = tri_mesh, _p2_fem()
    try:
        row = mod.solve_level(n, 1.0)
    finally:
        mod.mesh, mod.fem = q2_mesh, q2_fem
    row["element"] = "P2 triangles (independent discretization)"
    return row


PLATE_LEVELS = (2, 4, 8, 16, 32, 64)


def run(name, levels=None):
    mod = __import__(f"omar_pfem.verification_agent.{name}", fromlist=["solve_level"])
    key = "m" if name == "plate_hole" else "n"
    levels = levels or (PLATE_LEVELS if name == "plate_hole" else LEVELS)
    out = os.path.join(HERE, f"{name}_p2_ladder.json")
    rows = json.load(open(out)) if os.path.exists(out) else []
    done = {r[key] for r in rows}
    for n in levels:
        if n in done:
            continue
        r = solve_p2(mod, n)
        rows.append(r)
        rows.sort(key=lambda r: r[key])
        json.dump(rows, open(out, "w"), indent=1)
        print(f"{name} P2 {key}={n} dof={r['n_dof']} ({r['wall_clock_s']:.0f} s) "
              f"E={r['strain_energy']:.8g}", flush=True)
    return rows


if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")
    for name in (sys.argv[1:] or ["lshape", "crack", "plate_hole"]):
        run(name)
