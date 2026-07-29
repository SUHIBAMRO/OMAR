"""
Spatial (h-refinement) convergence study for the reference FE solver, per
the advisor's request: "the current discretization appears rather coarse
for nonlinear hyperelasticity ... investigate the influence of mesh
refinement and discuss the spatial convergence of the reference solution."

Why this uses FIXED analytic fields instead of the random (E, nu, load)
fields used everywhere else in this study: the GRF sampler in
omar_pfem/data/grf.py is a spectral method whose random phase array is
sized by (Nx, Ny) -- calling it at two different mesh resolutions with the
same seed does NOT resample the same continuous field at higher
resolution, it draws a statistically similar but different realization.
Comparing FE solutions across resolutions therefore requires a field that
is a genuine closed-form function of (x, y) (or (theta, r) for B2), so it
can be evaluated identically at any mesh resolution. This is standard
practice for verifying solver convergence (it checks the SOLVER, not the
stochastic ensemble) and is a separate question from the operator's
accuracy on the random-field benchmark used in the rest of this study.

The fixed fields below share the ensemble's mean and roughly its spatial
amplitude (mean E=1000, std 200; mean load magnitude 5, std 2) so the
convergence behavior seen here is representative of the typical training
sample, not a different problem regime.

Usage:
  python -m omar_pfem.mesh_convergence --geometry B1 --material neo_hookean \
      --resolutions 6,11,16,21,26,31,41,51 --out_json convergence_B1.json
  python -m omar_pfem.mesh_convergence --geometry B2 --material neo_hookean \
      --resolutions 6,11,16,21,26,31,41,51 --out_json convergence_B2.json
"""
import argparse
import json
import numpy as np
from scipy.interpolate import LinearNDInterpolator


class AnalyticFieldB1:
    """Smooth, closed-form (E, nu, ty) fields for B1's unit square, evaluable
    identically at any mesh resolution. Calling convention matches
    RegularGridInterpolator: pts is (N,2) Cartesian (x,y), returns (N,)."""
    def __init__(self, kind):
        self.kind = kind

    def __call__(self, pts):
        x, y = pts[:, 0], pts[:, 1]
        if self.kind == "E":
            return 1000.0 + 200.0 * np.sin(np.pi * x) * np.cos(np.pi * y)
        elif self.kind == "nu":
            return np.full_like(x, 0.3)
        elif self.kind == "ty":
            return -5.0 - 2.0 * np.sin(np.pi * x)
        raise ValueError(self.kind)


class AnalyticFieldB2:
    """Smooth, closed-form (E, nu, p) fields for B2's quarter ring. Calling
    convention matches RegularGridInterpolator: pts is (N,2) polar
    (theta, r), returns (N,)."""
    def __init__(self, kind):
        self.kind = kind

    def __call__(self, pts):
        theta = pts[:, 0]
        if self.kind == "E":
            return 1000.0 + 200.0 * np.sin(2.0 * theta)
        elif self.kind == "nu":
            return np.full_like(theta, 0.3)
        elif self.kind == "p":
            return 5.0 + 2.0 * np.sin(2.0 * theta)
        raise ValueError(self.kind)


def run_b1_convergence(resolutions, material):
    from omar_pfem.data.data_generate_B1 import generate_grid_Q4, solve_hyperelastic_TL_spatial

    Lx = Ly = 1.0
    E_fn, nu_fn, ty_fn = AnalyticFieldB1("E"), AnalyticFieldB1("nu"), AnalyticFieldB1("ty")

    # Query points fixed in physical space, evaluated at every resolution
    # via interpolation of that resolution's own nodal solution -- NOT mesh
    # nodes themselves, so the same physical quantity is compared across
    # meshes with different node layouts. Deliberately off the x=0.5
    # symmetry line of the analytic E(x,y) field above (sin(pi x) is
    # symmetric about x=0.5): a query point exactly on that line has a
    # true horizontal displacement u of exactly zero, which makes its
    # *relative* change between resolutions numerically meaningless
    # (dividing by a value that is itself just discretization noise).
    query_points = {"top_off_center": (0.7, 1.0), "mid_domain": (0.3, 0.6)}

    rows = []
    for N in resolutions:
        nodes, elements = generate_grid_Q4(Lx, Ly, N, N)
        u = solve_hyperelastic_TL_spatial(
            nodes, elements, E_fn, nu_fn, ty_fn, Ly,
            nsteps=10, newton_max=30, tol=1e-7, material=material,
        )
        u_mag = np.sqrt(u[:, 0] ** 2 + u[:, 1] ** 2)
        interp_u = LinearNDInterpolator(nodes, u[:, 0])
        interp_v = LinearNDInterpolator(nodes, u[:, 1])

        row = {"N": N, "n_nodes": len(nodes), "n_elements": len(elements),
               "max_disp_mag": float(u_mag.max())}
        for name, (qx, qy) in query_points.items():
            row[f"u_{name}"] = float(interp_u(qx, qy))
            row[f"v_{name}"] = float(interp_v(qx, qy))
        rows.append(row)
    return rows


def run_b2_convergence(resolutions, material):
    from omar_pfem.data.data_generate_B2 import generate_grid_Q4_ring, solve_hyperelastic_TL_ring

    R_in, R_out = 1.0, 2.0
    E_fn, nu_fn, p_fn = AnalyticFieldB2("E"), AnalyticFieldB2("nu"), AnalyticFieldB2("p")

    # Query points given in polar (theta, r), converted to Cartesian for
    # the interpolator (nodes are stored in Cartesian, see generate_grid_Q4_ring).
    # theta=pi/3 (not pi/4) deliberately avoids the symmetry line of
    # E(theta)=1000+200*sin(2*theta) about theta=pi/4 -- see B1's version
    # of this comment for why a query point exactly on a field symmetry
    # line gives a numerically meaningless relative-change ratio.
    query_points_polar = {"inner_arc_off_sym": (np.pi / 3, R_in), "mid_domain": (np.pi / 3, 1.5)}

    rows = []
    for N in resolutions:
        nodes, elements = generate_grid_Q4_ring(R_in, R_out, N, N)
        u = solve_hyperelastic_TL_ring(
            nodes, elements, E_fn, nu_fn, p_fn, R_in,
            nsteps=10, newton_max=30, tol=1e-7, material=material,
        )
        u_mag = np.sqrt(u[:, 0] ** 2 + u[:, 1] ** 2)
        interp_u = LinearNDInterpolator(nodes, u[:, 0])
        interp_v = LinearNDInterpolator(nodes, u[:, 1])

        row = {"N": N, "n_nodes": len(nodes), "n_elements": len(elements),
               "max_disp_mag": float(u_mag.max())}
        for name, (theta, r) in query_points_polar.items():
            qx, qy = r * np.cos(theta), r * np.sin(theta)
            row[f"u_{name}"] = float(interp_u(qx, qy))
            row[f"v_{name}"] = float(interp_v(qx, qy))
        rows.append(row)
    return rows


def add_relative_changes(rows):
    """Adds, to every row after the first, the relative change (vs. the
    previous, coarser resolution) of every non-(N/n_nodes/n_elements)
    scalar in the row -- the standard h-refinement convergence metric:
    a solver has converged once this relative change is small and
    monotonically shrinking as N increases."""
    keys = [k for k in rows[0].keys() if k not in ("N", "n_nodes", "n_elements")]
    for i in range(1, len(rows)):
        prev, cur = rows[i - 1], rows[i]
        cur["relative_change"] = {}
        for k in keys:
            denom = abs(prev[k]) if abs(prev[k]) > 1e-12 else 1e-12
            cur["relative_change"][k] = abs(cur[k] - prev[k]) / denom
    return rows


def main():
    parser = argparse.ArgumentParser("Mesh (h-refinement) convergence study for the reference FE solver")
    parser.add_argument("--geometry", type=str, required=True, choices=["B1", "B2"])
    parser.add_argument("--material", type=str, default="neo_hookean",
                         choices=["neo_hookean", "mooney_rivlin", "arruda_boyce"])
    parser.add_argument("--resolutions", type=str, default="6,11,16,21,26,31,41,51",
                         help="Comma-separated node-count-per-side values to test "
                              "(21 is this study's operator-learning mesh)")
    parser.add_argument("--out_json", type=str, default=None)
    args = parser.parse_args()

    resolutions = [int(n) for n in args.resolutions.split(",") if n.strip()]
    print(f"Mesh convergence study: geometry={args.geometry}, material={args.material}, "
          f"resolutions={resolutions}")
    print("Using fixed, smooth analytic (E, nu, load) fields, NOT the random-field "
          "ensemble used elsewhere in this study -- see this script's module docstring "
          "for why that substitution is necessary for a valid h-refinement comparison.")

    if args.geometry == "B1":
        rows = run_b1_convergence(resolutions, args.material)
    else:
        rows = run_b2_convergence(resolutions, args.material)

    rows = add_relative_changes(rows)

    print("\n" + "=" * 100)
    for row in rows:
        rc = row.get("relative_change")
        rc_str = (", ".join(f"{k}={v:.3%}" for k, v in rc.items())) if rc else "(reference, no prior resolution)"
        print(f"N={row['N']:>4d}  n_nodes={row['n_nodes']:>6d}  "
              f"max_disp_mag={row['max_disp_mag']:.6e}  relative_change: {rc_str}")
    print("=" * 100)

    if args.out_json:
        with open(args.out_json, "w") as f:
            json.dump({"geometry": args.geometry, "material": args.material,
                       "resolutions": resolutions, "rows": rows}, f, indent=2)
        print(f"Full report written to {args.out_json}")


if __name__ == "__main__":
    main()
