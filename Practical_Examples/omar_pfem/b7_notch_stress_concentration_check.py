"""
Demonstrates the actual physical claim behind the new B7 case (Timon
round-10, item 4: a geometry where "fine spatial resolution is actually
required by the geometry or physics/QoI, e.g. due to local stress
concentrations") before any Transolver training time is spent on it: a
mesh-convergence sweep showing that the notch's own peak PK1 stress has
NOT converged at coarse resolution while the global displacement field
has -- the exact signature Timon described (a smooth-looking global
field hiding a genuinely under-resolved local quantity), not assumed.

CPU-only, cheap (largest case here is 1,152 elements) -- this is a
correctness/physical-plausibility check, not a production run.
"""
import json

import numpy as np
import torch

from omar_pfem.data.data_generate_B7 import (
    generate_grid_Q4_ring_notch,
    solve_hyperelastic_TL_ring_notch,
)
from omar_pfem.physical_quantities_eval import gauss_quantities


def run(resolutions=((13, 7), (25, 13), (49, 25), (97, 49)),
        R_in_base=1.0, R_out=2.0, notch_depth=0.3, notch_width=0.15,
        E=1000.0, nu=0.3, p=-5.0, material="neo_hookean"):
    device = torch.device("cpu")
    dtype = torch.float64
    E_grid = lambda pts: np.full(len(pts), E)
    nu_grid = lambda pts: np.full(len(pts), nu)
    p_grid = lambda pts: np.full(len(pts), p)

    rows = []
    for Ntheta, Nr in resolutions:
        nodes, elements, r_in_theta = generate_grid_Q4_ring_notch(
            R_in_base, R_out, Ntheta, Nr, notch_depth=notch_depth, notch_width=notch_width)
        u = solve_hyperelastic_TL_ring_notch(
            nodes, elements, Nr, E_grid, nu_grid, p_grid,
            nsteps=10, newton_max=25, tol=1e-8, material=material, verbose=False)

        E_node = np.full(len(nodes), E)
        nu_node = np.full(len(nodes), nu)
        P, w, R = gauss_quantities(nodes, elements, u, E_node, nu_node, material,
                                    "plane_strain", "Q4", device, dtype)
        fro = np.sqrt((P.cpu().numpy() ** 2).sum(axis=(1, 2)))

        row = {
            "Ntheta": Ntheta, "Nr": Nr, "n_elements": int(len(elements)),
            "peak_pk1_stress": float(fro.max()),
            "max_abs_displacement": float(np.abs(u).max()),
        }
        rows.append(row)
        print(f"Ntheta={Ntheta:3d} Nr={Nr:3d}  n_elements={row['n_elements']:5d}  "
              f"peak_pk1_stress={row['peak_pk1_stress']:.4f}  "
              f"max|u|={row['max_abs_displacement']:.6f}")

    stress_change = [
        abs(rows[i + 1]["peak_pk1_stress"] - rows[i]["peak_pk1_stress"]) / rows[i]["peak_pk1_stress"]
        for i in range(len(rows) - 1)
    ]
    disp_change = [
        abs(rows[i + 1]["max_abs_displacement"] - rows[i]["max_abs_displacement"])
        / rows[i]["max_abs_displacement"]
        for i in range(len(rows) - 1)
    ]
    print("\nRelative change between consecutive resolutions:")
    for i, (ds, dd) in enumerate(zip(stress_change, disp_change)):
        print(f"  {rows[i]['n_elements']} -> {rows[i+1]['n_elements']} elements: "
              f"peak stress changed {ds:.1%}, max displacement changed {dd:.2%}")
    print("\nIf peak-stress change stays well above displacement change at the "
          "finest resolutions tested, the notch's own peak stress has genuinely "
          "not converged while the global field has -- the property this case is "
          "meant to demonstrate, not merely a slower-converging but still-smooth "
          "quantity.")

    return {
        "geometry": "B7_ring_notch", "material": material,
        "R_in_base": R_in_base, "R_out": R_out,
        "notch_depth": notch_depth, "notch_width": notch_width,
        "rows": rows,
        "peak_stress_relative_change_between_resolutions": stress_change,
        "max_displacement_relative_change_between_resolutions": disp_change,
    }


if __name__ == "__main__":
    result = run()
    out_path = "omar_pfem/b7_notch_stress_concentration_check.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print("\nSaved:", out_path)
