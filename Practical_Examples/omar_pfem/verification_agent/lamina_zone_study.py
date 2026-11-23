"""Why the lamina's sharp hole-zone average is contradicted (2026-11-23).

The hole zone 3 <= r <= 6 mm is selected by testing each QUADRATURE POINT, and
the circle r = 6 is not an element boundary in either discretization. So the
region over which the average is taken changes with the mesh: the quantity's
definition is discretization-dependent, and two families can converge to the
same continuum value along different paths and orders.

Measured here, for the FEM family (Q2, g = 1.5) and the spline family (p = 3,
beta = 2.8), from re-solves (identical to the stored ladders):
  zone_area_error   quadrature area of the zone against the exact pi (6^2 - 3^2)
  boundary_sensitivity
                    d(average)/d(outer radius) by moving the outer radius +-0.05 mm
                    (a definition sensitivity, no new physics)
Results: lamina_zone_study.json.

    python -m omar_pfem.verification_agent.lamina_zone_study
"""
import json
import os

import numpy as np

from omar_pfem.verification_agent import iga_lamina as G
from omar_pfem.verification_agent import lamina as L

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "lamina_zone_study.json")
EXACT_AREA = np.pi * (6.0 ** 2 - 3.0 ** 2)


def _with_zone(fn, outer):
    keep = L.HOLE_ZONE
    L.HOLE_ZONE = (3.0, outer)
    try:
        return fn()
    finally:
        L.HOLE_ZONE = keep


def main():
    res = {"exact_zone_area_mm2": EXACT_AREA, "fem": [], "spline": []}
    for m in (1, 2, 4, 8, 16, 32):
        rows = {o: _with_zone(lambda: L.solve_level(m, 1.5), o) for o in (5.95, 6.0, 6.05)}
        b = rows[6.0]
        res["fem"].append({"m": m, "hole_e22_avg": b["hole_e22_avg"],
                           "zone_area_error_rel": b["hole_zone_area_mm2"] / EXACT_AREA - 1,
                           "boundary_sensitivity_per_mm": (rows[6.05]["hole_e22_avg"]
                                                           - rows[5.95]["hole_e22_avg"]) / 0.1})
        print("fem", res["fem"][-1], flush=True)
    for m in (1, 2, 4, 8, 16):
        rows = {o: _with_zone(lambda: G.solve(m, 3, beta=2.8), o) for o in (5.95, 6.0, 6.05)}
        b = rows[6.0]
        res["spline"].append({"m": m, "hole_e22_avg": b["hole_e22_avg"],
                              "zone_area_error_rel": b["hole_zone_area_mm2"] / EXACT_AREA - 1,
                              "boundary_sensitivity_per_mm": (rows[6.05]["hole_e22_avg"]
                                                              - rows[5.95]["hole_e22_avg"])
                              / 0.1})
        print("spline", res["spline"][-1], flush=True)
    json.dump(res, open(OUT, "w"), indent=1)


if __name__ == "__main__":
    main()
