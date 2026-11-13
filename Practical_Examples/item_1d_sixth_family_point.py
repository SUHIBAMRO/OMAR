"""Omar explicitly asked for genuine, rigorous convergence proof, not a caveat.

A formal Richardson-extrapolation/GCI check (Celik et al. 2008) on the
existing 3 finest consistent-family points (10200, 15960, 23520 elements)
gave an apparent convergence order p~12.3 -- physically implausible for
quadratic elements (expected ~2-4) -- meaning these points are NOT yet in
the asymptotic convergence range required for that method to be trusted.
The only honest way to actually settle convergence is a genuine further
mesh point in the SAME refinement family (Ntheta+4, Nr+2, Nz+5, continuing
the established pattern): (33, 24, 46) = 33,120 elements.

This script solves ONLY the groove case (no nominal/no-groove counterfactual
needed for a pure peak-stress convergence check) to keep compute cost down.

Usage: python3 item_1d_sixth_family_point.py
"""
import json
import time

import torch

from omar_pfem.data.solve_quadratic_hex_groove_stress import (
    solve_and_measure_groove_region_quadratic)

Ntheta, Nr, Nz = 33, 24, 46
device = torch.device("cpu")

print(f"Solving GROOVE case at ({Ntheta},{Nr},{Nz}) ...", flush=True)
t0 = time.time()
groove = solve_and_measure_groove_region_quadratic(Ntheta, Nr, Nz, device=device)
elapsed = time.time() - t0
print(f"  done in {elapsed:.1f}s: n_elem={groove['n_elem']} n_nodes={groove['n_nodes']} "
      f"DOF={groove['n_nodes']*3} n_region_gauss={groove['n_region_gauss']} "
      f"avg|sigma_xx|={groove['region_avg_sigma_xx']:.4f} "
      f"max|sigma_xx|={groove['region_max_sigma_xx']:.4f}", flush=True)

out = {"resolution": [Ntheta, Nr, Nz], "n_elements": groove["n_elem"],
       "n_nodes": groove["n_nodes"], "dof": groove["n_nodes"] * 3, "groove": groove}
out_path = "item_1d_sixth_family_point_33120.json"
with open(out_path, "w") as f:
    json.dump(out, f, indent=2, default=lambda o: float(o) if hasattr(o, "item") else str(o))
print(f"\nWrote {out_path}")
