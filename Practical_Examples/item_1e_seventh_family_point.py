"""7th point in the same consistent refinement family (Ntheta+4, Nr+2,
Nz+5 pattern, continuing 2,376 -> 5,760 -> 10,200 -> 15,960 -> 23,520 ->
33,120), to further tighten the Richardson/GCI convergence estimate.
With 4 points (15960, 23520, 33120) the apparent order was already a
much more plausible p~6.0 (GCI +/-5.55%) than with only 3 (p~12.3,
unreliable); this point lets us redo the estimate with the 3 newest
points (23520, 33120, 45000) for a tighter, more trustworthy bound.

Groove-only solve (no nominal/no-groove counterfactual needed for a
pure peak-stress convergence check).

Usage: python3 item_1e_seventh_family_point.py
"""
import json
import time

import torch

from omar_pfem.data.solve_quadratic_hex_groove_stress import (
    solve_and_measure_groove_region_quadratic)

Ntheta, Nr, Nz = 37, 26, 51
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
out_path = "item_1e_seventh_family_point_45000.json"
with open(out_path, "w") as f:
    json.dump(out, f, indent=2, default=lambda o: float(o) if hasattr(o, "item") else str(o))
print(f"\nWrote {out_path}")
