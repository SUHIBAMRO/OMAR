"""The checkpoint comparison table Timon actually asked for (PI/DD vs
the Q20 reference) uses region-AVERAGE sigma_xx error, not max -- so it
is unaffected by the just-discovered max-stress non-monotonicity. But
the standalone "max|sigma_xx| convergence" table in both reports needs
an honest avg-based companion column across the SAME 4 original family
points (2,376 / 5,760 / 10,200 / 15,960 elements) for which avg was
never recorded, to confirm avg converges cleanly across the FULL
7-point family, not just the 3 newest points.

Groove-only solves (cheap at these sizes).

Usage: python3 item_1f_avg_for_original_four_points.py
"""
import json
import time

import torch

from omar_pfem.data.solve_quadratic_hex_groove_stress import (
    solve_and_measure_groove_region_quadratic)

device = torch.device("cpu")
resolutions = [(13, 12, 19), (17, 16, 25), (21, 18, 31), (25, 20, 36)]

results = []
for Ntheta, Nr, Nz in resolutions:
    print(f"Solving GROOVE case at ({Ntheta},{Nr},{Nz}) ...", flush=True)
    t0 = time.time()
    r = solve_and_measure_groove_region_quadratic(Ntheta, Nr, Nz, device=device)
    elapsed = time.time() - t0
    print(f"  done in {elapsed:.1f}s: n_elem={r['n_elem']} n_nodes={r['n_nodes']} "
          f"DOF={r['n_nodes']*3} avg|sigma_xx|={r['region_avg_sigma_xx']:.4f} "
          f"max|sigma_xx|={r['region_max_sigma_xx']:.4f}", flush=True)
    results.append({"resolution": [Ntheta, Nr, Nz], "n_elements": r["n_elem"],
                     "n_nodes": r["n_nodes"], "dof": r["n_nodes"] * 3, "groove": r})

out_path = "item_1f_avg_for_original_four_points.json"
with open(out_path, "w") as f:
    json.dump(results, f, indent=2, default=lambda o: float(o) if hasattr(o, "item") else str(o))
print(f"\nWrote {out_path}")
