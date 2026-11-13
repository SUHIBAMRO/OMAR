"""A careful external review of the 2026-11-12 report correctly caught two
real issues this script addresses directly:

1. The quadratic (Q20) convergence table mixed TWO different mesh
   families: the dedicated 4-point h-refinement sweep (13,12,19) ->
   (25,20,36), all built with consistent, smoothly-increasing
   Ntheta/Nr/Nz increments, PLUS one unrelated point -- the production
   PI/DD reference mesh (41,25,50) -- whose Ntheta/Nr/Nz ratios are NOT
   a consistent continuation of that same family (Ntheta jumps much
   more aggressively relative to Nr/Nz than every earlier step). The
   reported "last two points agree within 2.5%" was ALSO simply the
   wrong pair (it was actually the 3rd/4th points, 10,200 vs 15,960);
   the real 4th-vs-5th (15,960 vs 47,040) gap is about 14.9%, which is
   both a different mesh family AND a real, not-yet-explained gap.
   This script adds a genuinely consistent 5th point in the SAME
   refinement family (Ntheta +4, Nr +2, Nz +5, continuing the observed
   pattern of the last two steps): (29, 22, 41) = 23,520 elements.

2. The SCF reported everywhere so far is a sigma_xx-only ratio. This
   script computes the full per-component SCF (groove vs. no-groove,
   same point, same quadratic mesh) at this same new resolution, so at
   least one genuinely matched-depth, matched-element-type data point
   exists for every stress component, not only sigma_xx.

Usage: python3 item_1c_extra_point_and_percomponent_scf.py
"""
import json
import time

import numpy as np
import torch

from omar_pfem.data.solve_quadratic_hex_groove_stress import (
    solve_and_measure_groove_region_quadratic, R_IN0, LZ)

Ntheta, Nr, Nz = 29, 22, 41
device = torch.device("cpu")

print(f"Solving GROOVE case at ({Ntheta},{Nr},{Nz}) ...", flush=True)
t0 = time.time()
groove = solve_and_measure_groove_region_quadratic(Ntheta, Nr, Nz, device=device)
print(f"  done in {time.time()-t0:.1f}s: n_elem={groove['n_elem']} "
      f"n_region_gauss={groove['n_region_gauss']} avg|sigma_xx|={groove['region_avg_sigma_xx']:.4f} "
      f"max|sigma_xx|={groove['region_max_sigma_xx']:.4f}", flush=True)

same_point = np.array([R_IN0, 0.0, LZ / 2.0])
print(f"Solving NOMINAL (no-groove) case at ({Ntheta},{Nr},{Nz}), same point ...", flush=True)
t0 = time.time()
nominal = solve_and_measure_groove_region_quadratic(
    Ntheta, Nr, Nz, device=device, groove_depth=0.0, ref_point=same_point)
print(f"  done in {time.time()-t0:.1f}s: avg|sigma_xx|={nominal['region_avg_sigma_xx']:.4f} "
      f"max|sigma_xx|={nominal['region_max_sigma_xx']:.4f}", flush=True)

print("\n=== Per-component SCF (groove max / nominal avg), (29,22,41)=23,520 elements ===")
per_comp = {}
for comp, g in groove["per_component"].items():
    n = nominal["per_component"][comp]
    scf_max = g["max"] / (abs(n["avg"]) + 1e-12)
    scf_avg = g["avg"] / (abs(n["avg"]) + 1e-12)
    per_comp[comp] = {
        "groove_avg": g["avg"], "groove_max": g["max"],
        "nominal_avg": n["avg"], "nominal_max": n["max"],
        "SCF_max_over_nominal_avg": scf_max, "SCF_avg_over_nominal_avg": scf_avg,
    }
    print(f"  sigma_{comp}: groove avg={g['avg']:.4f} max={g['max']:.4f} | "
          f"nominal avg={n['avg']:.4f} max={n['max']:.4f} | "
          f"SCF(max/nom_avg)={scf_max:.3f}  SCF(avg/nom_avg)={scf_avg:.3f}")

out = {
    "resolution": [Ntheta, Nr, Nz], "n_elements": groove["n_elem"],
    "n_region_gauss": groove["n_region_gauss"],
    "groove": groove, "nominal": nominal, "per_component_scf": per_comp,
}
out_path = "item_1c_extra_point_23520_percomponent_scf.json"
with open(out_path, "w") as f:
    json.dump(out, f, indent=2, default=lambda o: float(o) if hasattr(o, "item") else str(o))
print(f"\nWrote {out_path}")
