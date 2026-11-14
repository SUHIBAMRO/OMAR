"""Omar's request (2026-11-14): try LOCAL refinement in BOTH directions
together -- r_grading (clusters radial nodes toward R_in, the groove
surface) AND z_beta (clusters z-layers toward the groove's axial
location) -- since the groove's sharp feature is a corner in the r-z
plane, not purely axial. Neither direction alone (z_beta sweep: 0.0,
1.0, 2.0, 3.0 -> max 0.826/0.615/0.684/0.783, non-monotonic) pinned
down a converged max|sigma_xx|; this tests whether combining both
directions, at the SAME fixed element count, does better.

Confirmed directly (data_generate_B2.generate_grid_Q4_ring's own
docstring): r_grading>1 clusters nodes toward R_in (the groove side),
via t -> t**r_grading on the radial parameter.

Usage: python3 item_1h_combined_rz_refinement.py <r_grading> <z_beta>
"""
import sys
import json
import time

from omar_pfem.data.solve_quadratic_hex_groove_stress_z_refined import (
    solve_and_measure_groove_region_quadratic_z_refined)

r_grading = float(sys.argv[1])
z_beta = float(sys.argv[2])
Ntheta, Nr, Nz = 21, 18, 31

print(f"Solving GROOVE case at ({Ntheta},{Nr},{Nz}), r_grading={r_grading}, z_beta={z_beta} ...",
      flush=True)
t0 = time.time()
r = solve_and_measure_groove_region_quadratic_z_refined(Ntheta, Nr, Nz, z_beta, r_grading=r_grading)
print(f"  done in {time.time()-t0:.1f}s: n_elem={r['n_elem']} n_nodes={r['n_nodes']} "
      f"avg|sigma_xx|={r['region_avg_sigma_xx']:.4f} max|sigma_xx|={r['region_max_sigma_xx']:.4f}",
      flush=True)

out_path = f"item_1h_rgrad_{r_grading:.1f}_zbeta_{z_beta:.1f}_result.json"
with open(out_path, "w") as f:
    json.dump(r, f, indent=2, default=lambda o: float(o) if hasattr(o, "item") else str(o))
print(f"Wrote {out_path}")
