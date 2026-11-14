"""Real z_beta sweep with Q20 (quadratic) elements, at a FIXED element
count (21,18,31)=10,200 -- same total elements throughout, just
clustered locally at the groove instead of spread uniformly in z.
z_beta=0.0 was already sanity-checked to reproduce the known uniform-
mesh value (avg=0.4306, max=0.8257) exactly. This script runs the
genuinely new points.

Usage: python3 item_1g_quadratic_zbeta_sweep.py <z_beta>
"""
import sys
import json
import time

from omar_pfem.data.solve_quadratic_hex_groove_stress_z_refined import (
    solve_and_measure_groove_region_quadratic_z_refined)

z_beta = float(sys.argv[1])
Ntheta, Nr, Nz = 21, 18, 31

print(f"Solving GROOVE case at ({Ntheta},{Nr},{Nz}), z_beta={z_beta} ...", flush=True)
t0 = time.time()
r = solve_and_measure_groove_region_quadratic_z_refined(Ntheta, Nr, Nz, z_beta)
print(f"  done in {time.time()-t0:.1f}s: n_elem={r['n_elem']} n_nodes={r['n_nodes']} "
      f"avg|sigma_xx|={r['region_avg_sigma_xx']:.4f} max|sigma_xx|={r['region_max_sigma_xx']:.4f}",
      flush=True)

out_path = f"item_1g_zbeta_{z_beta:.1f}_result.json"
with open(out_path, "w") as f:
    json.dump(r, f, indent=2, default=lambda o: float(o) if hasattr(o, "item") else str(o))
print(f"Wrote {out_path}")
