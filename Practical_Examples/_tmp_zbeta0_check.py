import time
from omar_pfem.data.solve_quadratic_hex_groove_stress_z_refined import solve_and_measure_groove_region_quadratic_z_refined
t0 = time.time()
r = solve_and_measure_groove_region_quadratic_z_refined(21, 18, 31, 0.0)
print(f"z_beta=0.0  n_elem={r['n_elem']}  avg|sigma_xx|={r['region_avg_sigma_xx']:.4f}  "
      f"max|sigma_xx|={r['region_max_sigma_xx']:.4f}  elapsed={r['elapsed_s']:.1f}s  "
      f"total_wall={time.time()-t0:.1f}s", flush=True)
