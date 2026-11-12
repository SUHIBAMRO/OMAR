"""Quick, readable summary printer for b3_quadratic_reference.json
(output of evaluate_b3_checkpoints_vs_quadratic.py) -- avoids dumping
the full per-sample JSON to the terminal.

Usage: python3 print_quadratic_reference_summary.py <path_to_json>
"""
import json
import sys

path = sys.argv[1] if len(sys.argv) > 1 else "/scratch/gole3157/pfem_run/b3_quadratic_reference.json"
with open(path) as f:
    d = json.load(f)

print("resolution:", d["resolution"], " n_elements:", d["n_elements"],
      " n_nodes:", d["n_nodes"], " DOF:", d["dof"])
print("n_region_gauss:", d["n_region_gauss"])
print()
print("SCF (max groove / nominal avg):", d["scf"]["SCF_max_sigma_xx_over_nominal_avg"])
print("SCF (avg basis):", d["scf"]["SCF_avg_sigma_xx"])
print()
for name, r in d["checkpoints"].items():
    s = r["summary"]
    print(f"{name:<65} disp={s['mean_disp_rel_l2_fine_combined']*100:6.2f}%  "
          f"region_field={s['mean_region_cauchy_field_rel_fine']*100:8.2f}%  "
          f"sigma_xx={s['mean_region_sigma_xx_rel_fine']*100:8.2f}%")
