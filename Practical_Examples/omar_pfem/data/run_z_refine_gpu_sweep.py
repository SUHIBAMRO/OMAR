"""GPU-scale version of solve_z_refined_groove_stress.py's CPU smoke
test (Timon's 2026-11-10 item 1a). Same fixed-element-count, increasing
z_beta design -- no cross-mesh interpolation needed, so no dependency
on mesh_convergence_B3._theta_t_axes' uniform-z-spacing limitation
(see PROJECT_STATUS.md). This just runs it at a resolution close to
the project's own production/training scale instead of a CPU smoke
test's small mesh, on GPU, and writes the results to JSON so Omar can
pull them back for the report.

Usage (on the HPC, inside the `omar` conda env):
    python -u -m omar_pfem.data.run_z_refine_gpu_sweep \
        --out_json /scratch/gole3157/pfem_run/item_1a_z_refine_gpu_sweep.json
"""
import argparse
import json

import torch

from omar_pfem.data.solve_z_refined_groove_stress import solve_and_measure_groove_region


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--Ntheta", type=int, default=41)
    ap.add_argument("--Nr", type=int, default=25)
    ap.add_argument("--Nz", type=int, default=50)
    ap.add_argument("--betas", type=float, nargs="+",
                     default=[0.0, 1.0, 2.0, 4.0, 6.0, 8.0, 10.0, 14.0, 18.0])
    ap.add_argument("--out_json", type=str, required=True)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}, Ntheta={args.Ntheta} Nr={args.Nr} Nz={args.Nz}", flush=True)

    results = []
    for beta in args.betas:
        r = solve_and_measure_groove_region(args.Ntheta, args.Nr, args.Nz, beta, device=device)
        results.append(r)
        print(f"beta={r['z_beta']:>5.1f} n_elem={r['n_elem']:>7} "
              f"n_region_gauss={r['n_region_gauss']:>6} "
              f"avg|sigma_xx|={r['region_avg_sigma_xx']:.4f} "
              f"max|sigma_xx|={r['region_max_sigma_xx']:.4f} "
              f"avg|sigma|_F={r['region_avg_sigma_frob']:.4f} "
              f"time={r['elapsed_s']:.2f}s", flush=True)

    with open(args.out_json, "w") as f:
        json.dump({"Ntheta": args.Ntheta, "Nr": args.Nr, "Nz": args.Nz, "results": results}, f, indent=2)
    print(f"wrote {args.out_json}")


if __name__ == "__main__":
    main()
