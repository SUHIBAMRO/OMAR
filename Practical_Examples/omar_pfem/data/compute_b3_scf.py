"""Timon's 2026-11-10 item 1b: Stress Concentration Factor (SCF) for
the B3 groove, now that local z-refinement (see
solve_z_refined_groove_stress.py / PROJECT_STATUS.md) gives a genuinely
CONVERGED local groove stress for the first time (global uniform
refinement never converged there, 274% non-convergence).

SCF = (stress at the groove) / (nominal stress at the SAME physical
location, SAME mesh, SAME loading, but with the groove geometrically
REMOVED -- groove_depth=0, flat R_in0 bore). This counterfactual
design (not the obvious-looking alternative of comparing against a
DIFFERENT far-away point, e.g. z=Lz/4) was chosen after that obvious
alternative was tried FIRST and gave a nonsensical SCF<1 result: this
problem's loading is a rigid ROTATION about an axis through z=Lz/2,
whose own dominant displacement term scales with dz=(z-Lz/2) -- EXACTLY
ZERO at the groove's own z=Lz/2 and sizeable at z=Lz/4. A point at a
different z is therefore under genuinely different local loading
intensity, independent of the groove, making a raw-stress comparison
between them meaningless for an SCF (confirmed directly, not assumed --
see PROJECT_STATUS.md for the z=Lz/4 attempt's actual numbers). Holding
z (and the mesh, and the loading) fixed and only removing the groove
geometrically is the standard, confound-free way to isolate a single
feature's own stress-concentration effect.

Usage: python -m omar_pfem.data.compute_b3_scf \
    --Ntheta 41 --Nr 25 --Nz 50 --groove_z_beta 18.0 \
    --out_json /scratch/gole3157/pfem_run/item_1b_scf.json
"""
import argparse
import json

import numpy as np
import torch

from omar_pfem.data.solve_z_refined_groove_stress import solve_and_measure_groove_region

R_IN0, LZ = 0.5, 1.0


def compute_scf(Ntheta, Nr, Nz, groove_z_beta, device=None):
    groove = solve_and_measure_groove_region(Ntheta, Nr, Nz, groove_z_beta, device=device)
    same_point = np.array([R_IN0, 0.0, LZ / 2.0])
    nominal = solve_and_measure_groove_region(
        Ntheta, Nr, Nz, groove_z_beta, device=device, groove_depth=0.0, ref_point=same_point)

    scf_avg_xx = groove["region_avg_sigma_xx"] / nominal["region_avg_sigma_xx"]
    scf_max_xx = groove["region_max_sigma_xx"] / nominal["region_avg_sigma_xx"]
    scf_avg_frob = groove["region_avg_sigma_frob"] / nominal["region_avg_sigma_frob"]

    return {
        "Ntheta": Ntheta, "Nr": Nr, "Nz": Nz, "groove_z_beta": groove_z_beta,
        "groove": groove, "nominal_no_groove": nominal,
        "SCF_avg_sigma_xx": scf_avg_xx, "SCF_max_sigma_xx_over_nominal_avg": scf_max_xx,
        "SCF_avg_sigma_frob": scf_avg_frob,
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--Ntheta", type=int, default=41)
    ap.add_argument("--Nr", type=int, default=25)
    ap.add_argument("--Nz", type=int, default=50)
    ap.add_argument("--groove_z_beta", type=float, default=18.0)
    ap.add_argument("--out_json", type=str, default=None)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}, Ntheta={args.Ntheta} Nr={args.Nr} Nz={args.Nz} "
          f"groove_z_beta={args.groove_z_beta}", flush=True)

    result = compute_scf(args.Ntheta, args.Nr, args.Nz, args.groove_z_beta, device=device)
    g, n = result["groove"], result["nominal_no_groove"]
    print(f"groove:      avg|sigma_xx|={g['region_avg_sigma_xx']:.4f}  "
          f"max|sigma_xx|={g['region_max_sigma_xx']:.4f}  avg|sigma|_F={g['region_avg_sigma_frob']:.4f}")
    print(f"no-groove nominal (same point, groove removed): avg|sigma_xx|={n['region_avg_sigma_xx']:.4f}  "
          f"avg|sigma|_F={n['region_avg_sigma_frob']:.4f}")
    print(f"SCF (avg groove sigma_xx / nominal avg sigma_xx)     = {result['SCF_avg_sigma_xx']:.3f}")
    print(f"SCF (max groove sigma_xx / nominal avg sigma_xx)     = {result['SCF_max_sigma_xx_over_nominal_avg']:.3f}")
    print(f"SCF (avg groove |sigma|_F / nominal avg |sigma|_F)   = {result['SCF_avg_sigma_frob']:.3f}")

    if args.out_json:
        with open(args.out_json, "w") as f:
            json.dump(result, f, indent=2)
        print(f"wrote {args.out_json}")
