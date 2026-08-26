"""Produces the advisor's explicit Q4-vs-Q9 agreement table ("Q4 and Q9
results should be nearly identical for such a refinement, meaning
difference smaller than 1e-5 in all norms") from two ALREADY-COMPLETED
fine-mesh solves -- pure post-processing (point location + quadrature),
no new Newton-CG solve.

Run this AFTER both high_dof_convergence_study.py --orders Q4 and --orders
Q9 runs (same --geometry/--material/--fine_N/--checkpoint_dir) have each
finished all their load steps. It reloads both checkpoints (solve_one's
checkpoint-resume path returns instantly since every step is already
done -- see matrix_free_solver.py's solve_matrix_free) and reports, for
L2 / H1-seminorm / energy, in both directions (Q4 as the integration
domain and Q9 as the integration domain -- these need not be numerically
identical since the two meshes differ, but should closely agree if both
have truly converged):
  - absolute difference |u_Q4 - u_Q9|
  - relative difference |u_Q4 - u_Q9| / |u_other|

Usage:
  python -m omar_pfem.compare_q4_q9 --geometry B1 --material neo_hookean \
      --fine_N 2236 --checkpoint_dir /content/drive/MyDrive/pfem_ckpt
"""
import argparse
import json

import numpy as np
import torch

from omar_pfem.high_dof_convergence_study import (
    solve_one, compute_l2_h1_errors_cross_order, compute_tangent_energy_error,
)


def main():
    parser = argparse.ArgumentParser("Q4 vs Q9 direct agreement check at a shared fine resolution")
    parser.add_argument("--geometry", type=str, required=True, choices=["B1", "B2"])
    parser.add_argument("--material", type=str, default="neo_hookean",
                         choices=["neo_hookean", "mooney_rivlin", "arruda_boyce"])
    parser.add_argument("--fine_N", type=int, required=True)
    parser.add_argument("--checkpoint_dir", type=str, required=True,
                         help="Same --checkpoint_dir the two finished runs used -- this script "
                              "only reads their checkpoint files, it does not solve anything new "
                              "unless a checkpoint is missing or incomplete.")
    parser.add_argument("--cg_tol", type=float, default=1e-8)
    parser.add_argument("--newton_tol", type=float, default=1e-8)
    # Defaults to saving next to the checkpoints rather than to None: this
    # comparison is the advisor's headline Q4-vs-Q9 check and takes a
    # multi-day solve to produce, but an earlier run was invoked WITHOUT
    # --out_json, so its numbers survived only as Colab stdout and no result
    # file was ever written to Drive. A result this expensive must never
    # depend on remembering to pass a flag.
    parser.add_argument("--out_json", type=str, default=None,
                        help="Where to write the JSON report. If omitted, defaults to "
                             "<checkpoint_dir>/q4_vs_q9_<geometry>_<material>_N<fine_N>.json. "
                             "Pass 'none' to explicitly disable saving.")
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    if args.out_json is None:
        args.out_json = (f"{args.checkpoint_dir}/q4_vs_q9_"
                         f"{args.geometry}_{args.material}_N{args.fine_N}.json")
        print(f"[auto-save] --out_json not given; results will be written to {args.out_json}")
    elif args.out_json.strip().lower() == "none":
        args.out_json = None

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    dtype = torch.float64

    results = {}
    for order in ("Q4", "Q9"):
        ckpt_path = f"{args.checkpoint_dir}/fine_{args.geometry}_{args.material}_{order}_N{args.fine_N}.pt"
        print(f"Loading {order} fine solution (resuming from checkpoint, expect an instant "
              f"'load steps 1-10 already done' if it is truly finished)...")
        sol = solve_one(args.geometry, order, args.fine_N, args.material, device, dtype,
                         args.cg_tol, args.newton_tol, verbose=False, checkpoint_path=ckpt_path)
        results[order] = sol
        if sol["stats"]["load_steps"] > 0:
            pass  # solve_one/solve_matrix_free already printed the resume status

    geom_kwargs = ({"Lx": 1.0, "Ly": 1.0} if args.geometry == "B1"
                    else {"R_in": 1.0, "R_out": 2.0, "theta_max": np.pi / 2})

    print(f"\n{'='*90}\nQ4 vs Q9 direct comparison at N_Q4={results['Q4']['N']} "
          f"(n_dof={results['Q4']['n_dof']}), N_Q9={results['Q9']['N']} (n_dof={results['Q9']['n_dof']})\n{'='*90}")

    # Direction A: Q4 mesh as the integration domain, Q9 as the "reference" evaluated there.
    errs_a = compute_l2_h1_errors_cross_order(results["Q4"], results["Q9"], "Q4", "Q9",
                                               args.geometry, **geom_kwargs)
    # Direction B: Q9 mesh as the integration domain, Q4 as the "reference" evaluated there.
    errs_b = compute_l2_h1_errors_cross_order(results["Q9"], results["Q4"], "Q9", "Q4",
                                               args.geometry, **geom_kwargs)
    # Advisor-confirmed energy norm: "the tangent/incremental energy norm;
    # relative errors are enough" -- sqrt(e^T K e) via the fine solution's own
    # tangent stiffness, evaluated once with each order's mesh as the K-operator
    # (direction A: Q9's K; direction B: Q4's K) -- not the old scalar
    # strain-energy difference, which was only ever a placeholder.
    energy_a = compute_tangent_energy_error(results["Q4"], results["Q9"], "Q4", "Q9",
                                             args.geometry, args.material, device, dtype, **geom_kwargs)
    energy_b = compute_tangent_energy_error(results["Q9"], results["Q4"], "Q9", "Q4",
                                             args.geometry, args.material, device, dtype, **geom_kwargs)

    def fmt(v):
        return f"{v:.3e}"

    print(f"\n{'Metric':<28}{'Abs (Q4-domain)':<18}{'Rel (Q4-domain)':<18}"
          f"{'Abs (Q9-domain)':<18}{'Rel (Q9-domain)':<18}")
    print(f"{'L2':<28}{fmt(errs_a['l2_abs']):<18}{fmt(errs_a['l2_rel']):<18}"
          f"{fmt(errs_b['l2_abs']):<18}{fmt(errs_b['l2_rel']):<18}")
    print(f"{'H1-seminorm':<28}{fmt(errs_a['h1_semi_abs']):<18}{fmt(errs_a['h1_semi_rel']):<18}"
          f"{fmt(errs_b['h1_semi_abs']):<18}{fmt(errs_b['h1_semi_rel']):<18}")
    print(f"{'Tangent energy norm':<28}{fmt(energy_a['tangent_energy_abs']):<18}"
          f"{fmt(energy_a['tangent_energy_rel']):<18}{fmt(energy_b['tangent_energy_abs']):<18}"
          f"{fmt(energy_b['tangent_energy_rel']):<18}")

    threshold = 1e-5
    worst_rel = max(errs_a['l2_rel'], errs_a['h1_semi_rel'], errs_b['l2_rel'], errs_b['h1_semi_rel'],
                     energy_a['tangent_energy_rel'], energy_b['tangent_energy_rel'])
    verdict = "PASS" if worst_rel < threshold else "FAIL"
    print(f"\nAdvisor's criterion (all relative norms < {threshold:.0e}): {verdict} "
          f"(worst observed relative difference = {worst_rel:.3e})")

    if args.out_json:
        with open(args.out_json, "w") as f:
            json.dump({
                "geometry": args.geometry, "material": args.material, "fine_N": args.fine_N,
                "q4_n_dof": results["Q4"]["n_dof"], "q9_n_dof": results["Q9"]["n_dof"],
                "q4_domain": errs_a, "q9_domain": errs_b,
                "energy_q4_domain": energy_a, "energy_q9_domain": energy_b,
                "threshold": threshold, "verdict": verdict,
            }, f, indent=2)
        print(f"\nFull report written to {args.out_json}")


if __name__ == "__main__":
    main()
