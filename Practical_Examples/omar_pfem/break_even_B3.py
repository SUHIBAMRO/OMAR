"""Break-even analysis for B3 (3D rocking bushing), matching B1/B2's own
two-comparison convention (break_even_resolution_matched.py /
break_even_accuracy_matched_N1401.json), built 2026-10-01 per Timon's
"finish the paper" request (report accuracy-matched FEM/VINO comparison
and break-even for the QoIs that are reproduced well: displacement,
energy, reaction).

TWO separate comparisons, both reported (neither replaces the other):

1. RESOLUTION-matched: NO and FEM both at B3's own training resolution
   (6,840 elements). "If you insist on FEM at the network's own
   resolution, does the NO still win." Computable right now -- every
   number is already a real, independently verified measurement:
     - NO inference: 6.32 ms/sample, 800.7x speedup vs FEM (evaluate_B3.py
       on checkpoint_50000.pt, PROJECT_STATUS.md 2026-09-27).
     - training_seconds: 6980.3s, real GPU wall-clock for run 4's full
       50,000-iteration training (B3_Transolver_Training.ipynb,
       PROJECT_STATUS.md 2026-09-27).

2. ACCURACY-matched: find the cheapest FEM mesh whose OWN error against
   an independent fine reference matches the NO's error against that
   SAME fine reference (not the NO's error vs. its own 6,840-element
   training mesh, which is not apples-to-apples -- see
   evaluate_B3_qois_finer.py's 2026-10-01 docstring addition for why).
   This needs a `--fine_qois_json` produced by running the UPDATED
   evaluate_B3_qois_finer.py (--fine_resolution 41 36 32, the project's
   established fine evaluation mesh) on real GPU data -- NOT yet run as
   of this script's creation. Without it, this script only reports the
   resolution-matched comparison.

   The FEM-vs-fine-reference lookup table below (element count -> rel
   error for disp/energy/reaction, and elapsed_s) is copied verbatim
   from mesh_convergence_B3.py's own already-verified, already-published
   required-resolution-table GPU run (2026-09-21,
   mesh_convergence_extended.json, Drive file id
   1Y2ldqky_YvYbG5ICFODO_O8o5iSzgkUM -- the same run that produced
   PROJECT_STATUS.md's "Final required-resolution table (5%/2%/1%)").
   No new GPU time is needed for this table itself.

Usage:
  python -m omar_pfem.break_even_B3
  python -m omar_pfem.break_even_B3 --fine_qois_json /path/to/qois_finer_resolution.json
"""
import argparse
import json
import os

# ---------------------------------------------------------------------------
# 1. RESOLUTION-matched break-even -- ready now, no new GPU run.
# ---------------------------------------------------------------------------
PRODUCTION_N_ELEMENTS = 6840
NO_MS_PER_SAMPLE = 6.32          # evaluate_B3.py, checkpoint_50000.pt, PROJECT_STATUS.md 2026-09-27
SPEEDUP_VS_FEM_SAME_RES = 800.7  # same source
FEM_MS_PER_SAMPLE_SAME_RES = NO_MS_PER_SAMPLE * SPEEDUP_VS_FEM_SAME_RES
TRAINING_SECONDS = 6980.3        # B3_Transolver_Training.ipynb run 4, real GPU wall-clock

# ---------------------------------------------------------------------------
# 2. ACCURACY-matched -- FEM-vs-fine-reference convergence ladder, real GPU
# data already on file (mesh_convergence_extended.json, 2026-09-21).
# Reference solved at (81,40,79)=243,360 elements.
# ---------------------------------------------------------------------------
CONVERGENCE_LADDER = [
    # n_elements, elapsed_s, disp_l2_rel, energy_rel, moment_rel
    (600,    3.723095655441284, 0.032495041216060616, 0.01848455414005086,  0.07380762674784835),
    (3240,   4.083858251571655, 0.016384167797588698, 0.00805606119560293,  0.030530555752764733),
    (9464,   4.672098636627197, 0.01022971206086564,  0.004688391178906972, 0.016915314815869192),
    (20808,  5.802619934082031, 0.006972502579457039, 0.0030564595361632814, 0.01062621388187719),
    (38808,  7.6209752559661865, 0.0049190731238141164, 0.0020877560324696624, 0.0070794538798823565),
    (65000,  10.355619668960571, 0.0035452014936181936, 0.00145182628833466,  0.004836785806474861),
    (123008, 16.57001519203186,  0.0021165463211680405, 0.0008329447239445438, 0.00271664733725233),
]
FINE_REFERENCE_N_ELEMENTS = 243360


def find_cheapest_matching_fem(no_rel_err, ladder, err_index):
    """Smallest-N row whose own error (vs the fine reference) is <= the
    NO's error (vs the SAME fine reference) -- "cheapest FEM at least as
    accurate as the NO," matching B1/B2's own accuracy-matched
    convention. Returns None if even the largest tested row does not
    reach the NO's accuracy (the honest "not yet matched within the
    tested ladder" case -- reported as such, not papered over)."""
    for row in ladder:
        if row[err_index] <= no_rel_err:
            return row
    return None


def main():
    p = argparse.ArgumentParser("B3 break-even: resolution-matched (always) + accuracy-matched (optional)")
    p.add_argument("--fine_qois_json", type=str, default=None,
                   help="output of evaluate_B3_qois_finer.py (--fine_resolution 41 36 32), "
                        "giving the NO's disp/energy/reaction error vs the independent fine "
                        "reference. Without this, only the resolution-matched result is reported.")
    p.add_argument("--out_json", type=str, default=None)
    args = p.parse_args()

    print("=" * 78)
    print("B3 BREAK-EVEN ANALYSIS")
    print("=" * 78)

    print("\n--- 1. RESOLUTION-matched (NO and FEM both at 6,840 elements) ---")
    saving_ms = FEM_MS_PER_SAMPLE_SAME_RES - NO_MS_PER_SAMPLE
    break_even_samples_res = TRAINING_SECONDS / (saving_ms / 1000.0)
    print(f"  FEM:            {FEM_MS_PER_SAMPLE_SAME_RES:.1f} ms/sample")
    print(f"  NO:             {NO_MS_PER_SAMPLE:.2f} ms/sample")
    print(f"  speedup:        {SPEEDUP_VS_FEM_SAME_RES:.1f}x")
    print(f"  training cost:  {TRAINING_SECONDS:.1f}s ({TRAINING_SECONDS / 3600.0:.2f}h)")
    print(f"  break-even:     {break_even_samples_res:.1f} samples "
          f"({break_even_samples_res * NO_MS_PER_SAMPLE / 1000.0 / 3600.0:.4f} GPU-hours of NO inference)")

    result = {
        "resolution_matched": {
            "n_elements": PRODUCTION_N_ELEMENTS,
            "fem_ms_per_sample": FEM_MS_PER_SAMPLE_SAME_RES,
            "no_ms_per_sample": NO_MS_PER_SAMPLE,
            "speedup": SPEEDUP_VS_FEM_SAME_RES,
            "training_seconds": TRAINING_SECONDS,
            "break_even_samples": break_even_samples_res,
        },
        "accuracy_matched": None,
    }

    print("\n--- 2. ACCURACY-matched (cheapest FEM at least as accurate as the NO, "
          f"both vs. the {FINE_REFERENCE_N_ELEMENTS:,}-element fine reference) ---")
    if args.fine_qois_json is None:
        print("  --fine_qois_json not given -- skipping. Run the updated "
              "evaluate_B3_qois_finer.py (--fine_resolution 41 36 32) on real GPU data "
              "first, then re-run this script with that output.")
    else:
        with open(args.fine_qois_json) as f:
            fine_data = json.load(f)
        s = fine_data["summary"]
        no_errs = {
            "displacement": (s["mean_disp_rel_l2_fine_combined"], 2),
            "energy": (s["pooled_rms_rel_err_energy_fine"], 3),
            "reaction_moment": (s["pooled_rms_rel_err_reaction_moment_y_fine"], 4),
        }
        acc_rows = {}
        for qoi, (no_err, err_index) in no_errs.items():
            match = find_cheapest_matching_fem(no_err, CONVERGENCE_LADDER, err_index)
            if match is None:
                print(f"  [{qoi}] NO error ({no_err:.4%}) is MORE accurate than even the largest "
                      f"tested FEM mesh ({CONVERGENCE_LADDER[-1][0]:,} elements, "
                      f"{CONVERGENCE_LADDER[-1][err_index]:.4%}) -- not matched within this ladder; "
                      f"using the largest tested row as a lower-bound break-even (conservative: "
                      f"the TRUE accuracy-matched FEM would be even more expensive, so this "
                      f"UNDERSTATES the NO's real advantage).")
                match = CONVERGENCE_LADDER[-1]
                matched_within_ladder = False
            else:
                matched_within_ladder = True
            n_el, elapsed_s, disp_e, energy_e, moment_e = match
            fem_ms = elapsed_s * 1000.0
            saving_ms_acc = fem_ms - NO_MS_PER_SAMPLE
            be_acc = (TRAINING_SECONDS / (saving_ms_acc / 1000.0)) if saving_ms_acc > 0 else None
            acc_rows[qoi] = {
                "no_rel_err_vs_fine_reference": no_err,
                "matched_fem_n_elements": n_el,
                "matched_within_tested_ladder": matched_within_ladder,
                "matched_fem_ms_per_sample": fem_ms,
                "speedup_vs_matched_fem": fem_ms / NO_MS_PER_SAMPLE,
                "break_even_samples": be_acc,
            }
            print(f"  [{qoi}] NO err={no_err:.4%} vs fine ref -> cheapest-matching FEM: "
                  f"{n_el:,} elements ({fem_ms:.1f} ms/sample) "
                  f"speedup={fem_ms / NO_MS_PER_SAMPLE:.1f}x "
                  f"break_even={('%.0f samples' % be_acc) if be_acc else 'never (FEM already cheaper)'}")
        result["accuracy_matched"] = acc_rows

    print("=" * 78)

    if args.out_json is None:
        args.out_json = os.path.join(os.path.dirname(os.path.abspath(__file__)), "break_even_B3.json")
    with open(args.out_json, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nWritten to {args.out_json}")


if __name__ == "__main__":
    main()
