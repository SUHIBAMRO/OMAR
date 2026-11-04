"""Per Prof. Rabczuk's 2026-11-04 email, points 3 and 4: the existing
discretization-error report only has two FEM mesh sizes (6,840 and
43,400) compared against the 950,400-element reference, but PI
baseline's own errors (2.78% disp, 5.33% energy, 1.38% reaction) are
NOT bracketed by those two points for every QoI:

  - displacement: 6,840-el FEM (2.52%) is already MORE accurate than
    PI's own 2.78% target -- the cheapest matching mesh must be
    COARSER than 6,840.
  - energy: same story, 6,840-el FEM (3.88%) already beats PI's 5.33%.
  - reaction moment: even 43,400-el FEM (1.63%) is LESS accurate than
    PI's own 1.38% target -- the cheapest matching mesh must be FINER
    than 43,400.

This script solves additional mesh resolutions (reusing
evaluate_B3_qois_finer.generate_fine_reference exactly as the existing
43,400/950,400 references already do -- genuine new FEM solves, cached
and resumable the same way) and compares each against the ALREADY-
CACHED 950,400-element reference, for the scalar QoIs (displacement
combined L2, energy, reaction moment -- the three "well-reproduced"
QoIs point 3 actually needs) plus H1 for completeness. Also records
each resolution's own mean per-sample FEM solve wall-clock time, needed
for the break-even table's FEM(ms) column.

Default candidate resolutions, all pre-existing rungs from this
project's own B3 groove mesh-convergence ladder
(cell_b3_groove_gpu_mesh_convergence.py's RESOLUTIONS list) so nothing
is newly invented: 336, 1,320, 3,360 (coarser, to bracket displacement/
energy) and 79,464, 201,780 (finer, to bracket reaction moment).

Usage:
  python -m omar_pfem.compute_b3_multi_resolution_vs_950400 \\
      --run_dir /scratch/gole3157/pfem_run --n_samples 10 \\
      --out_json /scratch/gole3157/pfem_run/b3_rabczuk_reeval_v2.json
"""
import argparse
import json

import numpy as np
import torch

from omar_pfem.evaluate_B3_qois_finer import generate_fine_reference, interpolate_node_field_to_finer_mesh, FINE_RESOLUTION_STRESS
from omar_pfem.evaluate_B3_qois import compute_H1_seminorm_rel, compute_energy, compute_reaction
from omar_pfem.data.data_generate_B3_dataset import R_GRADING

# (Ntheta, Nr, Nz) -> element count, all pre-existing rungs of this project's
# own B3 groove mesh-convergence ladder (cell_b3_groove_gpu_mesh_convergence.py).
CANDIDATE_RESOLUTIONS = {
    "336":     (9, 8, 7),
    "1320":    (13, 12, 11),
    "3360":    (17, 16, 15),
    "79464":   (45, 44, 43),
    "201780":  (61, 60, 58),
}


def _one_resolution_vs_fine(ref, fine_ref, device, dtype):
    Ntheta_c, Nr_c, Nz_c = ref["fine_resolution"]
    Ntheta_f, Nr_f, Nz_f = fine_ref["fine_resolution"]
    geom_fine = fine_ref["geom_fine"]
    n_samples = min(len(ref["samples"]), len(fine_ref["samples"]))

    rows = []
    for idx in range(n_samples):
        s_c = ref["samples"][idx]
        s_f = fine_ref["samples"][idx]
        assert s_c["sample_index"] == s_f["sample_index"], "sample ordering mismatch"

        u_c_np = s_c["u_true_f"][0].cpu().numpy()
        u_c_on_fine_np = interpolate_node_field_to_finer_mesh(
            u_c_np, Ntheta_c, Nr_c, Nz_c, Ntheta_f, Nr_f, Nz_f, R_GRADING, n_components=3)
        u_c_on_fine = torch.tensor(u_c_on_fine_np, dtype=dtype, device=device).unsqueeze(0)
        u_true_f = s_f["u_true_f"]
        E_f_t, nu_f_t = s_f["E_f_t"], s_f["nu_f_t"]

        err = u_c_on_fine - u_true_f
        l2_num = torch.sqrt(torch.mean(torch.sum(err ** 2, dim=2)))
        l2_den = torch.sqrt(torch.mean(torch.sum(u_true_f ** 2, dim=2))) + 1e-12
        disp_rel_l2 = float((l2_num / l2_den).item())

        h1_rel = float(compute_H1_seminorm_rel(u_true_f, u_c_on_fine, geom_fine)[0].item())

        geom_c = ref["geom_fine"]
        u_c_native = s_c["u_true_f"]
        E_c_t, nu_c_t = s_c["E_f_t"], s_c["nu_f_t"]
        energy_c = compute_energy(u_c_native, E_c_t, nu_c_t, geom_c)
        energy_f = compute_energy(u_true_f, E_f_t, nu_f_t, geom_fine)
        energy_rel = float((torch.abs(energy_c - energy_f) / (torch.abs(energy_f) + 1e-12))[0].item())

        _, Mr_c, _ = compute_reaction(u_c_native, E_c_t, nu_c_t, geom_c)
        _, Mr_f, _ = compute_reaction(u_true_f, E_f_t, nu_f_t, geom_fine)
        moment_rel = float((torch.abs(Mr_c - Mr_f) / (torch.abs(Mr_f) + 1e-12))[0].item())

        rows.append({"sample_index": s_c["sample_index"], "disp_rel_l2": disp_rel_l2,
                      "H1_seminorm_rel": h1_rel, "energy_rel": energy_rel,
                      "reaction_moment_y_rel": moment_rel,
                      "solve_elapsed_s": s_c.get("elapsed_s")})

    elapsed = [r["solve_elapsed_s"] for r in rows if r["solve_elapsed_s"] is not None]
    summary = {
        "resolution": [Ntheta_c, Nr_c, Nz_c],
        "n_elements": (Ntheta_c - 1) * (Nr_c - 1) * (Nz_c - 1),
        "n_samples": n_samples,
        "mean_disp_rel_l2": float(np.mean([r["disp_rel_l2"] for r in rows])),
        "mean_H1_seminorm_rel": float(np.mean([r["H1_seminorm_rel"] for r in rows])),
        "mean_energy_rel": float(np.mean([r["energy_rel"] for r in rows])),
        "mean_reaction_moment_y_rel": float(np.mean([r["reaction_moment_y_rel"] for r in rows])),
        "mean_solve_elapsed_s": float(np.mean(elapsed)) if elapsed else None,
    }
    return {"summary": summary, "per_sample": rows}


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--run_dir", type=str, required=True)
    parser.add_argument("--n_samples", type=int, default=10)
    parser.add_argument("--out_json", type=str, required=True,
                         help="same --out_json the real re-evaluation run used, so this "
                              "script reuses its cached 950,400-element reference")
    parser.add_argument("--resolutions", type=str, nargs="+", default=list(CANDIDATE_RESOLUTIONS),
                         help="labels from CANDIDATE_RESOLUTIONS to solve/compare, e.g. "
                              "--resolutions 336 1320 3360")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--results_out", type=str, default=None,
                         help="where to write this script's own JSON results "
                              "(default: <out_json>.multi_resolution_vs_950400.json)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    dtype = torch.float64
    print(f"device: {device}")

    R = args.run_dir
    HOLDOUT = f"{R}/b3_dataset_clean_holdout/dataset.h5"
    N_SAMPLES = args.n_samples
    results_out = args.results_out or (args.out_json + ".multi_resolution_vs_950400.json")

    print(f"\n{'=' * 90}\nLoading cached 950,400-element reference (no new solves) ...")
    fine_cache = args.out_json + ".fine_950400.cache.pt"
    fine_ref = generate_fine_reference(HOLDOUT, FINE_RESOLUTION_STRESS, N_SAMPLES, device, dtype,
                                        cache_path=fine_cache)

    results = {}
    try:
        with open(results_out) as f:
            results = json.load(f)
        print(f"Resuming from {results_out}: {list(results.keys())} already done.")
    except FileNotFoundError:
        pass

    for label in args.resolutions:
        if label not in CANDIDATE_RESOLUTIONS:
            print(f"  SKIPPED -- unknown resolution label {label!r}, "
                  f"known: {list(CANDIDATE_RESOLUTIONS)}")
            continue
        if label in results:
            print(f"\n[skip, already done] {label} elements")
            continue
        Ntheta, Nr, Nz = CANDIDATE_RESOLUTIONS[label]
        n_elem = (Ntheta - 1) * (Nr - 1) * (Nz - 1)
        print(f"\n{'=' * 90}\n=== Solving {label} elements ({(Ntheta, Nr, Nz)}, "
              f"n_elem={n_elem}) for {N_SAMPLES} samples ===")
        cache_path = args.out_json + f".res_{label}.cache.pt"
        ref = generate_fine_reference(HOLDOUT, (Ntheta, Nr, Nz), N_SAMPLES, device, dtype,
                                       cache_path=cache_path, require_region_mask=False)
        print(f"  comparing vs. the 950,400-element reference ...")
        results[label] = _one_resolution_vs_fine(ref, fine_ref, device, dtype)
        s = results[label]["summary"]
        print(f"  {label} el: disp={s['mean_disp_rel_l2']*100:.3f}%  "
              f"H1={s['mean_H1_seminorm_rel']*100:.3f}%  "
              f"energy={s['mean_energy_rel']*100:.3f}%  "
              f"reaction={s['mean_reaction_moment_y_rel']*100:.3f}%  "
              f"mean_solve_s={s['mean_solve_elapsed_s']}")
        with open(results_out, "w") as f:
            json.dump(results, f, indent=2)

    print(f"\n{'=' * 90}\n=== SUMMARY: every tested resolution vs. the 950,400-element reference ===")
    print(f"(for comparison -- PI baseline's own errors: disp=2.78% H1=12.30% "
          f"energy=5.33% reaction=1.38%)")
    for label, r in sorted(results.items(), key=lambda kv: kv[1]["summary"]["n_elements"]):
        s = r["summary"]
        print(f"  {s['n_elements']:>7,} el: disp={s['mean_disp_rel_l2']*100:.3f}%  "
              f"H1={s['mean_H1_seminorm_rel']*100:.3f}%  "
              f"energy={s['mean_energy_rel']*100:.3f}%  "
              f"reaction={s['mean_reaction_moment_y_rel']*100:.3f}%  "
              f"mean_solve_s={s['mean_solve_elapsed_s']}")
    print(f"\nResults in {results_out}")


if __name__ == "__main__":
    main()
