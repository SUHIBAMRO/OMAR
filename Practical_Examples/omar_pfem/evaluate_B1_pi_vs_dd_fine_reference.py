"""B1 PI-vs-DD accuracy comparison against an independent fine FEM
reference, per Professor Rabczuk's second email: "For B1, the DD model
was trained using the N=21 FEM data. Therefore, the PI-vs-DD accuracy
comparison should be repeated on an inference mesh different from N=21
and evaluated against the very fine N=2236 FEM reference already used
in our convergence study."

This project's own existing PI-vs-DD comparison
(point7b_results/comparison_B1_neo_hookean.json, "the clean comparison:
only the loss differs") evaluated both checkpoints at their OWN training
resolution/mesh -- exactly the limitation Timon is naming. A separate,
EARLIER round-12 bug (see no_accuracy_at_n1401.py's own
run_qoi_study_consistent_field_b1 docstring, 2026-09-18) already caught
and fixed the closely-related mistake of comparing FEM and a trained
operator on two DIFFERENT physical problems (AnalyticFieldB1 for FEM,
ParametricFieldB1(seed) for the operator) -- this module reuses that
same, already-validated fix (same field family on both sides, both
scored against ONE real fine reference) and extends it to compare TWO
models (PI and DD) against the SAME fine reference and the SAME coarse
FEM solution, instead of one model at a time, so the expensive fine
(and coarse) FEM solves are shared rather than redone per model.

Solver: reuses no_ground_truth_fast.solve_b1_fast_gpu
(assembled_direct_solver's single/few-step Newton solve, GPU-vectorized)
rather than the project's own default matrix-free solver -- the whole
reason that fast solver exists is that the matrix-free one measured
27,257s for a SINGLE N=1401 solve; fast_gpu was built and verified
(bit-for-bit against the slow CPU reference at N=11/N=21) specifically
to make resolutions this large tractable. N=2236 has not been attempted
with it before this module -- try it for real before assuming it is
fast enough; if it is not, fall back to a smaller (but still much finer
than N=21) --fine_N.

Usage:
  python -m omar_pfem.evaluate_B1_pi_vs_dd_fine_reference \\
      --pi_checkpoint /path/to/results/B1_neo_hookean/model_best.pt \\
      --dd_checkpoint /path/to/data_driven/B1_neo_hookean/model_best.pt \\
      --resolutions 13 29 49 --fine_N 2236 --n_seeds 5 \\
      --out_json /path/to/b1_pi_vs_dd_fine_reference.json
"""
import argparse
import hashlib
import json
import os
import time

import numpy as np
import torch

from omar_pfem.data.parametric_field import ParametricFieldB1
from omar_pfem.high_dof_convergence_study import (
    compute_l2_h1_errors, compute_reaction_resultant_error,
    compute_region_cauchy_stress_error, compute_tangent_energy_error,
    find_fine_peak_stress, select_fixed_region,
)
from omar_pfem.no_ground_truth_fast import solve_b1_fast_gpu
from omar_pfem.resolution_invariance_zeroshot import build_sample_b1
from omar_pfem.train_B1 import total_potential_energy_Q4_hyperelastic
from omar_pfem.model_dict import get_model


def sha256_of(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def build_model_b1(device, model_name="Transolver_Irregular_Mesh", n_hidden=256, n_layers=4,
                    n_heads=8, mlp_ratio=2, dropout=0.1, unified_pos=0, ref=16, slice_num=128,
                    fun_dim=4):
    """The architecture every B1 checkpoint in this project shares (see
    train_data_driven.py's own docstring: 'Same architecture, same mesh
    ... -- changes exactly one line: the loss') -- matches train_B1.py's
    own argparse defaults exactly. B1 checkpoints are saved as a bare
    state_dict (torch.save(model.state_dict(), ...)), not a {"args":...,
    "model_state":...} dict like B3's, so the architecture must be
    reconstructed from these known defaults rather than read back from
    the checkpoint file itself."""
    args = argparse.Namespace(model=model_name, n_hidden=n_hidden, n_layers=n_layers,
                               n_heads=n_heads, mlp_ratio=mlp_ratio, dropout=dropout,
                               unified_pos=unified_pos, ref=ref, slice_num=slice_num)
    return get_model(args).Model(
        space_dim=2, n_layers=n_layers, n_hidden=n_hidden, dropout=dropout,
        n_head=n_heads, Time_Input=False, mlp_ratio=mlp_ratio, fun_dim=fun_dim,
        out_dim=2, slice_num=slice_num, ref=ref, unified_pos=unified_pos,
    ).to(device)


def load_b1_checkpoint(checkpoint_path, device, **model_kwargs):
    model = build_model_b1(device, **model_kwargs)
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    fingerprint = sha256_of(checkpoint_path)
    return model, fingerprint


def _query_model_b1(model, sample, device, dtype, use_soft_dirichlet, mode, fun_dim, material, Ly):
    xy = torch.tensor(sample["xy"], device=device, dtype=dtype)
    quad = torch.tensor(sample["quad"], device=device, dtype=torch.long)
    top_edges = torch.tensor(sample["top_edges"], device=device, dtype=torch.long)
    bottom_nodes = torch.tensor(sample["bottom_nodes"], device=device, dtype=torch.long)
    E_b = torch.tensor(sample["E_node"][None], device=device, dtype=dtype)
    nu_b = torch.tensor(sample["nu_node"][None], device=device, dtype=dtype)
    f_b = torch.tensor(sample["node_forces"][None], device=device, dtype=dtype)
    with torch.no_grad():
        _, _, _, uv_pred, _ = total_potential_energy_Q4_hyperelastic(
            xy, quad, top_edges, bottom_nodes, model, E_b, nu_b, f_b,
            use_soft_dirichlet=bool(use_soft_dirichlet), mode=mode,
            dtype=dtype, fun_dim=fun_dim, material=material, Ly=Ly,
        )
    return uv_pred[0].double().cpu().numpy()


def compare_pi_dd_against_fine_reference(pi_model, dd_model, resolutions, fine_N, seeds, device,
                                          material="neo_hookean", Lx=1.0, Ly=1.0,
                                          region_radius=0.15, use_soft_dirichlet=1,
                                          mode="plane_strain", fun_dim=4,
                                          pi_fingerprint=None, dd_fingerprint=None,
                                          dtype=torch.float32, verbose=True, out_json=None):
    """THE SHARED-REFERENCE VERSION of no_accuracy_at_n1401.
    run_qoi_study_consistent_field_b1: for each seed, solves the fine
    (N=fine_N) and coarse (each N in resolutions) FEM references ONCE
    (via solve_b1_fast_gpu, same ParametricFieldB1(seed) realization
    both models were trained/tested on), then scores BOTH the PI and
    the DD model against that SAME shared fine reference and SAME
    coarse FEM solution -- avoiding the cost of re-solving the
    expensive fine_N reference twice (once per model) the way calling
    the per-model function separately would.

    out_json (added 2026-10-31, after a real ~10h HPC run was lost with
    nothing saved when an interactive node's session time limit killed
    it): when given, writes rows to this file IMMEDIATELY after each
    (seed, N) finishes, and skips any (seed, N) already present there
    on a re-run -- if every N for a seed is already done, that seed's
    own expensive fine_N solve is skipped entirely too."""
    geom_kwargs = {"Lx": Lx, "Ly": Ly}
    rows = []
    done_keys = set()
    if out_json and os.path.exists(out_json):
        with open(out_json) as f:
            prev = json.load(f)
        rows = prev.get("rows", [])
        done_keys = {(r["seed"], r["N"]) for r in rows}
        if verbose:
            print(f"Resuming from {out_json}: {len(rows)} row(s) already done.")

    def _save():
        if not out_json:
            return
        tmp = out_json + ".tmp"
        with open(tmp, "w") as f:
            json.dump({"material": material, "fine_N": fine_N, "resolutions": resolutions, "seeds": seeds,
                       "pi_fingerprint": pi_fingerprint, "dd_fingerprint": dd_fingerprint, "rows": rows},
                      f, indent=2)
        os.replace(tmp, out_json)  # atomic -- a crash mid-write never corrupts the real output

    for seed in seeds:
        if all((seed, N) in done_keys for N in resolutions):
            if verbose:
                print(f"[skip seed={seed}, all resolutions already done]")
            continue
        if verbose:
            print(f"\n{'=' * 90}\n=== seed={seed} ===")
        t0 = time.time()
        sample_fine, (E_fn, nu_fn, ty_fn) = build_sample_b1(
            fine_N, seed=seed, material=material, Lx=Lx, Ly=Ly, solve_fem=False)
        field_fns = {"E": E_fn, "nu": nu_fn, "ty": ty_fn}
        u_fine_flat, nodes_fine, elems_fine, _ = solve_b1_fast_gpu(
            fine_N, seed, material, device, torch.float64, Lx=Lx, Ly=Ly, nsteps=10)
        fine = {"nodes": nodes_fine, "elements": elems_fine,
                "u": u_fine_flat.reshape(-1, 2), "N": fine_N}
        if verbose:
            print(f"  fine reference (N={fine_N}) solved in {time.time() - t0:.1f}s")

        x_star, peak_ref = find_fine_peak_stress(fine, "Q4", "B1", material, E_fn, nu_fn,
                                                  device, torch.float64)
        region_pts, region_weights = select_fixed_region(fine, "Q4", x_star, region_radius)

        def _score(coarse):
            l2h1 = compute_l2_h1_errors(coarse, fine, "Q4", "B1", **geom_kwargs)
            energy = compute_tangent_energy_error(coarse, fine, "Q4", "Q4", "B1", material, device,
                                                   torch.float64, field_fns=field_fns, **geom_kwargs)
            reaction = compute_reaction_resultant_error(coarse, fine, "B1", material, E_fn, nu_fn,
                                                         device, torch.float64, order="Q4")
            cauchy = compute_region_cauchy_stress_error(coarse, fine, region_pts, region_weights,
                                                         "Q4", "B1", material, E_fn, nu_fn, device,
                                                         torch.float64, **geom_kwargs)
            return {
                "l2_rel": l2h1["l2_rel"], "h1_semi_rel": l2h1["h1_semi_rel"],
                "energy_rel": energy["tangent_energy_rel"],
                "reaction_resultant_rel_err": reaction["reaction_resultant_rel_err"],
                "cauchy_avg_rel_err": cauchy["region_cauchy_avg_rel_err"],
                "cauchy_p99_rel_err": cauchy["region_cauchy_p99_rel_err"],
                "cauchy_max_rel_err": cauchy["region_cauchy_max_rel_err"],
            }

        for N in resolutions:
            if (seed, N) in done_keys:
                if verbose:
                    print(f"  [skip N={N}, already done]")
                continue
            if verbose:
                print(f"  --- N={N} ---")
            t1 = time.time()
            u_fem_flat, nodes_np, elems_np, _ = solve_b1_fast_gpu(
                N, seed, material, device, torch.float64, Lx=Lx, Ly=Ly, nsteps=10)
            u_fem = u_fem_flat.reshape(-1, 2)
            sample, _ = build_sample_b1(N, seed=seed, material=material, Lx=Lx, Ly=Ly,
                                         solve_fem=False)
            assert np.allclose(nodes_np, sample["xy"]) and np.array_equal(elems_np, sample["quad"])

            u_pi = _query_model_b1(pi_model, sample, device, dtype, use_soft_dirichlet, mode,
                                    fun_dim, material, Ly)
            u_dd = _query_model_b1(dd_model, sample, device, dtype, use_soft_dirichlet, mode,
                                    fun_dim, material, Ly)

            fem_coarse = {"nodes": nodes_np.astype(np.float64), "elements": elems_np, "u": u_fem, "N": N}
            pi_coarse = {"nodes": nodes_np.astype(np.float64), "elements": elems_np, "u": u_pi, "N": N}
            dd_coarse = {"nodes": nodes_np.astype(np.float64), "elements": elems_np, "u": u_dd, "N": N}

            fem_score = _score(fem_coarse)
            pi_score = _score(pi_coarse)
            dd_score = _score(dd_coarse)
            if verbose:
                print(f"    FEM(N={N}) vs fine: L2={fem_score['l2_rel'] * 100:.2f}%  "
                      f"PI: L2={pi_score['l2_rel'] * 100:.2f}%  DD: L2={dd_score['l2_rel'] * 100:.2f}%  "
                      f"(elapsed {time.time() - t1:.1f}s)")

            rows.append({"seed": seed, "N": N, "fine_N": fine_N,
                         "fem_vs_fine": fem_score, "pi_vs_fine": pi_score, "dd_vs_fine": dd_score})
            _save()

    return {"material": material, "fine_N": fine_N, "resolutions": resolutions, "seeds": seeds,
            "pi_fingerprint": pi_fingerprint, "dd_fingerprint": dd_fingerprint, "rows": rows}


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--pi_checkpoint", type=str, required=True)
    parser.add_argument("--dd_checkpoint", type=str, required=True)
    parser.add_argument("--resolutions", type=int, nargs="+", default=[13, 29, 49],
                         help="inference meshes to evaluate at -- none of these are 21, "
                              "the DD model's own training resolution, per Timon's request")
    parser.add_argument("--fine_N", type=int, default=2236)
    parser.add_argument("--n_seeds", type=int, default=5)
    parser.add_argument("--seed_offset", type=int, default=0)
    parser.add_argument("--material", type=str, default="neo_hookean")
    parser.add_argument("--out_json", type=str, required=True)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    print(f"device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    pi_model, pi_fp = load_b1_checkpoint(args.pi_checkpoint, device)
    dd_model, dd_fp = load_b1_checkpoint(args.dd_checkpoint, device)
    print(f"PI checkpoint:  {args.pi_checkpoint}\n  sha256={pi_fp}")
    print(f"DD checkpoint:  {args.dd_checkpoint}\n  sha256={dd_fp}")

    seeds = list(range(args.seed_offset, args.seed_offset + args.n_seeds))
    result = compare_pi_dd_against_fine_reference(
        pi_model, dd_model, args.resolutions, args.fine_N, seeds, device,
        material=args.material, pi_fingerprint=pi_fp, dd_fingerprint=dd_fp, out_json=args.out_json)

    print(f"\nResults in {args.out_json} (already saved incrementally after every row)")

    print(f"\n{'=' * 90}\n=== SUMMARY (mean over {len(seeds)} seeds, per N) ===")
    for N in args.resolutions:
        rows_n = [r for r in result["rows"] if r["N"] == N]
        fem_l2 = np.mean([r["fem_vs_fine"]["l2_rel"] for r in rows_n])
        pi_l2 = np.mean([r["pi_vs_fine"]["l2_rel"] for r in rows_n])
        dd_l2 = np.mean([r["dd_vs_fine"]["l2_rel"] for r in rows_n])
        print(f"  N={N}: FEM(N)-vs-fine L2={fem_l2 * 100:.2f}%  "
              f"PI-vs-fine L2={pi_l2 * 100:.2f}%  DD-vs-fine L2={dd_l2 * 100:.2f}%")


if __name__ == "__main__":
    main()
