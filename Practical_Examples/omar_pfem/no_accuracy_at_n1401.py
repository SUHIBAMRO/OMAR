"""
NO accuracy (displacement L2, H1 semi-norm, tangent-energy norm, PK1 stress,
reaction forces) at N=1401, against a real FEM ground truth there for the
first time (Timon round-10, item 1: "the NO accuracy at N=1401 itself,
including these QoIs has not been checked").

Reuses, rather than reimplements, two already-verified pieces of this
project: omar_pfem.no_ground_truth_fast.solve_b1_fast_gpu (verified against
the slow CPU reference at N=11/N=21, see no_ground_truth_fast_correctness.json)
for the ground truth, and the exact same QoI machinery
(as_solved_field/gauss_quantities/stress_errors/reaction_errors/
compute_l2_h1_errors_cross_order/compute_tangent_energy_error) that already
produced Table 15-17's own QoI numbers at N=1001/1401 for the FEM-vs-FEM
comparison, applied here to NO-vs-FEM instead.

Whether ParametricFieldB1 (used by solve_b1_fast_gpu, and by
build_sample_b1's own input construction below) is a distributionally valid
test input for this checkpoint: YES, deliberately, per parametric_field.py's
own docstring -- its Fourier-series mechanism is calibrated to the exact
same (E_mean=1000, E_std=200)/(nu_mean=0.3, nu_std=0.05)/(ty mean/std) and
correlation-length regime as the GRF fields (data/grf.py) the main
checkpoint was actually trained on; only the SAMPLING MECHANISM differs (so
it can be evaluated at any resolution from the same seed), not the physical
regime. This was checked directly against parametric_field.py's own
docstring before writing this script, not assumed.
"""
import json

import numpy as np
import torch

from omar_pfem.high_dof_convergence_study import (
    compute_l2_h1_errors_cross_order,
    compute_tangent_energy_error,
)
from omar_pfem.no_ground_truth_fast import solve_b1_fast_gpu
from omar_pfem.physical_quantities_eval import (
    as_solved_field,
    gauss_quantities,
    reaction_errors,
    stress_errors,
)
from omar_pfem.resolution_invariance_zeroshot import build_sample_b1


def _score_prediction(u_pred, u_ref, nodes_np, elems_np, sample, args, material,
                       device, dtype):
    """Everything downstream of a displacement prediction: the same QoI set
    (disp L2, H1 semi-norm, tangent-energy norm, PK1 stress, reactions)
    Table 15-17 already reports, applied to whatever u_pred is handed in --
    factored out so fp32 and bf16 predictions can be scored identically
    without duplicating the QoI code."""
    rms = lambda a: np.sqrt(np.mean(a ** 2))
    e_u = rms(u_pred[:, 0] - u_ref[:, 0]) / (rms(u_ref[:, 0]) + 1e-12)
    e_v = rms(u_pred[:, 1] - u_ref[:, 1]) / (rms(u_ref[:, 1]) + 1e-12)
    rec = {"disp_rel_L2": 0.5 * (e_u + e_v)}

    fp = as_solved_field(nodes_np, elems_np, u_pred)
    fr = as_solved_field(nodes_np, elems_np, u_ref)
    h1 = compute_l2_h1_errors_cross_order(fp, fr, "Q4", "Q4", "B1", Lx=args.Lx, Ly=args.Ly)
    rec["L2_rel"] = float(h1["l2_rel"])
    rec["H1_semi_rel"] = float(h1["h1_semi_rel"])
    en = compute_tangent_energy_error(fp, fr, "Q4", "Q4", "B1", material, device, dtype,
                                       Lx=args.Lx, Ly=args.Ly)
    rec["energy_rel"] = float(en["tangent_energy_rel"])

    P_p, w, R_p = gauss_quantities(nodes_np, elems_np, u_pred, sample["E_node"],
                                    sample["nu_node"], material, "plane_strain", "Q4",
                                    device, dtype)
    P_r, _, R_r = gauss_quantities(nodes_np, elems_np, u_ref, sample["E_node"],
                                    sample["nu_node"], material, "plane_strain", "Q4",
                                    device, dtype)
    rec.update(stress_errors(P_p, P_r, w))
    rec.update(reaction_errors(R_p, R_r, np.asarray(sample["bottom_nodes"]), [0, 1]))
    return rec


def evaluate_no_accuracy_at_n1401(model, args, device, N=1401, seed=0,
                                   material="neo_hookean", dtype=torch.float64,
                                   also_bf16=True):
    """also_bf16: also score a bf16-autocast forward pass against the SAME
    ground truth, not just the fp32-vs-fp32 self-consistency check the
    profiling cell (Timon round-10, item 3) already did -- bf16 measured
    5.69x faster there (402.8ms vs. 2290.2ms/sample) but with a 4.6%
    self-consistency gap vs. fp32, which is not itself an accuracy
    verdict. This answers the real question directly: is bf16 still
    accurate enough against real ground truth to be worth adopting,
    while a real GPU is already being spent on this N=1401 sample anyway."""
    from omar_pfem.train_B1 import total_potential_energy_Q4_hyperelastic

    sample, _ = build_sample_b1(N, seed=seed, material=material, Lx=args.Lx, Ly=args.Ly,
                                 solve_fem=False)
    nodes_np = sample["xy"]
    elems_np = sample["quad"]

    print(f"Solving FEM ground truth at N={N} (fast GPU path, verified vs. the slow "
          f"CPU reference at small N -- see no_ground_truth_fast_correctness.json)...")
    u_ref_flat, nodes_gt, elems_gt, solve_stats = solve_b1_fast_gpu(
        N, seed, material, device, dtype)
    assert np.allclose(nodes_np, nodes_gt) and np.array_equal(elems_np, elems_gt)
    u_ref = u_ref_flat.reshape(-1, 2)

    xy = torch.tensor(nodes_np, device=device, dtype=torch.float32)
    quad = torch.tensor(elems_np, device=device, dtype=torch.long)
    top_edges = torch.tensor(sample["top_edges"], device=device, dtype=torch.long)
    bottom_nodes = torch.tensor(sample["bottom_nodes"], device=device, dtype=torch.long)
    E_b = torch.tensor(sample["E_node"][None], device=device, dtype=torch.float32)
    nu_b = torch.tensor(sample["nu_node"][None], device=device, dtype=torch.float32)
    f_b = torch.tensor(sample["node_forces"][None], device=device, dtype=torch.float32)

    def _forward(use_bf16):
        model.eval()
        with torch.no_grad():
            if use_bf16:
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    _, _, _, uv_pred, _ = total_potential_energy_Q4_hyperelastic(
                        xy, quad, top_edges, bottom_nodes, model, E_b, nu_b, f_b,
                        use_soft_dirichlet=bool(args.use_soft_dirichlet), mode="plane_strain",
                        dtype=torch.float32, fun_dim=args.fun_dim, material=material, Ly=args.Ly,
                    )
            else:
                _, _, _, uv_pred, _ = total_potential_energy_Q4_hyperelastic(
                    xy, quad, top_edges, bottom_nodes, model, E_b, nu_b, f_b,
                    use_soft_dirichlet=bool(args.use_soft_dirichlet), mode="plane_strain",
                    dtype=torch.float32, fun_dim=args.fun_dim, material=material, Ly=args.Ly,
                )
        return uv_pred[0].float().double().cpu().numpy()

    print("Running NO forward pass (fp32)...")
    u_pred_fp32 = _forward(use_bf16=False)
    result = {"N": N, "seed": seed, "material": material,
              "fp32": _score_prediction(u_pred_fp32, u_ref, nodes_np, elems_np, sample,
                                         args, material, device, dtype)}

    if also_bf16 and device.type == "cuda":
        print("Running NO forward pass (bf16 autocast, diagnostic)...")
        u_pred_bf16 = _forward(use_bf16=True)
        result["bf16"] = _score_prediction(u_pred_bf16, u_ref, nodes_np, elems_np, sample,
                                            args, material, device, dtype)
        result["bf16_vs_fp32_disp_rel_diff"] = float(
            np.linalg.norm(u_pred_bf16 - u_pred_fp32) / np.linalg.norm(u_pred_fp32))

    return result


if __name__ == "__main__":
    import argparse

    from omar_pfem.measure_inference_latency import build_model

    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--material", default="neo_hookean")
    ap.add_argument("--N", type=int, default=1401)
    ap.add_argument("--out_json", default="no_accuracy_at_n1401.json")
    ap.add_argument("--cpu", action="store_true")
    cli = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() and not cli.cpu else "cpu")
    args = argparse.Namespace(
        model="Transolver_Irregular_Mesh", n_hidden=256, n_layers=4, n_heads=8,
        mlp_ratio=2, dropout=0.1, unified_pos=0, ref=16, slice_num=128, fun_dim=4,
        use_soft_dirichlet=1, Lx=1.0, Ly=1.0, R_out=2.0,
    )
    model = build_model(args, device)
    model.load_state_dict(torch.load(cli.checkpoint, map_location=device))

    rec = evaluate_no_accuracy_at_n1401(model, args, device, N=cli.N, seed=cli.seed,
                                         material=cli.material)
    print(json.dumps(rec, indent=2))
    with open(cli.out_json, "w") as f:
        json.dump(rec, f, indent=2)
    print("Saved:", cli.out_json)
