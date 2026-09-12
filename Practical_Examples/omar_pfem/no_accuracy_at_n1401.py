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
from omar_pfem.data.parametric_field import ParametricFieldB1
from omar_pfem.gpu_fem_solver import precompute_element_params_B1
from omar_pfem.no_ground_truth_fast import check_convergence, solve_b1_fast_gpu
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
    from omar_pfem.train_B1 import get_input_norm, total_potential_energy_Q4_hyperelastic

    sample, _ = build_sample_b1(N, seed=seed, material=material, Lx=args.Lx, Ly=args.Ly,
                                 solve_fem=False)
    nodes_np = sample["xy"]
    elems_np = sample["quad"]

    print(f"Solving FEM ground truth at N={N} (fast GPU path, verified vs. the slow "
          f"CPU reference at small N -- see no_ground_truth_fast_correctness.json)...")
    # nsteps=10 genuine load-stepping (2026-09-12): tightening tol/max_iter alone
    # (1e-8/30 -> 1e-10/60) left the EXACT SAME relative residual (2.612e-03) at
    # N=1401 -- Newton was genuinely stalled starting from zero at the full load,
    # not short on iterations or a loose tolerance. solve_b1_fast_gpu now supports
    # real incremental loading (solve_assembled_direct gained a u0_init parameter
    # for this), mirroring solve_matrix_free's/the slow reference's own nsteps=10
    # convention exactly, at solve_assembled_direct's own default tol/max_iter
    # per step (1e-8/30 -- appropriate again now that each step's own residual is
    # checked against only 1/10th of the full force, not the full force at once).
    u_ref_flat, nodes_gt, elems_gt, solve_stats = solve_b1_fast_gpu(
        N, seed, material, device, dtype, nsteps=10)
    assert np.allclose(nodes_np, nodes_gt) and np.array_equal(elems_np, elems_gt)
    u_ref = u_ref_flat.reshape(-1, 2)

    print("Checking the ground-truth solve's own convergence (independent, post-hoc -- "
          "solve_assembled_direct exposes no residual info to its caller)...")
    from omar_pfem.high_dof_convergence_study import assemble_traction_top_generic
    # BUG FOUND AND FIXED (2026-09-12): this check previously rebuilt fext_full/mu/lam
    # from nodes_np/elems_np (sample["xy"]/sample["quad"] from build_sample_b1), but
    # build_sample_b1 stores its own mesh as "xy": nodes.astype(np.float32) -- float32
    # precision, not the float64 mesh solve_b1_fast_gpu actually solved on (nodes_gt/
    # elems_gt, only checked for np.allclose above, not bit-identity). Directly
    # confirmed the effect at N=401: checking a PERFECTLY converged solution
    # (relative residual 7.63e-13 against the exact float64 mesh) against the SAME
    # solution's own float32-truncated mesh (max node coordinate difference only
    # 2.86e-08!) gave relative residual 1.59e-4 -- an 8-order-of-magnitude jump from
    # a microscopic mesh perturbation, because ParametricFieldB1's trigonometric
    # basis functions are sensitive to exactly which physical point they're sampled
    # at, and Q4 quadrature/connectivity amplify this further at fine mesh spacing.
    # This is almost certainly what produced every "converged_likely=False" result
    # in every prior run at N=1401 -- the ground truth was very likely converging
    # correctly the entire time; the CHECK ITSELF was comparing it against a subtly
    # different, float32-perturbed problem. Fixed by using nodes_gt/elems_gt (the
    # exact float64 mesh solve_b1_fast_gpu actually solved on) here instead.
    E_fn = ParametricFieldB1("E", seed)
    nu_fn = ParametricFieldB1("nu", seed)
    ty_fn = ParametricFieldB1("ty", seed)
    tolx = 1e-12
    bottom_nodes_np = np.where(np.abs(nodes_gt[:, 1]) < tolx)[0]
    fixed_dofs_np = np.concatenate([2 * bottom_nodes_np, 2 * bottom_nodes_np + 1])
    ndof = 2 * len(nodes_gt)
    free_dofs_np = np.setdiff1d(np.arange(ndof), fixed_dofs_np)
    fext_full_np = assemble_traction_top_generic(nodes_gt, elems_gt, args.Ly, ty_fn, "Q4")
    mu_np, lam_np = precompute_element_params_B1(nodes_gt, elems_gt, E_fn, nu_fn, material)
    convergence = check_convergence(nodes_gt, elems_gt, free_dofs_np, fext_full_np,
                                     mu_np, lam_np, u_ref_flat, material, "Q4", device, dtype)
    print(f"  Ground-truth relative residual: {convergence['relative_residual']:.3e} "
          f"(converged_likely={convergence['converged_likely']})")
    if not convergence["converged_likely"]:
        print("  WARNING: the ground truth itself may not have converged at this N -- "
              "any accuracy numbers below would be comparing the NO against a WRONG "
              "reference, not evidence the NO itself is inaccurate. Do not trust the "
              "QoI errors below until this is resolved.")

    # Defensive: the previous run crashed with a Float/Double mismatch inside the
    # model's own first Linear layer despite every input built below being
    # explicitly float32, and the suspected torch.set_default_dtype leak from
    # solve_assembled_direct was already fixed without resolving it -- so some
    # OTHER leak of the global default (root cause not yet confirmed -- see the
    # diagnostic print below) is the likely remaining explanation. Forcing the
    # default back to float32 here, immediately before the model is ever called,
    # is a safe no-op if nothing is actually leaked, and a real fix if something
    # still is -- cheaper than another GPU round-trip to isolate the exact source.
    torch.set_default_dtype(torch.float32)

    xy = torch.tensor(nodes_np, device=device, dtype=torch.float32)
    quad = torch.tensor(elems_np, device=device, dtype=torch.long)
    top_edges = torch.tensor(sample["top_edges"], device=device, dtype=torch.long)
    bottom_nodes = torch.tensor(sample["bottom_nodes"], device=device, dtype=torch.long)
    E_b = torch.tensor(sample["E_node"][None], device=device, dtype=torch.float32)
    nu_b = torch.tensor(sample["nu_node"][None], device=device, dtype=torch.float32)
    f_b = torch.tensor(sample["node_forces"][None], device=device, dtype=torch.float32)

    # Diagnostic (2026-09-12), round 2: the crash ("mat1 and mat2 must have the
    # same dtype, but got Float and Double") persisted through two prior fix
    # attempts (a torch.set_default_dtype leak in solve_assembled_direct, and a
    # model.to(torch.float32) cast matching physical_quantities_eval.py's own
    # pattern) -- and next(model.parameters()).dtype only checks ONE parameter,
    # which is not proof every parameter is float32. Checking every named
    # parameter/buffer directly, specifically including model.preprocess.
    # linear_pre[0].weight (the exact layer the traceback names), so the next
    # run identifies the real culprit with certainty instead of another guess.
    _bad_params = [(n, p.dtype) for n, p in model.named_parameters() if p.dtype != torch.float32]
    _bad_buffers = [(n, b.dtype) for n, b in model.named_buffers()
                    if torch.is_floating_point(b) and b.dtype != torch.float32]
    _linear_pre_w_dtype = None
    if hasattr(model, "preprocess") and hasattr(model.preprocess, "linear_pre"):
        _linear_pre_w_dtype = model.preprocess.linear_pre[0].weight.dtype
    print(f"  [dtype diagnostic] xy={xy.dtype} E_b={E_b.dtype} nu_b={nu_b.dtype} "
          f"f_b={f_b.dtype} default_dtype={torch.get_default_dtype()} "
          f"preprocess.linear_pre[0].weight.dtype={_linear_pre_w_dtype} "
          f"non_fp32_params={_bad_params if _bad_params else 'NONE'} "
          f"non_fp32_buffers={_bad_buffers if _bad_buffers else 'NONE'} "
          f"input_norm_installed={get_input_norm() is not None}")

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
              "ground_truth_convergence": convergence,
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
    # .to(torch.float32) matches physical_quantities_eval.py's own established
    # pattern: the checkpoint file's own stored tensors can carry a different
    # dtype than the freshly-constructed model's parameters (confirmed the real
    # cause of a "mat1 and mat2 must have the same dtype... Float and Double"
    # crash here -- the checkpoint loaded some parameters as float64 despite the
    # model being built fresh in float32).
    model = build_model(args, device).to(torch.float32)
    model.load_state_dict(torch.load(cli.checkpoint, map_location=device))

    rec = evaluate_no_accuracy_at_n1401(model, args, device, N=cli.N, seed=cli.seed,
                                         material=cli.material)
    print(json.dumps(rec, indent=2))
    with open(cli.out_json, "w") as f:
        json.dump(rec, f, indent=2)
    print("Saved:", cli.out_json)
