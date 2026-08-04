"""
Per-sample accuracy diagnostics for an already-trained checkpoint, per the
advisor's request to investigate WHY B2's validation error is so much
higher than B1's, rather than only naming "the curved geometry" as a
hypothesis. No retraining -- this only evaluates an existing checkpoint on
its own test set and breaks the resulting error down several ways:

  - Error vs. pressure magnitude (B2) / traction magnitude (B1), and vs.
    mean E and mean nu -- does the network fail specifically on stiffer/
    softer or more/less loaded samples, or is the error roughly uniform
    across the training distribution?
  - Radial vs. circumferential displacement-error decomposition (B2 only,
    since B1's Cartesian geometry has no natural polar split): projects
    the per-node (u, v) error vector onto the local (r_hat, theta_hat)
    frame at each node, since a systematic radial-vs-circumferential bias
    would point at a specific mechanism (e.g. the pressure BC or the
    polar-to-Cartesian mesh mapping) rather than generic underfitting.
  - Boundary vs. interior error: inner-boundary nodes (loaded, B2) /
    top-boundary nodes (loaded, B1) vs. outer/bottom (far or fixed) vs.
    interior, since an error concentrated at the loaded boundary would
    implicate the boundary-condition/loading term specifically.
  - Worst-case samples: the K samples with the highest per-sample error,
    for closer inspection.
  - An error-contour scatter plot for the single worst sample (predicted,
    exact, and pointwise error side by side, matching this study's
    existing umag_combined visualization convention).

Usage:
  python -m omar_pfem.accuracy_diagnostics \
      --geometry B2 --material neo_hookean \
      --checkpoint /path/to/B2_neo_hookean/model_best.pt \
      --dataset /path/to/B2_neo_hookean/hyperelastic_training_data_q4.npz \
      --ntrain 800 --ntest 200 \
      --out_dir ./diagnostics_B2_neo_hookean
"""
import os
import json
import argparse

import numpy as np
import torch

from omar_pfem.model_dict import get_model


def build_model(args, device):
    return get_model(args).Model(
        space_dim=2, n_layers=args.n_layers, n_hidden=args.n_hidden, dropout=args.dropout,
        n_head=args.n_heads, Time_Input=False, mlp_ratio=args.mlp_ratio, fun_dim=args.fun_dim,
        out_dim=2, slice_num=args.slice_num, ref=args.ref, unified_pos=args.unified_pos,
    ).to(device)


@torch.no_grad()
def predict_all(geometry, samples, model, args, device, dtype):
    if geometry == "B1":
        from omar_pfem.train_B1 import total_potential_energy_Q4_hyperelastic
    else:
        from omar_pfem.train_B2 import total_potential_energy_Q4_hyperelastic

    s0 = samples[0]
    xy = torch.tensor(s0["xy"], device=device, dtype=dtype)
    quad = torch.tensor(s0["quad"], device=device, dtype=torch.long)
    if geometry == "B1":
        top_edges = torch.tensor(s0["top_edges"], device=device, dtype=torch.long)
        bottom_nodes = torch.tensor(s0["bottom_nodes"], device=device, dtype=torch.long)
        mesh_args = (xy, quad, top_edges, bottom_nodes)
    else:
        inner_edges = torch.tensor(s0["inner_edges"], device=device, dtype=torch.long)
        theta0_nodes = torch.tensor(s0["theta0_nodes"], device=device, dtype=torch.long)
        thetahalfpi_nodes = torch.tensor(s0["thetahalfpi_nodes"], device=device, dtype=torch.long)
        mesh_args = (xy, quad, inner_edges, theta0_nodes, thetahalfpi_nodes)

    model.eval()
    preds = []
    for chunk_start in range(0, len(samples), 32):
        chunk = samples[chunk_start:chunk_start + 32]
        E_b = torch.tensor(np.stack([c["E_node"] for c in chunk]), device=device, dtype=dtype)
        nu_b = torch.tensor(np.stack([c["nu_node"] for c in chunk]), device=device, dtype=dtype)
        f_b = torch.tensor(np.stack([c["node_forces"] for c in chunk]), device=device, dtype=dtype)
        if geometry == "B1":
            _, _, _, uv_pred, _ = total_potential_energy_Q4_hyperelastic(
                *mesh_args, model, E_b, nu_b, f_b,
                use_soft_dirichlet=args.use_soft_dirichlet, Ly=args.Ly, mode=args.mode,
                dtype=dtype, fun_dim=args.fun_dim, material=args.material)
        else:
            _, _, _, uv_pred, _ = total_potential_energy_Q4_hyperelastic(
                *mesh_args, model, E_b, nu_b, f_b,
                use_soft_dirichlet=args.use_soft_dirichlet, R_out=args.R_out, mode=args.mode,
                dtype=dtype, fun_dim=args.fun_dim, material=args.material)
        preds.append(uv_pred.cpu().numpy())
    return np.concatenate(preds, axis=0), s0["xy"]


def rel_l2(pred, exact):
    """Per-sample scalar error, identical definition to Section 7.1 of the
    report: 0.5*(e_u + e_v), each e_* a per-node RMS relative error."""
    err = pred - exact
    l2_u = np.sqrt(np.mean(err[:, 0] ** 2))
    l2_v = np.sqrt(np.mean(err[:, 1] ** 2))
    ref_u = np.sqrt(np.mean(exact[:, 0] ** 2)) + 1e-12
    ref_v = np.sqrt(np.mean(exact[:, 1] ** 2)) + 1e-12
    return 0.5 * (l2_u / ref_u + l2_v / ref_v)


def radial_circumferential_decomposition(xy, err_uv):
    """Projects each node's (u, v) error onto its local (r_hat, theta_hat)
    frame -- B2-specific (polar geometry); returns (err_radial, err_circumferential)
    of shape (n_nodes,)."""
    r = np.linalg.norm(xy, axis=1) + 1e-12
    r_hat = xy / r[:, None]
    theta_hat = np.stack([-r_hat[:, 1], r_hat[:, 0]], axis=1)
    err_radial = np.sum(err_uv * r_hat, axis=1)
    err_circ = np.sum(err_uv * theta_hat, axis=1)
    return err_radial, err_circ


def classify_boundary_nodes_b2(xy, R_in, R_out, tol=1e-6):
    r = np.linalg.norm(xy, axis=1)
    inner = np.abs(r - R_in) < tol * R_out
    outer = np.abs(r - R_out) < tol * R_out
    interior = ~(inner | outer)
    return inner, outer, interior


def classify_boundary_nodes_b1(xy, Ly, tol=1e-6):
    top = np.abs(xy[:, 1] - Ly) < tol
    bottom = np.abs(xy[:, 1]) < tol
    interior = ~(top | bottom)
    return top, bottom, interior


def main():
    parser = argparse.ArgumentParser("Per-sample accuracy diagnostics for an existing checkpoint")
    parser.add_argument("--geometry", type=str, required=True, choices=["B1", "B2"])
    parser.add_argument("--material", type=str, required=True,
                         choices=["neo_hookean", "mooney_rivlin", "arruda_boyce"])
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--ntrain", type=int, default=800)
    parser.add_argument("--ntest", type=int, default=200)
    parser.add_argument("--n_worst", type=int, default=5)

    parser.add_argument("--model", type=str, default="Transolver_Irregular_Mesh")
    parser.add_argument("--n_hidden", type=int, default=256)
    parser.add_argument("--n_layers", type=int, default=4)
    parser.add_argument("--n_heads", type=int, default=8)
    parser.add_argument("--mlp_ratio", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--unified_pos", type=int, default=0)
    parser.add_argument("--ref", type=int, default=16)
    parser.add_argument("--slice_num", type=int, default=128)
    parser.add_argument("--fun_dim", type=int, default=4)
    parser.add_argument("--use_soft_dirichlet", type=int, default=1)
    parser.add_argument("--mode", type=str, default="plane_strain")
    parser.add_argument("--Ly", type=float, default=1.0)
    parser.add_argument("--R_in", type=float, default=1.0)
    parser.add_argument("--R_out", type=float, default=2.0)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--out_dir", type=str, required=True)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    dtype = torch.float32

    if args.geometry == "B1":
        from omar_pfem.train_B1 import load_fem_dataset_Q4_with_materials_and_random_force
    else:
        from omar_pfem.train_B2 import load_fem_dataset_Q4_with_materials_and_random_force
    _, Test_samples = load_fem_dataset_Q4_with_materials_and_random_force(args.dataset, args.ntrain, args.ntest)
    print(f"Loaded {len(Test_samples)} test samples")

    model = build_model(args, device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    uv_pred_all, xy = predict_all(args.geometry, Test_samples, model, args, device, dtype)

    per_sample = []
    for i, s in enumerate(Test_samples):
        pred, exact = uv_pred_all[i], s["uv_exact"]
        err_uv = pred - exact
        fm = np.sqrt(np.sum(s["node_forces"] ** 2, axis=1))
        nz = fm[fm > 0]
        rec = {
            "sample_idx": i, "gid": s.get("gid", i),
            "rel_l2_error": float(rel_l2(pred, exact)),
            "mean_E": float(s["E_node"].mean()), "mean_nu": float(s["nu_node"].mean()),
            "mean_load_magnitude": float(nz.mean()) if len(nz) else 0.0,
            "max_abs_error": float(np.max(np.abs(err_uv))),
        }
        if args.geometry == "B2":
            err_r, err_c = radial_circumferential_decomposition(xy, err_uv)
            inner, outer, interior = classify_boundary_nodes_b2(xy, args.R_in, args.R_out)
            rec.update({
                "rms_error_radial": float(np.sqrt(np.mean(err_r ** 2))),
                "rms_error_circumferential": float(np.sqrt(np.mean(err_c ** 2))),
                "rms_error_inner_boundary": float(np.sqrt(np.mean(np.sum(err_uv[inner] ** 2, axis=1)))) if inner.any() else None,
                "rms_error_outer_boundary": float(np.sqrt(np.mean(np.sum(err_uv[outer] ** 2, axis=1)))) if outer.any() else None,
                "rms_error_interior": float(np.sqrt(np.mean(np.sum(err_uv[interior] ** 2, axis=1)))) if interior.any() else None,
            })
        else:
            top, bottom, interior = classify_boundary_nodes_b1(xy, args.Ly)
            rec.update({
                "rms_error_top_boundary": float(np.sqrt(np.mean(np.sum(err_uv[top] ** 2, axis=1)))) if top.any() else None,
                "rms_error_bottom_boundary": float(np.sqrt(np.mean(np.sum(err_uv[bottom] ** 2, axis=1)))) if bottom.any() else None,
                "rms_error_interior": float(np.sqrt(np.mean(np.sum(err_uv[interior] ** 2, axis=1)))) if interior.any() else None,
            })
        per_sample.append(rec)

    # Correlations: does error track material stiffness or load magnitude?
    errs = np.array([r["rel_l2_error"] for r in per_sample])
    Es = np.array([r["mean_E"] for r in per_sample])
    nus = np.array([r["mean_nu"] for r in per_sample])
    loads = np.array([r["mean_load_magnitude"] for r in per_sample])
    corr = {
        "corr_error_vs_E": float(np.corrcoef(errs, Es)[0, 1]),
        "corr_error_vs_nu": float(np.corrcoef(errs, nus)[0, 1]),
        "corr_error_vs_load_magnitude": float(np.corrcoef(errs, loads)[0, 1]),
    }
    if args.geometry == "B2":
        rad = np.array([r["rms_error_radial"] for r in per_sample])
        circ = np.array([r["rms_error_circumferential"] for r in per_sample])
        corr["mean_rms_error_radial"] = float(rad.mean())
        corr["mean_rms_error_circumferential"] = float(circ.mean())
        corr["radial_vs_circumferential_ratio"] = float(rad.mean() / (circ.mean() + 1e-12))
        inner_vals = [r["rms_error_inner_boundary"] for r in per_sample if r["rms_error_inner_boundary"] is not None]
        outer_vals = [r["rms_error_outer_boundary"] for r in per_sample if r["rms_error_outer_boundary"] is not None]
        interior_vals = [r["rms_error_interior"] for r in per_sample if r["rms_error_interior"] is not None]
        corr["mean_rms_error_inner_boundary"] = float(np.mean(inner_vals)) if inner_vals else None
        corr["mean_rms_error_outer_boundary"] = float(np.mean(outer_vals)) if outer_vals else None
        corr["mean_rms_error_interior"] = float(np.mean(interior_vals)) if interior_vals else None
    else:
        top_vals = [r["rms_error_top_boundary"] for r in per_sample if r["rms_error_top_boundary"] is not None]
        bot_vals = [r["rms_error_bottom_boundary"] for r in per_sample if r["rms_error_bottom_boundary"] is not None]
        interior_vals = [r["rms_error_interior"] for r in per_sample if r["rms_error_interior"] is not None]
        corr["mean_rms_error_top_boundary"] = float(np.mean(top_vals)) if top_vals else None
        corr["mean_rms_error_bottom_boundary"] = float(np.mean(bot_vals)) if bot_vals else None
        corr["mean_rms_error_interior"] = float(np.mean(interior_vals)) if interior_vals else None

    worst = sorted(per_sample, key=lambda r: -r["rel_l2_error"])[:args.n_worst]

    report = {
        "geometry": args.geometry, "material": args.material, "checkpoint": args.checkpoint,
        "n_test_samples": len(Test_samples),
        "mean_rel_l2_error": float(errs.mean()), "std_rel_l2_error": float(errs.std()),
        "correlations_and_breakdowns": corr,
        "worst_samples": worst,
        "per_sample": per_sample,
    }
    with open(os.path.join(args.out_dir, "accuracy_diagnostics.json"), "w") as f:
        json.dump(report, f, indent=2)

    print(f"\nMean test rel-L2 error: {errs.mean():.4e} (+/- {errs.std():.4e})")
    print(f"Correlations: {json.dumps(corr, indent=2)}")
    print(f"\nWorst {args.n_worst} samples:")
    for r in worst:
        print(f"  sample_idx={r['sample_idx']} rel_l2={r['rel_l2_error']:.4e} "
              f"E={r['mean_E']:.1f} nu={r['mean_nu']:.3f} load={r['mean_load_magnitude']:.2f}")

    _make_plots(args, per_sample, worst, uv_pred_all, Test_samples, xy)


def _make_plots(args, per_sample, worst, uv_pred_all, Test_samples, xy):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available -- skipping plots (JSON data is still written).")
        return

    errs = np.array([r["rel_l2_error"] for r in per_sample])
    Es = np.array([r["mean_E"] for r in per_sample])
    nus = np.array([r["mean_nu"] for r in per_sample])
    loads = np.array([r["mean_load_magnitude"] for r in per_sample])

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for ax, (x, xlabel) in zip(axes, [(Es, "Mean E"), (nus, "Mean nu"), (loads, "Mean load magnitude")]):
        ax.scatter(x, errs, s=12, alpha=0.6)
        ax.set_xlabel(xlabel); ax.set_ylabel("Per-sample rel-L2 error"); ax.grid(alpha=0.3)
    fig.suptitle(f"{args.geometry} x {args.material}: error vs. material/loading parameters")
    fig.tight_layout()
    fig.savefig(os.path.join(args.out_dir, "error_vs_parameters.png"), dpi=150)
    plt.close(fig)

    # Error contour for the single worst sample (prediction, exact, error)
    worst_idx = worst[0]["sample_idx"]
    pred = uv_pred_all[worst_idx]
    exact = Test_samples[worst_idx]["uv_exact"]
    umag_pred = np.linalg.norm(pred, axis=1)
    umag_exact = np.linalg.norm(exact, axis=1)
    err_mag = np.linalg.norm(pred - exact, axis=1)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for ax, field, title in zip(axes, [umag_pred, umag_exact, err_mag],
                                 ["Prediction |u|", "Exact |u|", f"Error |Δu| (Linf={err_mag.max():.3e})"]):
        sc = ax.scatter(xy[:, 0], xy[:, 1], c=field, cmap="coolwarm", s=10)
        ax.set_title(title); ax.set_aspect("equal")
        fig.colorbar(sc, ax=ax)
    fig.suptitle(f"{args.geometry} x {args.material}: worst test sample (idx={worst_idx}, "
                 f"rel_l2={worst[0]['rel_l2_error']:.3e})")
    fig.tight_layout()
    fig.savefig(os.path.join(args.out_dir, "worst_sample_error_contour.png"), dpi=150)
    plt.close(fig)

    print(f"Plots written to {args.out_dir}/error_vs_parameters.png and "
          f"{args.out_dir}/worst_sample_error_contour.png")


if __name__ == "__main__":
    main()
