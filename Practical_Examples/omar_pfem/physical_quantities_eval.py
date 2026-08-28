"""
Error in physically important quantities beyond displacement, per the
advisor's request:

  "examine error in physically important quantities beyond displacement,
   including error in H1 semi-norm, energy and also some local quantities
   such as stress components and reaction forces; maybe looking at maxima.
   I am referring to the transolver."

Every neural-operator number reported elsewhere in this codebase is a nodal
DISPLACEMENT error (a per-node RMS relative error, see train_B1.py's
`evaluate_dataset_hyperelastic_Q4`). That is the most forgiving quantity an
engineer could ask about: displacement is the primary unknown and the thing
the network is trained to produce, and it is smoother than its own
derivatives. Stress depends on grad(u) and so loses an order of accuracy;
reaction forces are an integral of that stress over the supports. A 9%
displacement error does not imply a 9% stress error -- quantifying that gap
is the entire point of this script.

Computed per held-out sample, for an already-trained checkpoint:

  1. Displacement relative L2, using the same definition as the rest of the
     report, so every other number below can be read against a familiar
     baseline.

  2. H1 semi-norm relative error -- the error in grad(u), integrated with the
     mesh's own Gauss quadrature.

  3. Tangent/incremental energy-norm relative error, ||e||_E =
     sqrt(e^T K(u_ref) e), induced by the reference solution's own tangent
     stiffness. This is the advisor-confirmed definition ("the
     tangent/incremental energy norm; relative errors are enough"), NOT a
     difference of two scalar total-energy values.

  (2) and (3) reuse `compute_l2_h1_errors_cross_order` and
  `compute_tangent_energy_error` from high_dof_convergence_study.py rather
  than reimplementing them, so the neural operator is scored with exactly the
  norms already used for the Q4-vs-Q9 finite-element study. Those take
  "already-solved field" dicts, so a prediction is simply packaged into the
  same dict shape an FE solution uses on the same mesh.

  4. First Piola-Kirchhoff stress components. P = dW/dF is obtained by
     autodiff of the same strain-energy density the solver and the training
     objective use (materials_torch), at F = I + grad(u) on the mesh's own
     Gauss points, with grad(u) built from the mesh's own shape-function
     gradients (dN_dX, returned by gauss_points_and_weights_physical) -- exact,
     not a finite difference of an interpolant. Autodiff rather than a
     hand-derived formula is what lets one code path cover all three
     materials: only Neo-Hookean has a closed-form PK1 in data/materials.py.
     Reported per component (P11, P12, P21, P22) as a quadrature-weighted
     relative L2, plus maximum pointwise error, since maxima were asked for.

     CAUTION when reading the per-component numbers: in both benchmarks the
     shear components P12/P21 are near zero almost everywhere, so their
     RELATIVE error divides by a tiny reference norm and is large even when
     the absolute error is negligible. For those two components the
     `*_max_abs_err` entries are the meaningful ones; the aggregate
     `P_rel_L2` (Frobenius norm of the full tensor) is not affected by this
     and is the right single number to quote for stress accuracy.

  5. Reaction forces on the constrained boundary. The internal force is
     assembled from the same Gauss-point stresses,
       R_ai = sum_g w_g detJ_g sum_j P[g,i,j] dN_dX[g,a,j],
     and restricted to the constrained nodes. On those nodes the external
     traction is zero for both benchmarks (B1 is loaded on the top edge, B2 on
     the inner arc), so the internal force there IS the reaction the supports
     must supply. Reported as the error in the total reaction resultant, in
     the nodal reaction magnitudes, and in the single largest nodal reaction.

Constrained nodes differ by geometry and are read from the dataset rather than
recomputed: B1 stores `bottom_nodes` (fully fixed); B2 is constrained by
symmetry on its two radial edges, where only the normal component is fixed, so
its reaction is reported on that component only.

Usage:
  python -m omar_pfem.physical_quantities_eval \
      --geometry B1 --material neo_hookean \
      --checkpoint .../model_best.pt --data_path .../hyperelastic_training_data_q4.npz \
      --ntrain 800 --ntest 50 --out_json .../physical_quantities_B1_neo_hookean.json
"""
import os
import time
import json
import argparse
import inspect

import numpy as np
import torch

from omar_pfem.model_dict import get_model
from omar_pfem.run_manifest import write_manifest
from omar_pfem.materials_torch import get_material_fns as get_material_fns_torch
from omar_pfem.high_dof_convergence_study import (
    compute_l2_h1_errors_cross_order,
    compute_tangent_energy_error,
    gauss_points_and_weights_physical,
)


def build_model(args, device):
    return get_model(args).Model(
        space_dim=2, n_layers=args.n_layers, n_hidden=args.n_hidden,
        dropout=args.dropout, n_head=args.n_heads, Time_Input=False,
        mlp_ratio=args.mlp_ratio, fun_dim=args.fun_dim, out_dim=2,
        slice_num=args.slice_num, ref=args.ref, unified_pos=args.unified_pos,
    ).to(device)


def as_solved_field(nodes, elements, u):
    """Package a displacement field into the dict shape `solve_one` returns,
    so the existing norm routines accept it unchanged."""
    return {"nodes": np.asarray(nodes), "elements": np.asarray(elements),
            "u": np.asarray(u), "N": int(round(np.sqrt(len(nodes))))}


def _material_params(material, E_gp, nu_gp, mode):
    energy_fn, to_params = get_material_fns_torch(material)
    kw = {}
    sig = inspect.signature(to_params).parameters
    if "mode" in sig:
        kw["mode"] = mode
    return energy_fn, to_params(E_gp, nu_gp, **kw)


def gauss_quantities(nodes, elements, u, E_node, nu_node, material, mode,
                     order, device, dtype):
    """PK1 stress and the assembled internal force, both at/from the mesh's own
    Gauss points. Returns (P, w_detJ, R_nodal)."""
    pts, detJ, w, N_sf, dN_dX, elem_idx = gauss_points_and_weights_physical(
        np.asarray(nodes), np.asarray(elements), order)

    elems = np.asarray(elements)[elem_idx]                   # (n_gp, nodes_per_elem)
    u_e = np.asarray(u)[elems]                               # (n_gp, npe, 2)
    dN = torch.tensor(dN_dX, dtype=dtype, device=device)     # (n_gp, npe, 2)
    u_et = torch.tensor(u_e, dtype=dtype, device=device)

    grad_u = torch.einsum("gai,gaj->gij", u_et, dN)          # (n_gp, 2, 2)
    F = torch.eye(2, dtype=dtype, device=device).expand_as(grad_u) + grad_u
    F = F.clone().requires_grad_(True)

    E_gp = torch.tensor(np.einsum("ga,ga->g", N_sf, np.asarray(E_node)[elems]),
                        dtype=dtype, device=device)
    nu_gp = torch.tensor(np.einsum("ga,ga->g", N_sf, np.asarray(nu_node)[elems]),
                         dtype=dtype, device=device)
    energy_fn, params = _material_params(material, E_gp, nu_gp, mode)

    W = energy_fn(F, *params, dtype=dtype).sum()
    P, = torch.autograd.grad(W, F)

    wdet = torch.tensor(np.asarray(w) * np.asarray(detJ), dtype=dtype, device=device)
    # internal force: R_ai = sum_g w_g detJ_g sum_j P[g,i,j] dN_dX[g,a,j]
    f_e = torch.einsum("g,gij,gaj->gai", wdet, P.detach(), dN)
    R = torch.zeros(len(nodes), 2, dtype=dtype, device=device)
    idx = torch.tensor(elems, dtype=torch.long, device=device)
    R.index_add_(0, idx.reshape(-1), f_e.reshape(-1, 2))
    return P.detach(), wdet, R


def stress_errors(P_pred, P_ref, w):
    P_pred = P_pred.cpu().numpy(); P_ref = P_ref.cpu().numpy(); w = w.cpu().numpy()
    out = {}
    for nm, i, j in [("P11", 0, 0), ("P12", 0, 1), ("P21", 1, 0), ("P22", 1, 1)]:
        a, b = P_pred[:, i, j], P_ref[:, i, j]
        out[f"{nm}_rel_L2"] = float(np.sqrt(np.sum(w * (a - b) ** 2))
                                    / (np.sqrt(np.sum(w * b ** 2)) + 1e-30))
        out[f"{nm}_max_abs_err"] = float(np.max(np.abs(a - b)))
    fro_err = np.sqrt(np.sum((P_pred - P_ref) ** 2, axis=(1, 2)))
    fro_ref = np.sqrt(np.sum(P_ref ** 2, axis=(1, 2)))
    out["P_rel_L2"] = float(np.sqrt(np.sum(w * fro_err ** 2))
                            / (np.sqrt(np.sum(w * fro_ref ** 2)) + 1e-30))
    out["P_max_pointwise_rel"] = float(np.max(fro_err / (fro_ref + 1e-30)))
    # peak stress is a design quantity in its own right: compare the maxima
    out["P_peak_pred"] = float(np.max(fro_ref * 0 + np.sqrt(np.sum(P_pred ** 2, axis=(1, 2)))))
    out["P_peak_ref"] = float(np.max(fro_ref))
    out["P_peak_rel_err"] = float(abs(out["P_peak_pred"] - out["P_peak_ref"])
                                  / (out["P_peak_ref"] + 1e-30))
    return out


def reaction_errors(R_pred, R_ref, nodes_idx, components):
    """components: which DOF components are actually constrained (B1: both;
    B2: symmetry fixes one component per edge)."""
    a = R_pred[nodes_idx][:, components]
    b = R_ref[nodes_idx][:, components]
    tot_a, tot_b = a.sum(0), b.sum(0)
    mag_a = torch.linalg.norm(a, dim=1) if a.dim() > 1 else a.abs()
    mag_b = torch.linalg.norm(b, dim=1) if b.dim() > 1 else b.abs()
    return {
        "reaction_resultant_rel_err": float(torch.linalg.norm(tot_a - tot_b)
                                            / (torch.linalg.norm(tot_b) + 1e-30)),
        "reaction_nodal_rel_L2": float(torch.linalg.norm(mag_a - mag_b)
                                       / (torch.linalg.norm(mag_b) + 1e-30)),
        "reaction_max_pred": float(mag_a.max()),
        "reaction_max_ref": float(mag_b.max()),
        "reaction_max_rel_err": float(abs(mag_a.max() - mag_b.max())
                                      / (mag_b.max() + 1e-30)),
    }


def summarize(per_sample):
    out = {}
    for k in per_sample[0]:
        v = np.array([s[k] for s in per_sample], dtype=float)
        out[k] = {"mean": float(v.mean()), "std": float(v.std()),
                  "max": float(v.max()), "min": float(v.min())}
    return out


def main():
    run_started_at = time.time()
    p = argparse.ArgumentParser(
        "Neural-operator error in H1 semi-norm, energy norm, stress and reactions")
    p.add_argument("--geometry", required=True, choices=["B1", "B2"])
    p.add_argument("--material", required=True,
                   choices=["neo_hookean", "mooney_rivlin", "arruda_boyce"])
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--data_path", required=True)
    p.add_argument("--ntrain", type=int, default=800)
    p.add_argument("--ntest", type=int, default=50)
    p.add_argument("--out_json", type=str, default=None)
    p.add_argument("--model", type=str, default="Transolver_Irregular_Mesh")
    p.add_argument("--n_hidden", type=int, default=256)
    p.add_argument("--n_layers", type=int, default=4)
    p.add_argument("--n_heads", type=int, default=8)
    p.add_argument("--mlp_ratio", type=int, default=2)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--unified_pos", type=int, default=0)
    p.add_argument("--ref", type=int, default=16)
    p.add_argument("--slice_num", type=int, default=128)
    p.add_argument("--fun_dim", type=int, default=4)
    p.add_argument("--use_soft_dirichlet", type=int, default=1)
    p.add_argument("--mode", type=str, default="plane_strain")
    p.add_argument("--Lx", type=float, default=1.0)
    p.add_argument("--Ly", type=float, default=1.0)
    p.add_argument("--R_out", type=float, default=2.0)
    p.add_argument("--cpu", action="store_true")
    args = p.parse_args()

    if args.out_json is None:
        args.out_json = f"physical_quantities_{args.geometry}_{args.material}.json"
        print(f"[auto-save] --out_json not given; writing to {args.out_json}")

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    # Derivative quantities (grad u, stress, reactions) are far more sensitive to
    # round-off than displacement is, so these are evaluated in FP64 even though
    # the network itself was trained in FP32.
    dtype = torch.float64

    # train_B1 and train_B2 expose the SAME function names in different
    # modules (each specialised to its own geometry's boundary handling), so
    # the geometry selects the module, not the symbol.
    if args.geometry == "B1":
        from omar_pfem.train_B1 import (
            load_fem_dataset_Q4_with_materials_and_random_force as load_ds,
            total_potential_energy_Q4_hyperelastic as energy_pred)
    else:
        from omar_pfem.train_B2 import (
            load_fem_dataset_Q4_with_materials_and_random_force as load_ds,
            total_potential_energy_Q4_hyperelastic as energy_pred)

    _, test = load_ds(args.data_path, args.ntrain, args.ntest)
    print(f"Evaluating {len(test)} held-out samples from {args.data_path}")

    model = build_model(args, device).to(torch.float32)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()
    print(f"Loaded checkpoint: {args.checkpoint}")

    s0 = test[0]
    xy = torch.tensor(s0["xy"], device=device, dtype=torch.float32)
    quad = torch.tensor(s0["quad"], device=device, dtype=torch.long)
    nodes_np, elems_np = np.asarray(s0["xy"]), np.asarray(s0["quad"])

    # Geometry-specific boundary arguments and constrained DOFs. B1 fixes both
    # components on the bottom edge; B2's two radial edges are symmetry planes,
    # so only the component normal to each plane is constrained -- u_y on the
    # theta=0 edge (which lies along the x-axis) and u_x on the theta=pi/2 edge
    # (along the y-axis), matching train_B2's own `free_v[theta0_nodes] = 0` /
    # `free_u[thetahalfpi_nodes] = 0`. The reaction is meaningful only on that
    # component: a full vector magnitude would mix in the free tangential
    # component, which carries no support force.
    if args.geometry == "B1":
        bnd = (torch.tensor(s0["top_edges"], device=device, dtype=torch.long),
               torch.tensor(s0["bottom_nodes"], device=device, dtype=torch.long))
        geo_kw = {"Ly": args.Ly}
        constrained = [(np.asarray(s0["bottom_nodes"]), [0, 1])]
    else:
        bnd = (torch.tensor(s0["inner_edges"], device=device, dtype=torch.long),
               torch.tensor(s0["theta0_nodes"], device=device, dtype=torch.long),
               torch.tensor(s0["thetahalfpi_nodes"], device=device, dtype=torch.long))
        geo_kw = {"R_out": args.R_out}
        constrained = [(np.asarray(s0["theta0_nodes"]), [1]),
                       (np.asarray(s0["thetahalfpi_nodes"]), [0])]

    geom_kwargs = ({"Lx": args.Lx, "Ly": args.Ly} if args.geometry == "B1"
                   else {"R_out": args.R_out})

    per_sample = []
    for i, s in enumerate(test):
        E_b = torch.tensor(s["E_node"][None], device=device, dtype=torch.float32)
        nu_b = torch.tensor(s["nu_node"][None], device=device, dtype=torch.float32)
        f_b = torch.tensor(s["node_forces"][None], device=device, dtype=torch.float32)
        with torch.no_grad():
            _, _, _, uv_pred, _ = energy_pred(
                xy, quad, *bnd, model, E_b, nu_b, f_b,
                use_soft_dirichlet=bool(args.use_soft_dirichlet),
                mode=args.mode, dtype=torch.float32,
                fun_dim=args.fun_dim, material=args.material, **geo_kw)
        u_pred = uv_pred[0].double().cpu().numpy()
        u_ref = np.asarray(s["uv_exact"], dtype=np.float64)

        # 1. displacement relative L2, same definition as the rest of the report
        rms = lambda a: np.sqrt(np.mean(a ** 2))
        e_u = rms(u_pred[:, 0] - u_ref[:, 0]) / (rms(u_ref[:, 0]) + 1e-12)
        e_v = rms(u_pred[:, 1] - u_ref[:, 1]) / (rms(u_ref[:, 1]) + 1e-12)
        rec = {"disp_rel_L2": 0.5 * (e_u + e_v)}

        # 2-3. H1 semi-norm and tangent-energy norm, via the existing routines
        fp = as_solved_field(nodes_np, elems_np, u_pred)
        fr = as_solved_field(nodes_np, elems_np, u_ref)
        h1 = compute_l2_h1_errors_cross_order(fp, fr, "Q4", "Q4", args.geometry, **geom_kwargs)
        rec["L2_rel"] = float(h1["l2_rel"])
        rec["H1_semi_rel"] = float(h1["h1_semi_rel"])
        en = compute_tangent_energy_error(fp, fr, "Q4", "Q4", args.geometry, args.material,
                                          device, dtype, **geom_kwargs)
        rec["energy_rel"] = float(en["tangent_energy_rel"])

        # 4-5. stress and reactions from the same Gauss-point machinery
        P_p, w, R_p = gauss_quantities(nodes_np, elems_np, u_pred, s["E_node"], s["nu_node"],
                                       args.material, args.mode, "Q4", device, dtype)
        P_r, _, R_r = gauss_quantities(nodes_np, elems_np, u_ref, s["E_node"], s["nu_node"],
                                       args.material, args.mode, "Q4", device, dtype)
        rec.update(stress_errors(P_p, P_r, w))
        for k, (idx, comps) in enumerate(constrained):
            tag = "" if len(constrained) == 1 else f"_edge{k}"
            for kk, vv in reaction_errors(R_p, R_r, idx, comps).items():
                rec[kk + tag] = vv

        per_sample.append(rec)
        if (i + 1) % 10 == 0 or i == len(test) - 1:
            print(f"  {i + 1}/{len(test)} samples")

    report = {"geometry": args.geometry, "material": args.material,
              "checkpoint": args.checkpoint, "data_path": args.data_path,
              "ntrain_skipped": args.ntrain, "ntest": len(test),
              "metrics": summarize(per_sample)}

    m = report["metrics"]
    print("\n" + "=" * 78)
    print(f"PHYSICAL-QUANTITY ERRORS  ({args.geometry} x {args.material}, {len(test)} samples)")
    print("=" * 78)
    print(f"{'quantity':<34}{'mean':>13}{'max over samples':>18}")
    for key in ["disp_rel_L2", "L2_rel", "H1_semi_rel", "energy_rel",
                "P_rel_L2", "P_peak_rel_err", "P11_rel_L2", "P22_rel_L2",
                "P12_rel_L2", "P21_rel_L2"]:
        if key in m:
            print(f"{key:<34}{m[key]['mean']:>13.4e}{m[key]['max']:>18.4e}")
    for key in sorted(k for k in m if k.startswith("reaction")):
        print(f"{key:<34}{m[key]['mean']:>13.4e}{m[key]['max']:>18.4e}")
    print("=" * 78)

    with open(args.out_json, "w") as f:
        json.dump(report, f, indent=2)
    print(f"Report written to {args.out_json}")

    write_manifest(
        os.path.dirname(os.path.abspath(args.out_json)) or ".",
        kind="physical_quantities", args=args, started_at=run_started_at,
        results={"geometry": args.geometry, "material": args.material,
                 "checkpoint": args.checkpoint, "ntest": len(test),
                 "metrics": report["metrics"]},
        outputs=[args.out_json],
        notes=("Advisor request: report the error in quantities beyond displacement L2 -- "
               "H1 semi-norm, tangent (incremental) energy norm, PK1 stress components and "
               "reaction forces -- on held-out samples. Stress is obtained by autodiff "
               "(P = dW/dF) at the Gauss points, so it is exact for all three materials "
               "rather than relying on a closed form only Neo-Hookean has. Reactions are "
               "the assembled internal forces summed over the constrained nodes. Errors "
               "are relative to the SAME-mesh FEM solution stored in the dataset."))


if __name__ == "__main__":
    main()
