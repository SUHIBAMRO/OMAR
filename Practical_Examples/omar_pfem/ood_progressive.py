"""Progressive out-of-distribution shift, one factor at a time.

The advisor's round-6 point 1: "characterize systematically where the 4-5x
deterioration comes from, ideally separating changes in material parameters
and loading and also looking at progressively increasing distribution shifts
rather than only one ID/OOD comparison."

Table 11 reports a single in-distribution/out-of-distribution pair per case,
with material stiffness and loading shifted together by roughly 2-2.5 sigma,
giving degradation factors of 3.94-5.58x. That number cannot say which factor
caused it, or whether the damage accumulates smoothly or has a threshold. This
script answers both by sweeping the shift and by shifting one factor at a time.

Design
------
`ParametricFieldB1/B2` take the distribution's mean and standard deviation as
constructor arguments, so a shift of k sigma is exactly `mean + k * std` --
no separate "OOD dataset" has to exist, and the shift is measured in the
training distribution's own units rather than in absolute stiffness.

Three factors are swept independently:

  material  E_mean shifted by +k*E_std (stiffer)
  loading   the traction/pressure mean shifted by k of its own std, in the
            direction that increases load magnitude -- more negative for B1's
            downward ty_mean = -5, more positive for B2's p_mean = +5
  both      the two together, which is what Table 11 measured

k = 0 is the in-distribution case and is identical for all three factors, so
it is computed once and shared. Poisson's ratio is deliberately NOT swept:
it is clipped to (0.2, 0.4) inside the field, so shifting its mean saturates
against the clip rather than producing a clean k-sigma shift, and any curve
drawn through it would be measuring the clip.

Metric
------
The per-component relative L2 of Tables 5 and 11,
`0.5*(rms(e_u)/rms(u) + rms(e_v)/rms(v))`, NOT the combined vector norm
pareto_analysis.py uses. That is deliberate: the whole point is to be read
against Table 11's degradation factors, so it must be the same measure they
were computed in.

Cost and resumption
-------------------
Every (factor, k) cell needs `--n_samples` fresh FEM solves, which dominate
the run: at N=21 one B1 solve is roughly 9-25 s depending on the machine. The
default grid is 3 factors x 6 non-zero levels + 1 shared zero = 19 cells. Each
cell is written to the output JSON the moment it finishes and skipped on a
re-run, so a Colab disconnect costs at most one cell.

Usage:
  python -m omar_pfem.ood_progressive \
      --geometry B1 --material neo_hookean \
      --checkpoint .../model_best.pt \
      --shifts 0,0.5,1.0,1.5,2.0,2.5,3.0 --n_samples 10 \
      --out_json .../ood_progressive_B1_neo_hookean.json
"""
import os
import json
import time
import argparse

import numpy as np
import torch

from omar_pfem.run_manifest import write_manifest
from omar_pfem.data.parametric_field import ParametricFieldB1, ParametricFieldB2
from omar_pfem.resolution_invariance_zeroshot import (
    build_model, mesh_tensors_of, loss_and_pred)

# The training distribution, as ParametricFieldB1/B2 default to it. Shifts
# below are expressed in these units, so "k sigma" means the same thing on
# both geometries even though their load fields differ in sign and scale.
TRAIN_DIST = {
    "B1": {"E_mean": 1000.0, "E_std": 200.0, "load_mean": -5.0, "load_std": 2.0},
    "B2": {"E_mean": 1000.0, "E_std": 200.0, "load_mean": 5.0, "load_std": 2.0},
}
# direction that makes the load HARDER: B1's ty is negative (downward), so
# more negative is a heavier pull; B2's p is positive, so more positive is a
# higher internal pressure.
LOAD_SIGN = {"B1": -1.0, "B2": +1.0}


def field_kwargs(geometry, factor, k):
    """Constructor kwargs for a k-sigma shift of one factor."""
    d = TRAIN_DIST[geometry]
    kw = {}
    if factor in ("material", "both"):
        kw["E_mean"] = d["E_mean"] + k * d["E_std"]
    if factor in ("loading", "both"):
        load_key = "ty_mean" if geometry == "B1" else "p_mean"
        kw[load_key] = d["load_mean"] + LOAD_SIGN[geometry] * k * d["load_std"]
    return kw


def build_shifted_b1(N, seed, material, kw, Lx=1.0, Ly=1.0):
    """build_sample_b1 with the field distribution shifted. Deliberately a
    separate function rather than an extra argument on build_sample_b1: that
    one is used by the zero-shot study and the Pareto analysis, and its
    defaults must not move under them."""
    from omar_pfem.data.data_generate_B1 import generate_grid_Q4, solve_hyperelastic_TL_spatial

    nodes, elements = generate_grid_Q4(Lx, Ly, N, N)
    E_fn = ParametricFieldB1("E", seed, **kw)
    nu_fn = ParametricFieldB1("nu", seed, **kw)
    ty_fn = ParametricFieldB1("ty", seed, **kw)

    tol = 1e-12
    bottom_nodes = np.where(np.abs(nodes[:, 1]) < tol)[0]
    top_edges = np.array([e for e in elements
                          if abs(nodes[e[2], 1] - Ly) < tol and abs(nodes[e[3], 1] - Ly) < tol])
    node_forces = np.zeros((len(nodes), 2), dtype=np.float32)
    if len(top_edges) > 0:
        top_nodes = np.unique(top_edges[:, 2:4].reshape(-1))
        node_forces[top_nodes, 1] = ty_fn(nodes[top_nodes]).astype(np.float32)

    uv = solve_hyperelastic_TL_spatial(nodes, elements, E_fn, nu_fn, ty_fn, Ly,
                                       nsteps=10, newton_max=30, tol=1e-7,
                                       material=material).astype(np.float32)
    return {"xy": nodes.astype(np.float32), "quad": elements.astype(np.int64),
            "top_edges": (top_edges.astype(np.int64) if len(top_edges)
                          else np.zeros((0, 4), dtype=np.int64)),
            "bottom_nodes": bottom_nodes.astype(np.int64),
            "E_node": E_fn(nodes).astype(np.float32),
            "nu_node": nu_fn(nodes).astype(np.float32),
            "node_forces": node_forces, "uv_exact": uv}


def build_shifted_b2(N, seed, material, kw, R_in=1.0, R_out=2.0):
    from omar_pfem.data.data_generate_B2 import (
        generate_grid_Q4_ring, solve_hyperelastic_TL_ring, assemble_traction_inner_curved)

    nodes, elements = generate_grid_Q4_ring(R_in, R_out, N, N)
    E_fn = ParametricFieldB2("E", seed, **kw)
    nu_fn = ParametricFieldB2("nu", seed, **kw)
    p_fn = ParametricFieldB2("p", seed, **kw)

    r = np.linalg.norm(nodes, axis=1)
    th = np.arctan2(nodes[:, 1], nodes[:, 0])
    tol = 1e-9
    theta0_nodes = np.where(np.abs(th) < tol)[0]
    thalf_nodes = np.where(np.abs(th - np.pi / 2) < tol)[0]
    inner_edges = np.array([e for e in elements
                            if abs(r[e[0]] - R_in) < 1e-6 and abs(r[e[1]] - R_in) < 1e-6])

    # the FEM-consistent nodal force, the same assembler the dataset uses --
    # NOT pressure times normal, which is mesh-dependent (see the B2 force
    # bug fixed on 2026-08-27)
    node_forces = assemble_traction_inner_curved(
        nodes, elements, R_in, p_fn).reshape(-1, 2).astype(np.float32)

    uv = solve_hyperelastic_TL_ring(nodes, elements, E_fn, nu_fn, p_fn, R_in,
                                    nsteps=10, newton_max=30, tol=1e-7,
                                    material=material).astype(np.float32)
    pts = np.stack([th, r], axis=1)
    return {"xy": nodes.astype(np.float32), "quad": elements.astype(np.int64),
            "inner_edges": (inner_edges.astype(np.int64) if len(inner_edges)
                            else np.zeros((0, 4), dtype=np.int64)),
            "theta0_nodes": theta0_nodes.astype(np.int64),
            "thetahalfpi_nodes": thalf_nodes.astype(np.int64),
            "E_node": E_fn(pts).astype(np.float32),
            "nu_node": nu_fn(pts).astype(np.float32),
            "node_forces": node_forces, "uv_exact": uv}


def per_component_rel_l2(pred, ref):
    """Tables 5 and 11's definition, so these numbers can be read against
    their degradation factors."""
    rms = lambda a: float(np.sqrt(np.mean(a ** 2)))
    e_u = rms(pred[:, 0] - ref[:, 0]) / (rms(ref[:, 0]) + 1e-12)
    e_v = rms(pred[:, 1] - ref[:, 1]) / (rms(ref[:, 1]) + 1e-12)
    return 0.5 * (e_u + e_v)


def main():
    p = argparse.ArgumentParser("Progressive OOD shift, one factor at a time")
    p.add_argument("--geometry", required=True, choices=["B1", "B2"])
    p.add_argument("--material", required=True,
                   choices=["neo_hookean", "mooney_rivlin", "arruda_boyce"])
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--N", type=int, default=21, help="the study's own mesh")
    p.add_argument("--shifts", default="0,0.5,1.0,1.5,2.0,2.5,3.0",
                   help="shift magnitudes in units of the training std")
    p.add_argument("--factors", default="material,loading,both")
    p.add_argument("--n_samples", type=int, default=10)
    p.add_argument("--seed_base", type=int, default=77_000_000,
                   help="kept disjoint from training (0-), zero-shot "
                        "(20,000,000-) and Pareto (900,000-) seed ranges")
    p.add_argument("--out_json", default=None)
    p.add_argument("--cpu", action="store_true")
    # model hyperparameters, the zero-shot study's own defaults
    p.add_argument("--model", default="Transolver_Irregular_Mesh")
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
    p.add_argument("--mode", default="plane_strain")
    p.add_argument("--Ly", type=float, default=1.0)
    p.add_argument("--R_out", type=float, default=2.0)
    args = p.parse_args()
    started = time.time()

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    dtype = torch.float32
    if args.out_json is None:
        args.out_json = f"ood_progressive_{args.geometry}_{args.material}.json"
    os.makedirs(os.path.dirname(os.path.abspath(args.out_json)) or ".", exist_ok=True)

    shifts = [float(s) for s in args.shifts.split(",") if s.strip()]
    factors = [f.strip() for f in args.factors.split(",") if f.strip()]
    build = build_shifted_b1 if args.geometry == "B1" else build_shifted_b2

    model = build_model(args, device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Device: {device}, N={args.N}, {args.n_samples} samples per cell")
    print(f"Metric: per-component relative L2 (Tables 5/11's definition)\n")

    rows = []
    if os.path.exists(args.out_json):
        try:
            rows = json.load(open(args.out_json)).get("rows", [])
            if rows:
                print(f"[resume] {len(rows)} cells already done\n")
        except Exception:
            rows = []
    done = {(r["factor"], r["shift_sigma"]) for r in rows}

    def save():
        rep = {"geometry": args.geometry, "material": args.material,
               "checkpoint": args.checkpoint, "N": args.N,
               "n_samples": args.n_samples, "seed_base": args.seed_base,
               "metric": "per-component relative L2, 0.5*(rms(e_u)/rms(u)+rms(e_v)/rms(v))",
               "train_distribution": TRAIN_DIST[args.geometry],
               "device": device.type, "rows": sorted(
                   rows, key=lambda r: (r["factor"], r["shift_sigma"]))}
        tmp = args.out_json + ".tmp"
        with open(tmp, "w") as f:
            json.dump(rep, f, indent=2)
        os.replace(tmp, args.out_json)
        return rep

    def run_cell(factor, k):
        kw = field_kwargs(args.geometry, factor, k)
        errs = []
        for i in range(args.n_samples):
            s = build(args.N, args.seed_base + i, args.material, kw)
            mesh_t = mesh_tensors_of(args.geometry, s, device, dtype)
            E = torch.tensor(s["E_node"][None], device=device, dtype=dtype)
            nu = torch.tensor(s["nu_node"][None], device=device, dtype=dtype)
            f = torch.tensor(s["node_forces"][None], device=device, dtype=dtype)
            with torch.no_grad():
                _, _, _, uv, _ = loss_and_pred(
                    args.geometry, mesh_t, model, E, nu, f, args, dtype)
            errs.append(per_component_rel_l2(uv[0].cpu().numpy(), s["uv_exact"]))
        return {"factor": factor, "shift_sigma": k,
                "field_kwargs": {kk: float(vv) for kk, vv in kw.items()},
                "n_samples": args.n_samples,
                "mean_rel_L2": float(np.mean(errs)),
                "std_rel_L2": float(np.std(errs)),
                "max_rel_L2": float(np.max(errs))}

    # k = 0 is the same problem for every factor -- compute it once, then
    # copy it into each factor's curve so every curve starts from the same
    # in-distribution point rather than from its own noisy estimate.
    baseline = next((r for r in rows if r["shift_sigma"] == 0.0), None)
    if 0.0 in shifts and baseline is None:
        print("[k=0] in-distribution baseline (shared by all factors)")
        baseline = run_cell(factors[0], 0.0)
        baseline["factor"] = "baseline"
        rows.append(baseline)
        save()
        print(f"       {baseline['mean_rel_L2']:.4f}\n")
    if baseline is not None:
        print(f"in-distribution baseline: {baseline['mean_rel_L2']:.4f}\n")

    for factor in factors:
        for k in shifts:
            if k == 0.0 or (factor, k) in done:
                continue
            t0 = time.time()
            row = run_cell(factor, k)
            if baseline is not None:
                row["degradation_vs_baseline"] = (
                    row["mean_rel_L2"] / baseline["mean_rel_L2"])
            rows.append(row)
            save()
            deg = row.get("degradation_vs_baseline")
            print(f"  {factor:<9} k={k:>4}sigma  err={row['mean_rel_L2']:.4f}"
                  + (f"  ({deg:.2f}x)" if deg else "")
                  + f"   [{time.time() - t0:.0f}s]")

    report = save()
    print("\n" + "=" * 62)
    print(f"{'factor':<11}{'k':>6}{'mean rel L2':>14}{'degradation':>14}")
    for r in report["rows"]:
        d = r.get("degradation_vs_baseline")
        print(f"{r['factor']:<11}{r['shift_sigma']:>6}{r['mean_rel_L2']:>14.4f}"
              + (f"{d:>13.2f}x" if d else f"{'--':>14}"))
    print("=" * 62)
    print(f"Written to {args.out_json}")

    write_manifest(
        os.path.dirname(os.path.abspath(args.out_json)) or ".",
        kind="ood_progressive", args=args, started_at=started, results=report,
        outputs=[args.out_json],
        notes=("Advisor round-6 point 1: separate material from loading and "
               "sweep the shift progressively instead of one ID/OOD pair. "
               "Shift is in units of the training distribution's own std, so "
               "k=2-2.5 is where Table 11's single measurement sits. Poisson's "
               "ratio is not swept because the field clips it to (0.2, 0.4), so "
               "a shifted mean saturates against the clip. Metric is Tables "
               "5/11's per-component relative L2, so the degradation column is "
               "directly comparable to Table 11's factors."))


if __name__ == "__main__":
    main()
