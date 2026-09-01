"""Re-score an already-trained MMS operator checkpoint per test-family
member, instead of only the family mean `mms_operator.py` keeps.

Why this needed writing: `mms_operator.py`'s own `evaluate()` computes one
error dict per test member internally and then collapses them to a mean
before returning -- the per-member numbers were never kept. This script does
not retrain anything; it rebuilds the exact same mesh, test family and
normalization `mms_operator.py --N <N>` built (same seeds, same defaults),
loads the checkpoint `model_best.pt` that run already wrote, and runs the 16
test members through the model one more time, keeping every member's row.

What this answers, which the family mean cannot: PROJECT_STATUS.md's open
question under point 9 -- "is the operator consistent across the family, or
merely consistent on average?" -- via each metric's stdev/mean across the
16 members, next to Q4's own (already known to be negligible, per
mms_family_fem_B1_neo_hookean.json's Q4_spread_stdev_over_mean).

Usage
-----
    python -m omar_pfem.mms_operator_per_member --N 17 \
        --checkpoint /path/to/operator_N17/model_best.pt \
        --out_json mms_operator_per_member_N17.json

Cost: no training, just 16 forward passes plus the mesh/dataset build that
mms_operator.py itself measured at a few seconds for N=9/17 and under a
minute for N=33 -- this is the inference-only fraction of that, so it is
faster still.
"""
import argparse
import json
import os
import time

import numpy as np
import torch

from omar_pfem.model_dict import get_model
from omar_pfem.materials_torch import get_material_fns
from omar_pfem.mms_study import build_mesh, compute_errors, DEFAULT_ALPHA, DEFAULT_BETA
from omar_pfem.mms_operator import (
    sample_family, build_dataset, dirichlet_mask, predict, ALPHA_RANGE, BETA_RANGE)


def main():
    p = argparse.ArgumentParser(
        "Per-member re-score of an existing MMS operator checkpoint")
    p.add_argument("--material", default="neo_hookean",
                   choices=["neo_hookean", "mooney_rivlin", "arruda_boyce"])
    p.add_argument("--N", type=int, required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--ntrain", type=int, default=64,
                   help="must match the training run -- only used to "
                        "reproduce its input normalization statistics, "
                        "no training happens here")
    p.add_argument("--ntest", type=int, default=16)
    p.add_argument("--seed", type=int, default=31_000_000)
    p.add_argument("--out_json", default=None)
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--model", default="Transolver_Irregular_Mesh")
    p.add_argument("--n_hidden", type=int, default=256)
    p.add_argument("--n_layers", type=int, default=4)
    p.add_argument("--n_heads", type=int, default=8)
    p.add_argument("--mlp_ratio", type=int, default=2)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--unified_pos", type=int, default=0)
    p.add_argument("--ref", type=int, default=16)
    p.add_argument("--slice_num", type=int, default=128)
    args = p.parse_args()

    assert os.path.exists(args.checkpoint), (
        f"checkpoint not found: {args.checkpoint}")
    args.fun_dim = 2
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    dtype = torch.float32
    torch.manual_seed(args.seed)
    np.random.seed(args.seed % (2 ** 32))

    # ---- rebuild exactly what mms_operator.py --N <N> built -------------
    nodes, elements = build_mesh("Q4", args.N)
    n_el = len(elements)
    _, E_nu_to_params = get_material_fns(args.material)
    E, nu = 1000.0, 0.3
    if args.material == "neo_hookean":
        params = E_nu_to_params(torch.tensor(E, dtype=torch.float64),
                                torch.tensor(nu, dtype=torch.float64),
                                mode="plane_strain")
    else:
        params = E_nu_to_params(torch.tensor(E, dtype=torch.float64),
                                torch.tensor(nu, dtype=torch.float64))
    mu_e = torch.full((n_el,), float(params[0]), dtype=torch.float64)
    lam_e = torch.full((n_el,), float(params[1]), dtype=torch.float64)

    print(f"[N={args.N}] {len(nodes)} nodes, {2 * len(nodes)} DOF, device {device}")

    # SAME seed, SAME ntrain/ntest, SAME order of calls as mms_operator.py's
    # main() (train_p first, then test_p) -- required for sample_family to
    # draw the identical 16 test members and for Ftr's statistics to match
    # the checkpoint's own normalization exactly.
    train_p = sample_family(args.ntrain, args.seed)
    test_p = sample_family(args.ntest, args.seed + 1)
    t0 = time.time()
    Ftr, _ = build_dataset(train_p, nodes, elements, "Q4", mu_e, lam_e, args.material)
    Fte, _ = build_dataset(test_p, nodes, elements, "Q4", mu_e, lam_e, args.material)
    print(f"dataset rebuilt in {time.time() - t0:.1f}s (must match the "
          f"training run's own family, not merely have the same size)")

    norm = {"mean": Ftr.mean(dim=(0, 1)), "std": Ftr.std(dim=(0, 1))}
    assert (norm["std"] > 0).all()

    xy_t = torch.tensor(nodes, dtype=dtype, device=device)
    mask = dirichlet_mask(nodes, torch.float64).to(device=device, dtype=dtype)
    normd = {k: v.to(device=device, dtype=dtype) for k, v in norm.items()}
    Fte_d = Fte.to(device, dtype)

    model = get_model(args).Model(
        space_dim=2, n_layers=args.n_layers, n_hidden=args.n_hidden,
        dropout=args.dropout, n_head=args.n_heads, Time_Input=False,
        mlp_ratio=args.mlp_ratio, fun_dim=args.fun_dim, out_dim=2,
        slice_num=args.slice_num, ref=args.ref, unified_pos=args.unified_pos,
    ).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()

    # ---- the one thing mms_operator.py's evaluate() throws away ---------
    per_member = []
    with torch.no_grad():
        for i in range(Fte_d.shape[0]):
            fb = Fte_d[i:i + 1]
            uv = predict(model, xy_t, fb, normd, mask)
            alpha, beta = test_p[i]
            e = compute_errors(nodes, elements, "Q4",
                               uv[0].double().cpu().numpy(), mu_e, lam_e,
                               args.material, alpha, beta)
            per_member.append({"alpha": alpha, "beta": beta,
                               **{k: float(e[k]) for k in
                                  ("L2_rel", "H1_semi_rel", "stress_rel_L2",
                                   "energy_rel")}})

    METRICS = ("L2_rel", "H1_semi_rel", "stress_rel_L2", "energy_rel")
    summary = {}
    for m in METRICS:
        vals = np.array([r[m] for r in per_member])
        summary[m] = {"mean": float(vals.mean()), "std": float(vals.std()),
                      "std_over_mean": float(vals.std() / vals.mean()),
                      "min": float(vals.min()), "max": float(vals.max())}

    print(f"\n{'metric':<16}{'mean':>12}{'std':>12}{'std/mean':>12}{'min':>12}{'max':>12}")
    for m in METRICS:
        s = summary[m]
        print(f"{m:<16}{s['mean']:>12.4e}{s['std']:>12.4e}"
              f"{s['std_over_mean']:>12.4f}{s['min']:>12.4e}{s['max']:>12.4e}")

    report = {
        "study": "MMS operator, per-test-family-member re-score (no retraining)",
        "material": args.material, "N": args.N, "n_dof": 2 * len(nodes),
        "checkpoint": args.checkpoint, "seed": args.seed,
        "family": {"alpha_range": list(ALPHA_RANGE), "beta_range": list(BETA_RANGE),
                   "ntrain": args.ntrain, "ntest": args.ntest},
        "per_member": per_member,
        "summary": summary,
        "note": ("std_over_mean here is the operator's own consistency across "
                 "the SAME 16-member family mms_family_fem_B1_neo_hookean.json "
                 "scores Q4 and Q9 on -- compare against that file's "
                 "Q4_spread_stdev_over_mean directly, same quantity, same "
                 "family, same N."),
    }
    out_json = args.out_json or f"mms_operator_per_member_N{args.N}.json"
    with open(out_json, "w") as fh:
        json.dump(report, fh, indent=2)
    print(f"\nWritten to {out_json}")


if __name__ == "__main__":
    main()
