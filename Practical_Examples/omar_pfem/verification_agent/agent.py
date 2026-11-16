"""Verification agent: deterministic core plus an optional LLM layer.

Tasks, all writing to the solved-case repository:

  mms       solve a manufactured-solution case on a mesh sequence and
            PASS/FAIL it on observed vs. theoretical convergence rates.
  replay-b3 judge the already-computed B3 groove-region QoIs (quadratic
            reference) with the QoI-convergence and strategy-robustness
            gates, deciding which QoIs may serve as validation targets.
  b1-ladder solve B1 on N=11..81 and gate global/pointwise/peak/average QoIs.
  orchestrate  LLM formalizer/proposer/critic/advisor (llm_roles.py) around
            the same gates; the LLM never sets a verdict (orchestrate.py).

    python -m omar_pfem.verification_agent.agent mms --orders Q4,Q9 --Ns 5,9,17 --cpu
    python -m omar_pfem.verification_agent.agent replay-b3
"""
import argparse
import json
import os

import torch

from omar_pfem.verification_agent import gates, repository
from omar_pfem.verification_agent.spec import ProblemSpec

EXPECTED_RATES = {
    "Q4": {"L2_rel": 2, "H1_semi_rel": 1, "stress_rel_L2": 1, "energy_norm_rel": 1},
    "Q9": {"L2_rel": 3, "H1_semi_rel": 2, "stress_rel_L2": 2, "energy_norm_rel": 2},
}

HERE = os.path.dirname(os.path.abspath(__file__))
PRACTICAL = os.path.abspath(os.path.join(HERE, "..", ".."))


def cmd_mms(args):
    from omar_pfem.mms_study import solve_mms, verify_derivation, DEFAULT_ALPHA, DEFAULT_BETA
    if not verify_derivation(args.material, DEFAULT_ALPHA, DEFAULT_BETA):
        raise SystemExit("manufactured-solution derivation check failed")
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    Ns = [int(n) for n in args.Ns.split(",")]
    for order in args.orders.split(","):
        rows = []
        for N in Ns:
            err, *_ = solve_mms(order, N, args.material, DEFAULT_ALPHA, DEFAULT_BETA,
                                device, torch.float64)
            rows.append(err)
            print(f"{order} N={N:<4} L2={err['L2_rel']:.3e} H1={err['H1_semi_rel']:.3e} "
                  f"stress={err['stress_rel_L2']:.3e} E-norm={err['energy_norm_rel']:.3e} "
                  f"({err['wall_clock_s']:.1f}s)", flush=True)
        hs = [r["h"] for r in rows]
        expected = EXPECTED_RATES[order]
        gate = gates.mms_rate_gate(hs, {k: [r[k] for r in rows] for k in expected},
                                   expected, tol=args.tol)
        spec = ProblemSpec(
            name=f"B1 MMS {args.material} {order}", geometry="unit_square", dim=2,
            material=args.material, loading="manufactured_body_force",
            regularity="smooth", element=order, reference="exact_manufactured",
            qois=tuple(expected))
        rec = repository.record(
            spec, method={"solver": "matrix-free Newton-CG (mms_study.solve_mms)",
                          "order": order, "Ns": Ns},
            observables={"rows": [{k: r[k] for k in ("N", "h", "n_dof", *expected)}
                                  for r in rows]},
            verdicts={"mms_rates": gate})
        print(f"\n{order}: {gate['verdict']}")
        for norm, v in gate["rates"].items():
            print(f"  {norm:<16} observed {v['observed']:.2f}  expected {v['expected']}"
                  f"  {'ok' if v['pass'] else 'OUT OF TOLERANCE'}")
        print(f"  stored -> accepted={rec['accepted']}\n")


def _load_b3_family():
    f = os.path.join(PRACTICAL, "item_1f_avg_for_original_four_points.json")
    rows = [r["groove"] for r in json.load(open(f))]
    for name in ("item_1c_extra_point_23520_percomponent_scf.json",
                 "item_1d_sixth_family_point_33120.json",
                 "item_1e_seventh_family_point_45000.json"):
        rows.append(json.load(open(os.path.join(PRACTICAL, name)))["groove"])
    return sorted(rows, key=lambda g: g["n_elem"])


def _load_b3_zbeta_10200(family):
    base = next(g for g in family if g["n_elem"] == 10200)
    out = [("uniform", base)]
    for zb in ("1.0", "2.0", "3.0"):
        out.append((f"z_beta={zb}", json.load(open(os.path.join(
            PRACTICAL, f"item_1g_zbeta_{zb}_result.json")))))
    return out


def cmd_replay_b3(args):
    family = _load_b3_family()
    hs = [g["n_elem"] ** (-1.0 / 3.0) for g in family]
    zbeta = _load_b3_zbeta_10200(family)
    print("B3 groove region, quadratic (Q20) global refinement family:",
          [g["n_elem"] for g in family])
    scale = family[-1]["region_avg_sigma_frob"]
    print(f"stress scale (finest-mesh region-average Frobenius norm): {scale:.4f}")
    print(f"{'QoI':<22}{'convergence':>14}{'robustness':>14}  admitted as validation target")
    admitted = {}
    series = [("frob", "avg", lambda g: g["region_avg_sigma_frob"])]
    for comp in ("xx", "yy", "zz", "xy", "xz", "yz"):
        for stat in ("avg", "max"):
            series.append((comp, stat, lambda g, c=comp, s=stat: g["per_component"][c][s]))
    for comp, stat, get in series:
            vals = [get(g) for g in family]
            g_conv = gates.qoi_convergence_gate(hs, vals, band_tol=args.band_tol,
                                                gci_tol=args.gci_tol, scale=scale)
            g_rob = gates.strategy_robustness_gate(
                [get(z) for _, z in zbeta], tol=args.robust_tol, scale=scale)
            qoi = f"sigma_{comp}_{stat}"
            spec = ProblemSpec(
                name=f"B3 groove {qoi}", geometry="bushing_groove", dim=3,
                material="neo_hookean", loading="prescribed_rotation",
                regularity="stress_concentration", element="Q20",
                reference="mesh_refinement_family", qois=(qoi,))
            rec = repository.record(
                spec, method={"solver": "torch-fem Q20, curved-surface midside nodes",
                              "refinement": "global family + local z-clustering at 10,200 el"},
                observables={"n_elements": [g["n_elem"] for g in family], "values": vals,
                             "stress_scale": scale,
                             "strategy_values": {k: get(z) for k, z in zbeta}},
                verdicts={"qoi_convergence": g_conv, "strategy_robustness": g_rob})
            admitted[qoi] = rec["accepted"]
            print(f"{qoi:<22}{g_conv['verdict']:>14}{g_rob['verdict']:>14}  "
                  f"{'YES' if rec['accepted'] else 'no'}  ({g_conv['reason']})")
    print(f"\nadmitted: {[q for q, a in admitted.items() if a]}")
    print(f"rejected: {[q for q, a in admitted.items() if not a]}")


B1_QOIS = ("max_disp_mag", "u_top_off_center", "v_top_off_center", "u_mid_domain",
           "v_mid_domain", "strain_energy", "interior_avg_PK1_frob", "peak_PK1_frob")


def cmd_b1_ladder(args):
    from omar_pfem.verification_agent.b1_ladder import run_ladder
    Ns = [int(n) for n in args.Ns.split(",")]
    cache = os.path.join(HERE, f"b1_ladder_{args.material}.json")
    rows = run_ladder(Ns, args.material, cache)
    hs = [r["h"] for r in rows]
    print(f"\nB1 {args.material}, Q4, N={Ns}")
    print(f"{'QoI':<24}{'verdict':>9}  detail")
    for qoi in B1_QOIS:
        vals = [r[qoi] for r in rows]
        g = gates.qoi_convergence_gate(hs, vals, band_tol=args.band_tol, gci_tol=args.gci_tol)
        spec = ProblemSpec(
            name=f"B1 {args.material} {qoi}", geometry="unit_square", dim=2,
            material=args.material, loading="top_edge_traction",
            regularity="clamped_free_corner", element="Q4",
            reference="mesh_refinement_family", qois=(qoi,))
        repository.record(spec, method={"solver": "data_generate_B1.solve_hyperelastic_TL_spatial",
                                        "fields": "mesh_convergence.AnalyticFieldB1", "Ns": Ns},
                          observables={"h": hs, "values": vals}, verdicts={"qoi_convergence": g})
        extra = (f", p={g['observed_order']:.2f}, extrapolated={g['richardson_extrapolate']:.6g}"
                 if "observed_order" in g else "")
        print(f"{qoi:<24}{g['verdict']:>9}  {g['reason']}{extra}")


def cmd_orchestrate(args):
    from omar_pfem.verification_agent import llm_roles
    from omar_pfem.verification_agent.orchestrate import orchestrate
    text = open(args.problem).read() if os.path.exists(args.problem) else args.problem
    backend = llm_roles.make_backend(args.backend, args.replay)
    trace = {}
    try:
        orchestrate(backend, text, trace=trace)
    except Exception as e:
        trace["error"] = f"{type(e).__name__}: {e}"
        raise
    finally:
        if hasattr(backend, "cost_usd"):
            trace["llm_calls"], trace["llm_cost_usd"] = backend.calls, backend.cost_usd
            print(f"LLM calls: {len(backend.calls)}, cost ${backend.cost_usd:.2f}")
        if args.trace_out:
            json.dump(trace, open(args.trace_out, "w"), indent=1)
            print(f"trace -> {args.trace_out}")


DEFAULT_SPECS = {
    "unit_square": dict(name="B1 neo_hookean", geometry="unit_square", dim=2,
                        material="neo_hookean", loading="top_edge_traction",
                        regularity="clamped_free_corner", element="Q4",
                        reference="mesh_refinement_family", qois=B1_QOIS),
    "bushing_groove": dict(name="B3 groove", geometry="bushing_groove", dim=3,
                           material="neo_hookean", loading="prescribed_rotation",
                           regularity="stress_concentration", element="Q20",
                           reference="mesh_refinement_family", qois=None),
}


def _default_spec(geometry):
    from omar_pfem.verification_agent.orchestrate import ADAPTER_VOCAB
    d = dict(DEFAULT_SPECS[geometry])
    d["qois"] = tuple(d["qois"] or ADAPTER_VOCAB[geometry][1])
    return ProblemSpec(**d)


def cmd_rules(args):
    """Deterministic rule-based plan (no LLM), executed and scored."""
    from omar_pfem.verification_agent import trees
    from omar_pfem.verification_agent.orchestrate import ADAPTERS, run_check
    for geometry in args.geometry.split(","):
        spec = _default_spec(geometry)
        sd = spec.to_dict()
        plan, analysis = trees.rule_plan(sd), trees.apply_rules(sd)
        adapter = ADAPTERS[geometry]()
        conv_params = {q: c["params"] for c in plan["checks"]
                       if c["check"] == "qoi_convergence" for q in c["qois"]}
        verdicts = {}
        for c in plan["checks"]:
            for q in c["qois"]:
                verdicts.setdefault(q, {})[c["check"]] = run_check(
                    adapter, c["check"], q, c["params"], element=spec.element,
                    conv_params=conv_params.get(q))
        print(f"\n{spec.name} (regularity {spec.regularity}, element {spec.element}, "
              f"order window {trees.order_window(spec.element)})")
        print(f"{'QoI':<24}{'class':<16}{'pred':<6}{'conv':<6}{'evid':<6}"
              f"{'strat':<7}{'anchor':<7}overall")
        hits = n_pred = 0
        for q in spec.qois:
            v = verdicts[q]
            vs = [x["verdict"] for x in v.values()]
            overall = "FAIL" if "FAIL" in vs else ("PASS" if all(x == "PASS" for x in vs)
                                                   else "INCONCLUSIVE")
            a = analysis[q]
            if a["predicted"]:
                n_pred += 1
                hits += a["predicted"] == overall
            cv = v.get("qoi_convergence", {})
            print(f"{q:<24}{a['qoi_class']:<16}{a['predicted'] or '-':<6}"
                  f"{cv.get('verdict', '-'):<6}{cv.get('evidence', '-'):<6}"
                  f"{v.get('strategy_robustness', {}).get('verdict', '-'):<7}"
                  f"{v.get('anchored_robustness', {}).get('verdict', '-'):<7}{overall}")
            print(f"{'':<4}conv: {cv.get('reason')}")
            for k in ("strategy_robustness", "anchored_robustness"):
                if k in v:
                    print(f"{'':<4}{k.split('_')[0]}: {v[k].get('reason')}")
        print(f"rule predictions correct: {hits}/{n_pred}")


def cmd_adaptive(args):
    from omar_pfem.verification_agent import adaptive
    from omar_pfem.verification_agent.b1_ladder import run_ladder
    spec = _default_spec("unit_square")
    print("Outcome prior from the repository (leave-one-out by QoI):")
    for q, p in adaptive.outcome_prior(spec).items():
        pp = "no data" if p["p_pass"] is None else f"P(PASS)={p['p_pass']:.2f}"
        print(f"  {q:<24}{p['qoi_class']:<16}{pp:<16}from {p['n_records']} records "
              f"(total similarity {p['weight']:.2f})")
    cache = os.path.join(HERE, f"b1_ladder_{args.material}.json")
    all_rows = json.load(open(cache)) if os.path.exists(cache) else []

    def solve(N):
        return run_ladder([N], args.material, cache)[0]

    timing = [r for r in all_rows if r.get("wall_clock_s")]
    res = adaptive.adaptive_refine(solve, list(B1_QOIS), budget_s=args.budget_s,
                                   timing_rows=timing)
    print(f"\nstopped: {res['stop']}; levels used {res['levels']}")
    for q, d in res["decisions"].items():
        print(f"  {q:<24}{d['verdict']:<13}levels {d['levels']}  {d['reason']}")


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("mms")
    m.add_argument("--material", default="neo_hookean")
    m.add_argument("--orders", default="Q4,Q9")
    m.add_argument("--Ns", default="5,9,17")
    m.add_argument("--tol", type=float, default=0.2)
    m.add_argument("--cpu", action="store_true")
    m.set_defaults(fn=cmd_mms)
    b = sub.add_parser("replay-b3")
    b.add_argument("--band_tol", type=float, default=0.01)
    b.add_argument("--gci_tol", type=float, default=0.05)
    b.add_argument("--robust_tol", type=float, default=0.02)
    b.set_defaults(fn=cmd_replay_b3)
    l = sub.add_parser("b1-ladder")
    l.add_argument("--material", default="neo_hookean")
    l.add_argument("--Ns", default="11,21,41,81")
    l.add_argument("--band_tol", type=float, default=0.01)
    l.add_argument("--gci_tol", type=float, default=0.05)
    l.set_defaults(fn=cmd_b1_ladder)
    o = sub.add_parser("orchestrate", help="LLM roles around the deterministic gates")
    o.add_argument("--problem", required=True, help="problem text, or a path to a text file")
    o.add_argument("--backend", choices=("anthropic", "claude-cli", "replay"), default="replay")
    o.add_argument("--replay", help="JSON file of canned role answers (replay backend)")
    o.add_argument("--trace_out")
    o.set_defaults(fn=cmd_orchestrate)
    r = sub.add_parser("rules", help="rule-based plan (no LLM), executed and scored")
    r.add_argument("--geometry", default="unit_square,bushing_groove")
    r.set_defaults(fn=cmd_rules)
    a = sub.add_parser("adaptive", help="outcome prior + adaptive refinement on B1")
    a.add_argument("--material", default="neo_hookean")
    a.add_argument("--budget_s", type=float, default=1800.0)
    a.set_defaults(fn=cmd_adaptive)
    args = p.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
