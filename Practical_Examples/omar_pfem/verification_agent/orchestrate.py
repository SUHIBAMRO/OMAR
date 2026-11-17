"""Orchestrated verification run: LLM roles around deterministic gates.

    formalize -> validate spec -> retrieve nearest solved cases
      -> propose plan -> validate plan (code) -> critic -> (one revision)
      -> execute catalogued checks (code) -> record verdicts (code)
      -> advisor explanation -> audit advisor against verdicts (code)

The data a check runs on comes from an adapter keyed by geometry; adapters
expose a fixed QoI vocabulary, so the formalizer cannot invent observables.
"""
import json
import os

from omar_pfem.verification_agent import gates, repository, trees
from omar_pfem.verification_agent import llm_roles as R
from omar_pfem.verification_agent.spec import ProblemSpec

HERE = os.path.dirname(os.path.abspath(__file__))


# ----------------------------------------------------------------------------
# Data adapters: geometry -> mesh family + QoI series (all precomputed or
# computed by deterministic solvers; no LLM involvement)
# ----------------------------------------------------------------------------


def _b1_uniform_Ns(material="neo_hookean", tol=1e-7):
    """Uniform (grading 0) levels present in the B1 ladder cache."""
    cache = os.path.join(HERE, f"b1_ladder_{material}.json")
    if not os.path.exists(cache):
        return (11, 21, 41, 81)
    rows = json.load(open(cache))
    return tuple(sorted(r["N"] for r in rows if r.get("grading", 0.0) == 0.0
                        and r.get("newton_tol", 1e-7) == tol))


class B1Adapter:
    geometry = "unit_square"
    description = (
        "B1: 2D unit square [0,1]^2, plane strain, compressible neo-Hookean "
        "psi = mu/2 (I1 - 2 - 2 ln J) + lam/2 (ln J)^2, total Lagrangian. "
        "E(x,y) = 1000 + 200 sin(pi x) cos(pi y) (800-1200), nu = 0.3 everywhere (no "
        "near-incompressibility). Bottom edge y=0 clamped (u=v=0); vertical edges "
        "traction-free; top edge y=1 loaded by a DEAD (reference-configuration) normal "
        "traction t_y = -5 - 2 sin(pi x) (compressive, 5-7), applied in 10 equal load "
        "increments, Newton tol 1e-7 (residual ~3e-11 reached; at N=41 tightening to 1e-10 "
        "changes no QoI); min det F 0.9917 at N=41, 0.9876 at N=161 (small strain). Q4 elements, full 2x2 Gauss "
        "integration, E/nu evaluated at element centroids. Uniform meshes N x N nodes, "
        "N=" + ",".join(map(str, _b1_uniform_Ns())) + " (h = 1/(N-1), ratio 2); "
        "strategies at N=81: uniform and corner grading a=0.5, 0.8 (x -> s - a sin(2 pi s)/"
        "(2 pi), y -> t - a sin(pi t)/pi; corner elements 2x and 5x smaller, same node "
        "count). QoIs: max_disp_mag = max nodal |u|; u/v_top_off_center at (0.7, 1.0) and "
        "u/v_mid_domain at (0.3, 0.6), linear interpolation of nodal values (both points are "
        "nodes of every uniform mesh); strain_energy = stored energy, "
        "integral of psi over the body (2x2 Gauss; not the total potential with the load "
        "work); "
        "interior_avg_PK1_frob = detJ-weighted mean of |P|_F over the Gauss points inside "
        "the circle centre (0.5,0.5), radius 0.15 (about 0.56 from the clamped corners); "
        "peak_PK1_frob = max of |P|_F over all Gauss points (corner singularity where the "
        "clamped edge meets the free edges).")
    qois = ("max_disp_mag", "u_top_off_center", "v_top_off_center", "u_mid_domain",
            "v_mid_domain", "strain_energy", "interior_avg_PK1_frob", "peak_PK1_frob")

    def __init__(self, material="neo_hookean", Ns=None, strategy_N=81):
        """Ns=None: every uniform level already in the cache (no new solves)."""
        from omar_pfem.verification_agent.b1_ladder import run_ladder
        cache = os.path.join(HERE, f"b1_ladder_{material}.json")
        Ns = _b1_uniform_Ns(material) if Ns is None else Ns
        self.rows = run_ladder(list(Ns), material, cache)
        self.hs = [r["h"] for r in self.rows]
        # same node count, three refinement strategies: uniform and corner
        # grading a=0.5, 0.8 (b1_ladder.grade_nodes), read from the cache only
        all_rows = json.load(open(cache))
        self.strategy_rows = sorted(
            (r for r in all_rows if r["N"] == strategy_N and r.get("newton_tol", 1e-7) == 1e-7),
            key=lambda r: r.get("grading", 0.0))

    def series(self, qoi):
        return self.hs, [r[qoi] for r in self.rows], None

    def strategies(self, qoi):
        if len(self.strategy_rows) < 2:
            return None
        return [r[qoi] for r in self.strategy_rows], None


class B3Adapter:
    geometry = "bushing_groove"
    description = (
        "B3: 3D bushing, inner radius 0.5, outer radius 1.0, length 1.0 (z in [0,1]); half "
        "model y >= 0 with symmetry plane y=0. A circumferential groove is cut into the INNER "
        "surface: r_in(z) = 0.5 - 0.1 (1 + cos(pi (z - 0.5)/0.15)) for |z - 0.5| < 0.15, "
        "depth 0.20, half-width 0.15, smooth cosine profile (C1 at the shoulders, NOT a "
        "sharp notch); root curvature radius rho = 0.0228. Loading: the whole inner surface "
        "(groove included) is RIGIDLY ROTATED by phi = 0.05 rad about the y axis through "
        "(0,0,0.5) (a tilting rotation, not torsion about the bushing axis z), applied in 10 "
        "equal increments; outer surface fixed; end faces z=0 and z=1 traction-free, so "
        "the fixed/rotated surfaces meet free end faces along four circular edges "
        "(possible edge singularities, which can pollute global convergence rates). The "
        "groove root, being part of the rotated inner surface, lies on a prescribed-"
        "displacement surface. "
        "Compressible neo-Hookean (same psi as the 2D "
        "form, 3D invariants), E=1000, nu=0.45. Elements: 20-node serendipity hexahedra "
        "(Q20) with curved-surface midside nodes, integrated with 2x2x2 = 8 Gauss points "
        "(the reduced rule for Q20, not the full 27). Newton-CG, rtol=atol=1e-8 "
        "(tightening to 1e-11 changes QoIs by <= 4e-14); min det F 0.914-0.948. Global "
        "family of 7 levels, 2,376-45,000 elements, refinement ratios (n_elem ratio)^(1/3) "
        "= 1.34, 1.21, 1.16, 1.14, 1.12, 1.11 (NOT constant; the Celik order uses the two "
        "actual ratios r21, r32). Strategies at 10,200 elements: uniform and AXIAL-only "
        "clustering of the z layers toward z=0.5 by sinh grading, z_beta = 1, 2, 3 "
        "(theta and r divisions unchanged, same element count; no radial clustering). "
        "Region QoIs: Cauchy stress "
        "(global Cartesian, or cylindrical about z), averaged with Gauss-point volume "
        "weights over the Gauss points inside a sphere of FIXED physical radius "
        "R0 = 2 rho = 0.0456 centred at (0.3, 0, 0.5) = the groove root on the inner "
        "surface, on the symmetry plane (theta = 0), so the sphere is cut by both the "
        "groove surface and y=0 (about a quarter sphere); 2-36 Gauss points inside from "
        "the coarsest to the finest level. At theta ~ 0, cyl_rr ~ sigma_xx, cyl_tt ~ "
        "sigma_yy, cyl_rz ~ sigma_xz and cyl_zz = sigma_zz exactly. *_max = max |component| "
        "over the Gauss points in the sphere. Region QoIs are judged RELATIVE TO A FIXED "
        "STRESS SCALE (the finest-level region average of the Frobenius norm, 0.910), not "
        "to their own magnitude, so near-zero components are not amplified; global QoIs "
        "(reaction_moment_y about the rotation axis, strain_energy; half model) are judged "
        "relative to themselves. Context: the best surrogates validated against these "
        "targets have errors of about 1.4-4%.")
    CYL = ("rr", "tt", "zz", "rt", "rz", "tz")
    GLOBAL = ("reaction_moment_y", "strain_energy")

    def __init__(self):
        ext = os.path.join(HERE, "b3_extended.json")
        rows = json.load(open(ext)) if os.path.exists(ext) else []
        fam = sorted((r for r in rows if r["z_beta"] is None and r["rtol"] == 1e-8),
                     key=lambda r: r["n_elem"])
        if len(fam) >= 7:
            # extended re-solve, checked to reproduce the stored data to <=5e-13
            self.family = fam
            base = next(r for r in fam if r["n_elem"] == 10200)
            self.zbeta = [("uniform", base)] + [
                (f"z_beta={r['z_beta']}", r) for r in sorted(
                    (r for r in rows if r["z_beta"] is not None), key=lambda r: r["z_beta"])]
            self.extended = True
        else:
            from omar_pfem.verification_agent.agent import _load_b3_family, _load_b3_zbeta_10200
            self.family = _load_b3_family()
            self.zbeta = _load_b3_zbeta_10200(self.family)
            self.extended = False
        self.hs = [g["n_elem"] ** (-1.0 / 3.0) for g in self.family]
        self.scale = self.family[-1]["region_avg_sigma_frob"]
        self.get = {"sigma_frob_avg": lambda g: g["region_avg_sigma_frob"]}
        for c in ("xx", "yy", "zz", "xy", "xz", "yz"):
            for st in ("avg", "max"):
                self.get[f"sigma_{c}_{st}"] = lambda g, c=c, st=st: g["per_component"][c][st]
        if self.extended:
            for k in self.GLOBAL:
                self.get[k] = lambda g, k=k: g[k]
            self.get["von_mises_avg"] = lambda g: g["region_avg_von_mises"]
            self.get["von_mises_max"] = lambda g: g["region_max_von_mises"]
            self.get["pressure_avg"] = lambda g: g["region_avg_pressure"]
            for c in self.CYL:
                for st in ("avg", "max"):
                    self.get[f"cyl_{c}_{st}"] = lambda g, c=c, st=st: g["per_cylindrical"][c][st]
        self.qois = tuple(self.get)

    def _scale(self, qoi):
        return None if qoi in self.GLOBAL else self.scale  # globals: self-relative

    def series(self, qoi):
        return self.hs, [self.get[qoi](g) for g in self.family], self._scale(qoi)

    def strategies(self, qoi):
        return [self.get[qoi](z) for _, z in self.zbeta], self._scale(qoi)

    def admissibility(self):
        rows = self.family + [z for _, z in self.zbeta]
        return {"min_detF": min(r.get("min_detF", float("nan")) for r in rows)}


ADAPTERS = {"unit_square": B1Adapter, "bushing_groove": B3Adapter}
ADAPTER_VOCAB = {
    "unit_square": (B1Adapter.description, B1Adapter.qois),
    "bushing_groove": (B3Adapter.description,
                       ("sigma_frob_avg",) + tuple(f"sigma_{c}_{s}" for c in
                                                   ("xx", "yy", "zz", "xy", "xz", "yz")
                                                   for s in ("avg", "max"))
                       + B3Adapter.GLOBAL + ("von_mises_avg", "von_mises_max", "pressure_avg")
                       + tuple(f"cyl_{c}_{s}" for c in B3Adapter.CYL for s in ("avg", "max"))),
}


# ----------------------------------------------------------------------------
# Execution of a validated plan (deterministic)
# ----------------------------------------------------------------------------


CONV_CHECKS = gates.CONV_CHECKS
combine = gates.combine_verdicts   # single policy, shared with adaptive.py


def convergence_params(plan):
    """{qoi: {check: params}} for the convergence checks of a plan; used to
    anchor anchored_robustness on the same tolerances."""
    out = {}
    for c in plan["checks"]:
        if c["check"] in CONV_CHECKS:
            for q in c["qois"]:
                out.setdefault(q, {})[c["check"]] = c["params"]
    return out


def run_check(adapter, check, qoi, params, element=None, conv_params=None):
    """Execute one catalogued check. The order window of the convergence
    checks comes from the element (trees.order_window), never from the plan.
    conv_params: {check: params} of the QoI's convergence checks (for the
    anchor of anchored_robustness)."""
    window = trees.order_window(element)
    if check == "qoi_convergence":
        hs, vals, scale = adapter.series(qoi)
        return gates.qoi_convergence_gate(hs, vals, scale=scale, order_window=window, **params)
    if check == "qoi_convergence_multilevel":
        hs, vals, scale = adapter.series(qoi)
        return gates.multilevel_convergence_gate(hs, vals, scale=scale, order_window=window,
                                                 **params)
    if check == "anchored_robustness":
        s = adapter.strategies(qoi)
        if s is None:
            return {"gate": "anchored_robustness", "verdict": "INCONCLUSIVE",
                    "reason": "no alternative refinement strategy computed for this problem"}
        hs, vals, scale = adapter.series(qoi)
        cp = conv_params or {}
        conv = gates.qoi_convergence_gate(hs, vals, scale=scale, order_window=window,
                                          **cp.get("qoi_convergence", {}))
        ml = gates.multilevel_convergence_gate(hs, vals, scale=scale, order_window=window,
                                               **cp.get("qoi_convergence_multilevel", {}))
        if conv["verdict"] != "PASS" and ml["verdict"] != "PASS":
            # a family that did not converge has no limit: comparing the
            # strategies with its finest value would say nothing either way
            return {"gate": "anchored_robustness", "verdict": "INCONCLUSIVE",
                    "anchor_kind": "none",
                    "reason": "refinement family not converged (3-level: "
                              f"{conv['verdict']}, 5-level: {ml['verdict']}), so there is no "
                              "limit to anchor the strategies to; uninformative"}
        # anchor: 3-level Richardson value if that has a rate, else the
        # multilevel fit limit if that has a rate, else the finest value
        anchor, kind = gates.family_anchor(conv)
        if conv.get("evidence") != "rate" and ml.get("evidence") == "rate":
            anchor, kind = gates.family_anchor(ml)
        return gates.anchored_robustness_gate(s[0], anchor, scale=s[1], anchor_kind=kind,
                                              **params)
    if check == "strategy_robustness":
        s = adapter.strategies(qoi)
        if s is None:
            return {"gate": "strategy_robustness", "verdict": "INCONCLUSIVE",
                    "reason": "no alternative refinement strategy computed for this problem"}
        vals, scale = s
        return gates.strategy_robustness_gate(vals, scale=scale, **params)
    if check == "mms_rates":
        return {"gate": "mms_rates", "verdict": "INCONCLUSIVE",
                "reason": "run `agent mms` for code verification; not a per-QoI check"}
    raise ValueError(check)


def orchestrate(backend, problem_text, log=print, trace=None):
    """`trace` (a dict) is filled in place, so a caller can save it even if
    a later step raises."""
    trace = {} if trace is None else trace
    trace["problem_text"] = problem_text

    # 1. formalize
    vocab = "\n".join(f"- geometry '{g}': {d}\n  available QoIs: {list(q)}"
                      for g, (d, q) in ADAPTER_VOCAB.items())
    spec_d, errs = R.formalize(backend, problem_text + "\n\nAVAILABLE DATA ADAPTERS "
                               "(geometry must be one of these; QoIs must come from its "
                               "list):\n" + vocab)
    if spec_d["geometry"] not in ADAPTERS:
        errs.append(f"no data adapter for geometry '{spec_d['geometry']}'")
    else:
        allowed = ADAPTER_VOCAB[spec_d["geometry"]][1]
        errs += [f"QoI '{q}' not available for {spec_d['geometry']}"
                 for q in spec_d["qois"] if q not in allowed]
    trace["spec"], trace["spec_errors"] = spec_d, errs
    log(f"[formalizer] {spec_d['name']}: {len(spec_d['qois'])} QoIs, "
        f"regularity={spec_d['regularity']}")
    for w in spec_d["well_posedness"]:
        log(f"   well-posedness: {w}")
    for q in spec_d["open_questions"]:
        log(f"   open question:  {q}")
    if errs:
        log(f"[guard] spec rejected: {errs}")
        return trace
    spec = ProblemSpec(**{k: (tuple(v) if k == "qois" else v) for k, v in spec_d.items()
                          if k in ProblemSpec.__dataclass_fields__})

    # 2. retrieve
    neigh = [{"similarity": round(s, 3), "problem": r["problem"]["name"],
              "verdicts": {k: v["verdict"] for k, v in r["verdicts"].items()}}
             for s, r in repository.nearest(spec, k=5)]
    trace["neighbours"] = neigh
    log(f"[repository] nearest: {[(n['problem'], n['similarity']) for n in neigh]}")

    # 3. propose / validate / critique, at most one revision
    baseline = trees.rule_plan(spec_d)
    trace["rule_plan"], trace["rule_analysis"] = baseline, trees.apply_rules(spec_d)
    log("[rules] baseline requires: " + "; ".join(
        f"{c['check']} on {len(c['qois'])} QoIs" for c in baseline["checks"])
        + f"; predicts FAIL: {[e['qoi'] for e in baseline['expected_failures']]}")
    plan, perrs = R.propose(backend, spec_d, neigh)
    perrs = perrs + trees.plan_rule_errors(plan, spec_d)
    crit = R.critique(backend, spec_d, plan) if not perrs else None
    if perrs or not crit["approve"]:
        feedback = {"validation_errors": perrs, "critic": crit}
        log(f"[guard/critic] revision requested: {json.dumps(feedback)}")
        plan, perrs = R.propose(backend, spec_d, neigh + [{"revise_previous_plan": feedback}])
        perrs = perrs + trees.plan_rule_errors(plan, spec_d)
        crit = R.critique(backend, spec_d, plan) if not perrs else None
    # Rules are authoritative: if the only remaining problems are checks the
    # rules require but the LLM left out, add them (catalogue defaults) and
    # record the augmentation, instead of discarding the whole plan.
    rule_only = perrs and all(e.startswith(("rules ", "ceiling: ")) for e in perrs)
    if rule_only:
        need = trees.required_checks(plan, spec_d)   # R1-R6 plus plan-level R7
        added = []
        for q, r in need.items():
            have = {c["check"] for c in plan["checks"] if q in c["qois"]}
            for chk in sorted(r["required"]):
                if chk not in have:
                    plan["checks"].append({"check": chk, "qois": [q],
                                           "params": trees.class_params(
                                               chk, r["qoi_class"]),
                                           "rationale": f"added by rules {r['rules']}"})
                    added.append(f"{chk}:{q}")
        # clamp tolerances looser than the class ceiling
        clamped = []
        for c in plan["checks"]:
            for q in list(c["qois"]):
                if q not in need:
                    continue
                ceil = trees.class_params(c["check"], need[q]["qoi_class"])
                if any(c["params"].get(k, 0) > v + 1e-12 for k, v in ceil.items()):
                    c["qois"].remove(q)
                    plan["checks"].append({"check": c["check"], "qois": [q],
                                           "params": {k: min(c["params"].get(k, v), v)
                                                      for k, v in ceil.items()},
                                           "rationale": "tolerance clamped to class ceiling"})
                    clamped.append(f"{c['check']}:{q}")
        plan["checks"] = [c for c in plan["checks"] if c["qois"]]
        trace["rule_augmented"], trace["ceiling_clamped"] = added, clamped
        log(f"[rules] plan augmented with required checks the LLM omitted: {added}; "
            f"tolerances clamped to class ceilings: {clamped}")
        perrs = R.validate_plan(plan, spec_d["qois"]) + trees.plan_rule_errors(plan, spec_d)
    if not perrs and crit is None:
        # the final plan has not been reviewed yet (e.g. after augmentation)
        crit = R.critique(backend, spec_d, plan)
    trace.update(plan=plan, plan_errors=perrs, critique=crit)
    if perrs:
        log(f"[guard] plan rejected after revision: {perrs}")
        return trace
    if not crit["approve"]:
        log(f"[critic] plan still not approved: {crit['objections']}; executing anyway, "
            "objections stored with the record")
    log(f"[proposer] {len(plan['checks'])} checks; expected failures: "
        f"{[e['qoi'] for e in plan['expected_failures']]}")

    # 4. execute (code only)
    adapter = ADAPTERS[spec.geometry]()
    per_qoi = {}
    conv_params = convergence_params(plan)
    for c in plan["checks"]:
        for q in c["qois"]:
            g = run_check(adapter, c["check"], q, c["params"], element=spec.element,
                          conv_params=conv_params.get(q))
            per_qoi.setdefault(q, {})[c["check"]] = g
    gate_summary = {}
    for q, verdicts in per_qoi.items():
        sub = ProblemSpec(**{**spec.to_dict(), "qois": (q,), "name": f"{spec.name} {q}"})
        rec = repository.record(sub, method={"orchestrated": True, "plan": plan,
                                             "critic": crit},
                                observables={}, verdicts=verdicts)
        overall, evidence, ev_src = combine(verdicts)
        gate_summary[q] = {"verdict": overall, "convergence_evidence": evidence,
                           "evidence_from": ev_src,
                           "gates": {k: {kk: vv for kk, vv in v.items()
                                         if kk in ("verdict", "reason", "relative_spread",
                                                   "observed_order", "gci_fine", "evidence",
                                                   "finest_three", "anchor_kind", "n_levels",
                                                   "fit_order", "fit_limit",
                                                   "uncertainty_fine", "failed_criteria",
                                                   "leave_one_out_orders")}
                                     for k, v in verdicts.items()}}
        log(f"[gates] {q:<24} {overall:<13} evidence={evidence:<5} accepted={rec['accepted']}")
    trace["gate_summary"] = gate_summary

    # prediction score: did the proposer anticipate the failures?
    failed = {q for q, g in gate_summary.items() if g["verdict"] == "FAIL"}
    trace["prediction"] = {}
    for who, exp in (("llm", plan["expected_failures"]),
                     ("rules", baseline["expected_failures"])):
        predicted = {e["qoi"] for e in exp}
        trace["prediction"][who] = {"predicted_fail": sorted(predicted),
                                    "actual_fail": sorted(failed),
                                    "missed": sorted(failed - predicted),
                                    "false_alarm": sorted(predicted - failed)}
        log(f"[score:{who}] predicted FAIL {sorted(predicted)} | actual FAIL {sorted(failed)}"
            f" | missed {sorted(failed - predicted)} | false alarm {sorted(predicted - failed)}")

    # 5. advise and audit
    advice, flags = R.advise(backend, spec_d, gate_summary)
    trace.update(advice=advice, advisor_flags=flags)
    log(f"[advisor] {advice['summary']}")
    log(f"[audit] {'advisor consistent with gates' if not flags else flags}")
    return trace
