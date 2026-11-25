"""Automatic sensitivity check of assumed inputs (25 Nov; rule and predictions pre-registered in
sensitivity_preregistration.json before any run).

An assumed input is a physical number of the description that is not in the source text
(autonomous.trace_numbers). Before, every such number needed a person. Here the agent measures how
much each assumed input changes each CERTIFIED quantity and asks a person only where that change
is larger than the certified numerical uncertainty of the quantity.

    python -m omar_pfem.verification_agent.sensitivity --description problems2d/x.json \
        --assume E nu "bcs[2].t" [--all-quantities]

Parameters: E, nu, bcs[k].t (the traction vector), bcs[k].p, bcs[k].u, hole[k].radius.
"""
import argparse
import copy
import json
import math
import os
import re
import time

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "sensitivity_runs")
REL = 0.10                      # +-10% for E, loads, prescribed displacement, hole radius
NU_ABS = 0.05                   # +-0.05 for nu
LEVELS = (8, 16)
MAX_N = 64


def perturbable(name):
    return bool(re.fullmatch(r"E|nu|bcs\[\d+\]\.(t|p|u)|hole\[\d+\]\.radius", name))


def param_name(flag):
    """'bcs[2].t = 0.0625 (declared assumption)' -> 'bcs[2].t'; vertices -> None."""
    n = flag.split(" = ")[0].strip()
    if n == "hole radius":
        return "hole[0].radius"
    return n if perturbable(n) else None


def _get(desc, name):
    if name in ("E", "nu"):
        return float(desc[name])
    m = re.fullmatch(r"bcs\[(\d+)\]\.(t|p|u)", name)
    if m:
        return desc["bcs"][int(m.group(1))][m.group(2)]
    m = re.fullmatch(r"hole\[(\d+)\]\.radius", name)
    return float(desc["domain"]["holes"][int(m.group(1))]["circle"][2])


def perturbed(desc, name, sign):
    """(description, relative size of the step dp/p) for a +1 / -1 step."""
    d = copy.deepcopy(desc)
    if name == "nu":
        nu = float(desc["nu"])
        new = min(0.49, max(0.0, nu + sign * NU_ABS))
        d["nu"] = new
        return d, (new - nu) / nu if nu else new - nu
    f = 1 + sign * REL
    if name == "E":
        d["E"] = float(desc["E"]) * f
    else:
        m = re.fullmatch(r"bcs\[(\d+)\]\.(t|p|u)", name)
        if m:
            b = d["bcs"][int(m.group(1))]
            k = m.group(2)
            b[k] = [v * f for v in b[k]] if isinstance(b[k], list) else b[k] * f
        else:
            m = re.fullmatch(r"hole\[(\d+)\]\.radius", name)
            h = d["domain"]["holes"][int(m.group(1))]["circle"]
            cx, cy, r = h
            h[2] = r * f
            # points and segment ends that lie on the hole edge move with it (radially)
            on = lambda p: abs(math.hypot(p[0] - cx, p[1] - cy) - r) <= 1e-9 * max(1.0, r)
            move = lambda p: [cx + (p[0] - cx) * f, cy + (p[1] - cy) * f] + list(p[2:])
            for b in d["bcs"]:
                if "segment" in b:
                    b["segment"] = [move(p) if on(p) else p for p in b["segment"]]
            for k, p in d.get("points", {}).items():
                if on(p):
                    d["points"][k] = move(p)
    return d, sign * REL


def _solve(desc, n):
    from omar_pfem.verification_agent.geometry2d import make
    r = make(desc).solve(n, None, "Q2")
    if r.get("failed"):
        raise RuntimeError(f"solve failed at n = {n}")
    return r


def _at_level(desc, name, qois, n):
    base = _solve(desc, n)
    dp, dm = perturbed(desc, name, +1), perturbed(desc, name, -1)
    rp, rm = _solve(dp[0], n), _solve(dm[0], n)
    out = {}
    for q in qois:
        q0, qp, qm = base[q], rp[q], rm[q]
        span = qp - qm
        rel_step = dp[1] - dm[1]
        S = (span / q0) / rel_step if q0 else float("nan")
        # POST HOC (25 Nov, after the first run gave S_E = -1.0101 for Q ~ 1/E): the central
        # difference is not exact for power laws; S_log = ln(Q+/Q-) / ln(p+/p-) is. Reported
        # in addition; the pre-registered S and the decision rule are unchanged.
        pr = (1 + dp[1]) / (1 + dm[1])
        S_log = (math.log(qp / qm) / math.log(pr)) if qp * qm > 0 and pr > 0 else float("nan")
        out[q] = {"Q0": q0, "Q+": qp, "Q-": qm, "S": S, "S_log_posthoc": S_log,
                  "change_rel": max(abs(qp - q0), abs(qm - q0)) / abs(q0) if q0 else float("inf"),
                  "asymmetry": abs((qp - q0) + (qm - q0)) / abs(span) if span else 0.0}
    return out


class _adopted_targets:
    """VERIF_ABS_TARGETS=adopted for the duration of a block."""

    def __enter__(self):
        self.old = os.environ.get("VERIF_ABS_TARGETS")
        os.environ["VERIF_ABS_TARGETS"] = "adopted"

    def __exit__(self, *a):
        if self.old is None:
            os.environ.pop("VERIF_ABS_TARGETS", None)
        else:
            os.environ["VERIF_ABS_TARGETS"] = self.old


def _agree(a, b):
    if not (math.isfinite(a) and math.isfinite(b)):
        return False
    return abs(a - b) <= max(0.02, 0.05 * abs(b))


def analyse(desc, params, certified=None, log=print):
    """certified: {quantity: {"U": ..., "limit": ...}} (None -> every quantity, no decision).
    Returns the record (also written to sensitivity_runs/<name>.json)."""
    from omar_pfem.verification_agent.geometry2d import make
    prob = make(desc)
    qois = sorted(certified) if certified is not None else list(prob.qois)
    qois = [q for q in qois if not q.startswith("_")]
    rec = {"problem": desc["name"], "rule": "sensitivity_preregistration.json",
           "range": {"relative": REL, "nu_absolute": NU_ABS}, "params": {}}
    t0 = time.time()
    for name in params:
        if not perturbable(name):
            rec["params"][name] = {"status": "not perturbable automatically",
                                   "needs_person": True}
            log(f"[sens] {name}: not perturbable automatically -> person")
            continue
        try:
            levels = list(LEVELS)
            res = {n: _at_level(desc, name, qois, n) for n in levels}
            while True:
                a, b = res[levels[-2]], res[levels[-1]]
                bad = [q for q in qois if not _agree(a[q]["S"], b[q]["S"])]
                if not bad or levels[-1] >= MAX_N:
                    break
                n = 2 * levels[-1]
                levels.append(n)
                res[n] = _at_level(desc, name, qois, n)
                log(f"[sens] {name}: S not settled for {bad}; level n = {n} added")
        except Exception as e:                       # e.g. a nonlinear solve fails
            rec["params"][name] = {"status": f"failed: {e!s:.200}", "needs_person": True}
            log(f"[sens] {name}: failed ({e!s:.120}) -> person")
            continue
        fin = res[levels[-1]]
        per_q, matters = {}, []
        for q in qois:
            r = dict(fin[q])
            r["S_previous_level"] = res[levels[-2]][q]["S"]
            r["settled"] = q not in bad
            if certified is not None:
                c = certified[q]
                # FIX (25 Nov): the framework's U is already RELATIVE (gates.uncertainty_fine);
                # the first version divided it by |limit| again
                u_rel = abs(c.get("U_rel", c["U"]))
                r["U_rel"] = u_rel
                r["ratio_to_U"] = r["change_rel"] / u_rel if u_rel else float("inf")
                if not r["settled"]:
                    r["decision"] = "unresolved"
                else:
                    r["decision"] = "matters" if r["change_rel"] > u_rel else "negligible"
                if r["decision"] != "negligible":
                    matters.append(q)
                # POST HOC (25 Nov, after the first runs: with U ~ 1e-9 the pre-registered rule
                # flags every input with any effect). Reported only; the decision above is the
                # one the loop uses until Professor Rabczuk decides.
                #  (a) adopted rule (24 Nov, question 2): U must be below one third of the
                #      half-width the input range causes (requirements.constitutive_requirement)
                #  (b) proposal: a person only if the half-width exceeds the adopted accuracy
                #      target of the quantity (requirements.absolute_target, adopted mode)
                from omar_pfem.verification_agent import requirements as RQ
                need, _ = RQ.constitutive_requirement(r["change_rel"]) if r["change_rel"] else (0.0, "")
                # not applicable when the input does not change the quantity beyond U
                r["posthoc_numerics_adequate"] = (bool(u_rel <= need)
                                                  if r["decision"] == "matters" else None)
                with _adopted_targets():
                    tgt, src = RQ.absolute_target(q)
                r["posthoc_target"] = tgt
                r["posthoc_person_needed"] = r["decision"] != "negligible" and (
                    r["change_rel"] > tgt if tgt is not None else True)
            per_q[q] = r
        post = [q for q in qois if per_q[q].get("posthoc_person_needed")]
        rec["params"][name] = {"value": _get(desc, name), "levels": levels, "quantities": per_q,
                               "posthoc_person_needed_for": post if certified is not None else None,
                               "matters_for": matters,
                               "needs_person": bool(matters) if certified is not None else None}
        log(f"[sens] {name}: " + ", ".join(f"{q} S={per_q[q]['S']:+.4f}"
                                           + (f" ({per_q[q]['decision']})" if certified is not None
                                              else "") for q in qois))
    rec["minutes"] = (time.time() - t0) / 60
    os.makedirs(OUT, exist_ok=True)
    json.dump(rec, open(os.path.join(OUT, desc["name"] + ".json"), "w"), indent=1)
    return rec


def main(argv=None):
    import warnings
    warnings.filterwarnings("ignore")
    ap = argparse.ArgumentParser()
    ap.add_argument("--description", required=True)
    ap.add_argument("--assume", nargs="+", required=True)
    ap.add_argument("--all-quantities", action="store_true",
                    help="analyse every quantity, without decisions (diagnostic)")
    ap.add_argument("--certified-from", help="autonomous_runs/<name>/run.json")
    a = ap.parse_args(argv)
    desc = json.load(open(a.description))
    cert = None
    if not a.all_quantities:
        run = json.load(open(a.certified_from or os.path.join(HERE, "autonomous_runs",
                                                              desc["name"], "run.json")))
        cert = run["certified_info"]
    analyse(desc, a.assume, cert)


if __name__ == "__main__":
    main()
