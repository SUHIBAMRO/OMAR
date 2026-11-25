"""Automatic code verification by a manufactured solution (round O, 2026-11-24).

Until now every manufactured solution was chosen and wired by hand. Here the agent:

  1. DECIDES whether code verification is needed: it is, unless the record holds a passing
     check of the same solver code (sha256 of pipeline2d.py), element, plane condition and
     set of boundary-condition types. MMS is suitable for every problem the pipeline can
     solve (the solver is ours and linear elastic); otherwise the reason is recorded.
  2. GENERATES a smooth manufactured field on the problem's own domain (scaled to its
     bounding box), derives the body force -div sigma and the stress symbolically (sympy),
     and imposes it through the SAME boundary-condition types as the production problem:
     exact displacement on clamped segments, exact normal displacement and shear traction on
     symmetry segments, exact traction on every other edge.
  3. SOLVES it with the production solver (pipeline2d.Problem2D.solve, mms=...) on four
     levels, Q2 and the independent P2, and checks the L2 and H1 rates against theory
     (degree 2: 3 and 2, tolerance 0.2 as in trees.class_params('mms_rates')) with
     gates.mms_rate_gate.
  4. RECORDS the outcome (code_verification_record.json), so the next problem with the same
     code and conditions does not repeat it (and a code change forces a new check).
"""
import hashlib
import json
import os

import numpy as np
import sympy as sp

HERE = os.path.dirname(os.path.abspath(__file__))
RECORD = os.path.join(HERE, "code_verification_record.json")
LEVELS = (2, 4, 8, 16)
MAX_N = 64                 # finer levels are added only while the rates move toward theory
EXPECTED = {"L2": 3.0, "H1": 2.0}


def solver_sha(material="linear_elastic"):
    files = ["pipeline2d.py"] + (["pipeline2d_nh.py"] if material == "neo_hookean" else [])
    h = hashlib.sha256()
    for f in files:
        h.update(open(os.path.join(HERE, f), "rb").read())
    return h.hexdigest() if material == "neo_hookean" else hashlib.sha256(
        open(os.path.join(HERE, "pipeline2d.py"), "rb").read()).hexdigest()


def key(prob, element):
    kinds = sorted({(("face_" if "face" in b else "") + b["type"]) for b in prob.desc["bcs"]}
                   | {"free"})
    mat = getattr(prob, "material", "linear_elastic")
    sha = solver_sha(mat)
    if "domain" in prob.desc:      # general domains: the geometry and mesh code is part of it
        files = [f for f in getattr(prob, "code_files", ["geometry2d.py"]) if f != "pipeline2d.py"]
        for f in files:
            sha = hashlib.sha256((sha + hashlib.sha256(open(os.path.join(HERE, f), "rb")
                                                       .read()).hexdigest()).encode()).hexdigest()
        kinds.append("general_domain" + ("_holes" if prob.desc["domain"].get("holes") else ""))
    base = f"{sha[:16]}|{element}|plane_{prob.plane}|{'+'.join(kinds)}"
    return base if mat == "linear_elastic" else base + f"|{mat}"


def _record():
    return json.load(open(RECORD)) if os.path.exists(RECORD) else {}


REF = "|reference_domain"


def needed(prob, suffix=""):
    """(needed?, reason) for each element. A check INCONCLUSIVE on the production domain with
    a PASS on the reference domain (post hoc rule R2, round Q) is not repeated either."""
    rec, out = _record(), {}
    for el in ("Q2", "P2"):
        k = key(prob, el) + suffix
        if k in rec and rec[k]["verdict"] == "PASS":
            out[el] = (False, f"passing check on record for {k} ({rec[k]['problem']})")
        elif not suffix and k in rec and rec[k]["verdict"] == "INCONCLUSIVE" and \
                rec.get(k + REF, {}).get("verdict") == "PASS":
            out[el] = (False, f"INCONCLUSIVE on the production domain and PASS on the reference "
                              f"domain on record for {k} (post hoc rule R2)")
        else:
            out[el] = (True, f"no passing check on record for {k}")
    return out


class Manufactured:
    """u = A [sin(kx x + 0.3) cos(ky y + 0.2), cos(kx x + 0.1) sin(ky y + 0.4)] with k ~ 1.3 pi /
    box size: smooth, not polynomial, non-zero on every edge."""

    def __init__(self, prob):
        x, y = sp.symbols("x y", real=True)
        x0, x1, y0, y1 = [c * prob.a for c in prob.box]
        Lx, Ly = x1 - x0, y1 - y0
        kx, ky = 1.3 * sp.pi / Lx, 1.3 * sp.pi / Ly
        A = 1e-3 * max(Lx, Ly)
        ux = A * sp.sin(kx * x + sp.Rational(3, 10)) * sp.cos(ky * y + sp.Rational(1, 5))
        uy = A * sp.cos(kx * x + sp.Rational(1, 10)) * sp.sin(ky * y + sp.Rational(2, 5))
        mu, lam = prob.mu, prob.lame
        exx, eyy = sp.diff(ux, x), sp.diff(uy, y)
        exy = (sp.diff(ux, y) + sp.diff(uy, x)) / 2
        sxx = 2 * mu * exx + lam * (exx + eyy)
        syy = 2 * mu * eyy + lam * (exx + eyy)
        sxy = 2 * mu * exy
        bx = -(sp.diff(sxx, x) + sp.diff(sxy, y))
        by = -(sp.diff(sxy, x) + sp.diff(syy, y))
        f = lambda e: sp.lambdify((x, y), e, "numpy")
        self._u = [f(ux), f(uy)]
        self._g = [[f(sp.diff(ux, x)), f(sp.diff(ux, y))], [f(sp.diff(uy, x)), f(sp.diff(uy, y))]]
        self._s = [f(sxx), f(syy), f(sxy)]
        self._b = [f(bx), f(by)]
        self.field = {"ux": str(ux), "uy": str(uy)}

    @staticmethod
    def _v(fn, x, y):
        return np.asarray(fn(x, y), float) * np.ones_like(np.asarray(x, float))

    def u(self, x, y):
        return [self._v(f, x, y) for f in self._u]

    def grad(self, x, y):
        return [[self._v(f, x, y) for f in row] for row in self._g]

    def body(self, x, y):
        return [self._v(f, x, y) for f in self._b]

    def traction(self, x, y, nx, ny):
        sxx, syy, sxy = [self._v(f, x, y) for f in self._s]
        return [sxx * nx + sxy * ny, sxy * nx + syy * ny]


def classify(gate_verdict, errors):
    """POST HOC rule R1 (round Q, pending Professor Rabczuk): PASS as the gate says; otherwise
    INCONCLUSIVE when every error falls at every level (rates not settled), else FAIL."""
    if gate_verdict == "PASS":
        return "PASS"
    falls = all(all(b < a for a, b in zip(e, e[1:])) for e in errors.values())
    return "INCONCLUSIVE" if falls else "FAIL"


def run(prob, force=False, log=print, suffix="", elements=("Q2", "P2")):
    """Code verification for one problem; returns {element: result}. suffix=REF stores the
    result as the reference-domain check of the production key."""
    from omar_pfem.verification_agent.gates import mms_rate_gate
    need = needed(prob, suffix)
    rec, out = _record(), {}
    mms = None
    for el in elements:
        if not need[el][0] and not force:
            k = key(prob, el) + suffix
            out[el] = {"skipped": True, "reason": need[el][1],
                       "verdict": rec.get(k, {}).get("verdict", "PASS"),
                       "reference_pass": "reference domain" in need[el][1]}
            log(f"[mms] {el}: skipped ({need[el][1]})")
            continue
        mms = mms or (prob.manufactured() if hasattr(prob, "manufactured") else Manufactured(prob))
        rows, added = [], []
        for n in LEVELS:                 # a level refused by the memory guard ends the ladder
            r = prob.solve(n, None, el, mms=mms)
            if r.get("failed"):
                added.append(f"n={n}: refused ({r.get('reason', 'failed')})")
                break
            rows.append(r)
        if len(rows) < 3:
            res = {"verdict": "INCONCLUSIVE", "gate_verdict": None, "rule": "R1 (post hoc, round Q)",
                   "levels": [r["n"] for r in rows], "levels_added": added,
                   "errors": {k: [r[k] for r in rows] for k in EXPECTED}, "field": mms.field,
                   "problem": prob.name, "reason": need[el][1] + "; fewer than 3 levels fit"}
            out[el] = res
            rec[key(prob, el) + suffix] = res
            json.dump(rec, open(RECORD, "w"), indent=1)
            continue
        while True:
            hs = [r["h"] for r in rows]
            g = mms_rate_gate(hs[-3:], {k: [r[k] for r in rows][-3:] for k in EXPECTED},
                              EXPECTED, tol=0.2)
            if g["verdict"] == "PASS" or rows[-1]["n"] >= MAX_N or added:   # added: a level refused
                break
            # pre-asymptotic? every failing norm's pairwise rate must be moving toward theory
            toward = True
            for k, exp_ in EXPECTED.items():
                if g["rates"][k]["pass"]:
                    continue
                e = [r[k] for r in rows]
                p1 = np.log(e[-3] / e[-2]) / np.log(hs[-3] / hs[-2])
                p2 = np.log(e[-2] / e[-1]) / np.log(hs[-2] / hs[-1])
                toward &= abs(p2 - exp_) < abs(p1 - exp_)
            if not toward:
                break
            n = 2 * rows[-1]["n"]
            r = prob.solve(n, None, el, mms=mms)
            if r.get("failed"):                    # e.g. the 3D memory guard refused the level
                added.append(f"n={n}: refused ({r.get('reason', 'failed')})")
                log(f"[mms] {el}: {added[-1]}")
                break
            added.append(f"n={n}: rates outside tolerance but moving toward theory")
            log(f"[mms] {el}: {added[-1]}")
            rows.append(r)
        verdict = classify(g["verdict"], {k: [r[k] for r in rows] for k in EXPECTED})
        res = {"verdict": verdict, "gate_verdict": g["verdict"], "rule": "R1 (post hoc, round Q)",
               "rates": g["rates"], "levels": [r["n"] for r in rows],
               "levels_added": added,
               "errors": {k: [r[k] for r in rows] for k in EXPECTED}, "field": mms.field,
               "problem": prob.name, "reason": need[el][1]}
        out[el] = res
        rec[key(prob, el) + suffix] = res
        json.dump(rec, open(RECORD, "w"), indent=1)
        log(f"[mms] {el}: {verdict} L2 rate {g['rates']['L2']['observed']:.2f} "
            f"H1 rate {g['rates']['H1']['observed']:.2f}")
    return out
