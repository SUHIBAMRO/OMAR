"""Reference domain for code verification (25 Nov, round Q; rule R2 of
round_q_preregistration.json - POST HOC, pending Professor Rabczuk's decision).

Code verification checks the CODE: the solver, the element and the kinds of boundary
condition. When the manufactured-solution check on the production domain is INCONCLUSIVE
(every error falls, but the rates have not settled at the finest affordable level - e.g. the
coarse unstructured meshes of Cook's membrane in 3D), the same check is run on a reference
domain that exercises the same code paths with regular meshes:

  the bounding rectangle of the section (same thickness and z-face conditions in 3D), with the
  SAME set of side-condition types (clamped, symmetry with the same orientations, traction,
  pressure), and a circular hole (radius 0.15 x the smaller side, centred) if the production
  domain has holes, carrying the condition of the production hole if it has one.

The loop continues only if the reference check PASSES, and every result of the run is then
labelled 'code verified on the reference domain (post hoc rule R2)'. A production FAIL (an
error that does not fall) still stops the loop.
"""
import copy


def reference_description(desc):
    """A description of the reference domain, or (None, reason)."""
    if "domain" not in desc:
        return None, "only general (gmsh) domains have a reference domain"
    pts = desc["domain"]["outer"]
    x0, x1 = min(p[0] for p in pts), max(p[0] for p in pts)
    y0, y1 = min(p[1] for p in pts), max(p[1] for p in pts)
    left, bottom = [[x0, y0], [x0, y1]], [[x0, y0], [x1, y0]]
    right, top = [[x1, y0], [x1, y1]], [[x0, y1], [x1, y1]]
    free = {"left": left, "bottom": bottom, "right": right, "top": top}
    side = [b for b in desc["bcs"] if "segment" in b]
    kinds = {b["type"] for b in side}
    sym_dirs = {("x" if abs(b["segment"][0][0] - b["segment"][1][0]) < 1e-12 else "y")
                for b in side if b["type"] == "symmetry"}
    bcs = []

    def take(prefer, kind, extra=None):
        for s in prefer:
            if s in free:
                bcs.append(dict({"segment": free.pop(s), "type": kind}, **(extra or {})))
                return True
        return False
    if "clamped" in kinds:
        take(["left", "bottom", "right", "top"], "clamped")
    if "x" in sym_dirs:
        take(["left", "right"], "symmetry")
    if "y" in sym_dirs:
        take(["bottom", "top"], "symmetry")
    if "traction" in kinds:
        t = next(b["t"] for b in side if b["type"] == "traction")
        take(["right", "top", "left", "bottom"], "traction", {"t": list(t)})
    if "pressure" in kinds:
        take(["top", "right", "bottom", "left"], "pressure", {"p": 1.0})
    holes = []
    if desc["domain"].get("holes"):
        r = 0.15 * min(x1 - x0, y1 - y0)
        holes = [{"circle": [0.5 * (x0 + x1), 0.5 * (y0 + y1), r]}]
        hb = [b for b in desc["bcs"] if "hole" in b]
        if hb:
            b = dict(hb[0])
            b["hole"] = 0
            bcs.append(b)
    ref = {k: copy.deepcopy(v) for k, v in desc.items()
           if k not in ("domain", "bcs", "points", "name", "assumed")}
    ref.update(name=desc["name"] + "_reference_domain",
               domain={"outer": [[x0, y0], [x1, y0], [x1, y1], [x0, y1]], "holes": holes},
               bcs=bcs + [copy.deepcopy(b) for b in desc["bcs"] if "face" in b], points={})
    from omar_pfem.verification_agent.geometry2d import make
    try:
        make(ref)
    except ValueError as e:
        return None, f"no valid reference domain: {e}"
    return ref, "ok"
