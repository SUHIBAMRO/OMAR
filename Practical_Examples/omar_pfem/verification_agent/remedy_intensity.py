"""Intensity-aware remedy (round L, 2026-11-23).

Lesson of round K (report Section 32.3): planning the grading from the corner exponents
alone missed the observed orders, because a corner with a strong exponent but a small
intensity does not dominate at practical resolutions, and grading every corner with one
exponent wastes elements. This planner adds the intensity, estimated from the uniform family
the agent already has:

  1. corner analysis (remedy.corner_analysis): the rate-limiting corners and their exponents;
  2. for each, the change of the strain energy inside d < rho between the two finest uniform
     levels, delta_c = |E_c(n) - E_c(n/2)|, a local error indicator (the energy error of a
     singular term concentrates at its corner);
  3. grade toward the lines through every corner whose share delta_c / sum(delta) is at
     least SHARE_MIN, each with its own exponent g_c > p / lam_c; leave the others uniform.

Rule constants (fixed before use on the held-out U-shape): SHARE_MIN = 0.10, rho = 0.25.
"""
import numpy as np

SHARE_MIN = 0.10


def coords(n, gmap, lo, hi):
    """Node coordinates on [lo, hi], n (even) elements per unit length; a half unit next to
    an integer line k in gmap is graded toward it as s ** gmap[k], the others are uniform."""
    t = np.linspace(0.0, 1.0, n // 2 + 1)[1:]
    pts = [float(lo)]
    for k in range(int(lo), int(hi)):
        gl, gr = gmap.get(k), gmap.get(k + 1)
        left = k + 0.5 * (t ** gl if gl else t)
        right = (k + 1) - 0.5 * ((1 - t) ** gr if gr else (1 - t))
        pts.extend(left.tolist())
        pts.extend(right.tolist())
    return np.array(pts)


def plan_intensity(polygon, nu, uniform_rows, degree=2, share_min=SHARE_MIN):
    """The plan from the corner analysis and the uniform family's local energies."""
    from omar_pfem.verification_agent.remedy import corner_analysis, plan_grading
    sing = [c for c in corner_analysis(polygon, nu, degree) if c["rate_limiting"]]
    rows = sorted(uniform_rows, key=lambda r: r["n"])
    fine, coarse = rows[-1], rows[-2]
    out = []
    for c in sing:
        key = f"local_energy_{c['x']}_{c['y']}"
        out.append({"x": c["x"], "y": c["y"], "lam": min(c["rate_limiting"]),
                    "delta": abs(fine[key] - coarse[key])})
    tot = sum(c["delta"] for c in out)
    gx, gy = {}, {}
    for c in out:
        c["share"] = c["delta"] / tot
        c["graded"] = c["share"] >= share_min
        if c["graded"]:
            c["g"] = plan_grading(c["lam"], degree)["g"]
            gx[c["x"]] = max(gx.get(c["x"], 0), c["g"])
            gy[c["y"]] = max(gy.get(c["y"], 0), c["g"])
    return {"corners": out, "plan": {"x": gx, "y": gy}, "share_min": share_min,
            "levels_used": [coarse["n"], fine["n"]]}
