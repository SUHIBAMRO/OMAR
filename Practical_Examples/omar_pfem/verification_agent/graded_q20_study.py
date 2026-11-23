"""Graded meshes with the reduced Q20 rule (2026-11-23).

Finding to explain: on the 3D benchmark (cylinder3d) the radially graded strategy
g = 2.0 produced an inverted Gauss point (min det F = -1.03), and on B3 strongly
graded meshes gave sign-alternating Gauss-point stresses. Question: is the reduced
2x2x2 rule the cause? Test on the 3D benchmark, where the exact solution is known:
k = 4 with g = 1.0, 1.5, 2.0, each with the reduced and the full (3x3x3, q20_full.py)
rule; errors against the exact values; min det F and where it occurs. Results are
written after every solve (graded_q20_study.json).

    python -m omar_pfem.verification_agent.graded_q20_study
"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "graded_q20_study.json")
CASES = [(4, 1.0, "reduced"), (4, 1.0, "full"), (4, 2.0, "reduced"), (4, 2.0, "full"),
         (4, 1.5, "reduced"), (4, 1.5, "full")]
QOIS = ("strain_energy", "reaction_inner_radial", "u_outer_radial", "gk_avg_hoop",
        "disc_avg_hoop", "peak_hoop")


def main():
    import warnings
    warnings.filterwarnings("ignore")
    from omar_pfem.verification_agent import cylinder3d as C
    ex = json.load(open(os.path.join(HERE, "cylinder3d_exact.json")))
    res = json.load(open(OUT)) if os.path.exists(OUT) else []
    done = {(r["k"], r["grading"], r["quadrature"]) for r in res}
    for k, g, q in CASES:
        if (k, g, q) in done:
            continue
        try:
            r = C.solve(k, g, quadrature=q)
            row = {"k": k, "grading": g, "quadrature": q, "seconds": r["seconds"],
                   "min_detF": r["min_detF"], "min_detF_location": r["min_detF_location"],
                   "errors": {x: (r[x] / ex[x] - 1) if r[x] == r[x] else None for x in QOIS}}
        except Exception as e:                    # a failed solve is a result too
            row = {"k": k, "grading": g, "quadrature": q, "error": repr(e)}
        res.append(row)
        json.dump(res, open(OUT, "w"), indent=1)
        print(json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
