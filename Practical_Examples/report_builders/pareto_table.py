"""The point-2 Pareto table, built once from the result JSON and shared by
the report and summary builders, so the two documents cannot diverge.

Timings come from **run4**, not run3. Both runs produced byte-identical
errors, but run3 ran on a much faster Colab instance and its operator
latency (1.610 ms at N=21) is a third of what Table 10a measures for the
same architecture on the same 441-node mesh (4.582 ms). run4's N=21 figure
is 4.584 ms. Using run3 would put a speed-up column in the report that
disagrees with the report's own inference-latency table.
"""
import os
import json

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "..", "omar_pfem", "point2_results",
                   "pareto_B1_neo_hookean.json")

with open(SRC) as f:
    DATA = json.load(f)

RUN = "run4"
HEAD = ["N", "Nodes", "FEM rel. L2 (%)", "FEM cost (s)",
        "Operator rel. L2 (%)", "Operator cost (ms)", "Speed-up"]


def rows():
    out = []
    for r in DATA["rows"]:
        fem_ms = r[RUN]["fem_ms_per_sample"]
        op_ms = r[RUN]["operator_ms_per_sample"]
        out.append([
            str(r["N"]), f"{r['n_nodes']:,}",
            f"{100 * r['fem_rel_L2']:.3f}", f"{fem_ms / 1000:.1f}",
            f"{100 * r['operator_rel_L2']:.2f}", f"{op_ms:.3f}",
            f"{fem_ms / op_ms:,.0f}×"])
    return out


def _v(key):
    return [r[key] for r in DATA["rows"]]


def facts():
    fem, op = _v("fem_rel_L2"), _v("operator_rel_L2")
    ns = _v("N")
    speed = [r[RUN]["fem_ms_per_sample"] / r[RUN]["operator_ms_per_sample"]
             for r in DATA["rows"]]
    op_ms = [r[RUN]["operator_ms_per_sample"] for r in DATA["rows"]]
    best_i = op.index(min(op))
    f = {
        "fem_lo": 100 * min(fem), "fem_hi": 100 * max(fem),
        "op_lo": 100 * min(op), "op_hi": 100 * max(op),
        "op_best_N": ns[best_i],
        "coarsest_fem": 100 * fem[0], "coarsest_fem_s": DATA["rows"][0][RUN]["fem_ms_per_sample"] / 1000,
        "advantage": min(op) / fem[0],
        "speed_lo": min(speed), "speed_hi": max(speed),
        "op_ms_lo": min(op_ms), "op_ms_hi": max(op_ms),
        "n_samples": DATA["n_samples"] if "n_samples" in DATA else 20,
    }
    return f


if __name__ == "__main__":
    print(" | ".join(HEAD))
    for r in rows():
        print(" | ".join(r))
    for k, v in facts().items():
        print(f"{k:<16} {v}")
