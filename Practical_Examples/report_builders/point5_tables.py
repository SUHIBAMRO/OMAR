"""The point-5 tables, built once from the result JSONs and shared.

Both `make_v29.py` (the report) and `make_summary_v3.py` (the parallel summary)
import their rows from here. Last round the same tables were typed into two
build scripts and then compared cell by cell afterwards to prove they matched;
building them once removes the possibility of a mismatch instead of checking
for one.

Nothing here is typed by hand. Every number is read from
`omar_pfem/point5_results/physical_quantities_{geometry}_{material}.json` and
formatted at build time, so a re-run of the evaluation only needs the JSONs
replaced.
"""
import os
import json

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "..", "omar_pfem", "point5_results")

CASES = [("B1", "neo_hookean", "B1 × Neo-Hookean"),
         ("B1", "mooney_rivlin", "B1 × Mooney-Rivlin"),
         ("B1", "arruda_boyce", "B1 × Arruda-Boyce"),
         ("B2", "neo_hookean", "B2 × Neo-Hookean"),
         ("B2", "mooney_rivlin", "B2 × Mooney-Rivlin"),
         ("B2", "arruda_boyce", "B2 × Arruda-Boyce")]


def load(geometry, material):
    path = os.path.join(RESULTS, f"physical_quantities_{geometry}_{material}.json")
    with open(path) as f:
        return json.load(f)


ALL = {(g, m): load(g, m) for g, m, _ in CASES}


def pct(metrics, key, field="mean"):
    """A metric as a percentage, to two decimals."""
    return f"{100.0 * metrics[key][field]:.2f}"


def pct_mean_max(metrics, key):
    """`mean (worst sample)`, both as percentages -- the advisor asked for
    maxima, and a mean alone hides that some samples are much worse."""
    return f"{100.0 * metrics[key]['mean']:.2f} ({100.0 * metrics[key]['max']:.1f})"


# ------------------------------------------------------------------ Table 15
NORMS_HEAD = ["Case", "Displacement rel. L2", "L2 rel.",
              "H1 semi-norm rel.", "Tangent-energy rel."]


def norms_rows():
    rows = []
    for g, m, label in CASES:
        M = ALL[(g, m)]["metrics"]
        rows.append([label] + [pct_mean_max(M, k) for k in
                               ("disp_rel_L2", "L2_rel", "H1_semi_rel", "energy_rel")])
    return rows


# ------------------------------------------------------------------ Table 16
STRESS_HEAD = ["Case", "‖P‖F", "P11", "P22", "P12", "P21", "peak ‖P‖"]


def stress_rows():
    rows = []
    for g, m, label in CASES:
        M = ALL[(g, m)]["metrics"]
        rows.append([label] + [pct(M, k) for k in
                               ("P_rel_L2", "P11_rel_L2", "P22_rel_L2",
                                "P12_rel_L2", "P21_rel_L2", "P_peak_rel_err")])
    return rows


# ------------------------------------------------------------------ Table 17
# B1 fixes both components on one edge; B2 fixes one component on each of two
# symmetry edges, so B2 contributes two rows. Only the two quantities present
# for every case and every edge are tabulated -- see point5_results/README.md.
REACTION_HEAD = ["Case", "Constrained boundary",
                 "Resultant rel. err.", "Nodal rel. L2"]

B2_EDGES = [("edge0", "θ = 0 (u_y fixed)"),
            ("edge1", "θ = π/2 (u_x fixed)")]


def reaction_rows():
    rows = []
    for g, m, label in CASES:
        M = ALL[(g, m)]["metrics"]
        if g == "B1":
            rows.append([label, "bottom edge (u_x, u_y fixed)",
                         pct(M, "reaction_resultant_rel_err"),
                         pct(M, "reaction_nodal_rel_L2")])
        else:
            for suf, name in B2_EDGES:
                rows.append([label, name,
                             pct(M, f"reaction_resultant_rel_err_{suf}"),
                             pct(M, f"reaction_nodal_rel_L2_{suf}")])
    return rows


# ------------------------------------------------- figures quoted in the text
def span(key, geometries=("B1", "B2"), field="mean"):
    """min-max of a metric across the named geometries, as a percentage
    string, so a sentence's range cannot drift from its table."""
    vals = [100.0 * ALL[(g, m)]["metrics"][key][field]
            for g, m, _ in CASES if g in geometries]
    return f"{min(vals):.2f}–{max(vals):.2f}"


def value(geometry, material, key, field="mean"):
    return ALL[(geometry, material)]["metrics"][key][field]


def reaction_span(base, field="mean"):
    """min-max of a reaction metric over every case and, for B2, every edge --
    `base` is the un-suffixed name, e.g. "reaction_resultant_rel_err"."""
    vals = []
    for g, m, _ in CASES:
        M = ALL[(g, m)]["metrics"]
        keys = [base] if g == "B1" else [f"{base}_edge0", f"{base}_edge1"]
        vals += [100.0 * M[k][field] for k in keys]
    return f"{min(vals):.2f}–{max(vals):.2f}"


if __name__ == "__main__":
    def show(head, rows):
        print(" | ".join(head))
        for r in rows:
            print(" | ".join(r))
        print()
    show(NORMS_HEAD, norms_rows())
    show(STRESS_HEAD, stress_rows())
    show(REACTION_HEAD, reaction_rows())
    for k in ("disp_rel_L2", "H1_semi_rel", "energy_rel", "P_rel_L2",
              "P_peak_rel_err"):
        print(f"{k:<16} B1 {span(k, ('B1',)):>14}   B2 {span(k, ('B2',)):>14}")
