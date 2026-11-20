"""F1: the manuscript's B3 groove-stress numbers (paper/main.tex, Sec. B3
stress, the passage "Two summary statistics of the same region ...")
recomputed with the stress-conversion fix (2026-11-19), next to the values
printed in the manuscript (= the pre-fix data, pre_stressfix_20261119/).

Sources: b3_extended.json (corrected, all B3 family/strategy levels),
pre_stressfix_20261119/b3_extended.json (old), manuscript_recompute/
q20_47040.json (the (41,25,50) level re-solved with the fix), contour_log.txt
(the inner-wall pointwise comparison re-run with the fix).

NOT affected (and therefore not recomputed): the 274% groove-region error of
the 839,040 vs 950,400 linear meshes and the surrogate errors (>1,700%):
both come from the evaluator (evaluate_B3_qois*.py), which computes P by
autograd from the displacement and converts once.

Note on wording: the stored `avg` is the SIGNED volume average of sigma_xx
(the old printout labelled it avg|sigma_xx|; the manuscript writes
<|sigma_xx|>): the manuscript should say <sigma_xx>.

    python -m omar_pfem.verification_agent.manuscript_recompute.f1_values
"""
import json
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
VA = os.path.dirname(HERE)


def family(path):
    rows = json.load(open(path))
    fam = {}
    zb = {}
    for r in rows:
        if r.get("quadrature", "reduced") != "reduced" or r.get("r_grading", 1.0) != 1.0:
            continue
        v = (r["per_component"]["xx"]["avg"], r["per_component"]["xx"]["max"])
        if r["z_beta"] is None:
            fam[r["n_elem"]] = v
        elif r["n_elem"] == 10200:
            zb[r["z_beta"]] = v
    zb[0.0] = fam[10200]
    return fam, zb


def main():
    new, new_zb = family(os.path.join(VA, "b3_extended.json"))
    old, old_zb = family(os.path.join(VA, "pre_stressfix_20261119", "b3_extended.json"))
    q47 = os.path.join(HERE, "q20_47040.json")
    extra = None
    if os.path.exists(q47):
        r = json.load(open(q47))
        extra = (r["per_component"]["xx"]["avg"], r["per_component"]["xx"]["max"])
    six = [2376, 5760, 10200, 15960, 23520, 33120]
    avgs_new = [new[n][0] for n in six + [45000]] + ([extra[0]] if extra else [])
    avgs_old = [old[n][0] for n in six + [45000]]
    contour = open(os.path.join(HERE, "contour_log.txt")).read()
    inside = [float(x) for x in re.findall(r"within existing region mask: max=([\d.]+)%", contour)]
    out = {
        "family_new": {n: {"avg_sigma_xx": a, "max_abs_sigma_xx": m} for n, (a, m) in new.items()},
        "family_old": {n: {"avg_sigma_xx": a, "max_abs_sigma_xx": m} for n, (a, m) in old.items()},
        "q20_47040_new": None if extra is None else {"avg_sigma_xx": extra[0],
                                                     "max_abs_sigma_xx": extra[1]},
        "manuscript_vs_new": {
            "avg range over eight resolutions 2,376-47,040 ('0.38-0.44')": {
                "old_data_7_family_levels": [min(avgs_old), max(avgs_old)],
                "new": [min(avgs_new), max(avgs_new)],
                "levels": six + [45000] + (["47040 (41,25,50)"] if extra else [])},
            "max first six levels ('4.13 to 0.706')": {
                "old": [old[2376][1], old[33120][1]], "new": [new[2376][1], new[33120][1]],
                "new_monotone_decreasing": all(new[a][1] > new[b][1]
                                               for a, b in zip(six, six[1:]))},
            "rise at the seventh level ('26%', '0.891 at 45,000')": {
                "old": [old[45000][1] / old[33120][1] - 1, old[45000][1]],
                "new": [new[45000][1] / new[33120][1] - 1, new[45000][1]]},
            "z_beta 0-3 at 10,200 ('0.615-0.826')": {
                "old": [min(v[1] for v in old_zb.values()), max(v[1] for v in old_zb.values())],
                "new": [min(v[1] for v in new_zb.values()), max(v[1] for v in new_zb.values())]},
            "inner wall, finest two levels inside the region ('0.3-1.0%')": {
                "new_per_component_max_percent": inside,
                "new": [min(inside), max(inside)] if inside else None},
            "59,400 (eighth family level, not in the manuscript)": {
                "new": {"avg_sigma_xx": new[59400][0], "max_abs_sigma_xx": new[59400][1]}},
        },
        "unaffected": ["274% (839,040 vs 950,400 linear meshes; evaluator path)",
                       "surrogate errors >1,700% (1,745-5,448%; evaluator path)"],
    }
    json.dump(out, open(os.path.join(HERE, "f1_values.json"), "w"), indent=1)
    print(json.dumps(out["manuscript_vs_new"], indent=1))


if __name__ == "__main__":
    main()
