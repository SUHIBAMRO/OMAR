"""Rebuilds the operator run's JSON from the Colab stdout it printed.

The run wrote its own JSON to Google Drive; only the console output came
back. Rather than retype eighty history rows and a results table -- which
is exactly how the point-8 README ended up quoting 3,215 where the run
printed 3,219 -- this parses the captured stdout and regenerates the same
schema `mms_operator.py` writes.

Two consequences of the source being stdout and not the JSON, both recorded
in `provenance` inside the output so no reader has to infer them:

* every number carries the **printed** precision (4-5 significant digits),
  not the run's full float64/float32 value;
* fields the run stores but never prints are **absent**, not guessed --
  the FEM references' `wall_clock_s` and the training wall clock in
  seconds (the cell printed "8.2 min" only).

Usage:
    python3 transcribe_operator_run.py <stdout.txt>
"""
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "mms_operator_B1_neo_hookean.json")

EPOCH_RE = re.compile(
    r"epoch\s+(\d+)\s+trainPi\(family mean\)\s+(-?[\d.]+)\s+"
    r"L2 ([\d.eE+-]+)\s+H1 ([\d.eE+-]+)\s+stress ([\d.eE+-]+)\s+"
    r"energy ([\d.eE+-]+)(\s+\*)?")
ROW_RE = re.compile(
    r"^(Q4 \(same mesh\)|Q9 \(same N\)|operator)\s+"
    r"([\d.e+-]+)\s+([\d.e+-]+)\s+([\d.e+-]+)\s+([\d.e+-]+)$")
KEYS = ("L2_rel", "H1_semi_rel", "stress_rel_L2", "energy_rel")


def main(path):
    raw = open(path).read()

    # ---- training history -------------------------------------------------
    history, best, best_epoch = [], None, None
    for m in EPOCH_RE.finditer(raw):
        ep, pi, l2, h1, st, en, star = m.groups()
        row = {"epoch": int(ep), "opt_steps": int(ep) * 8,
               "train_Pi": float(pi),
               "L2_rel": float(l2), "H1_semi_rel": float(h1),
               "stress_rel_L2": float(st), "energy_rel": float(en)}
        history.append(row)
        # The run flags "  *" exactly when a new best test-family L2 is hit
        # and model_best.pt is written, so the last star IS the checkpoint
        # the final table was scored with. Re-derive it here rather than
        # trusting the flag alone, and assert the two agree.
        if star:
            best_epoch = row["epoch"]
            best = {k: row[k] for k in KEYS}
    assert history, "no epoch lines parsed"
    assert history[-1]["epoch"] == 2000, history[-1]
    lowest = min(history, key=lambda r: r["L2_rel"])
    assert lowest["epoch"] == best_epoch, (lowest["epoch"], best_epoch)
    steps = history[-1]["opt_steps"]
    assert steps == 16000, steps          # the cell printed "16,000 steps"

    # ---- the three-way table ---------------------------------------------
    # The cell prints the table twice (the script's own, then the notebook's
    # re-print). Parse whichever form appears and require the two to agree.
    seen = {}
    for line in raw.replace("\r", "").split("\n"):
        line = " ".join(line.split())
        m = ROW_RE.match(line)
        if m:
            name, *vals = m.groups()
            vals = [float(v) for v in vals]
            if name in seen:
                assert seen[name] == vals, f"{name} printed twice, differently"
            seen[name] = vals
    # The pasted transcript is a single re-flowed line, so fall back to a
    # scan of the whole text when the line-based parse finds nothing.
    if len(seen) < 3:
        for name in ("Q4 (same mesh)", "Q9 (same N)", "operator"):
            hits = re.findall(
                re.escape(name) + r"(?: \(this run\))?\s+"
                r"([\d.]+e[+-]\d+)\s+([\d.]+e[+-]\d+)\s+"
                r"([\d.]+e[+-]\d+)\s+([\d.]+e[+-]\d+)", raw)
            assert hits, "table row not found: " + name
            vals = [float(v) for v in hits[0]]
            for h in hits[1:]:
                assert [float(v) for v in h] == vals, name + " differs"
            seen[name] = vals
    assert len(seen) == 3, sorted(seen)
    single = dict(zip(KEYS, seen["operator"]))

    # The FEM references are NOT transcribed at 3 digits. This run re-solved
    # the same two problems the FEM half already solved -- same mesh, same
    # solver, FP64 on CPU -- so the full-precision values already sit in
    # mms_B1_neo_hookean.json, which is what report Table 22 quotes. Adopt
    # those, and require the run's printed values to agree before doing so;
    # if they ever disagree this stops rather than silently preferring one.
    study = json.load(open(os.path.join(HERE, "mms_B1_neo_hookean.json")))
    ref = {}
    for order in ("Q4", "Q9"):
        row = next(r for r in study["rows"]
                   if r["order"] == order and r["N"] == 17)
        ref[order] = {k: row[k] for k in KEYS}
        ref[order]["n_dof"] = row["n_dof"]
        for k, v in zip(KEYS, seen[{"Q4": "Q4 (same mesh)",
                                    "Q9": "Q9 (same N)"}[order]]):
            assert abs(v / row[k] - 1) < 2e-3, (order, k, v, row[k])
        # the solver also prints a 4-digit line of its own; check that too
        m = re.search(order + r" N=17: L2 ([\d.e+-]+)\s+H1 ([\d.e+-]+)", raw)
        assert m, order
        assert abs(float(m.group(1)) / row["L2_rel"] - 1) < 1e-4
        assert abs(float(m.group(2)) / row["H1_semi_rel"] - 1) < 1e-4
    assert ref["Q4"]["n_dof"] == 578, ref["Q4"]["n_dof"]

    ratio = single["L2_rel"] / ref["Q4"]["L2_rel"]
    printed = float(re.search(r"operator / Q4 in L2: ([\d.]+)x", raw).group(1))
    assert abs(ratio - printed) < 0.005, (ratio, printed)
    assert ratio > 1.0, "operator/Q4 below 1.0 -- that is a bug, not a result"

    rep = {
        "study": "MMS, physics-informed operator third",
        "material": "neo_hookean", "N": 17, "n_dof": 578,
        "family": {"alpha_range": [0.03, 0.07], "beta_range": [0.5, 1.0],
                   "ntrain": 64, "ntest": 16},
        "training": {
            "principle": "physics-informed, Pi = U - W, no labels",
            "optimizer": "Adam lr=0.002 wd=0.0",
            "epochs": 2000, "opt_steps": steps, "batch_size": 8,
            "train_wall_clock_min": 8.2,
            "input_norm_bx_by": {"mean": [1.2482719732176717, 0.9440896721787019],
                                 "std": [1.3354366675853482, 1.2726065527134676]},
            "label_cost": "zero: u* is analytic, and it is not used in "
                          "training at all -- the loss is the energy"},
        "operator_mean_over_test_family": best,
        "operator_best_epoch": best_epoch,
        "operator_on_the_reference_member": single,
        "fem_reference_same_mesh": ref,
        "ceiling": (
            "The operator minimizes the SAME discrete functional over the SAME "
            "Q4 space as the Q4 solver, so the Q4 solution is the minimizer and "
            "the operator cannot beat it at this mesh. operator/Q4 is the "
            "quantity of interest: 1.0 would mean the network has fully solved "
            "the variational problem."),
        "functional_verified": {
            "by": "omar_pfem/test_mms_operator.py, run immediately before this "
                  "training run in the same cell",
            "Pi_at_u_FEM": -7.999050349300,
            "Pi_at_interpolated_u_star": -7.997639755811,
            "quadratic_excess_ratio": 4.000,
            "expected_quadratic_excess_ratio": 4.0,
            "best_scale_correct_W": 1.000,
            "best_scale_W_divided_by_8": 0.125,
            "note": "the last pair is the meta-check: a W wrongly divided by 8 "
                    "moves the minimum to 1/8, so the checks above can fail"},
        "device": "cuda", "gpu": "NVIDIA A100-SXM4-80GB",
        "dtype": "float32 (training), float64 (references)",
        "history": history,
        "operator_over_Q4_L2": ratio,
        "provenance": {
            "source": "Colab stdout of Round6_MMS_Operator on an A100, "
                      "transcribed by transcribe_operator_run.py",
            "why_not_the_run_s_own_json": "the run wrote its JSON to Google "
                                          "Drive; only the console output was "
                                          "returned to this repository",
            "precision": "operator values carry the PRINTED precision (4 "
                         "significant digits), not the run's full float value",
            "fem_references": "taken at full precision from "
                              "mms_B1_neo_hookean.json (the same mesh, solver "
                              "and FP64 the operator run re-solved), after "
                              "checking the run's printed values agree",
            "absent_not_guessed": [
                "fem_reference_same_mesh.*.wall_clock_s -- the operator run "
                "did not print it, and the FEM half's timings are a different "
                "machine",
                "training.train_wall_clock_s -- the cell printed 8.2 min only"],
        },
    }
    with open(OUT, "w") as f:
        json.dump(rep, f, indent=2)
    print(f"wrote {OUT}")
    print(f"  {len(history)} history rows, best L2 at epoch {best_epoch}")
    print(f"  operator/Q4 in L2 = {ratio:.4f}x  (run printed {printed}x)")
    for k in KEYS:
        print(f"  {k:<14} Q4 {ref['Q4'][k]:.3e}  Q9 {ref['Q9'][k]:.3e}  "
              f"op {single[k]:.3e}   op/Q4 {single[k]/ref['Q4'][k]:.2f}x")


if __name__ == "__main__":
    main(sys.argv[1])
