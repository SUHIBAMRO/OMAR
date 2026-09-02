"""Records the B1 x Arruda-Boyce accuracy/cost Pareto -- the third and last
of the three B1 materials (report Table 18 covers Neo-Hookean; this
directory's Mooney-Rivlin file covers the second).

SOURCE: the run's own JSON, fetched directly from Google Drive
(file id 10Jt2W0sCIHhoVG1Tb9NALMeEEhjaW3ua, `pareto_B1_arruda_boyce.json`
under pfem_run/zeroshot_B1_arruda_boyce/), not transcribed from stdout --
every field below is copied verbatim from that file. Run was resumed twice
(a container restart, then a stale-tab false-completion scare, both
resolved) but the JSON's own resume protocol means the final file is
identical to what an uninterrupted run would have produced: fingerprinted,
one row written to disk per completed resolution.

Usage: python3 record_pareto_b1_arruda_boyce.py
"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "pareto_B1_arruda_boyce.json")

NH = json.load(open(os.path.join(HERE, "pareto_B1_neo_hookean.json")))
MR = json.load(open(os.path.join(HERE, "pareto_B1_mooney_rivlin.json")))

# Verbatim from the Drive JSON (fetched directly, not transcribed from
# stdout -- see module docstring).
RAW_ROWS = [
    {"N": 13, "n_nodes": 169, "fem_rel_L2": 0.004743942256311409,
     "fem_ms_per_sample": 19810.790436495154, "operator_rel_L2": 0.059626544559446736,
     "operator_ms_per_sample": 5.738981002650689},
    {"N": 17, "n_nodes": 289, "fem_rel_L2": 0.0029037681365570696,
     "fem_ms_per_sample": 34741.10582900175, "operator_rel_L2": 0.05303469496537065,
     "operator_ms_per_sample": 5.712557001970708},
    {"N": 21, "n_nodes": 441, "fem_rel_L2": 0.002012941009657209,
     "fem_ms_per_sample": 54178.1589184975, "operator_rel_L2": 0.04749855096248155,
     "operator_ms_per_sample": 5.763072498666588},
    {"N": 25, "n_nodes": 625, "fem_rel_L2": 0.001481752122112069,
     "fem_ms_per_sample": 79048.4077295041, "operator_rel_L2": 0.042517629902418766,
     "operator_ms_per_sample": 6.034473000909202},
    {"N": 29, "n_nodes": 841, "fem_rel_L2": 0.0011297527147217266,
     "fem_ms_per_sample": 104617.07368700013, "operator_rel_L2": 0.03844309030755581,
     "operator_ms_per_sample": 5.581877000167879},
    {"N": 33, "n_nodes": 1089, "fem_rel_L2": 0.0008891523245995429,
     "fem_ms_per_sample": 136560.3936604998, "operator_rel_L2": 0.03599713154614258,
     "operator_ms_per_sample": 5.556230500133097},
    {"N": 37, "n_nodes": 1369, "fem_rel_L2": 0.000720817395245822,
     "fem_ms_per_sample": 173292.76901400043, "operator_rel_L2": 0.03531962684897363,
     "operator_ms_per_sample": 5.068118000053801},
    {"N": 41, "n_nodes": 1681, "fem_rel_L2": 0.0005918747252323142,
     "fem_ms_per_sample": 213884.34966000114, "operator_rel_L2": 0.03641835821599402,
     "operator_ms_per_sample": 5.562618000112707},
    {"N": 49, "n_nodes": 2401, "fem_rel_L2": 0.0004173146056831542,
     "fem_ms_per_sample": 308059.24427349964, "operator_rel_L2": 0.04274343674065294,
     "operator_ms_per_sample": 5.628570001135813},
]
CHECKPOINT_FINGERPRINT = "bff6d7f2af589477c00720d2aec0c7870f5b1d61ec344be57d70b4c01b2792eb"

NS = [r["N"] for r in RAW_ROWS]
assert NS == [13, 17, 21, 25, 29, 33, 37, 41, 49]

rows = [dict(r, speedup=r["fem_ms_per_sample"] / r["operator_ms_per_sample"])
        for r in RAW_ROWS]

op = {r["N"]: r["operator_rel_L2"] for r in rows}
fem = {r["N"]: r["fem_rel_L2"] for r in rows}
speedup = {r["N"]: r["speedup"] for r in rows}

# --- shape: where is the operator's minimum?
op_min_N = min(op, key=op.get)
IS_MONOTONE_DECREASING = all(op[NS[i]] >= op[NS[i + 1]] for i in range(len(NS) - 1))
assert not IS_MONOTONE_DECREASING, (
    "expected AB to bottom out before N=49, like Neo-Hookean and unlike "
    "Mooney-Rivlin -- re-check the shape claim below if this fires")
assert op_min_N == 37, op_min_N

# --- the operator never matches even the cheapest FEM solve
assert fem[13] < min(op.values()), (
    "the headline claim (coarsest FEM already beats the operator's best) "
    "does not hold")
FEM_ACCURACY_RATIO = min(op.values()) / fem[13]  # operator best error / FEM error at N=13

# --- speed-up range, and comparison to the other two B1 materials
SPEEDUP_LO, SPEEDUP_HI = min(speedup.values()), max(speedup.values())
NH_SPEEDUPS = [r["run4"]["fem_ms_per_sample"] / r["run4"]["operator_ms_per_sample"]
               for r in NH["rows"]]
MR_SPEEDUPS = [r["speedup"] for r in MR["rows"]]
NH_LO, NH_HI = min(NH_SPEEDUPS), max(NH_SPEEDUPS)
MR_LO, MR_HI = min(r["speedup"] for r in MR["rows"]), max(r["speedup"] for r in MR["rows"])

print(f"AB speed-up {SPEEDUP_LO:,.0f}x-{SPEEDUP_HI:,.0f}x "
      f"(NH {NH_LO:,.0f}x-{NH_HI:,.0f}x, MR {MR_LO:,.0f}x-{MR_HI:,.0f}x)")
print(f"AB operator error bottoms at N={op_min_N} ({op[op_min_N]:.4f}), "
      f"rises after -- same shape as Neo-Hookean, unlike Mooney-Rivlin's "
      f"monotone decrease to N=49")
print(f"cheapest FEM solve (N=13, {fem[13]:.4%}) is "
      f"{FEM_ACCURACY_RATIO:.1f}x more accurate than the "
      f"operator's best ({op[op_min_N]:.4%} at N={op_min_N})")

report = {
    "geometry": "B1",
    "material": "arruda_boyce",
    "checkpoint": "pfem_run/zeroshot_B1_arruda_boyce/model_best.pt",
    "checkpoint_fingerprint": CHECKPOINT_FINGERPRINT,
    "fine_N": 101,
    "batch_size": 1,
    "n_samples": 20,
    "solver_side": "the CPU reference solver, the cost of producing a new "
        "solution today; the GPU-native solver of section 8.5 is "
        "71.7-171.5x faster and shifts the speed-up column down by about "
        "two orders of magnitude without touching accuracy",
    "metric": "combined relative L2 over both displacement components, "
        "||e||/||u||, every row scored against the SAME N=101 reference -- "
        "the convergence-study convention of section 4.4, not the "
        "per-component average of Tables 5, 11 and 12",
    "rows": rows,

    "shape": {
        "operator_monotone_decreasing_in_N": False,
        "operator_best": {"N": op_min_N, "error": op[op_min_N]},
        "note": f"the operator error falls from {op[13]:.4f} at N=13 to "
            f"{op[op_min_N]:.4f} at N={op_min_N}, then RISES to "
            f"{op[49]:.4f} at N=49 -- a minimum inside the range, the same "
            f"shape Table 18 shows for Neo-Hookean (which bottoms at N=37 "
            f"too) and unlike Mooney-Rivlin, whose error falls monotonically "
            f"all the way to N=49. Of the three B1 materials, Mooney-Rivlin "
            f"is the exception, not the rule.",
    },

    "reading": {
        "the_two_methods_do_not_compete_on_accuracy":
            f"the finite-element solver at its coarsest setting -- N=13, "
            f"169 nodes, {fem[13]*1000:.1f} ms per sample -- reaches "
            f"{fem[13]:.4%}, already {FEM_ACCURACY_RATIO:.1f}x "
            f"more accurate than the operator at its best "
            f"({op[op_min_N]:.4%} at N={op_min_N}). There is no mesh in "
            f"this sweep at which the operator matches even the cheapest "
            f"finite-element solve, exactly as for the other two B1 "
            f"materials.",
        "against_the_other_two_B1_materials":
            f"Arruda-Boyce's speed-up spans {SPEEDUP_LO:,.0f}x to "
            f"{SPEEDUP_HI:,.0f}x, between Neo-Hookean's {NH_LO:,.0f}x-"
            f"{NH_HI:,.0f}x and Mooney-Rivlin's {MR_LO:,.0f}x-{MR_HI:,.0f}x "
            f"-- all three B1 materials sit in the same order-of-magnitude "
            f"band, and all three grow with resolution because operator "
            f"inference is flat in mesh size while the FEM solve is not.",
        "all_three_B1_pareto_sweeps_are_now_complete":
            "Neo-Hookean, Mooney-Rivlin and Arruda-Boyce all have full "
            "9-resolution Pareto sweeps. The B2 side (cell_pareto_B2.py) "
            "is separate and not yet complete for all three materials.",
    },

    "cost": "A100. Restarted twice across this session -- once for a "
        "container restart that (harmlessly) re-verified N=13-25 were "
        "already on disk, once to resume from N=29 after the sweep was "
        "interrupted -- final wall clock for the N=29-49 segment was "
        "5h12m30s. Nothing was lost or recomputed: the resume protocol "
        "in pareto_analysis.py reads completed resolutions back from the "
        "JSON and only computes what is missing.",

    "provenance": "fetched directly from Google Drive "
        "(pfem_run/zeroshot_B1_arruda_boyce/pareto_B1_arruda_boyce.json, "
        "file id 10Jt2W0sCIHhoVG1Tb9NALMeEEhjaW3ua), not transcribed from "
        "Colab stdout -- every row above is the run's own recorded value.",
}

with open(OUT, "w") as fh:
    json.dump(report, fh, indent=2)
print(f"\nwrote {OUT}")
