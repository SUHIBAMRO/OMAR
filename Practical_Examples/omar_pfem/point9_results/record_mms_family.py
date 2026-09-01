"""Records the Q4/Q9 family sweep that closes the family half of point 9.

WHAT IT IS. Tables 22-24 compare Q4, Q9 and the operator on ONE manufactured
solution, alpha=0.05, beta=0.7. The operator, however, is scored on a family
of 16 members. A single member against a family mean is not a comparison, so
this sweep solves Q4 and Q9 on the SAME 16 members the operator was scored on
-- drawn with the operator run's own call, `sample_family(16, 31_000_000 + 1)`
-- at N=9, 17 and 33.

The 16 members are recorded below so the identity of the set is checkable and
not merely asserted. (0.05, 0.7) is not among them.

SOURCE. The run wrote its JSON to Drive at
`pfem_run/mms/family/mms_family_fem_B1_neo_hookean.json`; only the Colab
stdout came back. Every number here is transcribed at its printed precision
and `provenance` says so. The operator columns are NOT from this run -- they
are `operator_mean_over_test_family` from the matching
`mms_operator_B1_neo_hookean*.json`, printed by the same cell, and the
single-member column is what Tables 22-24 publish.

Usage: python3 record_mms_family.py
"""
import json
import math
import os

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "mms_family_fem_B1_neo_hookean.json")

MEMBERS = [
    (0.03753, 0.72904), (0.06967, 0.87733), (0.06467, 0.79818),
    (0.04015, 0.69988), (0.05775, 0.68226), (0.05094, 0.73271),
    (0.04126, 0.68476), (0.03183, 0.91898), (0.05471, 0.58488),
    (0.05123, 0.61582), (0.03387, 0.79692), (0.06250, 0.61767),
    (0.06051, 0.96172), (0.05290, 0.63325), (0.03382, 0.61098),
    (0.05624, 0.77890),
]
SINGLE_MEMBER = (0.05, 0.7)   # what Tables 22-24 use

KEYS = ("L2_rel", "H1_semi_rel", "stress_rel_L2", "energy_rel")

# family means, printed by the run: N -> method -> the four numbers
FAMILY = {
    9: {
        "Q4":       (1.3515e-02, 1.1318e-01, 1.1442e-01, 1.2688e-02),
        "Q9":       (4.1550e-04, 5.7632e-03, 5.9812e-03, 3.3314e-05),
        "operator": (8.3792e-03, 1.1456e-01, 1.1593e-01, 1.0151e-02),
    },
    17: {
        "Q4":       (3.4049e-03, 5.6659e-02, 5.7323e-02, 3.1914e-03),
        "Q9":       (5.1628e-05, 1.4380e-03, 1.5008e-03, 2.0902e-06),
        "operator": (8.8261e-03, 5.8712e-02, 5.9368e-02, 9.5955e-03),
    },
    33: {
        "Q4":       (8.5298e-04, 2.8338e-02, 2.8676e-02, 7.9909e-04),
        "Q9":       (6.4423e-06, 3.5928e-04, 3.7557e-04, 1.3077e-07),
        "operator": (1.2358e-02, 4.0947e-02, 4.0003e-02, 6.4063e-03),
    },
}

# what Tables 22-24 print for the operator, on the SINGLE member
SINGLE = {
    9:  (5.0351e-03, 1.1414e-01, 1.1535e-01, 1.1331e-02),
    17: (8.2377e-03, 5.8306e-02, 5.8861e-02, 9.9141e-03),
    33: (1.1362e-02, 3.8498e-02, 3.7993e-02, 5.2892e-03),
}

# spread across the family, stdev/mean, as printed
SPREAD_Q4 = {
    9:  (0.002, 0.000, 0.007, 0.003),
    17: (0.003, 0.000, 0.007, 0.003),
    33: (0.003, 0.000, 0.007, 0.003),
}
# absolute stdev of Q9, as printed
STDEV_Q9 = {
    9:  (5.5e-08, 2.2e-07, 1.3e-04, 2.7e-07),
    17: (1.8e-09, 2.1e-08, 3.3e-05, 1.5e-08),
    33: (6.0e-11, 1.6e-09, 8.3e-06, 9.3e-10),
}

# --- checks
assert len(MEMBERS) == 16 and len(set(MEMBERS)) == 16
assert SINGLE_MEMBER not in MEMBERS, (
    "the single member Tables 22-24 use was drawn into the family, so the "
    "family mean is not independent of it and that must be said, not hidden")
for N, block in FAMILY.items():
    assert block["Q9"][0] < block["Q4"][0], f"N={N}: Q9 is not more accurate"

Ns = sorted(FAMILY)


def rate(a, b, Na, Nb):
    """Two-point observed order, from element size h = L/(N-1).

    h is L/(N-1) for N NODES per side, not L/N. The distinction is not
    cosmetic at these mesh counts: on Q4's L2 it is 1.99 against 2.13, and
    only 1.99 reproduces Table 23's measured 1.98 and the committed
    `fitted_rates_in_h` in mms_operator_rate_B1_neo_hookean.json. A 1/N
    convention was used once in the status file and is wrong.
    """
    return math.log(a / b) / math.log((Nb - 1) / (Na - 1))


def fitted_rate(errs, Ns_):
    """Least-squares slope of log(err) against log(h), h = 1/(N-1).

    This is the estimator `fitted_rates_in_h` uses, so the numbers here can
    be quoted beside it.

    NOTE, and it is why this function tells you nothing extra here: N=9, 17,
    33 are equally spaced in log h, and for three equally spaced abscissae
    the least-squares slope reduces algebraically to the endpoint slope --
    the middle point cancels. So it agrees with `rate` to the last digit on
    every method, INCLUDING the operator, whose three points are visibly not
    collinear. Use `per_interval_rates` to see that, not the agreement of
    these two.
    """
    xs = [math.log(1.0 / (N - 1)) for N in Ns_]
    ys = [math.log(e) for e in errs]
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    return (sum((x - mx) * (y - my) for x, y in zip(xs, ys))
            / sum((x - mx) ** 2 for x in xs))


rows = []
for N in Ns:
    b = FAMILY[N]
    rows.append({
        "N": N,
        "n_members": 16,
        "Q4": dict(zip(KEYS, b["Q4"])),
        "Q9": dict(zip(KEYS, b["Q9"])),
        "operator_family_mean": dict(zip(KEYS, b["operator"])),
        "operator_single_member_as_published": dict(zip(KEYS, SINGLE[N])),
        "operator_over_Q4_on_the_family": {
            k: round(o / q, 4)
            for k, o, q in zip(KEYS, b["operator"], b["Q4"])},
        "Q4_spread_stdev_over_mean": dict(zip(KEYS, SPREAD_Q4[N])),
        "Q9_stdev_absolute": dict(zip(KEYS, STDEV_Q9[N])),
        "operator_family_mean_over_single_member": {
            k: round(f / s, 4)
            for k, f, s in zip(KEYS, b["operator"], SINGLE[N])},
    })

METHODS = ("Q4", "Q9", "operator")
rates = {m: {k: round(rate(FAMILY[Ns[0]][m][i], FAMILY[Ns[-1]][m][i],
                           Ns[0], Ns[-1]), 4)
             for i, k in enumerate(KEYS)}
         for m in METHODS}
fitted = {m: {k: round(fitted_rate([FAMILY[N][m][i] for N in Ns], Ns), 4)
              for i, k in enumerate(KEYS)}
          for m in METHODS}
assert rates["operator"]["L2_rel"] < 0 and fitted["operator"]["L2_rel"] < 0, (
    "the operator's L2 rate came out non-negative, which contradicts the "
    "finding this file records")
# The control: Q4 must land on the rate Table 23 measured, or this sweep is
# not comparable to the published table and nothing here may be quoted beside
# it. Table 23 measured 1.98; mms_operator_rate_B1_neo_hookean.json fitted
# 1.99.
assert abs(rates["Q4"]["L2_rel"] - 1.99) < 0.05, rates["Q4"]["L2_rel"]
assert abs(fitted["Q4"]["L2_rel"] - 1.99) < 0.05, fitted["Q4"]["L2_rel"]
assert abs(rates["Q4"]["H1_semi_rel"] - 1.00) < 0.05, rates["Q4"]["H1_semi_rel"]
# The three meshes are equally spaced in log h, so the two estimators above
# coincide by construction. Their agreement is therefore NOT evidence that the
# error curves are straight; the per-interval rates are what shows that.
for m in METHODS:
    for k in KEYS:
        assert abs(rates[m][k] - fitted[m][k]) < 1e-9, (m, k)
per_interval = {
    m: {f"N{Ns[i]}_to_N{Ns[i + 1]}": {
        k: round(rate(FAMILY[Ns[i]][m][j], FAMILY[Ns[i + 1]][m][j],
                      Ns[i], Ns[i + 1]), 4)
        for j, k in enumerate(KEYS)}
        for i in range(len(Ns) - 1)}
    for m in METHODS}

report = {
    "study": "manufactured solution, Q4 vs Q9 vs the operator, over the "
             "operator's own 16-member test family",
    "geometry": "B1",
    "material": "neo_hookean",
    "Ns": Ns,
    "orders": ["Q4", "Q9"],
    "n_members": 16,
    "family_seed": 31_000_000,
    "family_draw": "sample_family(16, 31_000_000 + 1), verified identical to "
                   "the call mms_operator.py makes, so the FEM and operator "
                   "means are over the SAME 16 problems",
    "members": [{"i": i, "alpha": a, "beta": b}
                for i, (a, b) in enumerate(MEMBERS)],
    "single_member_of_tables_22_24": {"alpha": SINGLE_MEMBER[0],
                                      "beta": SINGLE_MEMBER[1],
                                      "in_family": False},
    "rows": rows,
    "observed_rates_N9_to_N33_two_point": rates,
    "fitted_rates_in_h": fitted,
    "per_interval_rates": per_interval,
    "rate_convention": "h = L/(N-1) for N nodes per side. The two-point rate "
                       "uses the N=9 and N=33 endpoints; fitted_rates_in_h is "
                       "the least-squares slope of log(err) on log(h) over all "
                       "three meshes, the same estimator "
                       "mms_operator_rate_B1_neo_hookean.json uses. THE "
                       "CONTROL: Q4's L2 comes out at "
                       f"{rates['Q4']['L2_rel']} (two-point) and "
                       f"{fitted['Q4']['L2_rel']} (fitted) against Table 23's "
                       "measured 1.98, so this sweep is comparable to the "
                       "published table. A 1/N convention would give 2.13 "
                       "instead and would not reproduce Table 23.",

    "reading": {
        "headline": "the operator does not converge, and refining makes it "
                    "relatively worse. Over N=9 to N=33 the family-mean L2 "
                    f"rates are Q4 {rates['Q4']['L2_rel']}, Q9 "
                    f"{rates['Q9']['L2_rel']}, operator "
                    f"{rates['operator']['L2_rel']} -- the operator's error "
                    "GROWS by 1.47x while Q4's falls by 15.8x. "
                    "operator/Q4 goes 0.62x, 2.59x, 14.49x.",
        "the_single_rate_hides_the_shape": "the operator's L2 rate is "
            f"{rates['operator']['L2_rel']} on both estimators, but that "
            "agreement is an artefact: N=9, 17, 33 are equally spaced in log h, "
            "so the least-squares slope reduces to the endpoint slope and the "
            "middle mesh cancels. Per interval the rate is "
            f"{per_interval['operator']['N9_to_N17']['L2_rel']} from N=9 to 17 "
            f"and {per_interval['operator']['N17_to_N33']['L2_rel']} from N=17 "
            "to 33 -- the error is nearly flat and then climbs, so a single "
            "fitted number understates the finer half. What this sweep "
            "establishes is the SIGN, on every interval, not a magnitude. The "
            "single-member run fitted -0.59 on the same quantity; that it is "
            "negative on a family as well as on one member is the point.",
        "the_0_62x_at_N9_is_not_a_bug": "the ceiling argument constrains PI. "
            "The operator minimises the same discrete functional over the same "
            "Q4 space, so nothing it produces can have lower PI than the Q4 "
            "solution. PI IS NOT L2. A field that does not minimise PI can sit "
            "closer to u* in L2 by partly cancelling Q4's own discretisation "
            "bias, and that is what N=9 does. A previous session raised a "
            "false alarm about exactly this; 8.11 already carries the "
            "correction. Do not 'fix' it.",
        "table_24_is_vindicated_at_N17": "its single-member 2.42x against the "
            "family's 2.59x, so the published comparison was representative "
            "there. At N=9 the single member was materially easier -- operator "
            "5.0351e-03 against the family's 8.3792e-03, 66% worse on the "
            "family -- so any N=9 statement should quote the family. N=33 "
            "moves 9%.",
        "the_FEM_is_member_independent": "Q4's spread across the family is "
            "0.002-0.003 in L2, 0.000 in H1 and 0.007 in stress. The "
            "comparison is therefore not being carried by a lucky member on "
            "the FEM side.",
        "what_is_still_open": "the OPERATOR's spread over the family is not "
            "recoverable. mms_operator*.json stores only "
            "operator_mean_over_test_family, no stdev, so 'is the operator "
            "consistent across the family, or only consistent on average?' "
            "cannot be answered from what is on disk. Answering it means "
            "re-running the operator with per-member output, about 8 minutes "
            "per mesh on an A100 at the measured N=17 time. Worth doing only "
            "if the report wants to make a consistency claim.",
    },

    "cost": "CPU runtime. The three meshes were solved for both orders on all "
            "16 members; the run writes each member's four numbers to "
            "<out_json>.progress as it goes and skips completed members on "
            "restart, so a disconnect costs one member and not the sweep.",

    "provenance": "transcribed from the Colab stdout of Round6_MMS_Family "
                  "(cell_mms_family.py) on 2026-09-01, run at commit f20a6ba. "
                  "The run's own JSON is on Drive at "
                  "pfem_run/mms/family/mms_family_fem_B1_neo_hookean.json. "
                  "Every value carries the PRINTED precision (4-5 significant "
                  "digits). The operator columns are NOT produced by this run: "
                  "they are operator_mean_over_test_family from "
                  "mms_operator_B1_neo_hookean*.json, and the single-member "
                  "column is what Tables 22-24 publish. The ratios and the "
                  "rates in this file are COMPUTED from the transcribed means, "
                  "not transcribed.",
}

with open(OUT, "w") as fh:
    json.dump(report, fh, indent=2)
print(f"wrote {OUT}")
print(f"  {'N':>4}{'Q4 L2':>14}{'Q9 L2':>14}{'operator L2':>14}{'op/Q4':>9}")
for r in rows:
    print(f"  {r['N']:>4}{r['Q4']['L2_rel']:>14.4e}{r['Q9']['L2_rel']:>14.4e}"
          f"{r['operator_family_mean']['L2_rel']:>14.4e}"
          f"{r['operator_over_Q4_on_the_family']['L2_rel']:>8.2f}x")
print(f"  L2 rates in h=1/(N-1), two-point N=9->33:  "
      f"Q4 {rates['Q4']['L2_rel']}  Q9 {rates['Q9']['L2_rel']}  "
      f"operator {rates['operator']['L2_rel']}")
print(f"  L2 rates, least-squares over all three:    "
      f"Q4 {fitted['Q4']['L2_rel']}  Q9 {fitted['Q9']['L2_rel']}  "
      f"operator {fitted['operator']['L2_rel']}")
print(f"  control: Table 23 measured Q4 L2 1.98, H1 1.00; here "
      f"{rates['Q4']['L2_rel']} and {rates['Q4']['H1_semi_rel']}")
