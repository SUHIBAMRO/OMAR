"""Human intervention in the verification-agent work, 2026-11-16 .. 11-19.

Every entry is taken from PROJECT_STATUS.md (the project log) and points to
the log entry it comes from. "Developer" means the people who wrote the
framework code (outside the framework); "Omar" means decisions/approvals by
the project owner; LLM roles (formalizer, proposer, critic, advisor) and
automated checks are parts of the framework.

The script counts the entries by category, by who FOUND the issue and by
whether the framework now does the step itself, and writes
HUMAN_INTERVENTION.md.

    python -m omar_pfem.verification_agent.human_intervention
"""
import collections
import os

HERE = os.path.dirname(os.path.abspath(__file__))

# categories
INPUT = "input/description corrected"
BUG = "framework bug fixed"
DESIGN = "design change"
INVEST = "manual diagnostic investigation"
OPS = "operations (compute, memory, restarts)"
DECIDE = "decision reserved for a human"

# found_by
DEV, OMAR, CRITIC, FORM, ADV, AUTO = ("developer", "Omar", "LLM critic", "LLM formalizer",
                                      "LLM advisor", "automated check")

# now: does the framework now do it without a person?
YES, PART, NO = "yes", "partly", "no"

E = [
    # ---- inputs / descriptions corrected --------------------------------------------------
    dict(id="I1", date="11-16", cat=INPUT, found=CRITIC, now=NO,
         what="B3 adapter description gave a wrong mesh family (5,040-47,040 el; true 2,376-45,000)",
         effect="part of the B3 critic's objections were caused by it"),
    dict(id="I2", date="11-17", cat=INPUT, found=FORM, now=PART,
         what="problems/b1_square.txt said Poisson's ratio varies (it is 0.3 constant)",
         effect="formalizer flagged the contradiction; fixed by hand"),
    dict(id="I3", date="11-17", cat=INPUT, found=DEV, now=NO,
         what="B3 description said surrogate errors are 1.4-4% (true for the moment only; "
              "groove stresses 600-13,000%)",
         effect="LLM tightened every tolerance to 0.3%; moment/energy and all stress averages "
                "FAILED in live run v3. Led to code-set tolerances (D11)"),
    dict(id="I4", date="11-17", cat=INPUT, found=DEV, now=NO,
         what="B3 rotation described as torsion; it is a tilt about the y axis",
         effect="description only"),
    dict(id="I5", date="11-17", cat=INPUT, found=CRITIC, now=NO,
         what="facts missing from the adapter descriptions (B1: nu, plane strain, load, region, "
              "query points; B3: phi, sphere radius/centre, groove shape, integration rule, "
              "stress scale, unequal ratios)",
         effect="open questions B1 9 -> 6, B3 10 -> 8 after adding them"),
    dict(id="I6", date="11-17", cat=INPUT, found=DEV, now=NO,
         what="trees.py comment carried the same wrong 1.4-4% range", effect="comment only"),
    dict(id="I7", date="11-17", cat=INPUT, found=DEV, now=NO,
         what="B3 Q20 elements use the reduced 2x2x2 rule (found while verifying the description)",
         effect="led to the full-integration check (V3)"),

    # ---- framework bugs ------------------------------------------------------------------
    dict(id="B1", date="11-16", cat=BUG, found=DEV, now=NO,
         what="convergence gate sent a divergent sequence into the Celik branch "
              "(printed p = 0.35 and a meaningless extrapolated value)",
         effect="verdict was right (FAIL), reason wrong; 8 repository records re-recorded"),
    dict(id="B2", date="11-16", cat=BUG, found=DEV, now=NO,
         what="pipeline crashed after rule augmentation (critic not called on the final plan; "
              "trace not saved)", effect="live B1 run 2 lost"),
    dict(id="B3", date="11-16", cat=BUG, found=ADV, now=PART,
         what="display bug: 'within 0% band' for a 0.5% band", effect="text only"),
    dict(id="B4", date="11-17", cat=BUG, found=DEV, now=NO,
         what="checks added by the rules took the tolerance of the last QoI class",
         effect="tolerances then clamped; fixed to each class's own ceiling"),
    dict(id="B5", date="11-17", cat=BUG, found=DEV, now=NO,
         what="multilevel reason text claimed a leave-one-out range with only 4 levels",
         effect="text only"),
    dict(id="B6", date="11-17", cat=BUG, found=DEV, now=NO,
         what="adaptive controller decided FAIL after the 3 coarsest B3 levels",
         effect="wrong early FAIL for sphere averages that pass on the full family"),
    dict(id="B7", date="11-17", cat=BUG, found=DEV, now=NO,
         what="claim audit (first version) gave 3 false positives on real traces",
         effect="rules refined; 0 flags on 6 live traces"),
    dict(id="B8", date="11-17", cat=BUG, found=DEV, now=NO,
         what="combine_verdicts accepted a 3-level rate although the 5-level gate found the "
              "sequence non-monotone", effect="2 verdicts rate -> band after the fix"),
    dict(id="B9", date="11-17", cat=BUG, found=DEV, now=NO,
         what="B1 default spec had a fixed QoI list without the Gaussian average",
         effect="interior_gk not checked until fixed"),
    dict(id="B10", date="11-17", cat=BUG, found=DEV, now=NO,
         what="B3 radial strategy r_grading 1.5/2/3 produced inverted or near-degenerate "
              "elements; CG did not finish", effect="replaced by 1.1/1.2/1.3 with a direct solver"),
    dict(id="B11", date="11-17", cat=BUG, found=DEV, now=NO,
         what="surface study extrapolated to corner nodes only; no node at the groove root "
              "on some levels", effect="root value sat 0.015 away; redone with 20-node extrapolation"),
    dict(id="B12", date="11-18", cat=BUG, found=DEV, now=NO,
         what="report: full-vs-reduced integration numbers given per level, labelled per quantity",
         effect="report text corrected"),
    dict(id="B13", date="11-18", cat=BUG, found=AUTO, now=YES,
         what="15 number mismatches in the technical report (stale 7-level values, rounding, "
              "a % of scale read as relative)",
         effect="found by verify_report.py (written by the developer); 373/373 after fixes"),
    dict(id="B14", date="11-18", cat=BUG, found=DEV, now=NO,
         what="B2 query on graded meshes took the nearest node (0.32% artificial spread)",
         effect="exact edge interpolation: 0.005%"),
    dict(id="B15", date="11-19", cat=BUG, found=DEV, now=NO,
         what="diagnosis engine, first version: D3 always supported, D1 for averages, M1 for "
              "B3 maxima, Q1 too strict", effect="over-attribution; refined before any result was used"),
    dict(id="B16", date="11-19", cat=BUG, found=DEV, now=NO,
         what="diagnosis action re-solved B2 at Newton tol 1e-11, below the residual floor "
              "(~8e-11): never converged, 867 s", effect="changed to 1e-10 (88 s, S1 refuted)"),
    dict(id="B17", date="11-19", cat=BUG, found=DEV, now=NO,
         what="diagnosis Q2 used combined verdicts of the signed components (strategy FAIL "
              "counted as non-convergence)", effect="the one mismatch with the manual findings; "
                                                    "fixed -> 16/16"),

    # ---- design changes from critic objections -------------------------------------------
    dict(id="D1", date="11-16", cat=DESIGN, found=CRITIC, now=NO,
         what="observed order must lie in the element's window; rate tried before band",
         effect="every B1 pass now has rate evidence"),
    dict(id="D2", date="11-16", cat=DESIGN, found=CRITIC, now=NO,
         what="anchored robustness (strategies vs the Richardson limit)", effect="new check"),
    dict(id="D3", date="11-16", cat=DESIGN, found=CRITIC, now=NO,
         what="per-QoI-class tolerance ceilings (5% GCI too loose)", effect="new ceilings"),
    dict(id="D4", date="11-16", cat=DESIGN, found=CRITIC, now=NO,
         what="global QoIs for B3 (reaction moment, strain energy), J > 0 and solver-tolerance "
              "checks", effect="~2.3 h CPU re-solve; best-verified B3 quantities"),
    dict(id="D5", date="11-17", cat=DESIGN, found=CRITIC, now=NO,
         what="anchored robustness INCONCLUSIVE when no converged family (both critics)",
         effect="no overall verdict changed"),
    dict(id="D6", date="11-17", cat=DESIGN, found=CRITIC, now=NO,
         what="rule R7: strategy robustness requires anchored robustness", effect="new rule"),
    dict(id="D7", date="11-17", cat=DESIGN, found=CRITIC, now=NO,
         what="reduced-integration check (spurious modes)", effect="see V3"),
    dict(id="D8", date="11-17", cat=DESIGN, found=CRITIC, now=NO,
         what="Gauss-point maxima are a biased proxy of the surface peak", effect="surface study (V5)"),
    dict(id="D9", date="11-17", cat=DESIGN, found=CRITIC, now=NO,
         what="radial refinement strategy for B3 (only axial existed)", effect="3 radial levels; "
                                                                             "3 sharp verdicts changed"),
    dict(id="D10", date="11-17", cat=DESIGN, found=CRITIC, now=NO,
         what="expected order from the corner/edge singular exponent", effect="singularity.py, R8"),
    dict(id="D11", date="11-17", cat=DESIGN, found=DEV, now=NO,
         what="tolerances set by code from the surrogate error (margin 3); LLM may not set them",
         effect="verdicts no longer depend on per-run LLM choices (32/96 pairs replaced in v4)"),
    dict(id="D12", date="11-17", cat=DESIGN, found=CRITIC, now=NO,
         what="Gaussian counterparts for cyl_zz and sigma_xz", effect="added to the B3 adapter"),

    # ---- manual diagnostic investigations ------------------------------------------------
    dict(id="V1", date="11-17", cat=INVEST, found=DEV, now=YES,
         what="region-definition hypothesis: sharp sphere vs Gaussian weights (region_study.py)",
         effect="diagnose.py Q1 now does this, incl. computing missing Gaussian averages"),
    dict(id="V2", date="11-17", cat=INVEST, found=DEV, now=YES,
         what="Newton/linear-solver tolerance checks (B1 tol 1e-7 vs 1e-10; B3 rtol 1e-8 vs 1e-11)",
         effect="diagnose.py S1/S2 read them; B2 tolerance and B3 direct solver run as actions"),
    dict(id="V3", date="11-17", cat=INVEST, found=CRITIC, now=PART,
         what="reduced vs full integration re-solve at 2,376 and 10,200 el",
         effect="diagnose.py D2 reads the result; the re-solve itself is not an action yet"),
    dict(id="V4", date="11-17", cat=INVEST, found=DEV, now=YES,
         what="Gauss-point pressure scatter (near-incompressibility, nu = 0.45)",
         effect="diagnose.py D3"),
    dict(id="V5", date="11-17", cat=INVEST, found=CRITIC, now=PART,
         what="groove-surface stress by nodal extrapolation (surface_study.py)",
         effect="diagnose.py Q3 reads it; not re-run by the agent"),
    dict(id="V6", date="11-17", cat=INVEST, found=DEV, now=YES,
         what="corner exponent (Williams) and B1 peak law peak ~ h_min^-0.281",
         effect="singularity.py + diagnose.py M1/M2"),
    dict(id="V7", date="11-19", cat=INVEST, found=DEV, now=YES,
         what="B3 edge singularity (lambda 0.6216 -> 2 lambda 1.243 vs fitted 1.24) and groove "
              "resolution h/rho ~ 1", effect="diagnose.py M2 and D1"),
    dict(id="V8", date="11-17", cat=INVEST, found=DEV, now=NO,
         what="B3 8th level: 100,815 el out of memory; 59,400 el chosen instead",
         effect="choice of the next level by hand"),

    # ---- operations ----------------------------------------------------------------------
    dict(id="O1", date="11-17", cat=OPS, found=DEV, now=NO,
         what="container restart killed a background chain after 3 of 7 levels; restarted",
         effect="stages skip finished files"),
    dict(id="O2", date="11-16", cat=OPS, found=DEV, now=NO,
         what="live LLM runs need the owner's account (no API key in the container)",
         effect="every live run started by hand"),

    # ---- decisions reserved for a human (by design) --------------------------------------
    dict(id="H1", date="11-17", cat=DECIDE, found=OMAR, now=NO,
         what="required margin between reference uncertainty and surrogate error (3)",
         effect="Rabczuk: OK as an ad-hoc start; add absolute requirements"),
    dict(id="H2", date="11-17", cat=DECIDE, found=OMAR, now=NO,
         what="redefine the B3 groove QoI (Gaussian) -- decided: keep sharp, add Gaussian separately",
         effect="manuscript not changed"),
    dict(id="H3", date="11-16", cat=DECIDE, found=OMAR, now=NO,
         what="manuscript wording 'region-averaged normal stresses' -- approved by Rabczuk",
         effect="not yet edited"),
    dict(id="H4", date="11-16", cat=DECIDE, found=OMAR, now=NO,
         what="approval of every billed live LLM run (~$6.10 so far)", effect="by design"),
]

# What the framework itself caught without a person (inside the pipeline)
IN_PIPELINE = [
    ("rule guard rejected an LLM plan that omitted a required check", "B1 run 1"),
    ("rules added checks the LLM plan lacked", "B1 v2: 2; B3 v4: 12"),
    ("code replaced LLM tolerances", "v4: 32 (B1) / 96 (B3) check-QoI pairs"),
    ("advisor audit caught a wrong verdict (fixture) and claim wording", "0 flags on 6 live traces"),
    ("formalizer open questions that exposed wrong inputs", "I2 and part of I5"),
    ("advisor found real redundancy (cyl_zz = sigma_zz; xx ~ rr; xz ~ rz)", "B3 v2"),
]


def summarise():
    by_cat = collections.Counter(e["cat"] for e in E)
    by_found = collections.Counter(e["found"] for e in E)
    by_now = collections.Counter(e["now"] for e in E)
    cross = collections.Counter((e["cat"], e["found"]) for e in E)
    return by_cat, by_found, by_now, cross


def markdown():
    by_cat, by_found, by_now, cross = summarise()
    cats = [INPUT, BUG, DESIGN, INVEST, OPS, DECIDE]
    founds = [DEV, OMAR, CRITIC, FORM, ADV, AUTO]
    L = ["# Human intervention in the verification-agent work (2026-11-16 to 11-19)", "",
         "Generated by `human_intervention.py` from the project log (PROJECT_STATUS.md). "
         "Each row is one intervention outside the framework; the ID links the tables. "
         "\"Developer\" = the people writing the framework code; \"now\" = whether the "
         "framework does this step itself today.", "",
         f"Total: {len(E)} interventions.", "",
         "## Counts by category and by who found the issue", "",
         "| category | " + " | ".join(founds) + " | total |",
         "|---|" + "---|" * (len(founds) + 1)]
    for c in cats:
        L.append(f"| {c} | " + " | ".join(str(cross[(c, f)] or "") for f in founds)
                 + f" | {by_cat[c]} |")
    L.append("| total | " + " | ".join(str(by_found[f]) for f in founds) + f" | {len(E)} |")
    L += ["", "Now done by the framework itself: " + ", ".join(
        f"{k} {by_now[k]}" for k in (YES, PART, NO)) + ".", ""]
    for c in cats:
        L += [f"## {c[0].upper() + c[1:]}", "", "| ID | date | found by | now automated | what | effect |",
              "|---|---|---|---|---|---|"]
        for e in E:
            if e["cat"] == c:
                L.append(f"| {e['id']} | {e['date']} | {e['found']} | {e['now']} | {e['what']} | "
                         f"{e['effect']} |")
        L.append("")
    L += ["## Caught inside the pipeline (no person needed)", "", "| what | where |", "|---|---|"]
    L += [f"| {a} | {b} |" for a, b in IN_PIPELINE]
    L += ["", "## Reading", "",
          "- Framework bugs were found almost entirely by the developer while running the "
          "framework on real data; one by the LLM advisor and one batch by the claims-check "
          "script. The framework does not yet test itself for such bugs.",
          "- Most design changes came from LLM critic objections, but every one was judged "
          "and implemented by the developer; the critics never approved a plan.",
          "- The manual diagnostic investigations of the B1/B3 non-convergence are the part "
          "that is now largely automated (diagnose.py, 16/16 agreement with the manual causes). "
          "The expensive re-solves (full integration, surface study, new mesh levels) are not "
          "yet actions the agent can start.",
          "- Wrong inputs (problem/adapter descriptions) were the main source of wrong LLM "
          "behaviour (I1, I3). The framework cannot detect a wrong fact that is internally "
          "consistent (I3 was found by the developer).",
          "- Decisions on acceptance margins, QoI definitions, manuscript wording and billed "
          "runs stay with people by design.",
          "", "Limits of this count: it is reconstructed from the log after the fact; small "
          "fixes not recorded in the log are missing, so the numbers are lower bounds. Effort "
          "(hours) per intervention was not recorded."]
    return "\n".join(L) + "\n"


def main():
    out = os.path.join(HERE, "HUMAN_INTERVENTION.md")
    open(out, "w").write(markdown())
    by_cat, by_found, by_now, _ = summarise()
    print(f"{len(E)} interventions -> {out}")
    print(dict(by_cat))
    print(dict(by_found))
    print(dict(by_now))


if __name__ == "__main__":
    main()
