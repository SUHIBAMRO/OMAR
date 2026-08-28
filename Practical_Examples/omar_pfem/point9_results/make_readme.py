"""Generates this directory's README from the MMS result JSON.

Written rather than typed because transcribing a results table by hand is
exactly how the point-8 README ended up quoting 3,215 us/DOF where the run
printed 3,219. Run it after dropping a new mms_*.json in here:

    python3 make_readme.py
"""
import glob
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))


def rate_row(d, norm):
    v = d[norm]
    pw = ", ".join(f"{x:.2f}" for x in v["pairwise"])
    flag = "OK" if abs(v["rate"] - v["expected"]) < 0.4 else "**OFF**"
    return f"| {norm} | {v['rate']:.2f} | {v['expected']} | {pw} | {flag} |"


def build(path):
    R = json.load(open(path))
    rows = R["rows"]
    orders = sorted({r["order"] for r in rows})
    L = []
    a = L.append

    a(f"# Point 9 — method of manufactured solutions ({R['material']})\n")
    a(f"Timon's point 9, and his \"this is the last thing to do\". His design: "
      f"*\"compare Q4, Q9 and the physics-informed Transolver against exactly "
      f"the same analytical solution in L2, H1 and energy norms and also "
      f"examine stress errors\"*.\n")
    a("**This is the FEM half — Q4 and Q9. The operator half is not done**; it "
      "needs a body-force term in the energy functional and a body-force input "
      "channel, neither of which the trained checkpoints have. See "
      "`omar_pfem/mms_study.py` and PROJECT_STATUS.md.\n")

    a("## The manufactured solution\n")
    a(f"```\n{R['manufactured_solution']}\n```\n")
    a(f"* **Body force**: {R['body_force']}.\n")
    a(f"* **Boundary conditions**: {R['boundary_conditions']}.\n")
    a(f"* **Material**: {R['material_field']}.\n")
    a(f"* **Precision**: {R['dtype']} on {R['device']}.\n")

    a("\n### Why a body force, and not a body-force-free exact solution\n")
    a("Timon left this fork open and it had to be settled to write any code. A "
      "body-force-free exact solution of finite-strain elasticity on this "
      "domain is, in practice, a homogeneous deformation — a constant "
      "deformation gradient — which a bilinear Q4 element reproduces to "
      "machine precision. The study would measure round-off, both orders would "
      "\"converge\" instantly, and it would distinguish nothing. Manufacturing "
      "the solution keeps the geometry, material and discretization exactly as "
      "they are everywhere else in the report. **This is a decision made here, "
      "not one Timon confirmed**, and it is the first thing to raise if he "
      "wants the study shaped differently.\n")

    a("\n## Results\n")
    a("| order | N | DOF | L2 | H1 semi | stress | energy |")
    a("|---|---|---|---|---|---|---|")
    for r in sorted(rows, key=lambda r: (r["order"], r["N"])):
        a(f"| {r['order']} | {r['N']} | {r['n_dof']:,} | {r['L2_rel']:.3e} | "
          f"{r['H1_semi_rel']:.3e} | {r['stress_rel_L2']:.3e} | "
          f"{r['energy_rel']:.3e} |")

    if "convergence_rates" in R:
        a("\n## Convergence rates — this is what validates the study\n")
        a("An MMS study is self-validating: if the body force were wrong by a "
          "sign or a factor, the discrete solution would converge to the wrong "
          "function and these rates would collapse. They do not.\n")
        for o in orders:
            if o not in R["convergence_rates"]:
                continue
            a(f"\n**{o}**\n")
            a("| norm | observed | theory | pairwise | |")
            a("|---|---|---|---|---|")
            for n in ("L2", "H1_semi", "stress", "energy"):
                a(rate_row(R["convergence_rates"][o], n))
        a("")
        for o, v in R.get("rate_check", {}).items():
            a(f"* **{o}: {v}**")

    # equal-DOF comparison, the thing the two orders are actually being
    # compared on -- computed here rather than asserted
    a("\n## Q4 vs Q9 at equal cost\n")
    byd = {}
    for r in rows:
        byd.setdefault(r["n_dof"], {})[r["order"]] = r
    shared = [d for d, v in byd.items() if len(v) > 1]
    if shared:
        a("Meshes where both orders have the same number of degrees of "
          "freedom, which is the only fair way to compare them:\n")
        a("| DOF | Q4 L2 | Q9 L2 | Q9 advantage | Q4 H1 | Q9 H1 | Q9 advantage |")
        a("|---|---|---|---|---|---|---|")
        for d in sorted(shared):
            q4, q9 = byd[d]["Q4"], byd[d]["Q9"]
            a(f"| {d:,} | {q4['L2_rel']:.3e} | {q9['L2_rel']:.3e} | "
              f"**{q4['L2_rel'] / q9['L2_rel']:.1f}×** | "
              f"{q4['H1_semi_rel']:.3e} | {q9['H1_semi_rel']:.3e} | "
              f"**{q4['H1_semi_rel'] / q9['H1_semi_rel']:.1f}×** |")
    else:
        a("No two runs share a DOF count; add matching resolutions to compare.")

    a("\n## What the error columns mean\n")
    for k, v in R.get("error_definitions", {}).items():
        a(f"* `{k}` — {v}")

    a("\n## Solver settings, and why they do not affect the numbers\n")
    s = R.get("solver", {})
    a(f"Newton tol {s.get('newton_tol')}, CG tol {s.get('cg_tol')}, "
      f"{s.get('load_steps')} load steps.\n")
    a(f"{s.get('tolerance_independence_checked', '')}\n")
    a("A caveat about the shared solver, recorded in PROJECT_STATUS: its CG "
      "stops on `||r||/||b|| < cg_tol`, and on the last Newton iteration of "
      "each load step `b` is the already-converged residual, so the target "
      "becomes unreachable and CG runs to its iteration cap. Harmless here — "
      "Newton has already converged and the cap is set proportional to the "
      "problem size — but it inflates the wall-clock column.\n")

    # The operator third, if a run is present. Generated rather than
    # appended by hand: this script rewrites README.md from scratch, so
    # anything appended manually would be silently lost on the next run.
    for p in sorted(glob.glob(os.path.join(HERE, "operator_*.json"))):
        O = json.load(open(p))
        demo = "demo" in os.path.basename(p).lower()
        op = O["operator_on_the_reference_member"]
        ref = O["fem_reference_same_mesh"]
        a("\n---\n")
        if demo:
            a("\n## The operator third — a DEMONSTRATION run only, not a result\n")
            a(f"`{os.path.basename(p)}`. **Do not quote these numbers in the "
              f"report.** It is a {O['training']['opt_steps']:,}-optimizer-step "
              f"{O['device'].upper()} run at N={O['N']}, kept only because it "
              f"proves the pipeline end to end and because its ceiling check "
              f"passed. For scale, the report's own physics-informed models "
              f"were trained for 75,000 steps.\n")
        else:
            a(f"\n## The operator third (N={O['N']})\n")
        a("| method | L2 | H1 semi | stress | energy |")
        a("|---|---|---|---|---|")
        for name, d in (("Q4 (same mesh)", ref["Q4"]), ("Q9 (same N)", ref["Q9"]),
                        ("operator" + (" (undertrained)" if demo else ""), op)):
            a(f"| {name} | {d['L2_rel']:.3e} | {d['H1_semi_rel']:.3e} | "
              f"{d['stress_rel_L2']:.3e} | {d['energy_rel']:.3e} |")
        r = O["operator_over_Q4_L2"]
        rh = op["H1_semi_rel"] / ref["Q4"]["H1_semi_rel"]
        a(f"\n**operator / Q4 in L2 = {r:.2f}×.** "
          + ("Above 1.0, which is the required outcome: "
             if r > 1 else "**BELOW 1.0, which should be impossible** — ")
          + "the operator minimizes the same functional over the same Q4 "
            "space, so the Q4 solution is the minimizer and a ratio below 1.0 "
            "would mean a bug, not a win.\n")
        a(f"In the H1 semi-norm the ratio is {rh:.2f}×. That it is so much "
          f"{'smaller' if rh < r else 'larger'} than the L2 ratio is worth "
          f"re-checking on a longer run — it is the opposite of the usual "
          f"ordering, where L2 is the more forgiving norm.\n")
        if demo:
            a("The error was **still falling at the last epoch** (see "
              "`history` in the JSON), so this ratio reflects the step budget, "
              "not the method. The reportable number needs "
              "`Round6_MMS_Operator.ipynb`: N=17, 2000 epochs, 20–40 min on "
              "any GPU.\n")

    a("\n## What is NOT here\n")
    a("* **The Transolver.** The comparison Timon asked for is three-way; this "
      "is two-way. The operator cannot be run on this problem as things stand: "
      "its energy functional has no body-force term and its inputs have no "
      "body-force channel.\n")
    a("* **One material and one geometry**, and a single manufactured solution "
      "rather than the parametrised family Timon called the ideal. The family "
      "is parametrised in the code by (alpha, beta); only one member is run.\n")
    return "\n".join(L) + "\n"


def main():
    # Deliberately narrow: this generator describes the FEM convergence
    # study only. A looser "mms_*.json" would also sweep up the operator
    # runs, which have a completely different schema.
    files = sorted(glob.glob(os.path.join(HERE, "mms_B1_*.json")))
    if not files:
        raise SystemExit("no mms_*.json in " + HERE)
    if len(files) > 1:
        raise SystemExit(f"more than one result file, pick one: {files}")
    text = build(files[0])
    out = os.path.join(HERE, "README.md")
    with open(out, "w") as f:
        f.write(text)
    print(f"wrote {out} from {os.path.basename(files[0])}")


if __name__ == "__main__":
    main()
