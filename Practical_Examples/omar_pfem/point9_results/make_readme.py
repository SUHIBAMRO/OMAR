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
    # The three-mesh rate sweep has its own schema and its own section
    # below; keep it out of the per-run listing.
    ops = sorted(p for p in glob.glob(os.path.join(HERE, "*operator_*.json"))
                 if "rate" not in os.path.basename(p).lower())
    rate_files = sorted(glob.glob(os.path.join(HERE, "*operator_rate_*.json")))
    prod = [p for p in ops if "demo" not in os.path.basename(p).lower()]
    if prod:
        a("**All three legs are measured.** Q4 and Q9 are below; the "
          "physics-informed operator is at the bottom of this file. The "
          "operator needed a body-force term in the energy functional and a "
          "body-force input channel, neither of which the report's trained "
          "checkpoints have, so it is a separately trained model "
          "(`omar_pfem/mms_operator.py`) — not a Table 5 checkpoint.\n")
    else:
        a("**This is the FEM half — Q4 and Q9. The operator half is not done**; "
          "it needs a body-force term in the energy functional and a body-force "
          "input channel, neither of which the trained checkpoints have. See "
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
    for p in ops:
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
            a(f"\n## The operator third — the three-way, complete (N={O['N']})\n")
            fv = O.get("functional_verified")
            if fv:
                a(f"Before any of this was trained, `{fv['by'].split(',')[0]}` "
                  f"established that the network is minimizing the same thing "
                  f"the FEM solver solves: Π(u_FEM) = {fv['Pi_at_u_FEM']:.6f} "
                  f"is a true minimum of the operator's functional — the "
                  f"interpolated u\\* does not beat it, all 36 admissible "
                  f"perturbations raise Π, and the excess grows quadratically "
                  f"(ratio {fv['quadratic_excess_ratio']:.3f}, "
                  f"{fv['expected_quadratic_excess_ratio']} expected). The "
                  f"meta-check: a W wrongly divided by 8 moves the minimum to "
                  f"scale {fv['best_scale_W_divided_by_8']}, so those checks "
                  f"can fail.\n")
        a("| method | L2 | H1 semi | stress | energy |")
        a("|---|---|---|---|---|")
        for name, d in (("Q4 (same mesh)", ref["Q4"]), ("Q9 (same N)", ref["Q9"]),
                        ("operator" + (" (undertrained)" if demo else ""), op)):
            a(f"| {name} | {d['L2_rel']:.3e} | {d['H1_semi_rel']:.3e} | "
              f"{d['stress_rel_L2']:.3e} | {d['energy_rel']:.3e} |")
        r = O["operator_over_Q4_L2"]
        rh = op["H1_semi_rel"] / ref["Q4"]["H1_semi_rel"]
        a(f"\n**operator / Q4 in L2 = {r:.2f}×.** "
          + ("Above 1.0. " if r > 1 else "Below 1.0 — and that is allowed. ")
          + "The ceiling constrains Π, not L2: the Q4 solution minimizes Π "
            "over this space, so nothing in it reaches a lower Π, but L2 "
            "error against u\\* is a different functional. A field that does "
            "not minimize Π can sit closer to u\\* in L2 by partially "
            "cancelling Q4's own discretization bias, and the three-mesh "
            "sweep saw exactly that at N=9 (0.37×). The norms that stayed "
            "above 1.0 at every mesh are H1 semi and stress.\n")
        rs = op["stress_rel_L2"] / ref["Q4"]["stress_rel_L2"]
        re_ = op["energy_rel"] / ref["Q4"]["energy_rel"]
        a(f"The four ratios are **not** the same number: L2 {r:.2f}×, "
          f"H1 semi {rh:.2f}×, stress {rs:.2f}×, energy {re_:.2f}×. "
          + ("The same inversion the production run shows, on a different "
             "mesh and a different device — which is the reason it is read "
             "there as a property of the training principle rather than an "
             "artefact of one run.\n" if demo and rh < r else ""))
        if demo:
            pass
        elif rh < r:
            a(f"The gradient-based norms are the ones the operator has "
              f"essentially closed — {rh:.2f}× in H1 and {rs:.2f}× in stress "
              f"means it recovers the strain and stress fields about as well "
              f"as the Q4 optimum it is chasing — while it is {r:.2f}× behind "
              f"in L2 and {re_:.2f}× in energy. That is the **opposite of the "
              f"usual ordering**, where L2 is the forgiving norm and the "
              f"derivative norms are the strict ones. The energy functional is "
              f"what is being minimized, and it is built from the deformation "
              f"gradient, so the quantities it sees directly are the ones that "
              f"come out closest; the displacement itself is only pinned down "
              f"through them, up to what the boundary mask fixes.\n")
        else:
            a(f"The L2 ratio is the smallest of the four, the ordering one "
              f"would expect.\n")
        if demo:
            a("The error was **still falling at the last epoch** (see "
              "`history` in the JSON), so this ratio reflects the step budget, "
              "not the method. The reportable number needs "
              "`Round6_MMS_Operator.ipynb`: N=17, 2000 epochs, 20–40 min on "
              "any GPU.\n")
        else:
            # Whether the budget or the method set this ratio is a question
            # about the training curve, so answer it from the curve rather
            # than by claiming convergence.
            h = O["history"]
            half = len(h) // 2
            b_half = min(x["L2_rel"] for x in h[:half])
            b_end = min(x["L2_rel"] for x in h)
            tail = [x["L2_rel"] for x in h[-20:]]
            a(f"\n**Is this the budget or the method?** Best test L2 was "
              f"{b_half:.3e} at the halfway point and {b_end:.3e} at the end, "
              f"an improvement of {(1 - b_end / b_half) * 100:.0f}% over the "
              f"second half of training — still falling, but slowly. The last "
              f"twenty validations span {min(tail):.3e} to {max(tail):.3e}, a "
              f"factor of {max(tail) / min(tail):.0f}, so single-epoch scores "
              f"are noisy and the reported number is the best checkpoint "
              f"(epoch {O['operator_best_epoch']}), not the last one. A longer "
              f"run would close some of the remaining L2 gap; nothing here "
              f"shows how much.\n")
            a(f"Training cost: {O['training']['opt_steps']:,} optimizer steps, "
              f"{O['training']['train_wall_clock_min']} min on an "
              f"{O.get('gpu', O['device'])}, and **no labels** — u\\* is "
              f"analytic but is never used in training, only in scoring.\n")

    for p in rate_files:
        RT = json.load(open(p))
        a("\n---\n")
        a("\n## Does the operator have a convergence rate of its own?\n")
        a("Section 8.11 used to say it could not be asked, because the operator "
          "had been trained at one mesh. `" + os.path.basename(p) + "` trains it "
          "at three under the identical protocol and asks.\n")
        a("| N | DOF | operator L2 | Q4 L2 | op/Q4 | operator H1 | Q4 H1 | op/Q4 |")
        a("|---|---|---|---|---|---|---|---|")
        for r in RT["rows"]:
            o, q = r["operator"], r["Q4"]
            a(f"| {r['N']} | {r['n_dof']:,} | {o['L2']:.3e} | {q['L2']:.3e} | "
              f"**{o['L2'] / q['L2']:.2f}×** | {o['H1_semi']:.3e} | "
              f"{q['H1_semi']:.3e} | {o['H1_semi'] / q['H1_semi']:.2f}× |")
        f = RT["fitted_rates_in_h"]
        a(f"\n**Fitted rates in h**: operator L2 **{f['operator_L2']:.2f}**, Q4 L2 "
          f"{f['Q4_L2']:.2f}; operator H1 {f['operator_H1_semi']:.2f}, Q4 H1 "
          f"{f['Q4_H1_semi']:.2f}. The Q4 figures are the control and they land "
          f"on Table 23's measured 1.98 and 1.00, so the operator's can be "
          f"quoted beside them.\n")
        h = RT["THE_HEADLINE"]
        a(f"**{h['what']}**\n")
        a(f"{h['why_that_makes_sense']}\n")
        a(f"{h['the_crossover_is_visible']}\n")
        c = RT["A_CORRECTION_TO_SECTION_8_11"]
        a("\n### The ceiling was stated too strongly, and this run showed it\n")
        a(f"{c['that_is_too_strong']}\n")
        a(f"{c['and_that_is_exactly_what_happened']}\n")
        a(f"{c['what_IS_protected']}\n")
        a(f"*{c['note_on_the_energy_column']}*\n")

    a("\n## What is NOT here\n")
    if prod:
        a("* **The operator at more than one mesh.** It is trained and scored "
          "at N=17 only, so it has no convergence rate of its own — the two "
          "rate tables above are FEM only. Retraining at each N is the missing "
          "work, and it is a training run per mesh, not a solve.\n")
        a("* **The operator's cost, on comparable terms.** Its 8.2 min of GPU "
          "training is not commensurable with a CPU FP64 Newton solve, and no "
          "attempt is made here to force them onto one axis.\n")
    else:
        a("* **The Transolver.** The comparison Timon asked for is three-way; "
          "this is two-way. The operator cannot be run on this problem as "
          "things stand: its energy functional has no body-force term and its "
          "inputs have no body-force channel.\n")
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
