# Timon's round-8 email — verbatim, 2026-09-06

A manuscript-level review, not just a reply to open questions — he
opens by saying he "went now to the work and manuscript." Sorts his
comments "to the points from last email," but does not repeat which
email is "last" or give a numbered list to map against; the mapping
below is inferred from content, and marked uncertain where it is.
Stored verbatim first; the reading follows below it.

**Revised after Omar's own review of the first reading** (same day):
point 1 had wrongly assumed "GPU native FEM" meant the CPU reference
solver — Timon's own heading says GPU, and points 1 and 5 are now read
as the same underlying concern; point 3 sharpened to the specific
coarse-vs-fine label-generation question, not a generic DD-NO study;
point 5 explicitly kept as "cite the existing caveat AND do the
remaining real work," not one instead of the other.

---

Dear Omar,

I went now to the work and manuscript. You did a tremendous amount of work. I think we should improve the presentation by converting several tables into figures as in my previous papers. I hope that I did not confuse now anything but here are my comments and suggestions which I sorted to the points from last email:

1. GPU native FEM: I think the implementation is still suboptimal. The assembly time is significantly too long and should be only a small portion of the solve time; you can see this for instance in TensorMesh. One critical aspect is then that if the FE solution can be done in miliseconds, we do not need NOs any more. So, ideally we should have a problem where the defined FE accuracy in terms of all norms and also defined QoIs (such as maximum stresses or similar) require FE simulations which are rather in the order of minutes, even for academic benchmark problems. In industrial applications, one could think of using a tire with different profiles. Here we could also use the nice feature that the resolution is independent of the parameters in NOs.

2. OOD is well addressed and for the paper we should do that for the other cases.

3. Well addressed and it seems that for case 2, VINO is suboptimal. For the paper, we should add this also for a DD-NO; ideally showing how different data generations (on two different discretizations) might affect the results. We should also present the results in form of a figure for the paper.

4. Can you please provide also the wall clock training time for DD-NO.

5. GPU FEM scaling and computational cost: The multi-million-DOF timings are still problematic: the large cases are dominated by CG, some runs hit the iteration cap, and the largest cases were not all rerun to convergence. Also, in the matrix-free solver the element-level work is largely hidden inside the Hessian-vector products in CG, so "assembly is negligible" is not the right interpretation. We should aim to improve/benchmark the linear solver/preconditioner before drawing conclusions about FEM scaling.

6. MMS: Very interesting but I'd suggest to have a richer family of MS containing the sum of several sine/cosine spatial modes while preserving the boundary conditions. Please also report the relative error in the energy norm (not the internal energy). For the paper, we should do it ideally for at least one other model/proble.

7. Break-even comparison: A FEM "batch size 128" means solving 128 independent FEM problems simultaneously; it is useful as a throughput experiment, but it should not be the main single-query comparison. For the primary comparison, we should use the same hardware and batch size: Comparison A: 1 for FEM versus 1 NO inference. Comparison B: Total budget: M FEM runs (for data generation) + training + N inference. For VINO, there is no data generation. Then we can compute the break even; VINO versus DDNO. Throughput experiments can be shown separately with the same hardware resources available to both methods.

Finally, as indicated at point 1, we should make sure that the FEM problem is expensive enough for an NO to be useful. We can do it by checking some QoIs and requesting stricter error tolerances versus your high fidelity ground truth or by adding a more complex problem.

Best regards,

Timon

---

## Reading

### General suggestion, applies broadly

**Convert several tables into figures**, matching his previous papers'
presentation style. Not tied to a specific point number; a presentation
change across the report, not a numeric one.

### What "VINO" means in this email

He uses "VINO" here for our own physics-informed operator (the
Variational/energy-based Neural Operator family his group's paper is
named after), not literally the external VINO codebase. Point 7's "For
VINO, there is no data generation" only makes sense this way: it is
describing OUR physics-informed Transolver (no labels needed), contrasted
with a data-driven neural operator (DD-NO, needs FEM-generated labels)
— exactly the point7b distinction already in the report.

### Point-by-point

**1 — GPU-native FEM implementation / benchmark scale.** Corrected
after Omar pushed back on the first reading, which had wrongly assumed
this was about the CPU reference solver (Table 4a): Timon's own heading
says **"GPU native FEM"** explicitly, not CPU. The first draft of this
reading substituted its own guess (the CPU/N=21 assembly-vs-solve split)
for what he actually wrote, which named neither a solver nor a
resolution. That was an assumption, not something confirmed from the
committed timing data, and it should have been flagged as such rather
than stated as if it were certain.

Read correctly, points 1 and 5 are almost certainly the **same
concern**, not two: the GPU-native matrix-free solver (Table 20/20a) is
the only thing in this report actually named "GPU native FEM," and it is
exactly where element-level ("assembly") work is not a separate,
measurable phase — it happens inside every Hessian-vector product of the
CG loop (Section 8.5 already says this; see point 5 below). So when
Timon says the assembly time is "significantly too long" for this
solver, he is most likely reacting to the fact that this hidden
per-iteration element work makes the effective cost of "assembly," done
implicitly thousands of times over, large relative to what a solver like
TensorMesh — which presumably assembles explicitly once and factorizes —
would spend. That reframes this as a real, open concern about the
GPU-native solver's cost structure at scale, not something already
explained by the small-mesh CPU case.
- **The separate, bigger point**: "if the FE solution can be done in
  milliseconds, we do not need NOs any more." This questions whether the
  whole benchmark problem is demanding enough, at the accuracy the
  advisor's own QoIs require, for a neural operator to be worth using at
  all. He suggests academic problems scaled so FE takes minutes, and
  names a concrete industrial example (a tire with different tread
  profiles, which also showcases the operator's resolution-independence).
  This is a benchmark-design question, not an implementation bug — it
  asks whether B1/B2 at their current sizes are the right test case for
  the paper's central claim.

**2 — OOD.** "Well addressed" — no further diagnosis needed on the one
case done (B1 × Neo-Hookean, Tables 19/19a). **For the paper**: repeat
the same progressive, factor-isolated study for the other five cases.
This is new measurement work (five more progressive sweeps), not an
edit.

**3 — mapped with real uncertainty, refined by Omar.** Likely means:
the resolution-invariance study (round-5 point 7), where B2's zero-shot
generalization is real but weaker than B1's (already reported, Table
12b/12c) — "it seems that for case 2 [B2], VINO [our physics-informed
operator] is suboptimal" fits this reading. **The precise experiment
this asks for, sharper than "a DD-NO resolution-invariance study" in
general**: train the DD-NO on FEM labels generated at a COARSE
discretization versus a FINER one, and measure how its accuracy and
zero-shot generalization across resolutions changes as a function of
that label-generation mesh. This targets something the physics-informed
operator structurally cannot suffer from — it never trains on FEM
labels at all — while the DD-NO's accuracy ceiling is inherited directly
from whatever mesh generated its training data. That asymmetry, not
just "does a DD-NO also generalize across resolutions," is the actual
comparison with the physics-informed operator that matters here. New
study either way — Table 21's data-driven comparison uses one
resolution, one case (B1 × Neo-Hookean), not the two-resolution
zero-shot protocol or the coarse-vs-fine label question this asks for.
Present as a figure.

**4 — DD-NO wall-clock training time.** Already measured and committed
(`point7b_results/comparison_B1_neo_hookean.json`); it just was not in
Table 21. **Done** — added in report v54 (`make_v54.py`): 1,458.3 s
(Adam) and 1,463.0 s (AdamW+OneCycleLR), alongside the physics-informed
run's 2,873.8 s and 3,108.9 s for the same two optimisers. The
data-driven number does not include the 5.65 h of CPU time spent
generating its 800 training labels, which the report now says
explicitly next to it.

**5 — GPU-FEM scaling.** Three separable claims, and — per Omar's
explicit caution — the first one being already stated in the report is
NOT a reason to treat this point as closed; two of its three parts are
real, unstarted work, and the reply to Timon needs to say both things at
once, not substitute the first for the other two:
- *"Assembly is negligible" is not the right interpretation* — **this
  exact caveat is already in the report**, near-verbatim: Section 8.5
  already says "It should not be read literally... the assembly has not
  become cheap, it has moved inside the CG loop, where this
  instrumentation cannot separate it... At the sizes in Table 20 the
  question simply does not have the clean answer it has at small
  scale." Worth pointing him to this paragraph directly — but as
  confirmation the concern is understood, alongside the two items below,
  not instead of them.
- *The largest cases were not all rerun to convergence* — **true and
  already flagged as a limit in the report** (Table 20b reruns only
  N=501 and N=701 to convergence; N=1001 and N=1401 were not). **Not yet
  done**: rerunning those two to convergence too.
- *Improve/benchmark the linear solver or preconditioner before drawing
  scaling conclusions* — **real, not-yet-started engineering work** (a
  better preconditioner than the current Jacobi one). This is also the
  concrete remedy for point 1's concern, since a better preconditioner
  directly reduces the CG-hidden cost point 1 is most likely reacting
  to — the two points converge on the same fix.

**6 — MMS.** Two requests:
- A richer manufactured-solution family (a sum of several sine/cosine
  spatial modes, boundary conditions preserved) — new work, replacing
  the current single-mode field.
- **Report the energy NORM, not the internal energy value.** Checked
  the source JSON directly: the current "Energy" column's own recorded
  convergence rate is double the H1 semi-norm's rate (Q4: energy
  expected-rate 2 vs. H1 expected-rate 1; Q9: 4 vs. 2) — the signature
  of a scalar energy VALUE comparison, which superconverges, not a norm
  of the error field, which would converge at the H1 rate. He is
  technically correct that these are different quantities. The proper
  energy norm already exists elsewhere in this codebase — Section 4.4's
  Table 6a reports a "tangent/incremental energy norm" — so this is a
  matter of applying that existing computation to the MMS study, not
  building it from scratch.
- Extend MMS to at least one other material/problem for the paper.

**7 — Break-even methodology.** Restructures how the comparison should
be framed, not just which numbers to report:
- *Comparison A* (primary): single query, same hardware, batch size 1
  both sides — FEM bs=1 vs. NO inference bs=1. Table 10c/10d's bs=1
  column already is this comparison; it has not been labeled as the
  PRIMARY one over the batched throughput figures.
- *Comparison B* (new): total cost of ownership — M FEM runs (data
  generation, DD-NO only) + training + N inferences — computed for the
  physics-informed operator (no data-generation cost) versus the DD-NO
  (which pays it), not against FEM directly. This break-even does not
  exist yet in the report; it needs point 3's DD-NO-across-two-
  discretizations study and point 4's wall-clock numbers as inputs.
- Batched throughput experiments (bs=8/32/128) stay, but presented
  separately, both methods on the same hardware, not as "the" headline
  break-even number.

### What this adds up to

Points 2, 3, 6 (family), and 7 (Comparison B) are new measurement work,
not edits — a fresh OOD sweep for five cases, a new DD-NO study on
coarse-vs-fine label generation, a richer MMS family, and a new
PI-vs-DD-NO break-even. Points 1 and 5 are most likely the same
underlying concern (the GPU-native solver's CG-hidden cost structure)
seen from two angles, with a shared concrete remedy — a better
preconditioner — plus rerunning the two largest cases to full CG
convergence. Point 1 additionally raises a separate benchmark-design
question (is B1/B2 at its current size demanding enough) that needs a
decision before redoing any of the above at a different scale. Only
point 4 (wall-clock) and part of point 6 (the energy-norm metric, which
already exists in this codebase) were quick, and point 4 is already
done.
