# Timon's round-6 email — verbatim, 2026-08-28

Reply to Omar's round-5 answers. Stored verbatim first; the reading follows
below it, with what is certain separated from what is inferred.

---

Dear Omar,

I am glad about your fast progress :). Since you developed your GPU native FEM
code from scratch, I suggest to document it separately and make it open source.
Ideally, we compare it to codes like Tensormesh in terms of computational
efficiency and also provide some breakdown of the cost. The key cost should be
the solver while the assembly should be minimal. Now to the points:

1. Please first characterize systematically where the 4–5× deterioration comes
from, ideally separating changes in material parameters and loading and also
looking at progressively increasing distribution shifts rather than only one
ID/OOD comparison. If this diagnosis suggests a relatively straightforward
mitigation, such as changing the training range or normalization, it would be
useful to test it. Otherwise, we would leave it for future research. You can
actually mention our paper on continual learning in this context if you wish.
It is now on arxiv.

2. Perfect :).

3. I'd start with one specific problem such as B1-Neo Hookean. Based on the
results, we can decide then. For the paper, we should ideally have a comparison
for all problems.

4. Perfect. We should also include some smaller intermediate numbers and also a
breakdown of the computational cost as I initially mentioned.

5. You understand correctly. Ideally, we do it for a parametrised family of
solutions which will be a bit more time consuming. We should discuss it in more
detail and this is the last thing to do. We should certainly start with one
representative problem, preferably Neo-Hookean. We can compare Q4, Q9 and the
physics-informed Transolver against exactly the same analytical solution in L2,
H1 and energy norms and also examine stress errors. If this produces a useful
result, we can consider extending the MMS study.

Finally, the GPU-FEM break-even result of approximately 7,600–96,000 samples is
very important. Please report both the CPU-FEM and GPU-FEM baselines side by
side. It also clarifies where the neural operator is useful.

Best regards,
Timon

---

## Reading

### New, and not previously on any list

**Open-source the GPU-native FEM solver as a separate documented artefact**,
and **benchmark it against codes like Tensormesh**. This is a new deliverable,
not a refinement of an existing point. It needs a decision from Omar before any
work starts: separate repository, license, and how much documentation.

**A cost breakdown, with a stated expectation**: "The key cost should be the
solver while the assembly should be minimal." He repeats this in item 4. This
is a testable claim about our own code, and the evidence we already have points
the other way at the study's own mesh — see below.

### Mapping of the numbered items

Items 1, 4 and 5 are unambiguous from their content. Items 2 and 3 are not, and
are marked as such rather than guessed.

| his # | what it is about | certainty |
|---|---|---|
| 1 | OOD degradation — the 3.94–5.58× factors of §8.6 | certain: "4–5× deterioration", "material parameters and loading", "ID/OOD comparison" |
| 2 | "Perfect :)" — no content to identify it by | **unknown** |
| 3 | a study to start on B1 × Neo-Hookean and extend to all six for the paper | **ambiguous** — fits either the Pareto (point 2) or the data-driven comparison (point 7) |
| 4 | the GPU-FEM scaling sweep (point 8) | certain: "smaller intermediate numbers" plus "breakdown of the computational cost as I initially mentioned" |
| 5 | MMS (point 9) | certain: Q4/Q9/Transolver against one analytical solution |

Item 3 most likely means the **data-driven comparison**, because that is the
one where "start with one problem, extend to all for the paper" carries real
cost (a second model trained per case), and because for the Pareto he was asked
about axes, not about which case. Not certain enough to act on without
confirming.

### What each item actually asks for

**1 — OOD.** Both of Omar's open questions are answered:
* *Direction*: separate material parameters from loading, and sweep
  **progressively increasing shift**, not one ID/OOD pair. So the deliverable
  is a curve of degradation against shift magnitude, per factor, not a single
  number.
* *Depth*: diagnose first. Mitigate **only if** the diagnosis points to
  something straightforward — he names two candidates, training range and
  normalization. Otherwise it becomes future work. This is explicit permission
  not to chase it.
* He offers his own continual-learning paper, now on arXiv, as a citation. The
  exact reference has not been located and nothing should be cited until it is.

**4 — the sweep.** Two additions: **smaller intermediate resolutions** (the
sweep currently starts at N=501 ≈ 0.5M DOF, so the interesting range below that
is missing entirely), and the **assembly-vs-solver cost breakdown** at each
size.

**5 — MMS.** Now unblocked in shape, and deliberately deferred: "this is the
last thing to do". The design he gives is specific — one representative
problem, Neo-Hookean, comparing **Q4, Q9 and the physics-informed Transolver
against exactly the same analytical solution**, in L2, H1 and energy norms,
plus stress errors. He did **not** resolve the body-force-versus-homogeneous
fork Omar raised; he says a parametrised family of solutions is the ideal and
that it needs more discussion. That fork is still open.

**The break-even.** He calls it "very important" and asks for CPU and GPU
baselines side by side. The report already does this (§8.5 gives both), so the
work here is presentation, not measurement — with one correction below.

## A number that needs correcting with him

Timon quotes the GPU break-even as "approximately 7,600–96,000 samples",
because that is what Omar's email said. That range is the **batch-size-128
column of Table 10c alone** — the least favourable of the four.

The report's full range is **1,133–95,038** across six cases and four batch
sizes, and in the deployment case the operator is actually aimed at, where
problems arrive one at a time, it is **1,133–19,410**. So the lower end Timon
is working from is five to seven times too pessimistic.

Since he says this figure "clarifies where the neural operator is useful", the
gap matters: it moves the break-even from tens of thousands of samples to
roughly a thousand for the best B1 case. The correction should be sent, with
the batch size each figure assumes named explicitly — quoted without that, the
number means nothing, which §8.5 already says.

## The cost-breakdown expectation, against what we already measured

Timon expects the solver to dominate and assembly to be minimal. At the study's
own mesh the report's Table 4b says the opposite for the CPU direct solver:
roughly **5.88×10⁷ assembly FLOPs per sample against 7.9×10⁵ solve FLOPs** —
assembly dominating by about 74×. That is unsurprising at 441 nodes, where a
dense direct solve is trivial and there is nothing for a solver to dominate;
his expectation is about the large-DOF regime, which is exactly why he attached
the request to item 4.

Two gaps to close before this can be answered:
* Those are **hand-counted FLOPs for the CPU solver**, not measured wall clock
  for the GPU one.
* `gpu_fem_solver.py` already separates `t_assembly_s` from `t_solve_s` in its
  `profile` dict, but that solver cannot reach the sweep's sizes.
  `matrix_free_solver.py`, which runs the sweep, has **no such breakdown** —
  it would need the same two buckets, synchronised, before item 4 can be
  answered at the sizes Timon cares about.

Worth being ready for the possibility that assembly does not turn out to be
minimal even at scale. In a matrix-free Newton-CG solver every CG iteration
costs one Hessian-vector product, which is an assembly-like operation, so
"solver time" and "assembly time" are not cleanly separable there in the way
they are for a solver that forms K once and factorises it. If that is what the
measurement shows, it is a real result about the method and should be reported
as one rather than forced into the expected shape.
