# Round-5 feedback from Prof. Timon Rabczuk — 2026-08-26

Stored verbatim. Everything below the rule is the advisor's own text,
unedited. Do not paraphrase it in place — paraphrases belong in
`PROJECT_STATUS.md`, which links back here. This file exists because the
round-5 requests were previously held only as a summary table, and a
summary cannot settle a question about what was actually asked.

---

…. not to the results, which are very interesting. I think we can wrap them up
in a paper but I still have a few comments and requests, i.e.

1. Can you please complete zero-shot resolution tests for the other five cases.
2. And construct the GPU FEM and transolver accuracy/cost Pareto comparison.
3. Then recompute break-even using GPU FEM.
4. Can you please also benchmark Transolver and GPU FEM under identical batch
   sizes and
5. examine error in physically important quantities beyond displacement,
   including error in H1 semi-norm, energy and also some local quantities such
   as stress components and reaction forces; maybe looking at maxima. I am
   referring to the transolver.
6. Can you please investigate OOD robustness, because the current 4–5×
   deterioration is probably the biggest obstacle to a strong
   "physics-informed operator" claim.
7. We should also demonstrate resolution invariance, i.e. train on 2 different
   resolutions and test then on 5 resolutions (coarser and finer).
   Theoretically the PI transolver should be resolution invariant, meaning we
   can train on a very coarse grid and inference on a finer grid. This could
   provide computational savings. In this context, we can also add a comparison
   to the data-driven version where we can generate data from two different
   (fine enough) simulations.
8. We should also test the performance of the GPU native FEM for finer
   discretizations up to a few million DOFs. Did you use Tensormesh or write
   the code yourself?
9. The last point would require actually additional work but I think it would
   be valuable: Instead of using a baseline FEM solution, we can use the MMS to
   generate a ground truth for a set of problems and then use it to test the
   physics informed neural operator compared to FEM.

There are many directions we could pursue subsequently.

Best regards,
Timon

---

## Details the earlier summary table had lost

Checked the verbatim text against the summary in `PROJECT_STATUS.md`. The
nine points were recorded faithfully, but three specifics were missing and
they change what the deliverable looks like:

1. **Point 7 names the motivation, not just the test.** "we can train on a
   very coarse grid and inference on a finer grid. This could provide
   computational savings." So resolution invariance is not the end product —
   the *savings* are. The study should quantify what training coarse and
   inferring fine actually saves, not only report that the error stays flat.

2. **Point 7 specifies how the data-driven baseline is built:** "generate data
   from two different (fine enough) simulations." That constrains the
   comparison — the data-driven model gets data from two resolutions, matching
   the physics-informed model's two training resolutions, and both must be
   fine enough to be credible. It is a matched comparison, not an arbitrary one.

3. **Point 9 measures FEM too:** MMS is used "to test the physics informed
   neural operator compared to FEM." Both are scored against the manufactured
   ground truth. The point is not only to grade the operator more honestly, it
   is to put the operator and FEM on the same footing for once — currently FEM
   *is* the reference and therefore cannot be graded at all.

Also note "a set of problems" (point 9) and "5 resolutions" (point 7) are
loose enough to need confirmation; our sweep uses 7 resolutions, which is a
superset and therefore satisfies the request.

Closing line — "There are many directions we could pursue subsequently" —
signals this is a first round, so scope discipline now matters more than
completeness.
