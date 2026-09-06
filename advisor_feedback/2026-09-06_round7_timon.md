# Timon's round-7 email — verbatim, 2026-09-06

Reply to Omar's email of 2026-09-04 (the report v51/v52 status update,
the break-even correction, and the three questions: TensorMesh vs.
torch-fem, open-sourcing the GPU-FEM code, and the continual-learning
citation — plus the open-ended commercialization question). Stored
verbatim first; the reading follows below it.

---

Dear Omar,

Thanks for your kind words. I am reading currently both files you sent me. To the different points:

1. I just mentioned TensorMesh as we tested it a bit and it seems quite efficient. I do not have any preference and torch-FEM is also fine. Most important is that it is an efficient GPU implementation which is necessary for a fair comparison to a NO.
2. There is no approval necessary. I prefer everything being open sourced and both MIT or Apache License are fine. Apache is more general and also TensorMesh is based on Apache license. I'd wait though until the paper is on arxiv and submitted to a journal. Then, we usually open source all codes on GitHub but not before :).
3. This is our continual learning paper: https://arxiv.org/abs/2605.04832.

Neural Operators are already incorporated into commercial software. ANSYS calls it for instance SIMAI. And though they barely describe what they are doing, it is most likely a (transformer based) pure data driven neural operator. There are already several start-up companies about CAE acceleration including neural operators. So, yes. This is certainly of industrial relevance. One key issue for industrial practice is 'trust' in my opinion, meaning how can you ensure that your inference predictions are sufficiently accurate in a quantity of interest without access to the ground truth? I attach a recent paper which is a step forward in this direction. So, this is something we can pursue in the future by trying to take advantage of concepts used in goal oriented error estimation (GOEE) in FEM though my personal experience is rather modest but in AI, the picture is a bit different.

Best regards,
Timon

---

## Reading

### What each item resolves

**1 — TensorMesh vs. torch-fem.** Resolved, no preference either way.
The only real requirement stated is that the comparison library have "an
efficient GPU implementation... necessary for a fair comparison to a
NO" — torch-fem satisfies this (PyTorch-native, GPU-accelerated,
Q4/Q9-equivalent quads). **Unblocked**: the GPU-FEM-vs-torch-fem
efficiency comparison can now be built.

**2 — Open-sourcing the GPU-FEM code.** Two separate answers, not one:
* No institutional/advisor sign-off is needed — this was the blocking
  question. **Resolved.**
* License: no strong preference, but he leans Apache 2.0 explicitly
  ("more general and also TensorMesh is based on Apache license").
* **A new gate, not previously known**: "I'd wait though until the
  paper is on arxiv and submitted to a journal... not before." This is
  a hard condition on the ACTION (making the repository public), not on
  the decision (license is settled). **Do not open-source or make any
  repository public until the paper is submitted/on arXiv, even though
  the license question is now answered.**

**3 — Continual-learning citation.** Given directly:
https://arxiv.org/abs/2605.04832. No longer blocked; nothing further to
ask for. Should be added to the report where OOD/mitigation is
discussed (Section 8.6), per his original suggestion in round 6 ("You
can actually mention our paper on continual learning in this context if
you wish").

**4 — Commercialization question.** He confirms industrial relevance
directly and concretely, not just in principle: names ANSYS's SIMAI as
an existing commercial neural-operator product (his own characterization
— "most likely a (transformer based) pure data-driven neural operator",
stated as inference, not confirmed fact) and notes "several start-up
companies about CAE acceleration including neural operators" exist
already. He then names the concrete obstacle he sees for this specific
line of work: **trust** — verifying inference accuracy in a quantity of
interest without access to ground truth — and proposes **goal-oriented
error estimation (GOEE)**, adapted from FEM, as a candidate future
direction, explicitly framed as future work ("something we can pursue
in the future"), not a request to act now.

**He says "I attach a recent paper"** on this GOEE/trust direction. The
attachment itself did not come through in the message text relayed to
this session — only his description of it. **Nothing about that paper's
content should be assumed or acted on until Omar shares the actual
file.**

### Not yet actioned, needs Omar's decision

- Whether to start the torch-fem comparison now that it's unblocked.
- Whether/when to add the continual-learning citation (straightforward,
  low-risk — likely fine to do immediately).
- The GOEE/trust idea is explicitly future work, not urgent, and cannot
  be scoped further without the attached paper.
