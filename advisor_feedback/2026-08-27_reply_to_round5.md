# Reply to Timon's round-5 email — drafted 2026-08-27

Asks about every place where the scope is genuinely open, rather than
guessing. The user's instruction was explicit: it is better to be slightly
tiresome than to spend days of GPU time in the wrong direction. Each item
states our default and says we proceed on it unless corrected, so nothing
blocks on a reply.

The substantive discovery behind question 9: `node_forces` is a *boundary*
field. It is allocated as `np.zeros((len(nodes), 2))` and written only at
`top_nodes` (B1) or `inner_nodes` (B2), so every interior node is exactly
zero in every training sample — and that field is one of the network's
four input channels. A body-force-driven manufactured solution is
therefore out of distribution on a whole input channel, and would require
retraining every model, plus a volumetric work term the energy functional
does not have (`W = sum(node_forces * uv) / len(top_edges)` is a boundary
approximation). A manufactured solution satisfying homogeneous equilibrium
avoids all of it. That fork is worth weeks, which is why it is asked
rather than assumed.

---

Subject: Round-5 requests — how I am proceeding, and where I need your steer

Dear Professor Rabczuk,

Thank you for the detailed feedback.

To your direct question: the GPU native FEM was written from scratch in
PyTorch, not with Tensormesh or any FEM library. It reuses the validated
CPU solver's assembly and material evaluation, and obtains the tangent by
automatic differentiation rather than a hand-derived formula.

Points 1 to 5 are underway exactly as you set them out. The zero-shot
tests are running for the remaining five cases; the identical-batch-size
benchmark and the physical-quantity errors for the Transolver — H1
semi-norm, tangent energy norm, PK1 stress components and their peak
values, and reaction forces — are close to finished.

Point 3 is computed, and the result is worth flagging early: against a GPU
FEM baseline the break-even rises to roughly 7,600–96,000 samples, against
52–1,245 for the CPU baseline. This weakens the efficiency claim
considerably, and I will present both baselines side by side rather than
only the favourable one.

For the remaining points the scope is open in several places, and my
choice would change the amount of work substantially. I have given my
default in each case and will proceed on it unless you tell me otherwise,
so nothing is waiting on your reply.

Point 2, the Pareto comparison. Two axes need fixing. On cost, my default
is inference cost alone for the operator, with the training cost shown
separately, since including amortised training makes the curve depend
entirely on an assumed number of future problems. On accuracy, my default
is the displacement relative L2 error, so the comparison stays comparable
with the existing tables, with the point-5 quantities reported alongside
rather than as the axis. Would you prefer either differently?

Point 6, OOD robustness. Two things are open. First, the direction of the
shift: material parameters outside the training range, load magnitude,
load spatial pattern, and geometry are separate experiments, and my
default is to take material parameters and load magnitude first, since
those are what the existing 4–5x figure was measured on. Second, the
depth: my default is to diagnose where and why the degradation
concentrates, and to attempt a mitigation only if the diagnosis points to
a direct one. Is the diagnosis what you have in mind for this paper, or
would you want the mitigation itself to be the deliverable?

Point 7, the data-driven comparison. Three things are open here, and the
first one determines the answer before any computation. On matching, the
data-driven model can be given the same number of training samples, or the
same total compute budget; these are different claims — the first is about
data efficiency, the second about cost — and my default is the same number
of samples, with the compute difference reported explicitly. On "fine
enough", my default is to generate the data at the same two resolutions
the physics-informed model trains on, 21 and 33, so the two are matched;
if you meant genuinely fine simulations, please say which resolutions. On
the model itself, my default is the identical Transolver architecture
trained with a data loss instead of the energy functional, so that the
comparison isolates the training principle rather than the architecture.

Point 8, the GPU FEM sweep. I read "a few million DOFs" as roughly 0.5, 1,
2 and 4M, run with the matrix-free Newton-CG solver already in the
repository — the one that produced the 10M and 40M DOF references. Two
smaller choices: my default is to run the sweep on one geometry and
material rather than all six, and to report solve time together with the
achieved residual, so the timings are not detached from accuracy.

Point 9, MMS. Here I would particularly like your steer, because two
readings differ by weeks of work. Our models take a nodal force field as
one of their four inputs, but in every training sample that field is
nonzero only on the loaded boundary; the interior values are identically
zero. A manufactured solution driven by a body force would therefore be
out of distribution on an entire input channel, so the operators would
need retraining, and the energy functional would need a volumetric work
term it does not currently have. If instead the manufactured solution is
chosen to satisfy the homogeneous equilibrium equation, with the load
carried entirely by the boundary, none of that is required and the
existing trained models can be tested directly. Which of the two do you
intend? Related, and cheaper to settle: by "a set of problems", would you
like one representative case or all six, and should the comparison be at a
single mesh or across a refinement sweep? My defaults are the homogeneous
formulation, one representative case, and a refinement sweep, since a
single mesh cannot separate discretisation error from operator error.

I will come back with results rather than more questions.

Best regards,
Omar
