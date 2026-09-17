# Draft reply to Timon's MMS normalization/resolution-invariance questions

Drafted 2026-09-17. NOT YET SENT -- Omar's call, per the project's own
standing convention (never send a reply without his review first).

Verified against the actual code before drafting, not answered from
memory: `omar_pfem/mms_operator.py` (the physics-informed operator's own
MMS training script) and `cell_mms_operator_rate.py` (the notebook that
runs it at N=9/17/33 to fit Table 24a's own convergence rate).

Point 1: confirmed correct as-is, no code change.
- `mms_operator.py` lines 221-227: `norm = {"mean": Ftr.mean(dim=(0,1)),
  "std": Ftr.std(dim=(0,1))}` -- computed ONCE from the training family
  (Ftr, all `ntrain` samples stacked), stored, and reused identically
  (`normd`) for both training and held-out test evaluation. Not
  recomputed per sample anywhere.

Point 2: also correct as things stand, but for a reason worth stating
explicitly -- checked whether it actually applies to anything already
published, not just to the general principle. Every MMS-operator result
so far (including Table 24a's own three-mesh rate) trains and evaluates
a SEPARATE network at one fixed N throughout -- confirmed directly in
`cell_mms_operator_rate.py`: it calls `python -m omar_pfem.mms_operator
--N 9`, `--N 17`, `--N 33` as three independent training runs, then fits
the rate externally across the three finished results. No single
trained model is ever evaluated zero-shot at a resolution other than its
own training mesh, so the network's own force-field input and the
energy loss's own nodal-force term are the same self-consistent object
throughout any one run -- exactly the case Timon's own email says is
fine. This is forward-looking guidance for an MMS-based coarse-to-fine
zero-shot study that does not exist yet, not a bug in what is published.

---

Dear Timon,

Thanks -- glad round-11 points 1-3 are clear now. Quick answers on both:

1. Yes, fixed from the training set, not sample-wise. In the MMS
operator study, the body-force channels' mean and standard deviation
are computed once from the training family and reused identically for
evaluation on the held-out test family -- the same fixed, dataset-wide
convention used everywhere else in the report that standardizes inputs.
A per-sample normalization is not in use anywhere, so the absolute load
magnitude is not being silently discarded.

2. Good point, and worth confirming explicitly before we build anything
further with MMS. As things stand today this is not yet an issue in
what is published: every MMS-operator result so far (including the
three-mesh convergence rate) trains and evaluates a separate network at
one fixed resolution throughout -- there is no single trained model
whose input is evaluated zero-shot at a resolution different from its
own training mesh, so the network's own force-field input and the
energy loss's own nodal-force term are always the same self-consistent
object within any one run.

If we do build an MMS-based coarse-to-fine resolution-generalization
test later -- an attractive option, since MMS gives an exact label at
any resolution with no reference FEM solve needed -- we will do exactly
as you describe: feed the network the continuous body-force field,
evaluated pointwise at each mesh's own node coordinates (so its
numerical representation stays resolution-independent), while
continuing to use the resolution-dependent consistent nodal force
vector (the same assembly routine already used) inside the energy
loss's own work term. That keeps the two roles separated -- the
network's own input stays consistent in magnitude across resolutions,
while the loss itself still integrates correctly against whichever mesh
is actually being trained or evaluated on.

No code changes needed right now, since this only bears on a
not-yet-built extension -- thank you for flagging it ahead of time.

Best regards,
Omar
