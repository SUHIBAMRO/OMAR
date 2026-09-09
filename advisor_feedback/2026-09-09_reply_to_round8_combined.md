# Combined reply to Timon's round-8 email — drafted 2026-09-09

Per Omar's own decision (2026-09-06): hold every round-8 point until
the whole work queue is finished, then send ONE combined reply rather
than partial answers point-by-point. The queue (items #1-13) is now
finished as of today (#12 figures, #13 torch-fem), so this draft
addresses all seven of Timon's points together.

**Status per point, condensed from PROJECT_STATUS.md** (see that file
for full derivation/caveats on each number below):
- Point 1 (benchmark scale): technical prerequisites now done (peak-
  stress bug fixed and re-run, preconditioner fixed); the deeper
  "is B1/B2 big enough" question is answered with Option A
  (justify current size) but explicitly flagged as provisional, not
  a closed decision — Timon may still prefer the tire-scale example.
- Point 2 (OOD, all 6 cases): DONE.
- Point 3 (DD-NO coarse-vs-fine): DONE.
- Point 4 (DD-NO wall-clock): DONE (already sent informally before).
- Point 5 (GPU-FEM scaling/preconditioner): DONE, one caveat kept.
- Point 6 (MMS): energy norm fixed correctly; richer sine/cosine
  family built but ONLY for Neo-Hookean, not yet merged with the
  three-material extension (which still uses the original single-mode
  field) — genuinely incomplete, stated as such below, not glossed
  over.
- Point 7 (break-even restructure): DONE in substance, one caveat kept
  (DD-NO inference cost assumed equal to the physics-informed
  operator's, not separately measured).

**Do not send without Omar's own review** — this is a first full draft,
not a final-checked one. In particular: point 2's exact B1×MR/B1×AB
degradation numbers should be pulled from Table 25 directly before
sending (this draft states the B2 numbers, which are confirmed, and
describes B1 only qualitatively); point 6's honesty about the
unmerged MMS family should be double-checked against how much detail
Omar wants to expose before the richer-family run is redone across all
three materials.

---

Subject: Round-8 feedback — combined reply on all seven points

Dear Professor Rabczuk,

Thank you for the detailed review — I've now worked through all seven
points, plus your general suggestion to convert more tables into
figures. Answering all seven together, as you sorted them.

**1. GPU-native FEM benchmark scale.** I used your suggested check
(stricter tolerances vs. a high-fidelity reference) rather than scaling
to a new problem for now. Table 6a (B1 x Neo-Hookean vs. a ~10M-DOF
reference) gives a clean answer in the well-behaved range: <=1% error
needs N=201 (~15 min), <=0.5% needs N=401 (~30 min) — tens of minutes,
not milliseconds, even at modest accuracy. I want to be upfront that
this doesn't fully close your question: even at N=1401 (the largest
mesh I ran, ~7.6h after fixing the preconditioner — see point 5), H1
and the energy norm still don't reach the 1e-4 target you'd asked for
earlier. I'm reading that as a point in the same direction (the problem
doesn't get "easy" even at large compute), but I'm not treating it as a
closed decision — if you'd still prefer the tire-profile example to
make this unambiguous, I'm ready to scope that as a separate, deliberate
effort.

I also implemented the peak-stress QoI you named as an example. The
first real run had a genuine bug — the "reference" peak value was
silently redefined at every mesh resolution instead of being one fixed
target, which produced meaningless noise I nearly reported as a real
finding. Fixed by locating the fine reference's own peak-stress point
once and evaluating every coarse resolution at that same fixed point.
The corrected sweep now converges monotonically and cleanly: peak-stress
relative error goes 62.6% -> 54.1% -> 43.9% -> 32.0% -> 21.2% -> 14.2%
-> 7.7% from N=51 to N=1401.

**2. Out-of-distribution, all six cases.** Done for all six geometry x
material combinations (previously only B1 x Neo-Hookean). For B2:
Neo-Hookean degrades to 4.75x at 3-sigma material shift (loading alone
barely moves it, 0.99x); Mooney-Rivlin is the most fragile at 5.47x;
Arruda-Boyce is the most robust at 2.27x, and its loading-shift case
actually improves slightly (0.79-0.93x). B1's Mooney-Rivlin and
Arruda-Boyce follow the same qualitative pattern as the B1 x
Neo-Hookean case already in the report. Full table (Table 25) and a
field-grid figure for all six cases are in the updated report.

**3. DD-NO, coarse-vs-fine label generation.** Trained the data-driven
operator on labels from a coarse mesh (N=13) versus a fine one (N=33),
same training budget as Table 21, then tested both across seven unseen
resolutions. Coarse-trained: most accurate at its own resolution
(10.5% at N=13) but degrades steadily as the test mesh moves away from
it, up to 25.9% at N=49 — a 2.5x spread. Fine-trained: more uniform,
10.9-13.7% across the same range, trading some peak accuracy for
resolution robustness. This is exactly the asymmetry you asked about:
the physics-informed operator never sees this failure mode at all,
since it never trains on FEM labels from any particular mesh.

**4. DD-NO wall-clock training time.** 1,458.3s (Adam) / 1,463.0s
(AdamW+OneCycle), versus the physics-informed operator's 2,873.8s /
3,108.9s for the same two optimizers. Not included: the 5.65h of CPU
time spent generating the DD-NO's 800 training labels — noted
explicitly next to the number rather than folded in.

**5. GPU-FEM scaling.** Built and validated a geometric multigrid
preconditioner (after a block-Jacobi attempt didn't help enough), then
reran the full resolution sweep: CG failures are now zero at all seven
resolutions (N=51 through N=1401), down from 20-80 failures at the four
largest sizes before the fix. Wall-clock at the largest sizes: N=701
~2.0h, N=1001 ~4.8h, N=1401 ~7.6h. One caveat I want to flag rather than
hide: the fitted convergence RATES (not the failure counts or
wall-clock, which are solid) at N=1001/1401 should be read cautiously,
since the fine reference mesh is only about 2.2x those resolutions,
which likely flattens the measured rate. Your point that "assembly is
negligible" isn't the right interpretation is already stated in the
report near-verbatim — the element-level work is hidden inside every
CG Hessian-vector product, not a separate phase that disappeared.

**6. MMS.** Two of the three things you asked for are done; I want to
flag honestly that they haven't been combined yet. The energy norm is
now computed correctly as a proper quadrature norm against the exact
continuous solution (the previous version compared against the nodal
interpolant, which silently superconverges — caught and fixed). The
richer manufactured-solution family (sum of several sine/cosine modes,
boundary conditions preserved) is built and verified — Q4/Q9 rates
match theory exactly — but so far only for Neo-Hookean. The extension
to Mooney-Rivlin and Arruda-Boyce that's currently in the report still
uses the original single-mode field, not the richer family. Re-running
the richer family across all three materials is the remaining work
here; I didn't want to present the two as already merged when they
aren't.

**7. Break-even methodology.** Restructured as you suggested. Comparison
A (single query, same hardware, batch size 1 both sides) is now labeled
explicitly as the primary comparison, with the batched throughput
numbers (bs=8/32/128) presented separately rather than as the headline
figure. Comparison B (total cost of ownership) is new: the data-driven
operator's 800-solve label-generation cost (5.65h CPU) makes it more
expensive than the physics-informed operator by a fixed 18,924s (Adam)
/ 18,694s (AdamW+OneCycle) at every problem count, not a variable
break-even threshold — since the physics-informed operator pays no
data-generation cost at all. One caveat: this assumes both operators'
own inference cost is equal (same architecture), which I haven't
separately measured for the DD-NO specifically.

**Tables into figures.** Per your general suggestion, every table in
both the report and the summary that lacked a figure now has one — 39
figures in total, covering all seven points above plus the earlier
results.

Happy to walk through any of this on a call if useful, especially
point 1's benchmark-scale decision and point 6's remaining MMS work.

Best regards,

Omar
