# Standalone question to Timon — GPU-native solver vs. torch-fem comparison

Drafted 2026-09-09, at Omar's request: a short, focused email asking
only about this one open item (the two comparison caveats below),
rather than the full combined round-8 reply.

---

Subject: Quick question — GPU-native FEM vs. torch-fem comparison

Dear Professor Rabczuk,

I ran the direct comparison you asked for between our matrix-free
GPU-native solver and torch-fem (an established GPU-accelerated FEM
library), at N=401/701/1001/1401. torch-fem wins wall-clock by a
large, non-monotonic margin (418x/655x/1197x/936x, peaking at N=1001).

Before reading too much into that number, I want to flag two things
that make it not a clean apples-to-apples comparison:

1. The two solvers aren't run at matched precision — torch-fem's own
   near-null-space construction hardcodes float32, forcing loosened
   tolerances (rtol=atol=1e-3) against our float64/1e-8. This plausibly
   explains a large share of the gap on its own.
2. Our solver resumed from an already-converged checkpoint rather than
   solving fresh, so I don't have a real peak-memory number for it to
   set against torch-fem's measured one.

My reading is that this is a genuine architectural trade-off
(matrix-free vs. explicitly-assembled), not a defect in either solver
— consistent with what you predicted before I ran it. But I wanted to
check directly: is this comparison good enough as it stands, or would
you like me to re-run it at matched precision/tolerance and with a
fresh (non-resumed) solve so a real peak-memory number exists, before
this goes into the final report?

Separately, and not urgent: the MMS richer-family gap you'd flagged
(round-8 point 6) is now fully closed too — I've re-run the richer
sine/cosine family for Mooney-Rivlin and Arruda-Boyce as well, and
Q4/Q9 convergence rates now match theory for all three materials. No
action needed on that one, just flagging it as done.

Best regards,

Omar
