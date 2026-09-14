# Reply to Timon's provenance/database request — drafted 2026-09-14

Do not send without Omar's own review.

---

Subject: Re: saving results/setups — what already existed, what I closed, and what's still coming

Dear Professor Rabczuk,

Thank you for the note — this is a real gap worth closing properly, and
good timing given how much round-10 produced.

Checking first: the project already had infrastructure for exactly
this. Every script that produces a final number can call a shared
`write_manifest()` function that records, in one place next to that
run's own output: the exact git commit (and whether the working tree
was clean), the full command line and every argument, the environment
(GPU, CUDA, PyTorch version), start/end time, the headline results
themselves, and every file the run wrote. It is append-only per
directory, so re-running a script adds to the history rather than
overwriting it. Most of the project's scripts already call it.

What I found and fixed: round-10's own newest scripts (the ones behind
the accuracy-matched comparison, the batch-size/throughput benchmark,
and the accuracy-matched break-even) had not been wired to it yet.
Fixed all of them, smoke-tested locally before trusting the change.

As an interim measure until you share more about the platform your
group is building, I put together `EXPERIMENT_LOG.md` — one line per
final round-10 result, each tied to its exact git commit and pointing
at the JSON file with the real numbers. I made a point of keeping the
two genuine negative findings from this round visible in it rather than
only the favorable ones: the operator's default (unoptimized) inference
mode never actually repays its own training cost against a FEM baseline
matched to its own accuracy, and peak PK1 stress converges slowly for
both methods (likely a boundary-singularity artifact), not just for
ours.

I am now extending the same treatment backward across the rest of the
project's history (round 5 through round 10) and will share that once
it's put together properly rather than something assembled in a rush.

I have deliberately not tried to design anything beyond this — you
mentioned your group is building a more structured platform, and I did
not want to build something that might duplicate or conflict with that.
Happy to adapt whatever we have here to whatever format works best for
it once you share more.

Best regards,

Omar
