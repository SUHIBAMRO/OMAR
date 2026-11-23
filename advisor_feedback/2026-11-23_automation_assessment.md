# Which verification decisions can be automated reliably? (2026-11-23)

Evidence: `three_way_result.json`, `crosscheck_result.json`, `cylinder3d_result.json`,
`followup_l_shape_v_right_top.json`, the diagnosis engine (`diagnose.py`) and the human-intervention
log (`HUMAN_INTERVENTION.md`). "Reliable" means: measured on problems the decision was not tuned on,
with no wrong decision observed. Post hoc items are marked.

| Decision | Automated now | Evidence of reliability | Safeguard | Status |
|---|---|---|---|---|
| Accept a quantity with a demonstrated rate (one family) | yes (gates) | 3D benchmark: true error inside the certified uncertainty for 4 of 4 rate-certified quantities (pre-registered) | U must meet the class tolerance; peak with U > tol refused (true error 2.03% > 2%) | reliable for the family; NOT a statement about the continuum value (B3: ~2% spline gap) |
| Make that uncertainty final | yes (cross-check v2, post hoc) | 27 of 38 rate-certified quantities cross-checked; 11 left INCONCLUSIVE | independent sequence must itself pass the gate; unresolved never certified | reliable as a gate; needs independent solvers per problem class (human-written today) |
| Refuse a quantity with no finite limit (singular peak) | yes (singular-order check + diagnosis M1) | missed failures 0 of 16 for every method; M1 correct on Cook, plate, L-shape, crack (held out) | none needed | reliable |
| Name the cause of non-convergence | partly (16 hypotheses, 6 classes) | causes established independently: 12 of 12 after the changes (9 of 9 before, on 5 problems); B1/B3 agreement 16 of 16 is a consistency check only | evidence "sufficient" only if no applicable test is open; otherwise NEEDS_DATA | reliable where facts exist; on problems with derived facts most solver/physics tests stay NEEDS_DATA |
| Recognise a missing check (capability gap, C1, post hoc) | yes | plate: 0 of 6 before (read as pre-asymptotic), now recognised | C1 never changes a verdict | trivially reliable (the check states it); not evidence of diagnostic skill |
| Decide that a sequence is pre-asymptotic (D4 via pairwise orders, post hoc) | yes | L-shape displacement: predicted next pairwise order 0.97-1.15, observed 1.01 | no verdict change from D4 | one test case; promising |
| Add a finer level within a budget | yes for 2D ladders with a level runner (`followup.py`) | cost estimate 29 s / 4.7 GB, measured 41 s; verdict re-evaluated with frozen checks (stayed FAIL at 0.938 < 0.94) | budget; new levels stored apart from pre-registered ladders | reliable mechanics; 3D needs the cluster (manual submission) |
| Tighter-tolerance / linear-solver / MMS re-runs (S1, S2, S3) | B1, B2, Cook, B3 only (problem-specific actions) | consistent with manual results | — | needs a generic solver hook per problem class |
| Write the problem specification | LLM (formalizer) | recovered correctly in 9 of 9 billed runs | code validates the spec | reliable on B1-B3; the only clear LLM gain |
| Choose quantities and predict verdicts | LLM | B3 coverage 61%; 3-7 false-alarm predictions per run, varying between runs | code decides every verdict | not reliable; increases human checking (proxy 19 vs 9 decisions on B1-B3) |
| Write the data adapter and problem facts | no | — | — | the main remaining manual step per new problem |
| Find errors in the reference itself (double stress conversion, 2% gap) | no | found only by independent implementations | cross-check rule now enforces the comparison where an independent solver exists | requires independent code, not an agent |

Most important missing automation, in order: (1) generic independent-discretization runners
(so the cross-check is not limited to problems with a hand-built second solver);
(2) generic solver hooks for S1/S2/S3 follow-ups; (3) adapters and facts from a problem
description (today written by hand); (4) 3D follow-ups that need more than 15 GB, which by
policy go to the cluster only by authorized manual submission.
