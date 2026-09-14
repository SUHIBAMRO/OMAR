# Experiment Log

Per Timon's note (2026-09-14): "please note the exact git commit and a
one-line description of the setup for every run you consider final,"
pending the more structured platform he is building. This file is the
interim version of that for every round-10 result currently cited in
the Report/Summary/reply draft. Going forward, every script that
produces a final number should also call `omar_pfem.run_manifest
.write_manifest(...)` (already the case for most of this project's
scripts; round-10's own gap is closed as of commit `7681812` below --
see the commits in that range for exactly which scripts gained the
call and why). A manifest entry captures far more than this file does
(exact command, full args, environment, timing) — this file is only the
one-line index into that, git-commit by git-commit, for a human
skimming what happened when.

Repo: `SUHIBAMRO/OMAR`, branch `claude/claude-code-question-d307wp`.

## Round-10 final results

| Commit | Date | One-line description | Result file(s) |
|---|---|---|---|
| `4dd4484` | 2026-09-12 | **Bug, not a result**: fixed a checkpoint-resolution defect that had every round-10 NO accuracy script silently evaluating the wrong (data-driven) model instead of the physics-informed one. | `resolve_b1_checkpoint.py` |
| `c2586c1` | 2026-09-13 | NO accuracy at N=1401 vs. real FEM ground truth, original (N=21,33) checkpoint, corrected: 44.65% disp_rel_L2, degrading monotonically away from training resolutions. | `no_accuracy_degradation_sweep_B1_neo_hookean.json` |
| `cd043e4` | 2026-09-13 | Corrected peak-stress metric (fixed physical location from a fine reference, not the coarse mesh's own sample-max) and the resulting coarsest-suitable-FEM crossover, original checkpoint. | `no_peak_stress_fixed_location_B1_neo_hookean.json` |
| `01e50e4` | 2026-09-14 | Multi-resolution retrained checkpoint (N=21,33,101,201): N=1401 disp_rel_L2 drops from 44.65% to 5.85% (7.6x) vs. real FEM ground truth. | `no_accuracy_multires_retrain_comparison.json`, `no_accuracy_degradation_sweep_multires.json` |
| `086db1d` | 2026-09-14 | **Finding, not just an improvement**: the retrained checkpoint's fixed-location peak-stress accuracy now BEATS torch-fem's own best low-N (N<=49) peak-stress accuracy for every NO resolution N>=29 -- flips which QoI binds the coarsest-suitable-FEM crossover. | `no_peak_stress_fixed_location_multires.json` |
| `4467d34` | 2026-09-14 | Confirmed real final training wall-clock for the multi-res checkpoint: 41,881.28s (~11.6h), early-stopped exactly as configured (8 checks, no improvement). | `metrics_history_multires.json` |
| `ae7ee83` | 2026-09-14 | Batch-size/throughput and torch.compile/TF32 re-measured with the corrected (but still original N=21,33) checkpoint -- confirmed unchanged from the wrong-checkpoint-era numbers. | (Drive-side JSONs, see PROJECT_STATUS.md 2026-09-14 entries) |
| `732cf47` | 2026-09-14 | Batch-size/throughput re-verified specifically against the multi-res checkpoint: max-bs throughput 4,182.27 vs. 4,203.79 samples/s (0.5% difference) -- checkpoint-independent. | `max_feasible_batch_multires.json` |
| `9ba82d8` | 2026-09-14 | torch.compile/TF32 re-verified specifically against the multi-res checkpoint: all four variants within 0.01-0.8% of the original numbers -- checkpoint-independent. | `no_inference_torch_compile_N1401_multires.json` |
| `2feef0d` | 2026-09-14 | **Negative result, reported as such**: accuracy-matched break-even, NO@N=1401 vs. FEM@N=11 (its own coarsest-suitable mesh). In default eager fp32 mode the NO NEVER breaks even -- the accuracy-matched FEM is already cheaper per sample. Only with torch.compile+TF32 does it break even, after 34,005 samples (~3.72 GPU-hours) against the 11.6 GPU-hour training cost. | `break_even_accuracy_matched_N1401.json` |
| `7681812`..`10f7e78` | 2026-09-14 | Housekeeping: added `write_manifest()` calls to the round-10 scripts that lacked them (`gpu_fem_benchmark.py`, `no_accuracy_at_n1401.py`'s two sweep functions, both torch.compile cell scripts, the break-even cell script) so every future run of these captures git commit + full setup automatically, per Timon's own request. Also folded all of the above into the canonical Report/Summary deliverables and fixed one process gap (a claim written into the Report before its own verification had actually run -- caught and corrected with the real numbers once they existed). | n/a (code + document changes) |

## Known negative/failure results worth keeping visible

These are genuine, real findings that came out unfavorably (not
development bugs) -- kept here explicitly per Timon's own request not
to bury failures:

- **NO's default (eager fp32) inference never economically justifies
  its own training cost** against a FEM mesh that is only as accurate
  as the NO itself (see `2feef0d` above). Only holds with the
  torch.compile+TF32 optimization applied.
- **Peak PK1 stress remains a slow-converging QoI for BOTH methods**
  (NO and native FEM), likely because the reference point sits at a
  domain-corner boundary-condition transition prone to a stress
  singularity -- neither method's peak-stress error should be read as
  a clean, well-posed target without this caveat (see `cd043e4`,
  `086db1d`).
