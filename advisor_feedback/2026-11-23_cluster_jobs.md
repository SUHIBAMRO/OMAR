# Cluster jobs, ready for manual submission (2026-11-23)

No agent connects to the TU Ilmenau HPC. You submit these jobs yourself. All three
are CPU-only jobs and none needs a GPU.

## 0. Once, on the login node

```bash
cd ~/OMAR
git fetch origin claude/claude-code-question-d307wp
git checkout claude/claude-code-question-d307wp
git pull --ff-only
cd Practical_Examples
```

In the commands below, `<queue>` is your CPU batch queue. Earlier runs used
`BatchGPU` with `--cpu`. Any queue that allows the stated memory and wall time will do.

## 1. Spline cylindrical averages, B3 groove: `iga_b3_cyl_job.sh` (priority 1)

- **Purpose:** produce the spline values of `gk2_cyl_rr_avg` and `gk2_cyl_tt_avg`.
  These are the two B3 quantities that still have no independent cross-check.
- **Starting point:** displacement levels 1–3 are already stored in
  `iga_b3_cyl_result.json` and are skipped.
- **What it computes:** P1 levels 1–4, then displacement level 4.
  - Every level must reproduce the stored Cartesian value to 1e-6, or the job stops.
- **Cost:** about 5–8 h and 16 GB. Local runs never finish because the development
  machine restarts.

```bash
bsub -n 8 -R "rusage[mem=16G]" -q <queue> -W 12:00 \
     -o iga_b3_cyl.%J.out -e iga_b3_cyl.%J.err < iga_b3_cyl_job.sh
```

**Output:** `omar_pfem/verification_agent/iga_b3_cyl_result.json`

## 2. Follow-up level k = 12, clamped 3D block: `block3d_followup_job.sh` (priority 2)

- **Purpose:** run the level that the agent chose but declined within its own budget.
- **Predictions:** committed beforehand in `block3d_followup_preregistration.json`.
- **Cost:** 0.3–2 h and 48 GB.

```bash
bsub -n 8 -R "rusage[mem=48G]" -q <queue> -W 4:00 \
     -o block3d_followup.%J.out -e block3d_followup.%J.err < block3d_followup_job.sh
```

**Output:** `omar_pfem/verification_agent/clamped_block_3d_followup.json`

## 3. Spline level 5, B3 groove: `iga_b3_job.sh` (optional, the largest)

- **Purpose:** decide whether the groove peak converges.
  - Orders at levels 2–4 so far: 1.71, 1.06, 1.00.
  - The report states this quantity as *not certified*. The job can only strengthen
    that statement or overturn it.
- **Cost:** about 13 h, 32 GB, 607,104 unknowns.

```bash
bsub -n 8 -R "rusage[mem=32G]" -q <queue> -W 24:00 \
     -o iga_b3.%J.out -e iga_b3.%J.err < iga_b3_job.sh
```

**Output:** keys `L5:p2:fbar_p1` and `L5:p2` are added to
`omar_pfem/verification_agent/iga_b3_result.json`.

## 4. Returning the results

Use either of these:

- **(a) Push from the cluster** (only the result files and the `.out` logs):
  ```bash
  git add -f omar_pfem/verification_agent/iga_b3_cyl_result.json iga_b3_cyl.*.out
  git commit -m "Cluster results"
  git push origin claude/claude-code-question-d307wp
  ```
- **(b) Copy them over:** copy the JSON files and the `.out` logs back to me.

After that, the integration is mechanical:

1. crosscheck
2. three_way
3. verify_report
4. report section 31.7
5. scoring the follow-up with `block3d_followup --score`
6. PDF
7. change log

The acceptance criteria are those already committed. Any change made after seeing the
results will be labelled post hoc.

## Not a cluster job: billed LLM ablation (needs your approval)

- **Already done:** b1, b2 and b3 (3 runs each, measured $0.55–1.09 per run).
- **Prepared but not run** (problem cards added): cook, plate, lamina, lshape, crack,
  cylinder, block.
- **Command, after your approval:**

  ```bash
  python -m omar_pfem.verification_agent.ablation --backend claude-cli \
      --problems cook,plate,lamina,lshape,crack,cylinder,block --repeats 3 \
      --out omar_pfem/verification_agent/ablation_result_round_M.json
  ```

- **Estimated cost:** 7 problems × 3 runs × $0.55–1.09 = **about $12–23**.
- **Estimated time:** about 2 h at 250–300 s per run.
- **Rule-only baseline** (free; already computed for the new problems):
  - misses: cook `strain_energy`, cylinder `peak_hoop`, block `root_gk_avg_von_mises`;
  - false alarm: plate `hole_peak_sigma_xx`;
  - these misses are what the LLM step would be tested against.
