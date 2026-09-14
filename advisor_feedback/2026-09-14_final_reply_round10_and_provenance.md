# FINAL reply to Timon — round-10 points + provenance request, as sent

Finalized and sent by Omar 2026-09-14. Supersedes the two separate
drafts this merges: `2026-09-13_reply_to_round10_draft.md` (the 5
round-10 points) and `2026-09-14_reply_to_provenance_request.md` (the
write_manifest/EXPERIMENT_LOG.md provenance response) — Omar combined
both into one email rather than sending two.

Attachments: the updated full report (`PFEM_Transolver_Report_2026-09-14.docx`)
and the short Round-10 summary (`PFEM_Round10_Summary_2026-09-14.docx`).

---

Subject: (round-10 follow-up + provenance request reply)

Dear Prof. Rabczuk,

Thank you for the note. I agree that this is an important point, particularly for traceability, reproducibility, and the long-term reliability of the computational results.

The project already has a shared write_manifest() function that records the exact git commit, run configuration, environment, timing, outputs, and headline results for each final run. I found that several of the newest Round-10 scripts had not yet been connected to it, so I have now fixed and tested those.

As an interim step, I also created an EXPERIMENT_LOG.md, with each final Round-10 result linked to its exact git commit and corresponding JSON output. I have kept the relevant negative findings visible as well, rather than recording only successful results. I am now extending the same documentation systematically across the entire project history, so that the final computational experiments, configurations, results, and relevant failures or negative findings are traceable consistently from the beginning of the project onward.

Regarding the five Round-10 points, Points 1, 2, 3, and 5 are now completed. The accuracy-matched FEM comparison was updated after the multi-resolution retraining, the memory/throughput study was completed, the N=1401 inference was profiled and optimized, and the accuracy-matched break-even was recalculated. The multi-resolution retraining was particularly effective, reducing the displacement error at N=1401 from 44.65% to 5.85%.

I have attached the updated full report and the short Round-10 summary.

For Point 4, before committing substantial GPU time to full data generation and Transolver training, I first built and verified a practical candidate geometry. I extended the existing B2 pressure-vessel-like ring with a smooth local notch (Gaussian dimple) on the inner wall. This preserves the already-verified B2 physics and solver framework while introducing a genuine local stress concentration of the type you suggested.

I then ran a mesh-convergence study specifically to check whether fine spatial resolution is genuinely required for the local quantity:

Mesh resolution (elements) | Peak PK1 stress at notch | Change | Max displacement | Change
72 | 11.70 | — | 0.008193 | —
288 | 12.72 | +8.7% | 0.008384 | +2.3%
1,152 | 13.45 | +5.7% | 0.008455 | +0.85%
4,608 | 13.9372 | +3.7% | 0.008478 | +0.27%

The two quantities separate clearly with refinement: the global displacement is already essentially converged, while the local peak stress at the notch is still increasing. The ratio between the local-stress change and the displacement change also grows across the refinements, which suggests a genuine local resolution requirement rather than a generally slow-converging problem.

The geometry and nonlinear solver have also been smoke-tested and converge cleanly. I have deliberately paused here before generating the full dataset and training a new Transolver checkpoint, because this next step would require considerably more computational time.

Would you consider this ring-with-local-notch example sufficient and worthwhile for Point 4 and for the first paper, or would you prefer a different practical geometry or local quantity of interest before I proceed with the full training?

I would appreciate your view on this before I proceed with the full training.

Thank you again for the feedback.

Best regards,
Omar
