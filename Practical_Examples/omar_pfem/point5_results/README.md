# Point 5 results — error in physically important quantities

The six output files of `omar_pfem/physical_quantities_eval.py`, one per
(geometry, material) case. These are the numbers written into report §8.8
(Tables 15–17) and into section 6 of the parallel summary document.

## Where they came from

Each file was produced on Colab by `physical_quantities_eval.py` and written to
Google Drive; the copies here were taken from a Colab dump of those Drive files
on 2026-08-27. The `checkpoint` field inside each file records exactly which
trained model was scored, and every one of them is the checkpoint the rest of
the report already uses:

| case | checkpoint |
|---|---|
| B1 × Neo-Hookean | `pfem_run/results/B1_neo_hookean/model_best.pt` |
| B1 × Mooney-Rivlin | `pfem_run/results/B1_mooney_rivlin/model_best.pt` |
| B1 × Arruda-Boyce | `pfem_run/results/B1_arruda_boyce/model_best.pt` |
| B2 × Neo-Hookean | `pfem_run/B2_accuracy_search/lossnorm/train/model_best.pt` |
| B2 × Mooney-Rivlin | `pfem_run/B2_accuracy_search_mooney_rivlin/lossnorm/train/model_best.pt` |
| B2 × Arruda-Boyce | `pfem_run/B2_accuracy_search_arruda_boyce/lossnorm/train/model_best.pt` |

The three B2 entries are the corrected loss-normalized runs of report §9.1, not
the superseded pre-fix ones — the same distinction that had to be fixed in
Table 7 at v26.

All six ran with `--ntrain 800 --ntest 50`, so the 50 scored samples are
held out from the 800 the network trained on.

## One incompleteness, stated rather than filled in

The Colab dump printed the first 4,000 characters of each file. The three B1
files are shorter than that and are complete. The three B2 files are longer,
because B2 reports its reactions separately on each of its two symmetry edges,
and each was cut off part-way through the second edge's block. What survives is
every quantity through `reaction_nodal_rel_L2_edge1`; what is missing is
`reaction_max_pred_edge1`, `reaction_max_ref_edge1` and
`reaction_max_rel_err_edge1` for those three cases.

Nothing was reconstructed or estimated to fill the gap. Table 17 in the report
uses only the resultant and nodal reaction errors, which are present for every
case and both edges, and the largest-single-nodal-reaction error is quoted in
the text for B1 only, where it is complete. Re-pulling the three B2 files from
Drive in full would close the gap; it changes nothing already reported.
