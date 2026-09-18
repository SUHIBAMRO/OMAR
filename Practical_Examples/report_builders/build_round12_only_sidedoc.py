"""Builds a standalone side document containing ONLY the three points from
Timon's round-12 email (2026-09-18), in his own point order, with their
full tables/figures/text -- copied VERBATIM out of the current Report
(PFEM_Transolver_Report_2026-09-18.docx), not re-derived or re-typed, so
there is zero risk of the side document drifting from the already-
verified Report content.

Copying mechanism: works directly on the underlying document XML
(lxml elements), operating on doc.element.body's own children in true
document order (a flat list of w:p/w:tbl elements, unlike python-docx's
own doc.paragraphs/doc.tables, which are separate lists that lose
interleaving). Each copied element is deep-copied; any embedded image
(<a:blip r:embed="...">) has its relationship re-created in the
destination document (source image bytes copied via
part.relate_to(image_part, ...)), so figures render correctly in the
new file, not just as broken references.
"""
import copy
import os

from docx import Document
from docx.opc.constants import RELATIONSHIP_TYPE as RT
from docx.oxml.ns import qn

DELIV = '/home/user/OMAR/advisor_feedback'
REPORT = os.path.join(DELIV, 'PFEM_Transolver_Report_2026-09-18.docx')
OUT = os.path.join(DELIV, 'Round12_Reply_to_Timon_Points_2026-09-18.docx')

src = Document(REPORT)


def body_children(doc):
    return list(doc.element.body)


def elem_text(el):
    if el.tag != qn('w:p'):
        return None
    return ''.join(t.text or '' for t in el.iter(qn('w:t')))


def find_index(children, text):
    matches = [i for i, el in enumerate(children)
               if elem_text(el) is not None and elem_text(el).strip() == text]
    assert len(matches) == 1, f'{len(matches)} matches for {text!r}'
    return matches[0]


def slice_by_anchor_text(doc, start_text, end_text):
    children = body_children(doc)
    i0 = find_index(children, start_text)
    i1 = find_index(children, end_text)
    assert i1 >= i0
    # If the end anchor is itself a table's own caption, the table follows
    # immediately after it in the body and must be pulled in too -- a bare
    # paragraph-text slice would otherwise silently drop it. Extend by AT
    # MOST one table: a second table right after the first (as happens
    # here -- an unrelated, pre-existing table with its own caption
    # BELOW it, not above) belongs to different content and must not be
    # swept in too.
    if i1 + 1 < len(children) and children[i1 + 1].tag == qn('w:tbl'):
        i1 += 1
    return children[i0:i1 + 1]


def append_heading(dst_doc, text):
    from docx.shared import Pt
    p = dst_doc.add_paragraph()
    r = p.add_run(text)
    r.bold = True
    r.font.size = Pt(14)


def append_subheading(dst_doc, text):
    from docx.shared import Pt
    p = dst_doc.add_paragraph()
    r = p.add_run(text)
    r.bold = True
    r.italic = True
    r.font.size = Pt(11)


def append_quote(dst_doc, text):
    from docx.shared import Pt
    p = dst_doc.add_paragraph()
    r = p.add_run(text)
    r.italic = True
    r.font.size = Pt(10)
    r.font.color.rgb = None


def copy_elements(dst_doc, src_doc, elements):
    """Deep-copies each element into dst_doc's body (before its own
    sectPr, so page setup stays valid), fixing up image relationships
    along the way."""
    body = dst_doc.element.body
    sect_pr = body.find(qn('w:sectPr'))
    for el in elements:
        new_el = copy.deepcopy(el)
        for blip in new_el.iter(qn('a:blip')):
            rId = blip.get(qn('r:embed'))
            if not rId:
                continue
            image_part = src_doc.part.related_parts[rId]
            new_rId = dst_doc.part.relate_to(image_part, RT.IMAGE)
            blip.set(qn('r:embed'), new_rId)
        if sect_pr is not None:
            sect_pr.addprevious(new_el)
        else:
            body.append(new_el)


dst = Document()
# Match the source document's default table style so copied tables keep
# their look (borders etc.) -- tables reference styles by id, and that
# id must exist in the destination too.
if src.tables:
    src_style = src.tables[0].style
    try:
        dst.styles.add_style(src_style.name, src_style.type)
    except ValueError:
        pass  # style already exists (e.g. built-in "Table Grid")

append_heading(dst, "Round-12 reply to Timon -- the three points from his 2026-09-18 email, in full")
p = dst.add_paragraph()
p.add_run(
    "This document contains ONLY the material answering Timon's three round-12 points, "
    "copied verbatim (same tables, same figures, same text) from the full Report "
    "(PFEM_Transolver_Report_2026-09-18.docx) -- nothing summarized or re-derived. "
    "See that document for the complete report."
).italic = True

# =======================================================================
# POINT 1 -- Cauchy-stress fixed-region QoI
# =======================================================================
append_heading(dst, "Point 1 -- Cauchy-stress fixed-region QoI, final checkpoints")
append_quote(dst,
    "“We need one final accuracy-versus-resolution table using the final retrained "
    "checkpoints only... For the stress comparison, I would prefer Cauchy stress rather "
    "than PK1... I suggest defining a fixed physical region around the stress "
    "concentration and reporting, in addition to the Cauchy-stress field error, a robust "
    "local quantity such as a volume/area-weighted average or 95/99th percentile... A "
    "true maximum can still be reported separately.” — Timon, 2026-09-18"
)
elements = slice_by_anchor_text(
    src,
    "Round-12 point 1 (Cauchy-stress fixed-region QoI, final checkpoints): the advisor "
    "asked for one final accuracy-versus-resolution table -- for EACH resolution, FEM "
    "vs. operator vs. the same fine reference, in displacement L2, H1/energy norm, "
    "reaction force, and stress -- with Cauchy stress (not PK1) as the main engineering "
    "quantity, explicitly rejecting the bare pointwise maximum as the primary QoI "
    "(singular/mesh-dependent at corners even for FEM) in favour of a FIXED physical "
    "region around the stress concentration, reporting a robust local statistic (a "
    "weighted average or 99th percentile) with the true max still available separately, "
    "the region fixed in physical space across every resolution. All new code "
    "(fixed-region selection, the Cauchy push-forward from the already-validated PK1 "
    "stress, and the weighted-percentile/top-fraction statistics) was verified on CPU "
    "before any GPU time was spent (5 checks, all passed). Below: two tables per case, "
    "all six cases, all sixteen resolutions tested (N=3..49) -- the established "
    "accuracy QoIs, then the new Cauchy-stress QoI. Scope, stated plainly: these "
    "resolutions are the LOW_N range already established for this project's own "
    "coarsest-suitable-mesh crossover analysis above, against a fine reference at "
    "N=201 -- this does NOT extend to N=1401, which would need a fresh, expensive "
    "fine_N=2236 reference for the five cases that have never had one, a separate, "
    "bigger ask that should be scoped on its own rather than silently bundled in here.",
    "Two findings worth stating plainly. First, the true max is markedly noisier and "
    "slower-converging than the region average for BOTH methods and every case -- "
    "B1xNeo-Hookean's own FEM region average falls to 0.30% by N=49 (displacement L2 "
    "is already down to 0.05% there), while its FEM true max is still 62.8% at N=3 "
    "and only reaches 26.5% at N=49 -- a real, data-grounded illustration of exactly "
    "why the advisor asked to avoid the bare pointwise maximum as the primary design "
    "QoI: even FEM's own true max, with no operator involved at all, converges far "
    "more slowly than the region-averaged statistic he asked to use instead. Second, "
    "the operator's own region-averaged stress error is consistently worse than its "
    "displacement L2 error at matched resolution (e.g. B1xMooney-Rivlin at N=13: L2 "
    "7.39% vs. region average 10.69%) -- stress is a genuinely harder, noisier target "
    "for the operator than displacement, not merely a rescaled version of the same "
    "accuracy. B2xNeo-Hookean is the extreme case: its operator's own region-averaged "
    "stress error reaches 245.59% at N=3 and stays in the 3.2%-245.6% range across "
    "the whole sweep, never settling anywhere close to FEM's own sub-1% region "
    "average at the same resolutions -- confirming that displacement accuracy alone "
    "should not be treated as a stand-in for stress accuracy.",
)
copy_elements(dst, src, elements)

# =======================================================================
# POINT 2 -- training-cost summary
# =======================================================================
append_heading(dst, "Point 2 -- training-cost summary (resolutions, samples, epochs, wall-clock, cost)")
append_quote(dst,
    "“We should add a table showing training resolutions, number of samples, "
    "epochs/steps, total wall-clock, training cost per sample/step and peak memory... "
    "I think we should not waste time on the fine resolution training here but only for "
    "a complex geometry problem where resolution might matter.” — Timon, 2026-09-18"
)
elements = slice_by_anchor_text(
    src,
    "Round-12 point 2 (training-cost summary): the advisor separately asked for a table "
    "of every training run's own resolutions, sample counts, epochs, wall-clock, and "
    "cost per sample, so the direct-N1401-vs-multi-resolution comparison above is "
    "legible in its own training-cost terms, not only through the retrained-checkpoint "
    "accuracy comparison in Table 18-R10i below. Table 18-R10n reports exactly that for "
    "all seven completed training runs (six multi-resolution retrains plus the "
    "direct-N1401 ablation just discussed), read directly from each run's own "
    "already-saved metrics_history.json -- nothing retrained for this table. Peak GPU "
    "memory during training was never instrumented for any of these runs; that gap is "
    "reported honestly as \"not measured\" rather than guessed. The advisor's own "
    "further suggestion in the same point -- a controlled re-run matching sample counts "
    "and training budget between the direct-N1401 and multi-resolution recipes, to "
    "isolate the resolution effect cleanly -- is intentionally not attempted here, per "
    "his own explicit statement that it is not worth the time for this toy problem, "
    "reserved instead for a future complex-geometry case where resolution genuinely "
    "matters. The clearest finding in the table: the direct-N1401 ablation's own cost "
    "PER SAMPLE (294.85 s) is 9-12x higher than any multi-resolution case's per-sample "
    "cost (24.36-33.31 s), since N=1401 samples are themselves far more expensive to "
    "generate -- exactly the training-budget confound the advisor's point 2 flagged, "
    "not merely a difference in total wall-clock hours.",
    "Table 18-R10n. Training-cost summary, all seven completed training runs (round-12 "
    "point 2): resolutions used, samples per resolution, total samples, final "
    "epoch/optimizer steps, wall-clock, and cost per sample/optimizer step, read from "
    "each run's own metrics_history.json. B2xArruda-Boyce has no multi-resolution "
    "retrain; its own original (fixed-selection, N=21,33) run is reported instead using "
    "the same convention, so the table is honest that this row is not a "
    "multi-resolution case.",
)
copy_elements(dst, src, elements)

# =======================================================================
# POINT 3 -- compile+TF32 for the paper, distinguish the two break-evens
# =======================================================================
append_heading(dst, "Point 3 -- use the compile+TF32 result; distinguish same-resolution from accuracy-matched")
append_quote(dst,
    "“For inference timing let's use the optimized compile+TF32 result for the "
    "paper. We then clearly distinguish the same-resolution comparison from the "
    "accuracy-matched comparison.” — Timon, 2026-09-18"
)
append_subheading(dst, "Where the compile+TF32 number itself comes from (round-10 point 3, background)")
elements = slice_by_anchor_text(
    src,
    "Point 3 asked whether inference at N=1401 admits further optimization before "
    "treating 2.29 s/sample as final. Profiling attributes about 67% of the time to "
    "batched matrix multiplications (the slice-based attention operating on roughly two "
    "million mesh nodes) and about 22% to the MLP layers, concentrated in genuine core "
    "compute at this token count rather than an obvious inefficiency. Peak GPU memory "
    "(23.16 GB) is notably higher than the GPU-native finite-element solver's own peak "
    "memory at the same N (15.6-20.5 GB), so the operator is not the more "
    "memory-frugal option here either. Table 18-R10d summarizes the optimization "
    "attempts, each correctness-checked against strict fp32 eager mode rather than "
    "assumed successful.",
    "torch.compile alone gives 1.07x, correctness-checked with a negligible output "
    "difference. Enabling TF32 matmul precision (Ampere tensor cores) gives 4.67x alone "
    "and 5.82x combined with torch.compile -- the best result found, at a real but small "
    "accuracy cost (about 3.2e-3 relative difference vs. strict fp32), checked directly "
    "rather than assumed. Re-measured separately against the retrained multi-resolution "
    "checkpoint: all four timing variants within 0.01-0.8% of the original checkpoint's "
    "own numbers (eager 2,286.9 vs. 2,292.1 ms; torch.compile 2,145.2 vs. 2,145.1 ms; "
    "eager+TF32 487.3 vs. 491.2 ms; compile+TF32 395.0 vs. 394.0 ms), again confirming "
    "checkpoint-independence.",
)
copy_elements(dst, src, elements)

append_subheading(dst, "Both break-even tables, now using the SAME compile+TF32 operator number")
elements = slice_by_anchor_text(
    src,
    "Point 5 asked for an updated break-even analysis. Using the retrained checkpoint's "
    "own coarsest-suitable finite-element mesh at N=1401 (N=11, bound by the "
    "tangent-energy norm per the crossover above) against the operator's own real "
    "inference cost there, and the real training wall-clock for the retrained checkpoint "
    "(41,881 s, about 11.6 hours, confirmed above): in its default eager fp32 deployment "
    "mode, the operator never recovers its own training cost against this baseline -- "
    "the accuracy-matched finite-element mesh is already cheaper per sample (1,625.6 ms) "
    "than the operator's default forward pass (2,292.1 ms). With the torch.compile plus "
    "TF32 optimization above (394.0 ms/sample), the operator is instead 4.13x faster per "
    "sample and recovers its training cost after 34,005 solved problems, about 3.72 "
    "GPU-hours of inference against the 11.6 GPU-hours spent training. This is the only "
    "point in the whole round-10 investigation where the operator's default, unoptimized "
    "deployment mode does not come out ahead; its economic case against a genuinely "
    "accuracy-matched baseline rests on deploying it with the optimizations found under "
    "point 3, not on its default settings.",
    "Both Arruda-Boyce cases fail at N=1401 in every run attempted so far, for a genuine "
    "and already root-caused reason distinct from the memory-cleanup issue above: "
    "torch-fem's own Newton-Raphson solve does not converge (\"did not converge in "
    "increment 8 after 10 cutbacks\"), and the real underlying cause, read directly from "
    "the exception's own cause chain rather than assumed, is a CUDA out-of-memory error "
    "inside torch-fem's own Hessian assembly for this material specifically -- "
    "Arruda-Boyce's own strain-energy density is more expensive to differentiate twice "
    "than Neo-Hookean's or Mooney-Rivlin's, and at N=1401's problem size that cost alone "
    "exhausts the GPU before torch-fem's own solve can complete. This is reported as a "
    "real finding about torch-fem's own scaling limit for this material at this "
    "resolution, not a defect in this project's own comparison code.",
)
copy_elements(dst, src, elements)

dst.save(OUT)
print('Saved', OUT)

# ---- sanity check ----
check = Document(OUT)
print('paragraphs:', len(check.paragraphs))
print('tables:', len(check.tables))
print('images:', len(check.inline_shapes))
