"""Omar is about to send the Summary/Report to Timon and asked for an
explicit accuracy check first: does the Summary's own round-10/11
section intro list every one of round-11's four points, and does each
point have an explicit, findable label in the body (not just content
that happens to answer it)?

Audit result: points 2 and 4 were already explicitly labeled
"Round-11 point N" in the body; points 1 and 3 were only answered via
generic phrasing ("the advisor's round-11 feedback asked...", "exactly
as the advisor asked") with no explicit point number, and the section's
own intro paragraph (post-round-9) never mentioned round-11 at all --
it still read as if round-10's own point 4 (complex geometry) was the
single open item, with no acknowledgement that round-11 points 1-3 have
since been asked and finished. This is a real clarity gap worth fixing
before sending, not a content gap (the numbers themselves were already
correct) -- Timon should be able to see at a glance that all of round-11
points 1-3 are done and only point 4 remains, exactly as his own most
recent email already treats it.

Fixes, all in place (no numbers changed, no new results, no case for
Omar to double-check the underlying data -- only the labeling around
already-published data):
  1. New paragraph after the round-10/11 section's own intro, explicitly
     listing round-11's four points and their status.
  2. Round-11 point 3 (protocol) now explicitly labeled.
  3. Round-11 point 1 (both break-evens) now explicitly labeled.
  4. Round-11 point 2 is in two parts (extend to all cases; clarify
     which quantity binds each case in one table) -- both parts now
     explicitly labeled as such, instead of only the second part
     carrying the "Round-11 point 2" tag.
"""
import os

from docx import Document
from docx.text.paragraph import Paragraph
import copy

DELIV = '/home/user/OMAR/advisor_feedback'
SRC = os.path.join(DELIV, 'PFEM_Work_Summary_2026-09-17e.docx')
DST = os.path.join(DELIV, 'PFEM_Work_Summary_2026-09-17f.docx')

doc = Document(SRC)
ORIGINAL = list(doc.paragraphs)


def find_para_exact(text):
    hits = [p for p in ORIGINAL if p.text.strip() == text]
    assert len(hits) == 1, f'{len(hits)} paragraphs exactly match {text!r}'
    return hits[0]


def replace_paragraph_text(para, new_text):
    for r in list(para.runs):
        r._r.getparent().remove(r._r)
    para.add_run(new_text)
    return para


def insert_after(anchor, text):
    new_p_el = copy.deepcopy(anchor._p)
    anchor._p.addnext(new_p_el)
    np = Paragraph(new_p_el, anchor._parent)
    for r in list(np.runs):
        r._r.getparent().remove(r._r)
    np.add_run(text)
    return np


# 1) New paragraph right after the section's own intro (para 46), recapping
# round-11's own four points and their status.
intro = find_para_exact(
    "New since the round-9 reply already sent: (1) a checkpoint-loading bug "
    "found and fixed, which had been silently corrupting every round-10 accuracy "
    "result with a completely different (data-driven baseline) model; (2) the "
    "operator's own accuracy at N=1401 checked directly against real FEM ground "
    "truth for the first time, and the coarsest FEM mesh that matches it; (3) a "
    "multi-resolution retraining that reduced the error at N=1401 from 44.65% to "
    "5.85% and inverted the peak-stress crossover finding; (4) batch-size/"
    "throughput and profiling/torch.compile/TF32 re-verified against both the "
    "corrected and the retrained checkpoint, confirming both are "
    "checkpoint-independent; (5) a real accuracy-matched break-even. One "
    "question remains open and blocks further work: point 4's complex-geometry "
    "design (ring with a local notch), already verified with real "
    "mesh-convergence evidence, is awaiting Timon's confirmation before any "
    "training time is committed to it."
)
insert_after(
    intro,
    "Timon's round-11 feedback (received after the round-10 reply above) asked "
    "four further things, all now finished except the last: (1) keep BOTH "
    "break-even comparisons, accuracy-matched and resolution-matched -- done, "
    "Tables R10-4 and R10-4'; (2) extend the N=1401 accuracy/QoI/break-even "
    "analysis to all six geometry-material cases, and clarify in one table "
    "which quantity actually defines each case's own coarsest-suitable FEM mesh "
    "-- done, Tables R10-6 and R10-7; (3) describe the multi-resolution "
    "training protocol in full detail -- done, in the retrain discussion below; "
    "(4) train a B1xNeo-Hookean checkpoint directly at N=1401 as an ablation "
    "against the zero-shot multi-resolution result -- done, Table R10-5. Point "
    "4 (the complex-geometry example) remains the one open item, exactly as "
    "Timon's own instruction was to discuss it separately once points 1-3 above "
    "were finished."
)

# 2) Round-11 point 3 (protocol) -- explicit label.
p_protocol = find_para_exact(
    "The same fix (retrain on N=21,33,101,201 instead of N=21,33 only) was "
    "since extended to every other case with the same degradation signature. "
    "Protocol, exactly as the advisor asked: weighting is equal by construction "
    "(400 train / 100 val samples per resolution, identical count at all four); "
    "each batch is homogeneous in one resolution, but the full batch list "
    "spanning all four is shuffled together so different resolutions interleave "
    "randomly through the epoch; the loss carries no explicit cross-resolution "
    "weight. N=101/201 were added because they were exactly where the "
    "degradation sweep had already measured real error (15.0%/22.3%) before any "
    "retraining ran. Two methods changes made the later retrains cheaper: TF32 "
    "training (verified safe, 2.08x at N=201, the resolution it actually "
    "helps), and a real bug found and fixed this session -- the GPU fast "
    "data-generation path had only ever been wired up for B1, so B2's own "
    "--fast_solver/--nsteps did nothing at all until now; every earlier B2 job "
    "silently ran on the slow CPU solver regardless of its own launch command."
)
replace_paragraph_text(
    p_protocol,
    "The same fix (retrain on N=21,33,101,201 instead of N=21,33 only) was "
    "since extended to every other case with the same degradation signature. "
    "Round-11 point 3 (multi-resolution training protocol, in full detail, as "
    "requested): weighting is equal by construction (400 train / 100 val "
    "samples per resolution, identical count at all four); each batch is "
    "homogeneous in one resolution, but the full batch list spanning all four "
    "is shuffled together so different resolutions interleave randomly through "
    "the epoch; the loss carries no explicit cross-resolution weight. N=101/201 "
    "were added because they were exactly where the degradation sweep had "
    "already measured real error (15.0%/22.3%) before any retraining ran. Two "
    "methods changes made the later retrains cheaper: TF32 training (verified "
    "safe, 2.08x at N=201, the resolution it actually helps), and a real bug "
    "found and fixed this session -- the GPU fast data-generation path had only "
    "ever been wired up for B1, so B2's own --fast_solver/--nsteps did nothing "
    "at all until now; every earlier B2 job silently ran on the slow CPU solver "
    "regardless of its own launch command."
)

# 3) Round-11 point 1 (both break-evens) -- explicit label.
p_breakeven = find_para_exact(
    "The advisor's round-11 feedback asked to keep a SECOND break-even "
    "comparison too: both methods at the SAME resolution, N=1401, rather than "
    "each at its own accuracy-matched mesh -- expected, correctly, to look "
    "much more favourable to the operator. Table R10-4', all six cases, "
    "operator's own default eager fp32 forward pass at N=1401 vs. torch-fem's "
    "own real GPU solve time at the same N."
)
replace_paragraph_text(
    p_breakeven,
    "Round-11 point 1 (keep BOTH break-even comparisons): the accuracy-matched "
    "one above (Table R10-4) is one half; the advisor asked to also keep a "
    "SECOND comparison, both methods at the SAME resolution N=1401 rather than "
    "each at its own accuracy-matched mesh -- expected, correctly, to look much "
    "more favourable to the operator. Table R10-4', all six cases, operator's "
    "own default eager fp32 forward pass at N=1401 vs. torch-fem's own real GPU "
    "solve time at the same N."
)

# 4) Round-11 point 2, part A (extend QoI to all cases) -- explicit label.
p_peakstress = find_para_exact(
    "Task #22's peak-stress QoI, the five cases beyond B1xNeo-Hookean (Table "
    "R10-6). Caveat: B1xMooney-Rivlin, B1xArruda-Boyce and B2xMooney-Rivlin "
    "here reflect their ORIGINAL pre-retrain checkpoints (their disp_rel_L2 "
    "matches the \"OLD\" column in Table R10-2 exactly) -- the peak-stress "
    "metric has not been recomputed against this week's new retrained "
    "checkpoints for those three. B2xNeo-Hookean and B2xArruda-Boyce have only "
    "one checkpoint each, no caveat."
)
replace_paragraph_text(
    p_peakstress,
    "Round-11 point 2, part A (extend the N=1401 QoI analysis to all remaining "
    "cases): task #22's peak-stress QoI, the five cases beyond B1xNeo-Hookean "
    "(Table R10-6). Caveat: B1xMooney-Rivlin, B1xArruda-Boyce and "
    "B2xMooney-Rivlin here reflect their ORIGINAL pre-retrain checkpoints "
    "(their disp_rel_L2 matches the \"OLD\" column in Table R10-2 exactly) -- "
    "the peak-stress metric has not been recomputed against this week's new "
    "retrained checkpoints for those three. B2xNeo-Hookean and B2xArruda-Boyce "
    "have only one checkpoint each, no caveat."
)

# 5) Round-11 point 2, part B (clarify binding quantity in one table) --
# already labeled "Round-11 point 2", refine to "part B" for consistency
# with part A above.
p_crossover = find_para_exact(
    "Round-11 point 2: does the same QoI determine the coarsest-suitable-FEM "
    "crossover every time? Already answered for B1xNeo-Hookean (N=11, bound by "
    "tangent energy) -- repeated here for the other five cases (Table R10-7). "
    "Checkpoint caveat as above: B1xMooney-Rivlin, B1xArruda-Boyce, "
    "B2xMooney-Rivlin and B2xNeo-Hookean reflect their ORIGINAL pre-retrain "
    "checkpoints here (task #22 predates all four retrains)."
)
replace_paragraph_text(
    p_crossover,
    "Round-11 point 2, part B (clarify, in one table, which quantity actually "
    "defines each case's own coarsest-suitable FEM mesh): does the same QoI "
    "determine the crossover every time? Already answered for B1xNeo-Hookean "
    "(N=11, bound by tangent energy) -- repeated here for the other five cases "
    "(Table R10-7). Checkpoint caveat as above: B1xMooney-Rivlin, "
    "B1xArruda-Boyce, B2xMooney-Rivlin and B2xNeo-Hookean reflect their "
    "ORIGINAL pre-retrain checkpoints here (task #22 predates all four "
    "retrains)."
)

doc.save(DST)
print('Saved', DST)
