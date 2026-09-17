"""Closes out the last open multi-resolution-retrain case: B2 x
Neo-Hookean finished training on real GPU (2026-09-17). Replaces the
"still mid-training" placeholder paragraph with the real result,
following the exact same structure already used for B1xMooney-Rivlin/
B1xArruda-Boyce/B2xMooney-Rivlin (Tables 18-R10f/g/h): an intro
paragraph, a full 16-resolution old-vs-new table (new letter, 18-R10l,
since a/b/c/d/e/f/g/h/i/j/k are all already taken), and a caption.
Also updates the already-existing cross-case summary (Table 18-R10i)
to add this 5th row, fixes its own caption text (no longer needs to
say B2xNeo-Hookean is omitted), and replaces Round-10 Figure D's
embedded image with the updated 5-case version (Omar reviewed and
approved the draft figure before this script ran).

Every number transcribed verbatim from the real GPU log pasted into the
working conversation 2026-09-17 (both the OLD-checkpoint sweep, run
#52, and the NEW-checkpoint sweep, run #3 of
zeroshot_B2_neo_hookean_multires/run_manifest.json). Nothing here is
estimated or interpolated.
"""
import copy
import os

from docx import Document
from docx.oxml.ns import qn
from docx.shared import Inches, Pt
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.text.paragraph import Paragraph
from docx.table import Table

FIG = '/home/user/OMAR/Practical_Examples/report_builders/figures'
DELIV = '/home/user/OMAR/advisor_feedback'
SRC = os.path.join(DELIV, 'PFEM_Transolver_Report_2026-09-17c.docx')
DST = os.path.join(DELIV, 'PFEM_Transolver_Report_2026-09-17d.docx')

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


def insert_table_after(anchor_para, header, rows):
    tbl = doc.add_table(rows=1 + len(rows), cols=len(header))
    tbl.style = doc.tables[0].style
    pr = doc.tables[0]._tbl.find(qn('w:tblPr'))
    if pr is not None:
        old = tbl._tbl.find(qn('w:tblPr'))
        if old is not None:
            tbl._tbl.remove(old)
        tbl._tbl.insert(0, copy.deepcopy(pr))
    for j, h in enumerate(header):
        c = tbl.cell(0, j)
        c.text = ''
        r = c.paragraphs[0].add_run(h)
        r.bold = True
    for i, row in enumerate(rows, start=1):
        for j, v in enumerate(row):
            tbl.cell(i, j).text = str(v)
    tbl._tbl.getparent().remove(tbl._tbl)
    anchor_para._p.addnext(tbl._tbl)
    return tbl


def insert_after_table(table, style_para, text):
    new_p_el = copy.deepcopy(style_para._p)
    table._tbl.addnext(new_p_el)
    np = Paragraph(new_p_el, style_para._parent)
    for r in list(np.runs):
        r._r.getparent().remove(r._r)
    np.add_run(text)
    return np


def insert_figure_after(anchor_para, image_path, caption_text, width_in=6.0):
    img_p = doc.add_paragraph()
    img_p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    img_p.add_run().add_picture(image_path, width=Inches(width_in))
    img_elem = img_p._p
    img_elem.getparent().remove(img_elem)

    cap_p = doc.add_paragraph()
    cap_run = cap_p.add_run(caption_text)
    cap_run.italic = True
    cap_run.font.size = Pt(9)
    cap_elem = cap_p._p
    cap_elem.getparent().remove(cap_elem)

    anchor_para._p.addnext(cap_elem)
    anchor_para._p.addnext(img_elem)
    return Paragraph(cap_elem, anchor_para._parent)


def add_table_row(table, values):
    from docx.oxml.ns import qn as _qn
    new_tr = copy.deepcopy(table.rows[-1]._tr)
    table._tbl.append(new_tr)
    new_row = table.rows[-1]
    for j, v in enumerate(values):
        cell = new_row.cells[j]
        for p in cell.paragraphs:
            for r in list(p.runs):
                r._r.getparent().remove(r._r)
        cell.paragraphs[0].add_run(str(values[j]))
    return new_row


# ------------------------------------------------------------------
# 1) Replace the "still mid-training" placeholder with the real result.
# ------------------------------------------------------------------
placeholder = find_para_exact(
    "One case remains open as of this writing and is not included in the tables "
    "above: B2 x Neo-Hookean's own multi-resolution retrain was still mid-training "
    "(most recently observed at epoch 575 of a 2000-epoch cap, validation error "
    "still improving, no sign of a plateau yet). It will be added to this section "
    "once complete, rather than reported early from an unfinished run."
)
replace_paragraph_text(
    placeholder,
    "B2 x Neo-Hookean: Table 18-R10l, same protocol, the fourth and last "
    "(geometry, material) case carried through this multi-resolution retrain "
    "(B2 x Arruda-Boyce has no retrain planned). Both accuracy sweeps ran at "
    "essentially identical wall-clock (about 1h5m each, old and new checkpoint), "
    "and every one of the 32 solves (16 resolutions, both checkpoints) converged "
    "cleanly, relative residual at or below about 3.7e-11 throughout. The same "
    "tradeoff shape holds a fourth time: worse than the original checkpoint across "
    "N=13-33 (its own original training range), decisively better everywhere from "
    "N=37 upward. Unlike B2 x Mooney-Rivlin's near-flat new-checkpoint curve, here "
    "the new checkpoint's own error is lowest right where it was explicitly trained "
    "(12.98% at N=101, its best point across the whole sweep) and rises again "
    "beyond that trained range, to about 36-40% from N=401 through N=1401 -- still "
    "substantially better than the original checkpoint's own 46.4-46.7% across that "
    "same range, but not as resolution-invariant as the Mooney-Rivlin case. At the "
    "target resolution, N=1401: 46.37% -> 36.05%, a relative reduction of about "
    "22.3%, the smallest proportional improvement of the five cases measured so far "
    "(the other four range from about 45% to 87%)."
)

HEADER = ['N', 'OLD (N=21,33 only)', 'NEW (N=21,33,101,201)', 'Better?']
t_nh = insert_table_after(placeholder, HEADER, [
    ['13', '10.67%', '50.78%', 'no'],
    ['17', '5.62%', '16.25%', 'no'],
    ['21', '1.23%', '5.11%', 'no'],
    ['25', '6.37%', '13.34%', 'no'],
    ['29', '2.96%', '3.80%', 'no'],
    ['33', '2.33%', '4.65%', 'no'],
    ['37', '8.37%', '7.56%', 'YES'],
    ['41', '15.52%', '9.19%', 'YES'],
    ['45', '22.11%', '11.94%', 'YES'],
    ['49', '27.87%', '17.94%', 'YES'],
    ['101', '48.53%', '12.98%', 'YES'],
    ['201', '47.75%', '15.98%', 'YES'],
    ['401', '46.67%', '39.95%', 'YES'],
    ['701', '46.51%', '37.98%', 'YES'],
    ['1001', '46.44%', '36.50%', 'YES'],
    ['1401', '46.37%', '36.05%', 'YES -- the target resolution'],
])

insert_after_table(
    t_nh, placeholder,
    "Table 18-R10l. disp_rel_L2 vs. real finite-element ground truth, B2 x "
    "Neo-Hookean, original vs. retrained checkpoint, all 16 resolutions reported "
    "individually."
)

# ------------------------------------------------------------------
# 2) Update the cross-case summary table (Table 18-R10i) with the 5th row,
#    and fix its own caption (no longer needs to exclude B2xNeo-Hookean).
# ------------------------------------------------------------------
summary_caption_old = find_para_exact(
    'Table 18-R10i. Summary across every completed multi-resolution retrain case: '
    'disp_rel_L2 at the target resolution N=1401, old vs. new checkpoint. '
    '"Relative reduction" is (old - new) / old. B2 x Neo-Hookean and the '
    'direct-N1401 ablation are omitted here for the reason given just above, not '
    'because they were measured and found unfavourable.'
)
replace_paragraph_text(
    summary_caption_old,
    'Table 18-R10i. Summary across every completed multi-resolution retrain case: '
    'disp_rel_L2 at the target resolution N=1401, old vs. new checkpoint. '
    '"Relative reduction" is (old - new) / old. The direct-N1401 ablation is a '
    'different comparison (direct training vs. this same zero-shot checkpoint, not '
    'old-vs-new) and is reported separately in Table 18-R10j above.'
)

# Find the actual summary table (5 rows after the edit, 4 before) -- locate it
# by its own header row, which is unique in this document.
summary_table = None
for tbl in doc.tables:
    if [c.text for c in tbl.rows[0].cells] == [
            'Case', 'Old checkpoint (N=1401)', 'New checkpoint (N=1401)', 'Relative reduction']:
        summary_table = tbl
        break
assert summary_table is not None, 'could not find the cross-case summary table'
assert len(summary_table.rows) == 5, f'expected 4 data rows before edit, got {len(summary_table.rows) - 1}'
add_table_row(summary_table, ['B2 x Neo-Hookean', '46.37%', '36.05%', '~22.3%'])

# ------------------------------------------------------------------
# 3) Replace Round-10 Figure D's embedded image with the updated 5-case
#    version (Omar reviewed and approved the draft before this ran).
#    Caption text is unchanged (still accurate), so only the image
#    paragraph is removed and replaced.
# ------------------------------------------------------------------
fig_caption = find_para_exact(
    "Round-10 Figure D. Multi-resolution retrain fix, old vs. new checkpoint "
    "disp_rel_L2 at N=1401, every case completed as of this writing (Table "
    "18-R10i). Figure left without a main-sequence number for the same reason as "
    "Round-10 Figures A/B/C: it is a direct extension of that same subsection, not "
    "a new numbered section of its own."
)
# The image paragraph immediately precedes this caption in document order.
old_img_p = fig_caption._p.getprevious()
assert old_img_p is not None and old_img_p.findall('.//' + qn('w:drawing')), \
    'expected an image paragraph immediately before Round-10 Figure D\'s caption'
fig_caption_text = fig_caption.text
old_img_p.getparent().remove(old_img_p)
fig_caption._p.getparent().remove(fig_caption._p)

# Re-find the summary table's caption paragraph (Table 18-R10i's own caption,
# just edited above) to anchor the new figure after it, same position as before.
anchor_for_fig = find_para_exact(
    'Table 18-R10i. Summary across every completed multi-resolution retrain case: '
    'disp_rel_L2 at the target resolution N=1401, old vs. new checkpoint. '
    '"Relative reduction" is (old - new) / old. The direct-N1401 ablation is a '
    'different comparison (direct training vs. this same zero-shot checkpoint, not '
    'old-vs-new) and is reported separately in Table 18-R10j above.'
)
insert_figure_after(anchor_for_fig, os.path.join(FIG, 'fig_multires_retrain_summary.png'),
                     fig_caption_text)

doc.save(DST)
print('Saved', DST)
