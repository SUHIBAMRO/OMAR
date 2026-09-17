"""Omar's explicit instruction (2026-09-17): the Work Summary he sends to
Timon should contain ONLY the round-10/11 feedback-response section --
drop the round-9 response section entirely, and drop the long separate
"Summary of what was done and what came out" narrative (which restates
much of the project's whole history in prose form, not organized around
a specific currently-active feedback round). Round-10/11's own section
is self-contained (its own R10-* table/figure numbering never collides
with anything in the removed sections, and the one "round-9" mention
left inside it is just narrative context, not a cross-reference to
content that is being deleted), so this is a pure deletion, not a
rewrite of what stays.

Deletes at the body-XML level (paragraphs AND tables/images in between,
which doc.paragraphs alone does not enumerate) so nothing from the
removed sections survives as an orphaned table or image.
"""
import os

from docx import Document
from docx.text.paragraph import Paragraph
from docx.table import Table

DELIV = '/home/user/OMAR/advisor_feedback'
SRC = os.path.join(DELIV, 'PFEM_Work_Summary_2026-09-17g.docx')
DST = os.path.join(DELIV, 'PFEM_Work_Summary_2026-09-17h.docx')

doc = Document(SRC)


def replace_paragraph_text(para, new_text):
    for r in list(para.runs):
        r._r.getparent().remove(r._r)
    para.add_run(new_text)
    return para


body = doc.element.body
items = []
for child in body.iterchildren():
    if child.tag.endswith('}p'):
        items.append(('p', Paragraph(child, doc)))
    elif child.tag.endswith('}tbl'):
        items.append(('t', Table(child, doc)))

def find_header_index(text):
    for i, (kind, obj) in enumerate(items):
        if kind == 'p' and obj.text.strip() == text:
            return i
    raise AssertionError(f'header not found: {text!r}')

i_round9 = find_header_index(
    "Response to Timon's round-9 feedback (torch-fem comparison + follow-up email)"
)
i_round10 = find_header_index(
    "Response to Timon's round-10 feedback (accuracy-matched comparison, "
    "throughput, profiling, geometry, break-even)"
)
i_narrative = find_header_index('Summary of what was done and what came out')

print(f'round-9 header at item {i_round9}, round-10/11 header at {i_round10}, '
      f'narrative header at {i_narrative}, total items {len(items)}')
assert i_round9 < i_round10 < i_narrative

# Delete round-9 section: items[i_round9 : i_round10]
for kind, obj in items[i_round9:i_round10]:
    el = obj._p if kind == 'p' else obj._tbl
    el.getparent().remove(el)

# Delete the narrative section: items[i_narrative:] (to the end of the doc)
for kind, obj in items[i_narrative:]:
    el = obj._p if kind == 'p' else obj._tbl
    el.getparent().remove(el)

# Reword the top intro sentence: it previously described a two-part
# document (a "what was done" narrative followed by point-by-point
# results); only the point-by-point round-10/11 material remains now.
intro = doc.paragraphs[2]
assert intro.text.strip() == (
    "A plain summary of what was done and what came out, followed by the "
    "completed results (tables and figures) organized against your feedback "
    "points. Full methodology and discussion are in the main report."
), f'unexpected intro text: {intro.text!r}'
replace_paragraph_text(
    intro,
    "A plain summary of the completed results (tables and figures) for "
    "Timon's round-10 and round-11 feedback, organized against his own "
    "feedback points. Full methodology and discussion are in the main "
    "report."
)

doc.save(DST)
print('Saved', DST)
