"""Item #12: embed the remaining 10 notebook-generated field-grid
figures (produced by Round6_Project_Figures.ipynb on Colab, fetched
from Drive) into the already-updated Report/Summary docx files (the
2026-09-09 versions that already carry Figures 17-29 from
embed_all_figures.py).

Anchor coverage differs between geometry/material combos and even
between documents, checked directly rather than assumed:
- Table 6a (FEM mesh-convergence field, B1 NH) exists in both docs.
- Table 12 / "Table 12 (revised)" (zero-shot) exists in both docs.
- OOD: Table 19 is B1 x Neo-Hookean ONLY in the Report; Table 25 is the
  one that actually covers all six geometry x material combinations
  (checked each table's own caption text, not just its header) -- so
  all six OOD field grids anchor to Table 25 in the Report. The Summary
  has NO Table 25 or Table 26 at all (checked: absent from its own
  table map) -- its OOD discussion is condensed under Table 19, which
  in the Summary already covers all six cases (different content than
  the Report's own Table 19), so that's the Summary anchor instead.
- DD-NO: Table 26 exists in the Report only. The Summary's DD-NO
  discussion is a single short paragraph with no dedicated table
  (searched for "DD-NO"/"data-driven"/"coarse-vs-fine" text directly)
  -- anchored there via its own intro sentence instead of a table.

For figures sharing one anchor (the six OOD cases; the two DD-NO
cases), list order is REVERSED here on purpose: insert_figure() always
splices new content in immediately after the anchor, so repeated calls
against the same anchor end up in reverse-of-call-order in the final
document -- verified this mechanically before relying on it, not
assumed.
"""
import os
import shutil
import sys

sys.path.insert(0, os.path.dirname(__file__))
from embed_all_figures import (
    DELIV, FIG, load_items, group_anchor, text_anchor, insert_figure,
)
from docx_table_map import build_table_map

REPORT = os.path.join(DELIV, 'PFEM_Transolver_Report_updated_2026-09-09.docx')
SUMMARY = os.path.join(DELIV, 'PFEM_Work_Summary_updated_2026-09-09.docx')

# (figure file, base caption, report anchor labels, summary anchor labels)
# summary anchor labels: a list of table labels, OR a 2-tuple
# ('text', needle) to anchor on paragraph text instead of a table.
FIGURES = [
    ('fig_B1_resolution_grid.png',
     'Mesh-convergence field grid (B1, Neo-Hookean): predicted |u| across '
     'seven resolutions, N=51-1401, one converged solution each (Table 6a).',
     ['Table 6a'], ['Table 6a']),
    ('fig_B1_zeroshot_grid.png',
     'Zero-shot resolution invariance field grid (B1, Neo-Hookean): predicted '
     '|u| at seven unseen resolutions from one trained checkpoint (Table 12).',
     ['Table 12 (revised)', 'Table 12'], ['Table 12 (revised)', 'Table 12']),
    # OOD field grids, six cases -- listed in REVERSE of desired reading
    # order (B2 AB first) since they share one anchor; see module docstring.
    ('fig_B2_arruda_boyce_ood_grid.png',
     'Out-of-distribution shift field grid (B2, Arruda-Boyce): predicted |u| '
     'under material shift, k=0-3sigma (Tables 19/25).',
     ['Table 25'], ('text', 'Loading causes no degradation anywhere')),
    ('fig_B2_mooney_rivlin_ood_grid.png',
     'Out-of-distribution shift field grid (B2, Mooney-Rivlin): predicted |u| '
     'under material shift, k=0-3sigma (Tables 19/25).',
     ['Table 25'], ('text', 'Loading causes no degradation anywhere')),
    ('fig_B2_neo_hookean_ood_grid.png',
     'Out-of-distribution shift field grid (B2, Neo-Hookean): predicted |u| '
     'under material shift, k=0-3sigma (Tables 19/25).',
     ['Table 25'], ('text', 'Loading causes no degradation anywhere')),
    ('fig_B1_arruda_boyce_ood_grid.png',
     'Out-of-distribution shift field grid (B1, Arruda-Boyce): predicted |u| '
     'under material shift, k=0-3sigma (Tables 19/25).',
     ['Table 25'], ('text', 'Loading causes no degradation anywhere')),
    ('fig_B1_mooney_rivlin_ood_grid.png',
     'Out-of-distribution shift field grid (B1, Mooney-Rivlin): predicted |u| '
     'under material shift, k=0-3sigma (Tables 19/25).',
     ['Table 25'], ('text', 'Loading causes no degradation anywhere')),
    ('fig_B1_neo_hookean_ood_grid.png',
     'Out-of-distribution shift field grid (B1, Neo-Hookean): predicted |u| '
     'under material shift, k=0-3sigma (Tables 19/25).',
     ['Table 25'], ('text', 'Loading causes no degradation anywhere')),
    # DD-NO, two cases -- reversed (fine first) so final order is coarse,
    # then fine, matching Table 26's own coarse-vs-fine framing.
    ('fig_B1_dd_no_fine_grid.png',
     'DD-NO fine-trained field grid (B1, Neo-Hookean): predicted |u| at seven '
     'unseen resolutions from a checkpoint trained on the FINE mesh (Table 26).',
     ['Table 26'], ('text', 'how the resolution of the TRAINING mesh itself')),
    ('fig_B1_dd_no_coarse_grid.png',
     'DD-NO coarse-trained field grid (B1, Neo-Hookean): predicted |u| at '
     'seven unseen resolutions from a checkpoint trained on the COARSE mesh '
     '(Table 26).',
     ['Table 26'], ('text', 'how the resolution of the TRAINING mesh itself')),
]


def next_figure_number(doc):
    import re
    nums = []
    for p in doc.paragraphs:
        m = re.match(r'^Figure (\d+)\.', p.text.strip())
        if m:
            nums.append(int(m.group(1)))
    return max(nums) + 1 if nums else 1


def resolve_anchor(items, tmap, spec_labels):
    if isinstance(spec_labels, tuple) and spec_labels[0] == 'text':
        return text_anchor(items, spec_labels[1])
    return group_anchor(items, tmap, spec_labels)


PLACEHOLDER = 'Figure ##N##.'


def process(docx_in, docx_out, is_summary):
    doc, tmap = build_table_map(docx_in)
    items = load_items(doc)
    start_num = next_figure_number(doc)

    resolved = []
    for fname, base_caption, report_labels, summary_labels in FIGURES:
        labels = summary_labels if is_summary else report_labels
        anchor_elem, idx = resolve_anchor(items, tmap, labels)
        resolved.append([fname, base_caption, anchor_elem, idx])

    # Stable sort: ties (shared anchors) keep their FIGURES list order,
    # which the reverse-order trick above relies on for PLACEMENT.
    resolved.sort(key=lambda r: r[3])

    for fname, base_caption, anchor_elem, idx in resolved:
        caption_text = f'{PLACEHOLDER} {base_caption}'
        img_path = os.path.join(FIG, fname)
        assert os.path.isfile(img_path), img_path
        insert_figure(anchor_elem, doc, img_path, caption_text)
        print(f'  inserted {fname}  (anchor idx {idx})')

    # NUMBERING is resolved separately, in a second pass over the
    # document's own final paragraph order -- NOT the insertion-call
    # order above. Repeated insertions against one shared anchor land
    # in REVERSE of call order (each new pair splices in right after
    # the anchor, pushing the previous pair further down), so for the
    # six OOD figures / two DD-NO figures, call order != reading order.
    # Assigning numbers by call order there produced "Figure 36" right
    # before "Figure 31" in one draft of this script -- caught by
    # reading the saved file back before trusting it, fixed by
    # deferring numbering to this fresh, physically-ordered pass.
    n = start_num
    for p in doc.paragraphs:
        if p.text.strip().startswith(PLACEHOLDER):
            for run in p.runs:
                if PLACEHOLDER in run.text:
                    run.text = run.text.replace(PLACEHOLDER, f'Figure {n}.')
                    break
            n += 1

    doc.save(docx_out)
    print(f'Saved {docx_out} ({len(resolved)} figures inserted, numbered {start_num}-{n - 1})')


if __name__ == '__main__':
    print('=== Report ===')
    report_tmp = REPORT.replace('.docx', 'b.docx')
    process(REPORT, report_tmp, is_summary=False)

    print('\n=== Summary ===')
    summary_tmp = SUMMARY.replace('.docx', 'b.docx')
    process(SUMMARY, summary_tmp, is_summary=True)
