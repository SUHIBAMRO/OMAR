"""Item #12: mesh-convergence figure (Tables 1/1a/1b/2/2a/2b), built
from the numbers already published in the real Report docx -- these
six tables have no separate JSON source anywhere in the repo (checked
directly: grepping the whole codebase for Table 1's own first value
found nothing), so the report's own already-verified numbers are the
only available source, extracted directly rather than retyped by hand.

A real ordering trap was found and fixed before trusting any of this:
for this specific block of six tables, the CAPTION PARAGRAPH comes
AFTER its table, not before (the opposite convention Table 6a/20c and
others in this same document use) -- confirmed by reading the text
immediately following each of the six matching-header tables, not
assumed. Extracting by "nearest preceding caption" instead (the
convention that works everywhere else) would have silently mislabeled
every one of these six tables one slot off (e.g. attributing Table 2's
own numbers to Table 1) -- caught here by cross-checking the semantic
lead-in sentences ("B1 (Neo-Hookean), full resolution sweep", etc.)
against which caption follows which table, not by trusting either
position rule blindly.
"""
import os
import re

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from docx import Document
from docx.table import Table
from docx.text.paragraph import Paragraph
from docx.oxml.ns import qn

from plot_style import MATERIAL_COLOR, add_line_labels

DELIV = '/tmp/claude-0/-home-user/64d7c4d8-d5f0-5686-a58f-aa87abfd4ba4/scratchpad/deliverables'
REPORT = os.path.join(DELIV, 'PFEM_Transolver_Report_updated_2026-09-08.docx')
OUT = 'report_builders/figures'
os.makedirs(OUT, exist_ok=True)

TARGET_HDR = ['N', 'Nodes', 'Max |u|', 'Δ vs. prev.', 'Strain energy U', 'Δ vs. prev.']

doc = Document(REPORT)
body = doc.element.body
items = []
for child in body.iterchildren():
    if child.tag == qn('w:p'):
        items.append(('p', Paragraph(child, doc)))
    elif child.tag == qn('w:tbl'):
        items.append(('tbl', Table(child, doc)))

matches = [(i, obj) for i, (kind, obj) in enumerate(items)
           if kind == 'tbl' and [c.text for c in obj.rows[0].cells] == TARGET_HDR]
assert len(matches) == 6, f'expected exactly 6 mesh-convergence tables, found {len(matches)}'

label_for = {}
for idx, tbl in matches:
    caption = None
    for j in range(idx + 1, min(idx + 3, len(items))):
        k, o = items[j]
        if k == 'p':
            m = re.match(r'^(Table\s+\d+[a-z]?)\.', o.text.strip())
            if m:
                caption = m.group(1)
                break
    assert caption is not None, f'no caption found immediately after table at {idx}'
    label_for[caption] = tbl

CASES = [
    ('Table 1', 'B1', 'Neo-Hookean'), ('Table 1a', 'B1', 'Mooney-Rivlin'),
    ('Table 1b', 'B1', 'Arruda-Boyce'), ('Table 2', 'B2', 'Neo-Hookean'),
    ('Table 2a', 'B2', 'Mooney-Rivlin'), ('Table 2b', 'B2', 'Arruda-Boyce'),
]
assert set(label_for) == {c[0] for c in CASES}, sorted(label_for)


def pct(s):
    s = s.strip().rstrip('%')
    return float(s) if s else None


fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), dpi=200)
# Staggered vertically (index-dependent dy) since the three materials'
# finest-resolution points sit very close together, especially for B2.
LABEL_DY = {'Neo-Hookean': 10, 'Mooney-Rivlin': 20, 'Arruda-Boyce': -16}

for cap, geometry, material in CASES:
    tbl = label_for[cap]
    N, delta_energy = [], []
    for row in tbl.rows[1:]:
        cells = [c.text for c in row.cells]
        n = int(cells[0].split()[0])
        d = pct(cells[5])
        if d is not None:
            N.append(n)
            delta_energy.append(d)
    ax = axes[0] if geometry == 'B1' else axes[1]
    ax.loglog(N, delta_energy, marker='o', color=MATERIAL_COLOR[material], linewidth=1.6,
               markersize=5, label=material)
    # Value label on the finest-resolution point only (the headline
    # number) -- labelling all 7 points per line would be unreadable.
    add_line_labels(ax, [N[-1]], [delta_energy[-1]], fmt='{:.3f}%',
                     color=MATERIAL_COLOR[material], dy=LABEL_DY[material])

for ax, geometry in zip(axes, ('B1', 'B2')):
    ax.set_xlabel('N')
    ax.set_ylabel('Strain-energy change per refinement (%)')
    ax.set_title(geometry)
    ax.legend(frameon=False, fontsize=8)
    ax.grid(True, which='both', alpha=0.25)
    xlo, xhi = ax.get_xlim()
    ax.set_xlim(xlo, xhi * 1.6)

fig.suptitle('Mesh convergence: strain energy vs. resolution', fontsize=12)
fig.tight_layout(rect=[0, 0, 1, 0.93])

out_path = os.path.join(OUT, 'fig_mesh_convergence.png')
fig.savefig(out_path)
print('Saved', out_path)
