"""Reconstructs round12_final_accuracy_cauchy_summary.json's own
B1_neo_hookean rows directly from the REAL, already-published Table
18-R10p in advisor_feedback/PFEM_Transolver_Report_2026-10-05.docx --
the original scratchpad JSON this script was built from is lost (only
ever existed in a now-gone past session's /tmp), but this same data
was written into the report as a literal table and has survived every
revision since, verified by reading it directly via python-docx rather
than retyped by hand from the rendered page.
"""
import json
import os

from docx import Document
from docx.table import Table
from docx.text.paragraph import Paragraph

doc = Document('/home/user/OMAR/advisor_feedback/PFEM_Transolver_Report_2026-10-05.docx')
body = doc.element.body

rows_raw = None
found_caption = False
for child in body.iterchildren():
    if child.tag.endswith('}p'):
        p = Paragraph(child, doc)
        if 'Table 18-R10p' in p.text:
            found_caption = True
            continue
    if found_caption and child.tag.endswith('}tbl'):
        t = Table(child, doc)
        rows_raw = [[c.text for c in row.cells] for row in t.rows]
        break

assert rows_raw is not None, "Table 18-R10p not found"
header = rows_raw[0]
assert header == ['N', 'FEM avg', 'Op. avg', 'FEM p99', 'Op. p99', 'FEM max', 'Op. max'], header

def pct_to_frac(s):
    return float(s.rstrip('%')) / 100.0

rows = []
for r in rows_raw[1:]:
    N, fem_avg, op_avg, fem_p99, op_p99, fem_max, op_max = r
    rows.append({
        'N': int(N),
        'fem_cauchy_avg_rel_err': pct_to_frac(fem_avg),
        'no_cauchy_avg_rel_err': pct_to_frac(op_avg),
        'fem_cauchy_p99_rel_err': pct_to_frac(fem_p99),
        'no_cauchy_p99_rel_err': pct_to_frac(op_p99),
        'fem_cauchy_max_rel_err': pct_to_frac(fem_max),
        'no_cauchy_max_rel_err': pct_to_frac(op_max),
    })

print(f'Reconstructed {len(rows)} rows, N={[r["N"] for r in rows]}')

out = {'cases': {'B1_neo_hookean': rows}}

OUT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'round12_final_accuracy_cauchy_summary_RECONSTRUCTED.json')
os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
with open(OUT_PATH, 'w') as f:
    json.dump(out, f, indent=2)
print('Saved:', OUT_PATH)
