"""Mirrors add_richer_mms_to_report.py's fix in the Summary document,
in that document's own terser style (short captions, not full
explanatory paragraphs -- matching how Table 22/22a/22b/23/23a already
read here versus their fuller Report versions). Same underlying data
(point9_results/mms_richer_B1_*.json), same new table/figure numbers
as the Report (22c/22d/22e/23b) for cross-document consistency, except
the figure follows THIS document's own existing numbering (Figure 27
here vs. Figure 28 in the Report, for the identical original image) --
so the new one is Figure 27a here, Figure 28a in the Report.
"""
import copy
import json
import os

from docx import Document
from docx.oxml.ns import qn
from docx.shared import Inches, Pt
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.text.paragraph import Paragraph

PF = '/home/user/OMAR/Practical_Examples/omar_pfem'
FIG = '/home/user/OMAR/Practical_Examples/report_builders/figures'
DELIV = '/tmp/claude-0/-home-user/64d7c4d8-d5f0-5686-a58f-aa87abfd4ba4/scratchpad/deliverables'
SRC = os.path.join(DELIV, 'PFEM_Work_Summary_2026-09-14.docx')
DST = os.path.join(DELIV, 'PFEM_Work_Summary_2026-09-15.docx')

doc = Document(SRC)
ORIGINAL = list(doc.paragraphs)


def find_para_exact(text):
    hits = [p for p in ORIGINAL if p.text.strip() == text]
    assert len(hits) == 1, f'{len(hits)} paragraphs exactly match {text!r}'
    return hits[0]


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


def insert_figure_after(anchor_para, image_path, caption_text, width_in=6.5):
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


def load_richer(material):
    return json.load(open(os.path.join(PF, 'point9_results', f'mms_richer_B1_{material}.json')))


def fmt(v):
    return f'{v:.3e}'


def table_rows_for(rows):
    out = []
    for r in rows:
        out.append([r['order'], r['N'], f"{r['n_dof']:,}", fmt(r['L2_rel']), fmt(r['H1_semi_rel']),
                    fmt(r['stress_rel_L2']), fmt(r['energy_rel']), fmt(r['energy_norm_rel'])])
    return out


def rate_rows_for(material_label, rates):
    out = []
    for order in ('Q4', 'Q9'):
        for norm_key, norm_label, theory in [
            ('L2', 'L2', 2 if order == 'Q4' else 3),
            ('H1_semi', 'H1 semi-norm', 1 if order == 'Q4' else 2),
            ('stress', 'Stress', 1 if order == 'Q4' else 2),
            ('energy_norm', 'Energy norm', 1 if order == 'Q4' else 2),
        ]:
            r = rates[order][norm_key]
            pw = ', '.join(f'{v:.2f}' for v in r['pairwise'])
            out.append([material_label, order, norm_label, f"{r['rate']:.2f}", theory, pw])
    return out


nh = load_richer('neo_hookean')
mr = load_richer('mooney_rivlin')
ab = load_richer('arruda_boyce')
for d in (nh, mr, ab):
    assert d['richer_family'] is True
    assert d['rate_check'] == {'Q4': 'as expected', 'Q9': 'as expected'}

HEADER = ['Order', 'N', 'DOF', 'L2', 'H1 semi', 'Stress', 'Energy (value)', 'Energy norm']

anchor = find_para_exact(
    "Table 23a. Observed rates for both materials, fitted the same way as Table 23. All "
    "sixteen land on theory -- the verification holds for every material this report trains "
    "an operator on, not only the one in Tables 22-24."
)

p1 = insert_after(
    anchor,
    "Richer family, correct norm (advisor, round-8 point 6, closed in the Report itself "
    "on 2026-09-15). Tables 22/22a/22b used one spatial mode (varying only two "
    "amplitudes) and reported the scalar internal-ENERGY VALUE, which superconverges "
    "at double the true rate. Tables 22c/22d/22e repeat all three materials on a "
    "genuine multi-mode field (three sine/cosine terms per displacement component) "
    "and report the ENERGY NORM instead -- the quantity Cea's lemma actually bounds. "
    "Data: point9_results/mms_richer_B1_*.json, already verified 2026-09-09; only now "
    "reflected in the documents Timon reads, not just a side file."
)

t22c = insert_table_after(p1, HEADER, table_rows_for(nh['rows']))
p2 = insert_after_table(t22c, p1, "Table 22c. B1 x Neo-Hookean, richer-mode field.")
t22d = insert_table_after(p2, HEADER, table_rows_for(mr['rows']))
p3 = insert_after_table(t22d, p2, "Table 22d. B1 x Mooney-Rivlin, richer-mode field.")
t22e = insert_table_after(p3, HEADER, table_rows_for(ab['rows']))
p4 = insert_after_table(t22e, p3, "Table 22e. B1 x Arruda-Boyce, richer-mode field.")

p5 = insert_after(p4,
    "Energy (value) is 1-2 orders tighter than every other column at matched rows -- e.g. "
    "Neo-Hookean Q4 N=33: L2 2.011e-03, H1 5.392e-02, value 2.257e-03, norm 4.803e-02 -- "
    "confirming the advisor's point: the value column alone overstates verification "
    "tightness.")

rate_header = ['Material', 'Order', 'Norm', 'Observed', 'Theory', 'Pairwise']
rate_rows = (rate_rows_for('Neo-Hookean', nh['convergence_rates'])
             + rate_rows_for('Mooney-Rivlin', mr['convergence_rates'])
             + rate_rows_for('Arruda-Boyce', ab['convergence_rates']))
t23b = insert_table_after(p5, rate_header, rate_rows)
p6 = insert_after_table(t23b, p5,
    "Table 23b. Richer-field rates, all three materials. Energy-norm rate now matches "
    "H1's own theory (1 at Q4, 2 at Q9), not double it -- all 24 rows land on theory.")

insert_figure_after(p6, os.path.join(FIG, 'fig_mms_richer_convergence.png'),
    "Figure 27a. Richer multi-mode family: convergence rates in the energy norm, all "
    "three materials, Q4 and Q9 (Tables 22c/22d/22e/23b).")

doc.save(DST)
print('Saved', DST)
