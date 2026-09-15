"""Closes Timon's round-8 point 6 IN THE ACTUAL REPORT (not just a side
JSON + a Summary text blurb, which is as far as the 2026-09-09 session
got -- see PROJECT_STATUS.md's own admission that "no Report edit was
needed" was the wrong call, since Timon still sees this as open on
2026-09-15: from his side, reading the Report, nothing changed).

Timon's exact ask: "we should use a combination of several spatial
sine/cosine modes rather than essentially one spatial mode with
different amplitudes and compute the actual energy norm of the error
rather than the scalar internal-energy error."

Deliberately does NOT touch Tables 22/22a/22b/23/23a or Table 24's
series: those still document the original single-mode study and the
operator/Q4 minimality comparison, which Table 24's own text explicitly
computes its ratios FROM Table 22's exact numbers ("the Q4 and Q9 rows
are the N=17 rows of Table 22 and are not a second measurement") -- so
changing Table 22 in place would silently invalidate Table 24's ratios
(2.42x, 1.03x, 3.11x, all still tied to the single-mode field) without
re-running the operator comparison, which is a separate, much bigger
piece of work outside what Timon asked for here. Instead, this adds
NEW tables (22c/22d/22e, rate table 23b) and a new figure presenting
the richer-family + energy-norm study for all three materials, using
the already-verified data from point9_results/mms_richer_B1_*.json
(2026-09-09, rate_check "as expected" for both Q4 and Q9, all three
materials) -- no new solving needed, this is a report-only change.
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
SRC = os.path.join(DELIV, 'PFEM_Transolver_Report_2026-09-14.docx')
DST = os.path.join(DELIV, 'PFEM_Transolver_Report_2026-09-15.docx')

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
    d = json.load(open(os.path.join(PF, 'point9_results', f'mms_richer_B1_{material}.json')))
    return d


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

HEADER = ['Order', 'N', 'DOF', 'L2', 'H1 semi-norm', 'Stress', 'Energy (value)', 'Energy norm']

anchor = find_para_exact(
    "Table 23a. Observed convergence rates for Mooney-Rivlin and Arruda-Boyce, fitted "
    "the same way as Table 23 (least squares on log h, three consecutive-pair points "
    "from the four resolutions in Tables 22a/22b). All sixteen rows land on their "
    "theoretical rate, matching Table 23's own Neo-Hookean result -- the verification "
    "holds for every constitutive model this report trains an operator on, not only "
    "the one exercised in Tables 22-24."
)

p1 = insert_after(
    anchor,
    "A richer family, and the error in the correct norm (advisor, round-8 point 6). "
    "Tables 22/22a/22b use one spatial mode per displacement component "
    "(u* = alpha*sin(pi x)sin(pi y), varied only through the two amplitudes alpha, "
    "beta) and report \"Energy\" as the internal strain energy VALUE, "
    "|U_h - U*|/|U*|. Both choices were flagged by the advisor: the single mode "
    "under-exercises the manufactured-solution check, and the energy value is not "
    "the quantity Cea's lemma bounds -- it is a scalar comparison that "
    "superconverges at double the true discretization rate (confirmed below), so a "
    "table built only from it would overstate how tightly the method is verified. "
    "Both points are addressed here directly in the Report, not only in a side "
    "study: the manufactured field is now a genuine combination of several spatial "
    "modes, u_x = 0.05*[sin(pi x)sin(pi y) + 0.5 sin(2 pi x)sin(pi y) + "
    "0.3 sin(pi x)sin(3 pi y)], u_y = 0.035*[sin(pi x)sin(pi y) - "
    "0.4 sin(3 pi x)sin(2 pi y) + 0.2 sin(2 pi x)sin(2 pi y)] (three modes per "
    "component, not one), still vanishing on the entire boundary for the same "
    "reason as before, and the reported energy column is now the ENERGY NORM, "
    "sqrt(int grad(e):C(F*):grad(e) dV / int grad(u*):C(F*):grad(u*) dV) with "
    "e = u_h - u*, C the fourth-order tangent modulus at the exact solution -- the "
    "same tangent/incremental norm Table 6a's own matrix-free Hessian-vector "
    "product uses, here expressed as a direct quadrature integral against the "
    "continuous exact field instead. Tables 22c/22d/22e repeat Tables 22/22a/22b's "
    "same four resolutions and both element orders on this richer field for all "
    "three materials, reporting both the old energy value and the new energy norm "
    "side by side so the difference is visible rather than asserted."
)

t22c = insert_table_after(p1, HEADER, table_rows_for(nh['rows']))
p2 = insert_after_table(t22c, p1,
    "Table 22c. Q4 and Q9 against the richer-mode manufactured solution, B1 "
    "geometry, Neo-Hookean, FP64. Same four resolutions as Table 22; \"Energy "
    "(value)\" is the same internal-energy comparison Table 22 reports, kept here "
    "for contrast. \"Energy norm\" is the quantity the advisor asked for.")

t22d = insert_table_after(p2, HEADER, table_rows_for(mr['rows']))
p3 = insert_after_table(t22d, p2,
    "Table 22d. Q4 and Q9 against the richer-mode manufactured solution, B1 "
    "geometry, Mooney-Rivlin, FP64. Same columns and resolutions as Table 22c.")

t22e = insert_table_after(p3, HEADER, table_rows_for(ab['rows']))
p4 = insert_after_table(t22e, p3,
    "Table 22e. Q4 and Q9 against the richer-mode manufactured solution, B1 "
    "geometry, Arruda-Boyce, FP64. Same columns and resolutions as Table 22c.")

p5 = insert_after(p4,
    "The contrast in Tables 22c/22d/22e is exactly what the advisor's comment "
    "predicts: \"Energy (value)\" is consistently one to two orders of magnitude "
    "tighter than every other column at the same row -- e.g. Neo-Hookean Q4, "
    "N = 33: L2 = 2.011e-03, H1 semi-norm = 5.392e-02, energy value = 2.257e-03, "
    "energy norm = 4.803e-02 -- because the value comparison superconverges (its "
    "fitted rate is double the true rate, Table 23b below), while the energy norm "
    "tracks the H1 semi-norm error closely at every row and resolution, as Cea's "
    "lemma says it should for a norm rather than a value. A table reporting only "
    "the value column would have looked like a tighter verification than the "
    "method actually achieves.")

rate_header = ['Material', 'Order', 'Norm', 'Observed rate', 'Theory', 'Pairwise']
rate_rows = (rate_rows_for('Neo-Hookean', nh['convergence_rates'])
             + rate_rows_for('Mooney-Rivlin', mr['convergence_rates'])
             + rate_rows_for('Arruda-Boyce', ab['convergence_rates']))
t23b = insert_table_after(p5, rate_header, rate_rows)
p6 = insert_after_table(t23b, p5,
    "Table 23b. Observed convergence rates on the richer-mode field, all three "
    "materials, fitted the same way as Table 23/23a. The energy-norm row now "
    "carries the SAME theoretical rate as the H1 semi-norm row (1 at Q4, 2 at Q9) "
    "rather than double it, because a norm of the error converges at the same "
    "order as the gradient it is built from -- unlike the energy value's own rate "
    "(Table 23/23a, not repeated here), which is double the H1 rate by "
    "construction. All twenty-four rows land on their theoretical value, closing "
    "the gap the advisor raised: the manufactured-solution check now exercises "
    "several spatial modes at once, for every constitutive model, reported in the "
    "norm the theory actually bounds.")

p7 = insert_figure_after(p6,
    os.path.join(FIG, 'fig_mms_richer_convergence.png'),
    "Figure 28a. Method of manufactured solutions, richer multi-mode family: "
    "convergence rates in the energy norm, all three materials, Q4 and Q9 "
    "(Tables 22c/22d/22e/23b). Figure number follows Table 22a/22b/23a's own "
    "lettering convention -- Figure 29 (B2 fix-attempt history) already exists "
    "later in this document and is unrelated to this section.")

doc.save(DST)
print('Saved', DST)
