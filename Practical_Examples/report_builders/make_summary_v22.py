"""v21 -> v22. Mirrors make_v48.py's fix into the summary: the
"operator/Q4, single member" ratio at N=33 in the Q4/Q9/operator
16-member-family table is corrected from 13.32x to 13.33x, matching the
very next table's (operator/Q4 by norm) already-correct 13.33x and the
source JSON's own `ratios_operator_over_Q4.L2["33"] = 13.33`.
"""
from docx import Document

SRC, DST = 'PFEM_Summary_Completed_Work.docx', 'PFEM_Summary_Completed_Work.docx'
BACKUP = 'PFEM_Summary_Completed_Work.pre_v22.docx'

doc = Document(SRC)
doc.save(BACKUP)

t = doc.tables[42]
assert [c.text for c in t.rows[0].cells] == [
    'N', 'Q4 (family)', 'Q9 (family)', 'operator (family)',
    'operator/Q4, family', 'operator/Q4, single member'], 'not the family table'
row33 = t.rows[3]
cells = row33.cells
assert cells[0].text == '33', 'not the N=33 row'
assert cells[-1].text == '13.32×', f'expected stale 13.32×, found {cells[-1].text!r}'
cells[-1].text = '13.33×'
assert cells[-1].text == '13.33×'

doc.save(DST)
print('overwrote', DST, '(backup at', BACKUP, ')')
