# ============================================================
# التحقّق من جدول 10: أي مجموعة توقيت GPU FEM يستعملها التقرير؟
# ============================================================
# البند الوحيد الي لسا مش متحقّق منه. على Drive مجموعتين توقيت
# بأرقام مختلفة قليلًا، وملاحظاتي بتقول إن جدول 10 بياخد
# gpu_fem_solver/ — بس التقرير مش بالمستودع فما انفحص.
#
# هاد بيقرا التقرير نفسه وبيقارن أرقامه بالمجموعتين، وبيقول أيهما
# يطابق. ما بده أي نسخ يدوي.
import glob
import os
import re
import json
import subprocess
import sys

R = '/content/drive/MyDrive/pfem_run'

try:
    import docx  # noqa: F401
except ImportError:
    subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'python-docx'], check=True)
from docx import Document
from docx.oxml.ns import qn

# --- find the newest report ---------------------------------------------
reports = []
for root in ('/content/drive/MyDrive',):
    reports += glob.glob(f'{root}/**/PFEM_Transolver_Report_v*.docx', recursive=True)
if not reports:
    raise SystemExit('ما لقيت ملف PFEM_Transolver_Report_v*.docx — ابعتلي مساره')


def version(p):
    m = re.search(r'_v(\d+)\.docx$', p)
    return int(m.group(1)) if m else -1


reports.sort(key=version)
path = reports[-1]
print('التقرير:', path)
print('(كل النسخ الي لقيتها:', [os.path.basename(p) for p in reports], ')\n')

# --- every table, found recursively -------------------------------------
# python-docx's document.tables misses tables nested inside other elements
# -- it missed one in this very report before -- so walk the XML instead.
doc = Document(path)
tbls = doc.element.body.findall('.//' + qn('w:tbl'))
print(f'{len(tbls)} جدول بالملف\n')


def cells(tbl):
    out = []
    for tr in tbl.findall('.//' + qn('w:tr')):
        row = []
        for tc in tr.findall('.//' + qn('w:tc')):
            row.append(''.join(t.text or '' for t in tc.findall('.//' + qn('w:t'))).strip())
        out.append(row)
    return out


# --- the two candidate timing sets --------------------------------------
CAND = {
    'gpu_fem_solver (ملاحظاتي بتقول هاي)': {
        'B1_neo_hookean': [1651.6, 477.9, 381.3, 354.6],
        'B2_neo_hookean': [1665.9, 479.0, 381.3, 354.7]},
    'gpu_fem_timing_{B1,B2} (الأقدم)': {
        'B1_neo_hookean': [1649.9, 479.2, 382.6, 354.8],
        'B2_neo_hookean': [1672.0, 477.4, 381.7, 354.5]},
}
targets = sorted({round(v, 1) for s in CAND.values() for r in s.values() for v in r})

# --- find which table carries them --------------------------------------
print('=' * 70)
print('الجداول الي فيها أرقام توقيت GPU FEM')
print('=' * 70)
for i, tbl in enumerate(tbls, 1):
    rows = cells(tbl)
    flat = ' '.join(' '.join(r) for r in rows)
    nums = {round(float(x), 1) for x in re.findall(r'\d+\.\d+', flat)}
    hits = nums & set(targets)
    if not hits:
        continue
    print(f'\n--- جدول رقم {i} بالملف --- ({len(rows)} صف)')
    for r in rows[:12]:
        print('   ', ' | '.join(c[:22] for c in r))
    if len(rows) > 12:
        print(f'    ... و{len(rows) - 12} صف كمان')
    print('    أرقام التوقيت الي فيه:', sorted(hits))
    for name, sets in CAND.items():
        matched = sorted(hits & {round(v, 1) for r_ in sets.values() for v in r_})
        print(f'      يطابق {name}: {matched if matched else "لا شيء"}')

print()
print('=' * 70)
print('الحكم')
print('=' * 70)
alltxt = ' '.join(' '.join(' '.join(r) for r in cells(t)) for t in tbls)
allnums = {round(float(x), 1) for x in re.findall(r'\d+\.\d+', alltxt)}
for name, sets in CAND.items():
    own = {round(v, 1) for r_ in sets.values() for v in r_}
    # numbers unique to this set are what actually decide it; the two sets
    # agree on some values and those cannot distinguish them
    other = set().union(*[{round(v, 1) for r_ in s.values() for v in r_}
                          for n, s in CAND.items() if n != name])
    decisive = own - other
    found = sorted(allnums & decisive)
    print(f'{name}:')
    print(f'   أرقام فارقة موجودة بالتقرير: {found if found else "ولا واحد"}')
