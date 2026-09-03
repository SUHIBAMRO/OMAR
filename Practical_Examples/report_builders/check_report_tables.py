"""Safety-net checker for the report/summary .docx files: lists every
table caption found ("Table N. ..."), flags any table NUMBER used for
more than one caption, and reports how many times each number is
referenced elsewhere in the prose (not the caption itself) -- so a
renumbering decision can be made by seeing which usage is the one
readers actually follow, not by guessing.

Exists because the report was found (2026-09-03 audit) to have Table 3,
4 and 5 each captioning two different tables -- a genuine ambiguity a
reader has no way to resolve from the number alone. Run this after any
change that adds, removes, or renumbers a table, on BOTH documents,
before treating either as final.

Usage:
    python3 check_report_tables.py <path-to-docx> [<path-to-docx> ...]

Exit code is 0 if no duplicate numbers are found in any file, 1
otherwise -- safe to use in a pre-flight check before sending a file
anywhere.
"""
import re
import sys
from collections import defaultdict

from docx import Document

# Matches "Table 5. ..." and also "Table 6 (revised). ..." -- a caption
# variant this checker's first version missed entirely, which is exactly
# why it exists: it must not silently trust its own first cut at parsing
# either.
CAPTION_RE = re.compile(r'^Table\s+(\d+[a-z]?)(?:\s*\([^)]*\))?\.\s*(.*)$')
# a reference elsewhere in the text: "Table 5", "Table 5,", "Table 5)",
# "Table 5's", etc. -- NOT immediately followed by another digit or letter
# that would make it a different number (so "Table 5" doesn't match inside
# "Table 51").
REF_RE_TEMPLATE = r'Table\s+{num}\b(?![a-zA-Z0-9])'


def check_file(path):
    print(f'\n{"=" * 70}\n{path}\n{"=" * 70}')
    doc = Document(path)
    paras = [p.text for p in doc.paragraphs]

    captions = defaultdict(list)  # number -> [(para_index, full_caption_text)]
    for i, t in enumerate(paras):
        m = CAPTION_RE.match(t.strip())
        if m:
            captions[m.group(1)].append((i, t.strip()))

    print(f'{len(captions)} distinct table numbers, '
          f'{sum(len(v) for v in captions.values())} caption paragraphs total\n')

    ok = True
    for num in sorted(captions, key=lambda x: (int(re.match(r'\d+', x).group()), x)):
        entries = captions[num]
        ref_re = re.compile(REF_RE_TEMPLATE.format(num=re.escape(num)))
        # count inline references: any paragraph containing "Table {num}"
        # that is NOT itself one of this number's own caption paragraphs.
        caption_para_indices = {i for i, _ in entries}
        ref_count = 0
        ref_examples = []
        for i, t in enumerate(paras):
            if i in caption_para_indices:
                continue
            for m in ref_re.finditer(t):
                ref_count += 1
                snippet = t[max(0, m.start() - 35):m.start() + 45].strip()
                ref_examples.append((i, snippet))

        if len(entries) > 1:
            ok = False
            print(f'DUPLICATE  Table {num}  -- used for {len(entries)} different '
                  f'tables, {ref_count} inline reference(s) elsewhere:')
            for i, cap in entries:
                print(f'    para {i:4d}: {cap[:90]}')
            for i, snippet in ref_examples[:6]:
                print(f'    ref  {i:4d}: ...{snippet}...')
            if len(ref_examples) > 6:
                print(f'    ... and {len(ref_examples) - 6} more reference(s)')
            print()
        else:
            i, cap = entries[0]
            print(f'ok         Table {num:<4} para {i:4d}  refs={ref_count:<3}  {cap[:70]}')

    return ok


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(2)
    all_ok = True
    for path in sys.argv[1:]:
        all_ok &= check_file(path)
    print(f'\n{"=" * 70}')
    if all_ok:
        print('ALL CLEAR -- no duplicate table numbers in any file checked.')
        sys.exit(0)
    else:
        print('DUPLICATES FOUND -- see above. Fix before treating any file as final.')
        sys.exit(1)
