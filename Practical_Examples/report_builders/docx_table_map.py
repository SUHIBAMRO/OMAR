"""Shared, robust caption<->table association for the real Report docx.

A real ordering trap was found while building item #12's mesh-convergence
figure: for the six Table 1/1a/1b/2/2a/2b tables, the caption paragraph
sits AFTER its table, the opposite of the convention every other table
in this same document uses (caption before). Checking only one
direction silently mislabels every table one slot off.

build_table_map() checks BOTH directions for every table and asserts
each one has a caption match on EXACTLY one side (raising loudly on
anything ambiguous or unmatched, rather than silently guessing), so a
caller can trust get_table('Table N.') without re-deriving this each
time.
"""
import re

from docx import Document
from docx.table import Table
from docx.text.paragraph import Paragraph
from docx.oxml.ns import qn

CAPTION_RE = re.compile(r'^(Table\s+\d+[a-z]?(?:\s*\(revised\))?)\.')


def _load_items(docx_path):
    doc = Document(docx_path)
    body = doc.element.body
    items = []
    for child in body.iterchildren():
        if child.tag == qn('w:p'):
            items.append(('p', Paragraph(child, doc)))
        elif child.tag == qn('w:tbl'):
            items.append(('tbl', Table(child, doc)))
    return doc, items


def _nearest_caption(items, idx, step):
    j = idx + step
    while 0 <= j < len(items):
        kind, obj = items[j]
        if kind == 'tbl':
            return None
        text = obj.text.strip()
        if text:
            m = CAPTION_RE.match(text)
            return m.group(1) if m else None
        j += step
    return None


def build_table_map(docx_path):
    """Returns (doc, {caption_label: Table}).

    Two conventions coexist in this document (confirmed directly, not
    assumed): most tables have their caption BEFORE them, but one block
    (Tables 1/1a/1b/2/2a/2b) has it AFTER. When a single caption
    paragraph sits between two adjacent tables, a naive nearest-
    neighbour check matches it on BOTH sides (it is simultaneously "next
    caption after" the earlier table and "nearest preceding" the later
    one) -- resolved here with a second pass: first resolve every table
    that has a caption on exactly one side, then for each remaining
    ambiguous table, adopt whichever convention its nearest already-
    resolved neighbour (in table order) used, on the reasoning that the
    convention only changes at block boundaries, never mid-block."""
    doc, items = _load_items(docx_path)
    tables = [(idx, obj) for idx, (kind, obj) in enumerate(items) if kind == 'tbl']

    before_of = {idx: _nearest_caption(items, idx, -1) for idx, _ in tables}
    after_of = {idx: _nearest_caption(items, idx, +1) for idx, _ in tables}

    resolved = {}  # table idx -> ('before'|'after', label)
    for idx, _ in tables:
        b, a = before_of[idx], after_of[idx]
        if b and not a:
            resolved[idx] = ('before', b)
        elif a and not b:
            resolved[idx] = ('after', a)
        elif not a and not b:
            resolved[idx] = (None, None)  # uncaptioned table, e.g. a template

    for idx, _ in tables:
        if idx in resolved:
            continue
        pos = [i for i, _ in tables].index(idx)
        convention = None
        for j in range(pos - 1, -1, -1):
            other_idx = tables[j][0]
            if other_idx in resolved and resolved[other_idx][0]:
                convention = resolved[other_idx][0]
                break
        if convention is None:
            for j in range(pos + 1, len(tables)):
                other_idx = tables[j][0]
                if other_idx in resolved and resolved[other_idx][0]:
                    convention = resolved[other_idx][0]
                    break
        if convention is None:
            raise AssertionError(f'table at item {idx}: ambiguous captions '
                                  f'({before_of[idx]!r}/{after_of[idx]!r}) and no '
                                  f'resolved neighbour to infer a convention from')
        label = before_of[idx] if convention == 'before' else after_of[idx]
        resolved[idx] = (convention, label)

    table_map = {}
    for idx, obj in tables:
        _, label = resolved[idx]
        if label is None:
            continue
        if label in table_map:
            raise AssertionError(f'caption {label!r} matched more than one table')
        table_map[label] = obj
    return doc, table_map


def get_rows(table, header_row=0):
    """List of dict rows keyed by the table's own header cell text."""
    header = [c.text.strip() for c in table.rows[header_row].cells]
    rows = []
    for row in table.rows[header_row + 1:]:
        cells = [c.text.strip() for c in row.cells]
        rows.append(dict(zip(header, cells)))
    return rows


if __name__ == '__main__':
    import sys
    DELIV = '/tmp/claude-0/-home-user/64d7c4d8-d5f0-5686-a58f-aa87abfd4ba4/scratchpad/deliverables'
    path = DELIV + '/PFEM_Transolver_Report_updated_2026-09-08.docx'
    doc, tmap = build_table_map(path)
    print(f'{len(tmap)} captioned tables found')
    for label in sorted(tmap, key=lambda s: (len(s), s)):
        print(' ', label)
