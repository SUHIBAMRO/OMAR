"""Named standard benchmark geometries (POST HOC rule, 26 Nov, decided by Omar after the round-2
literature search: papers name these benchmarks but give their coordinates only in a figure).

A vertex of a description may be taken from this table instead of the source text only if
  * the source text names the benchmark (one of `names`, case-insensitive), and
  * the description's outer boundary is exactly the table's polygon (same vertex set),
and the run log then states that the geometry came from the table, with its citation.
Physical inputs (E, nu, loads) are never taken from here.
"""
import re

TABLE = {
    "cook_membrane": {
        "names": ("cook's membrane", "cook’s membrane", "cook membrane", "cook's problem",
                  "cook’s problem", "cook's cantilever", "cook-type"),
        "outer": [[0, 0], [48, 44], [48, 60], [0, 44]],
        "citation": "R.D. Cook, Improved two-dimensional finite element, J. Struct. Div. ASCE 100 "
                    "(1974) 1851-1863: tapered panel with vertices (0,0), (48,44), (48,60), (0,44)",
    },
}


def named_in(text):
    low = re.sub(r"\s+", " ", text.lower())
    return [k for k, g in TABLE.items() if any(n in low for n in g["names"])]


def match(desc, text):
    """The table entry whose polygon the description uses and whose name the text contains."""
    outer = (desc.get("domain") or {}).get("outer")
    if not outer:
        return None
    pts = {(round(float(x), 9), round(float(y), 9)) for x, y in outer}
    for k in named_in(text):
        if pts == {(float(x), float(y)) for x, y in TABLE[k]["outer"]} and len(outer) == len(pts):
            return k
    return None
