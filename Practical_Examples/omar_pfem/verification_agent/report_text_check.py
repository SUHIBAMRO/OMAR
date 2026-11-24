"""Do the report's printed numbers actually appear in the report? (2026-11-23, round J)

verify_report.py compares each claim's printed value (typed into the script) with the
stored data; it never reads the report, so a number typed differently into the report
would pass unnoticed. This script closes that gap: it extracts the text of the exported
PDF, splits it by section, and looks for every claim's printed value in its section,
in the forms the report uses (thousands separators, the Unicode minus, "a x 10^-n" with
superscripts, the number of decimals implied by the claim's tolerance, percentages).

Claims whose value is a yes/no flag (0 or 1 with a label that is a test, not a count)
cannot be found as text and are listed apart; every other claim not found is listed for
review. The result is a list, not a verdict: a number can be found by coincidence.

    python -m omar_pfem.verification_agent.report_text_check [report.pdf]
"""
import json
import math
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
PDF = os.path.join(ROOT, "advisor_feedback", "2026-11-24_Verification_Framework_Technical_Report_v11.pdf")
OUT = os.path.join(HERE, "report_text_check.json")
SUP = str.maketrans("-0123456789", "⁻⁰¹²³⁴⁵⁶⁷⁸⁹")
# Reviewed 2026-11-23: claims that back a figure or a derived statement and whose raw value
# the text does not print. Each group with the reason; a claim matched here is reported as
# "backing", never as found.
BACKING = (
    (r"^IGA3D \w+ L\d$", "per-level spline values: plotted in Figures 25.1/25.2, the text prints "
                         "limits, orders and selected levels"),
    (r"^splineP1 .* L\d$", "per-level spline P1 values: behind the plotted sequences and the "
                           "stated orders and limits"),
    (r"^splineP1 peak 3-level order$", "backs 'the peak converges with neither definition'"),
    (r"^B2 level \d elements$", "the text gives the range of levels, not every element count"),
    (r"^(FEM m\d+ (dof|compliance err|gk e22 err)|IGA p2 m4 dof|lamina dof m=32)$",
     "back derived statements ('1.05 million unknowns', 'about 4,300 unknowns', "
     "'about 5,000 times', '20 times')"),
)
FLAG_WORDS = ("rate", "no rate", "PASS", "FAIL", "cross-checked", "is ", "has ", "covered",
              "monotone", "reproduces", "matches", "within", "agree")


def text_by_section(pdf):
    t = subprocess.run(["pdftotext", "-layout", pdf, "-"], capture_output=True, text=True).stdout
    t = t.replace(" ", " ").replace(" ", " ")
    heads = [(m.start(), m.group(1)) for m in re.finditer(r"(?m)^(\d+(?:\.\d+)?)\.?\s+[A-Z]", t)
             if re.fullmatch(r"([1-9]|[1-3]\d)(\.([1-9]|1[0-2]))?", m.group(1))]   # not table rows
    # the body starts at the LAST occurrence of "1 " heading chain; take for every key the
    # last position (the table of contents comes first)
    last = {}
    for pos, key in heads:
        last[key] = pos
    order = sorted(last.items(), key=lambda kv: kv[1])
    out = {}
    for i, (key, pos) in enumerate(order):
        if "." in key:
            nxt = next((p for k, p in order[i + 1:]), len(t))
        else:
            nxt = next((p for k, p in order[i + 1:] if "." not in k), len(t))
        out[key] = t[pos:nxt]
    return out, t


def forms(v, tol):
    """Strings the report may use for the printed value v (tolerance tol)."""
    out = set()
    from decimal import Decimal
    ds = {max(0, -Decimal(repr(float(v))).as_tuple().exponent)}               # as typed
    if tol and tol > 0:
        ds.add(max(0, int(round(-math.log10(2 * tol)))))       # as the tolerance implies
    x = v
    for d in (d for d in ds if d <= 8):
        s = f"{abs(x):,.{d}f}" if abs(x) >= 1000 else f"{abs(x):.{d}f}"
        for core in {s, s.replace(",", ""), s.rstrip("0").rstrip(".") if "." in s else s}:
            out.add(("−" if x < 0 else "") + core)
            out.add(("-" if x < 0 else "") + core)
    if v < 0:                              # written as "x% below" in the text
        out |= {f.lstrip("-\u2212") for f in out}
    if v != 0 and (abs(v) < 0.1 or abs(v) >= 1e6):
        e = int(math.floor(math.log10(abs(v))))
        m = abs(v) / 10 ** e
        for md in (0, 1, 2):
            ms = f"{m:.{md}f}"
            sign = "−" if v < 0 else ""
            out.add(f"{sign}{ms} × 10{str(e).translate(SUP)}")
            if ms in ("1", "1.0"):
                out.add(f"10{str(e).translate(SUP)}")
    return out


def found(sec_text, v, tol):
    sec_text = re.sub(r"\s+", " ", sec_text)          # a number may wrap across lines
    for f in forms(v, tol):
        pat = r"(?<![\d.,])" + re.escape(f) + r"(?![\d])(?!\.\d)"
        if re.search(pat, sec_text):
            return f
    return None


def is_flag(c):
    return c["printed"] in (0, 1) and c["tolerance"] == 0 and any(
        w in c["claim"] for w in FLAG_WORDS)


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    pdf = argv[0] if argv else PDF
    secs, whole = text_by_section(pdf)
    claims = json.load(open(os.path.join(HERE, "verify_report_result.json")))
    rows = []
    for c in claims:
        if is_flag(c):
            rows.append({**c, "text": "flag (not textual)"})
            continue
        why = next((w for pat, w in BACKING if re.search(pat, c["claim"])), None)
        sec = secs.get(c["section"])
        hit = found(sec, c["printed"], c["tolerance"]) if sec else None
        where = "section" if hit else None
        if not hit:
            hit = found(whole, c["printed"], c["tolerance"])
            where = "elsewhere" if hit else None
        if not where and why:
            where = "backing"
        rows.append({**c, "text": where or "NOT FOUND", "form": hit, "backing_reason": why,
                     "section_found": sec is not None})
    summ = {k: sum(r["text"] == k for r in rows)
            for k in ("section", "elsewhere", "backing", "NOT FOUND", "flag (not textual)")}
    json.dump({"pdf": os.path.basename(pdf), "summary": summ, "rows": rows}, open(OUT, "w"),
              indent=1, default=str)
    return summ, rows


if __name__ == "__main__":
    s, rows = main()
    print(s)
    for r in rows:
        if r["text"] in ("NOT FOUND", "elsewhere"):
            print(f"{r['text']:<10}[{r['section']}] {r['claim']}: {r['printed']}")
