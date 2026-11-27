"""POST HOC (24 Nov 2026, after the pre-registered autonomy test): choosing the passage of a
paper that describes the example, without a section number.

Seen in the test: literature_agent.select_section, given only a hint, took 'section 2: The
linear elasticity model' (QTT paper) and a running page header 'S EPTEMBER 30, 2026' (PoliVEM
paper), because headings are found by a regular expression that also matches page headers,
formulas and figure labels, and the plate-with-hole section of PoliVEM was not detected at all.

This selector does not depend on headings. It scores overlapping windows of the text by
  hint words (x3) + words of a problem statement (Young, Poisson, modulus, traction, clamped,
  symmetry, boundary, load, plane stress/strain, mesh, elements) + numbers with units,
restricted to the part of the paper after the first heading that names numerical examples
(if there is one), and returns the best window. The literature agent then fills the card
from that passage exactly as before (every quote is still checked against the whole paper).
"""
import re

KEY = ("young", "poisson", "modulus", "traction", "clamped", "fixed", "symmetry", "boundary",
       "load", "plane stress", "plane strain", "mesh", "elements", "displacement", "pressure")
START = re.compile(r"^\s*\d+\.?\s+(numerical (examples|experiments|results|studies|tests)|"
                   r"examples|test cases|benchmarks?)\b", re.I | re.M)


COMMON = {"case", "the", "and", "with", "plate", "example", "test", "problem", "under", "block",
          "membrane", "panel", "domain", "linear", "elastic", "elasticity", "nu", "e"}


def anchors(text, hint, lo):
    """26 Nov (after round 3: paper 2206.14610, hint "Cook's membrane", the best sliding window
    began with the L-shape example before it): window starts just before the occurrences of the
    hint's most distinctive word (the rarest non-common hint word in the paper)."""
    words = [w for w in re.findall(r"[a-z]+", (hint or "").lower()) if len(w) > 2
             and w not in COMMON]
    low = text.lower()
    hits = {w: [m.start() for m in re.finditer(r"\b" + re.escape(w), low) if m.start() >= lo]
            for w in words}
    hits = {w: v for w, v in hits.items() if v}
    if not hits:
        return []
    w = min(hits, key=lambda k: len(hits[k]))
    starts = []
    for i in hits[w]:
        nl = text.rfind("\n", max(lo, i - 400), i)        # start of the line that names it
        starts.append(max(lo, (nl if nl >= 0 else i - 200)))
    return sorted(set(starts))


def select_passage(text, hint, width=7000, step=500):
    m = START.search(text)
    lo = m.start() if m else 0
    words = [w for w in re.findall(r"\w+", (hint or "").lower()) if len(w) > 2]
    best, where = None, None
    starts = anchors(text, hint, lo) or range(lo, max(lo + 1, len(text) - width // 2), step)
    for a in starts:
        w = text[a:a + width].lower()
        s = 3 * sum(w.count(x) for x in words) + sum(w.count(k) for k in KEY) \
            + 0.2 * len(re.findall(r"\d+(?:\.\d+)?\s*(?:mm|m|mpa|gpa|pa|kpa|n)\b", w))
        if best is None or s > best:
            best, where = s, a
    return text[where:where + width], f"passage at character {where} (score {best:.1f})"


def run_card(backend, pdf, hint):
    """literature_agent.run_card with the passage selector (post hoc)."""
    from omar_pfem.verification_agent import literature_agent as LA
    import os
    text = LA.extract_text(pdf)
    sec, where = select_passage(text, hint)
    # 26 Nov: the passage may hold several examples; name the one wanted
    focus = (f"THE EXAMPLE TO EXTRACT: {hint}. If the text below contains several examples, "
             "describe only this one and ignore the others.\n\n") if hint else ""
    card = backend.ask("librarian", LA.PROMPT + focus + sec, LA.CARD_SCHEMA)
    ok = LA.check_quotes(card, text)
    vocab, gaps = LA.vocabulary(card)
    return {"pdf": os.path.basename(pdf), "section_used": where, "card": card,
            "quotes_found": ok, "quote_rate": sum(ok.values()) / max(len(ok), 1),
            "qoi_classes": vocab, "vocabulary_gaps": gaps, "needs": LA.needs(card, ok),
            "problem_text": LA.problem_text(card, ok)}
