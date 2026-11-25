# Pre-registration: automatic literature search (2026-11-25)

Written before any LLM call of the search (the "scout", the card and the definition steps). Free
steps already run while building it: arXiv queries, downloads and the code screen on 28 papers
(results seen: the code ranking, not any LLM judgement). Frozen file hashes are in
`literature_search_preregistration.json`.

Run: `literature_search.search(backend="claude-cli", max_read=3, budget_usd=5)`; the material
class with fewer recorded runs is searched first (neo-Hookean).

Predictions
- **L1 (HIGH)** Step L needs no person: queries, ranking, the choice of papers and the hints are
  made by code or the LLM.
- **L2 (LOW)** At least one of the papers read is accepted by the problem definition (supported,
  every number traced) and reaches verdicts. Low because none of the three hand-picked papers of
  round O was representable from its text.
- **L3 (LOW-MEDIUM)** The scout's "likely representable" agrees with the definition step's decision
  (supported or not) for at least 2 of the papers read.
- **L4 (HIGH)** LLM cost at most USD 5.
- **L5 (HIGH)** No paper is solved with a number that is not in its text (the trace check).
