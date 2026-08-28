# Point 7b — physics-informed against data-driven

B1 × Neo-Hookean. Same architecture, same mesh, same 800/200 split, same
75,000-optimizer-step budget. Error is the per-component relative L2 of
Tables 5 and 11.

| Run | Error | Optimizer | Label cost |
|---|---|---|---|
| Physics-informed | **0.0959** | Adam 2e-3, wd 0 | **0 h** |
| Data-driven, matched optimizer | **0.1307** | Adam 2e-3, wd 0 | 5.65 h |
| Data-driven, its own recipe | **0.0826** | AdamW 1e-3 + OneCycleLR | 5.65 h |

## The clean comparison says the physics-informed loss wins

When the *only* difference is the loss, the physics-informed model is **36%
more accurate** — 0.0959 against 0.1307. The energy functional is not a
handicap here; it is the better training signal at equal everything else.

And it is free of a cost the other side pays. The data-driven model needs a
finite-element solution per training sample: 800 solves at Table 4a's
measured rate is **5.65 hours of CPU** that the physics-informed model never
spends. Including it, the data-driven run costs 6.06 h against 0.80 h — 7.6×
more compute for a worse result.

## The first run said the opposite, and that is worth recording

The first data-driven run reached 0.0826 and appeared to beat the
physics-informed model. It used AdamW at lr=1e-3 with weight decay and
OneCycleLR, where `train_B1` uses plain Adam at lr=2e-3 with no decay. The
optimizer was confounded with the loss, and the script's own docstring
claimed "the ONLY difference is the loss", which was false. Both runs are
kept here; neither is deleted.

## The 2×2 has an empty cell, and it matters

|  | Adam 2e-3 | AdamW + OneCycleLR |
|---|---|---|
| **Physics-informed** | 0.0959 | **not measured** |
| **Data-driven** | 0.1307 | 0.0826 |

The matched column is a clean comparison of training principles. The
unmatched pair — 0.0959 against 0.0826 — is **not**: it compares each method
under a different optimizer, so it cannot rank the principles at all.

Two further asymmetries, both worth stating rather than resolving in our own
favour:

* The matched optimizer was **tuned for the physics-informed loss**. Section
  8.2's protocol screened batch sizes against the energy objective, so
  forcing the data-driven model onto that recipe is not obviously neutral
  either. "Matched" removes one confound and introduces a milder one.
* The unmatched data-driven run's best value came at step 75,000 of 75,000 —
  **the last one**. It had not converged, so 0.0826 is a lower bound on that
  recipe rather than its ceiling. The matched run peaked at step 56,000 and
  then wandered, so it had converged within budget.

## The run that would settle it

Train the **physics-informed** model with AdamW + OneCycleLR, same 75,000
steps: **about 48 minutes**, no new finite-element data. If it beats 0.0826
the physics-informed principle wins under both recipes and the conclusion is
unambiguous. If it does not, the ranking depends on the optimizer and the
report has to say so.

Until that cell is filled, the defensible claim is the matched one: **at
equal optimizer, equal budget and equal data, the physics-informed loss is
more accurate and needs no labels.**
