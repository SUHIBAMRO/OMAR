"""
Find the RIGHT B1 x Neo-Hookean checkpoint on Drive, by fingerprint, instead
of guessing a path.

WHY THIS EXISTS (2026-09-12): every round-10 GPU cell that needed this
checkpoint hardcoded a path (`results/checkpoints/B1_neo_hookean/model_best.pt`)
that never existed on Drive, silently fell back to
`data_driven/B1_neo_hookean/model_best.pt` -- a COMPLETELY DIFFERENT model
(train_data_driven.py's own data-driven-loss baseline from the round-5/6
comparison study) -- and produced a real GPU result (~620-640% error at
EVERY resolution from N=13 to N=1401, flat regardless of N) that looked
exactly like "the operator breaks down far from its training range" but
was actually "the wrong model was loaded". That false alarm survived
several real GPU rounds because nothing ever checked WHICH checkpoint was
actually on disk.

Fixing the hardcoded path once is not enough on its own: there turned out
to be at least three plausible candidate directories for this same nominal
model (`results/B1_neo_hookean/`, `zeroshot_B1_neo_hookean/`,
`data_driven/B1_neo_hookean/`), and nothing on disk names which one is
"the" checkpoint that produced the numbers already published in Table 5 /
the zero-shot study. The one fact that DOES pin this down unambiguously is
already sitting in a committed JSON: point7a_results/zeroshot_B1_neo_hookean.json
records `"checkpoint_fingerprint": "<sha256 of the exact checkpoint file that
produced those numbers>"`. This module tries every known candidate path,
hashes whichever ones exist, and returns the one whose hash matches --
verified by content, not assumed by path.
"""
import hashlib
import os

# The zero-shot study's own already-published, trusted numbers
# (mean_rel_L2_vs_fine_reference = 5.2%-9.7% at N=13-49) came from exactly
# this checkpoint file. Any candidate path that does NOT hash to this is
# NOT the checkpoint those numbers describe, whatever its path suggests.
ZEROSHOT_B1_NEO_HOOKEAN_FINGERPRINT = (
    "86030f4f05ea74f83079cee6b74485b30f2c1a5acac6acd5bc7a06e3adaa88f4"
)

CANDIDATE_RELPATHS = [
    "results/B1_neo_hookean/model_best.pt",
    "zeroshot_B1_neo_hookean/model_best.pt",
    "results/checkpoints/B1_neo_hookean/model_best.pt",
    "data_driven/B1_neo_hookean/model_best.pt",
]


def sha256_of(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def resolve_b1_neo_hookean_checkpoint(run_dir, target_fingerprint=ZEROSHOT_B1_NEO_HOOKEAN_FINGERPRINT):
    """run_dir: the Drive run directory (e.g. /content/drive/MyDrive/pfem_run).
    Hashes every candidate path that exists under it and returns
    (path, fingerprint) for the one matching target_fingerprint. Raises with a
    full diagnostic (every candidate found, and its own hash) if none match
    or none exist -- so the failure mode is a clear list to look at, not
    another silent wrong guess."""
    found = []
    for rel in CANDIDATE_RELPATHS:
        p = os.path.join(run_dir, rel)
        if os.path.exists(p):
            fp = sha256_of(p)
            found.append((p, fp))
            print(f"  [checkpoint candidate] {p}  sha256={fp}"
                  f"{'  <-- MATCHES zero-shot study' if fp == target_fingerprint else ''}")

    if not found:
        raise FileNotFoundError(
            f"No B1 x Neo-Hookean checkpoint found under {run_dir} at any of "
            f"{CANDIDATE_RELPATHS} -- update CANDIDATE_RELPATHS in "
            f"resolve_b1_checkpoint.py with the real path.")

    if target_fingerprint is not None:
        match = [p for p, fp in found if fp == target_fingerprint]
        if match:
            return match[0], target_fingerprint
        raise RuntimeError(
            f"None of the {len(found)} checkpoint(s) found under {run_dir} match the "
            f"zero-shot study's own fingerprint ({target_fingerprint}) -- see the "
            f"candidates printed above. This means either the zero-shot study's own "
            f"checkpoint isn't on this Drive under a known path, or the fingerprint "
            f"constant here is stale. DO NOT silently pick one of the mismatched "
            f"candidates -- resolve this before trusting any accuracy number.")

    print(f"  [checkpoint] no target fingerprint given -- using the first candidate found: "
          f"{found[0][0]}")
    return found[0]
