"""What accuracy a reference value must have: derived from its purpose.

A reference value is used to measure surrogate errors. The measured error
is (surrogate error) +- (reference uncertainty), so the reference
uncertainty must be clearly smaller than the smallest surrogate error it
has to resolve. Requirement used here (fixed in code, never chosen by an
LLM):

    required uncertainty  =  best recorded surrogate error / MARGIN,   MARGIN = 3

so the reference shifts a measured error by at most about a third. The
requirement only ever TIGHTENS the per-class ceilings in trees.py
(trees.check_params takes the minimum of both); a QoI without a recorded
surrogate error keeps its class ceiling, and that is stated, not hidden.

Surrogate errors are the best (smallest) relative error over all
checkpoints reported in this project, with the source of every number.
B1's per-QoI errors other than displacement exist only as figures in the
2026-11-13 re-evaluation report, so they are NOT recorded here.
"""
import re

MARGIN = 3.0

# geometry -> list of (QoI regex, best surrogate relative error, source)
SURROGATE_ERRORS = {
    "bushing_groove": [
        (r"^reaction_moment_y$", 0.0138,
         "PI baseline vs the 950,400-element reference (Rabczuk email 2026-11-04 item 2; "
         "PROJECT_STATUS 'reaction 1.38%')"),
        (r"^strain_energy$", 0.0413,
         "best energy error, local-integration-refinement seed 1 vs the 950,400-element "
         "reference (PROJECT_STATUS table: 4.13/5.22/4.41%, PI 5.33%)"),
        (r"(sigma|cyl|von_mises|pressure)", 5.995,
         "groove-region stress: smallest error of any checkpoint and component, "
         "sigma_xz 599.54% (seed 2 vs 950,400 el); other components 1,200-13,000%, "
         "region sigma_xx vs the Q20 reference >= 2,102%"),
    ],
    "unit_square": [
        (r"(disp|^u_|^v_)", 0.0410,
         "best PI displacement relative L2 error vs the N=2236 reference (N=49 inference, "
         "re-evaluation report 2026-11-13, Section 4); a field norm, used as the "
         "displacement-class level"),
    ],
}


def surrogate_error(geometry, qoi):
    """(best surrogate relative error, source) or (None, reason)."""
    for pat, err, src in SURROGATE_ERRORS.get(geometry, []):
        if re.search(pat, qoi):
            return err, src
    return None, "no surrogate error recorded for this QoI"


def required_uncertainty(geometry, qoi):
    """(required relative uncertainty or None, explanation)."""
    err, src = surrogate_error(geometry, qoi)
    if err is None:
        return None, src + "; the class ceiling applies"
    return err / MARGIN, f"best surrogate error {err:.2%} / {MARGIN:g} ({src})"
