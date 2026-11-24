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
        (r"^gk2_sigma_xx_avg$", 1.7336,
         "Gaussian-kernel definition: best mean self-relative error over 11 u-only checkpoints, "
         "10 held-out samples, vs the 950,400-el hex8 reference (local-integration-refinement seed 3; cluster job 3271674, "
         "2026-11-20, evaluate_b3_checkpoints_gaussian.py)"),
        (r"^gk2_sigma_yy_avg$", 1.4575,
         "Gaussian-kernel definition: best mean self-relative error over 11 u-only checkpoints, "
         "10 held-out samples, vs the 950,400-el hex8 reference (local-integration-refinement seed 3; cluster job 3271674, "
         "2026-11-20, evaluate_b3_checkpoints_gaussian.py)"),
        (r"^gk2_sigma_zz_avg$", 1.5512,
         "Gaussian-kernel definition: best mean self-relative error over 11 u-only checkpoints, "
         "10 held-out samples, vs the 950,400-el hex8 reference (local-integration-refinement seed 3; cluster job 3271674, "
         "2026-11-20, evaluate_b3_checkpoints_gaussian.py)"),
        (r"^gk2_pressure_avg$", 1.5508,
         "Gaussian-kernel definition: best mean self-relative error over 11 u-only checkpoints, "
         "10 held-out samples, vs the 950,400-el hex8 reference (local-integration-refinement seed 3; cluster job 3271674, "
         "2026-11-20, evaluate_b3_checkpoints_gaussian.py)"),
        (r"^gk2_cyl_rr_avg$", 1.6528,
         "Gaussian-kernel definition: best mean self-relative error over 11 u-only checkpoints, "
         "10 held-out samples, vs the 950,400-el hex8 reference (local-integration-refinement seed 3; cluster job 3271674, "
         "2026-11-20, evaluate_b3_checkpoints_gaussian.py)"),
        (r"^gk2_cyl_tt_avg$", 1.4943,
         "Gaussian-kernel definition: best mean self-relative error over 11 u-only checkpoints, "
         "10 held-out samples, vs the 950,400-el hex8 reference (local-integration-refinement seed 3; cluster job 3271674, "
         "2026-11-20, evaluate_b3_checkpoints_gaussian.py)"),
        (r"^gk2_cyl_zz_avg$", 1.5512,
         "Gaussian-kernel definition: best mean self-relative error over 11 u-only checkpoints, "
         "10 held-out samples, vs the 950,400-el hex8 reference (local-integration-refinement seed 3; cluster job 3271674, "
         "2026-11-20, evaluate_b3_checkpoints_gaussian.py)"),
        (r"^gk2_cyl_rz_avg$", 2.8015,
         "Gaussian-kernel definition: best mean self-relative error over 11 u-only checkpoints, "
         "10 held-out samples, vs the 950,400-el hex8 reference (DD rel_l2 N=400; cluster job 3271674, "
         "2026-11-20, evaluate_b3_checkpoints_gaussian.py)"),
        (r"^gk2_sigma_xz_avg$", 2.8731,
         "Gaussian-kernel definition: best mean self-relative error over 11 u-only checkpoints, "
         "10 held-out samples, vs the 950,400-el hex8 reference (DD rel_l2 N=400; cluster job 3271674, "
         "2026-11-20, evaluate_b3_checkpoints_gaussian.py)"),
        (r"^gk2_sigma_frob_avg$", 5.765,
         "Gaussian-kernel definition: best mean self-relative error over 11 u-only checkpoints, "
         "10 held-out samples, vs the 950,400-el hex8 reference (DD hybrid; cluster job 3271674, "
         "2026-11-20, evaluate_b3_checkpoints_gaussian.py)"),
        (r"^gk2_von_mises_avg$", 1.7636,
         "Gaussian-kernel definition: best mean self-relative error over 11 u-only checkpoints, "
         "10 held-out samples, vs the 950,400-el hex8 reference (DD rel_l2 N=400; cluster job 3271674, "
         "2026-11-20, evaluate_b3_checkpoints_gaussian.py)"),
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


# ABSOLUTE ACCURACY TARGETS: an engineering requirement on the reference value
# itself, independent of any surrogate (e.g. a stiffness used in design must be
# known to 1%). The required uncertainty is then the smaller of
# (surrogate error / MARGIN) and the target. The values below are a PROPOSAL,
# not decided: pending Professor Rabczuk (question 3 of the 20 Nov draft).
# They are used only with VERIF_ABS_TARGETS=proposal; by default no absolute
# target is applied and every stored verdict is unchanged.
ABSOLUTE_TARGETS_PROPOSAL = [
    (r"(reaction|strain_energy)", 0.01,
     "global stiffness/energy quantities used in design: 1% (proposal)"),
    (r"(disp|^u_|^v_|^w_)", 0.01, "displacements: 1% (proposal)"),
    (r"(avg|gk)", 0.05, "region-averaged stresses (fatigue/strength screening): 5% "
                        "(proposal)"),
    (r"(peak|max)", 0.05, "peak stresses, only where a finite limit exists: 5% (proposal)"),
]


# ADOPTED 24 November 2026: Professor Rabczuk approved the one-third margin and the limits
# of question 2 (1% for global quantities and displacements, 5% for region-averaged
# stresses), QoI-dependent. Peak stresses were not part of the question: no target is
# adopted for them. Used with VERIF_ABS_TARGETS=adopted (adopted.py sets it).
ABSOLUTE_TARGETS_ADOPTED = [
    (r"(reaction|strain_energy|compliance)", 0.01,
     "global quantities: 1% (adopted 24 Nov 2026, advisor decision on question 2)"),
    (r"(disp|^u_|^v_|^w_)", 0.01,
     "displacements: 1% (adopted 24 Nov 2026, advisor decision on question 2)"),
    (r"(avg|gk)", 0.05,
     "region-averaged stresses: 5% (adopted 24 Nov 2026, advisor decision on question 2)"),
]


def constitutive_requirement(half_width_rel):
    """Adopted 24 Nov 2026 (question 2): the discretization uncertainty of a quantity with a
    constitutive (CSM/CFM) range must be below one third of the range's relative half-width."""
    return half_width_rel / MARGIN, (f"one third of the constitutive half-width "
                                     f"{half_width_rel:.2%} (adopted 24 Nov 2026)")


def absolute_target(qoi):
    import os
    mode = os.environ.get("VERIF_ABS_TARGETS")
    if mode == "adopted":
        for pat, t, src in ABSOLUTE_TARGETS_ADOPTED:
            if re.search(pat, qoi):
                return t, src
        return None, "no absolute target adopted for this QoI"
    if mode != "proposal":
        return None, "no absolute target set"
    for pat, t, src in ABSOLUTE_TARGETS_PROPOSAL:
        if re.search(pat, qoi):
            return t, src
    return None, "no absolute target proposed for this QoI"


def required_uncertainty(geometry, qoi):
    """(required relative uncertainty or None, explanation)."""
    err, src = surrogate_error(geometry, qoi)
    tgt, tsrc = absolute_target(qoi)
    cands = []
    if err is not None:
        cands.append((err / MARGIN, f"best surrogate error {err:.2%} / {MARGIN:g} ({src})"))
    if tgt is not None:
        cands.append((tgt, f"absolute target {tgt:.2%} ({tsrc})"))
    if not cands:
        return None, src + "; the class ceiling applies"
    return min(cands)
