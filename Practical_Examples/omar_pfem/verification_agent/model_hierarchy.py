"""Hierarchy of physical and numerical models known to the verification agent.

Each model records its assumptions, validity limits, the accuracy EVIDENCE
measured in this project (with the file it comes from), its cost, and its
availability. Two uses:

  admissible(facts)  which models' assumptions hold for a problem
  escalate(hyp)      for a supported diagnosis hypothesis (diagnose.py), the
                     model one step up the hierarchy that addresses it, and
                     whether the agent can run it

The physical hierarchy is backed by one measured discrepancy: B1 solved
with small-strain linear elasticity vs the compressible neo-Hookean model
(same mesh, material sampling and load), b1_model_discrepancy().

    python -m omar_pfem.verification_agent.model_hierarchy
"""
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.dirname(HERE)
OUT = os.path.join(HERE, "model_hierarchy_result.json")


def _mms(material, element):
    p = os.path.join(PKG, "point9_results", f"mms_B1_{material}.json")
    if not os.path.exists(p):
        return None
    r = json.load(open(p))["convergence_rates"].get(element, {})
    return {k: round(v["rate"], 3) for k, v in r.items() if k in ("L2", "H1_semi", "energy")}


PHYSICAL = {
    "linear_elastic": dict(
        level=0, assumptions="small strains and rotations; linear constitutive law",
        validity="|grad u| << 1 (strains below ~1%); no geometric stiffening",
        evidence="b1_model_discrepancy() (measured here)", available=True),
    "neo_hookean_compressible": dict(
        level=1, assumptions="isotropic hyperelastic, psi = mu/2 (I1-3-2 ln J) + lam/2 (ln J)^2; "
                             "quasi-static; no rate effects",
        validity="moderate strains (no strain stiffening); compressible (nu well below 0.5)",
        evidence="2D MMS Q4/Q9 (point9_results), 3D Q20 MMS (mms_q20_result.json); used for "
                 "B1, B2, B3, Cook", available=True),
    "mooney_rivlin": dict(
        level=2, assumptions="two-invariant hyperelastic (2D calibration, I2 = J^2)",
        validity="moderate strains; better shear response at larger stretch",
        evidence="2D MMS rates (point9_results/mms_B1_mooney_rivlin.json)", available=True),
    "arruda_boyce": dict(
        level=2, assumptions="8-chain, limited chain extensibility N",
        validity="large strains up to the locking stretch (I1_bar < 3N)",
        evidence="2D MMS rates (point9_results/mms_B1_arruda_boyce.json)", available=True),
    "nearly_incompressible_mixed": dict(
        level=2, assumptions="volumetric/isochoric split with an independent pressure",
        validity="nu -> 0.5 (elastomers, bushings)",
        evidence="mixed_b3.py (F-bar / Q20-P0) on B3", available=True),
    "viscoelastic_or_damage": dict(
        level=3, assumptions="rate/history dependence (Mullins, viscoelasticity)",
        validity="cyclic or rate-dependent rubber loading",
        evidence="none in this project", available=False),
}

NUMERICAL = {
    "Q4_disp": dict(element="Q4", degree=1, quadrature="2x2", formulation="displacement",
                    validity="2D; locks in bending/near-incompressibility (nu >~ 0.45)",
                    evidence=lambda: {"mms_neo_hookean": _mms("neo_hookean", "Q4"),
                                      "independent_solver": "B1/B2 reproduced to 2e-12 "
                                      "(independent_solver_result.json)"},
                    cost="B1 N=161 (51,842 dof): 783 s project solver, 34 s scikit-fem",
                    available=True),
    "Q9_disp": dict(element="Q9", degree=2, quadrature="3x3", formulation="displacement",
                    validity="2D; much less locking-prone than Q4",
                    evidence=lambda: {"mms_neo_hookean": _mms("neo_hookean", "Q9")},
                    cost="~4x the dof of Q4 at equal h", available=True),
    "Q2_skfem": dict(element="Q2 (scikit-fem)", degree=2, quadrature="3x3",
                     formulation="displacement, exact arcs (B2)",
                     validity="2D", evidence=lambda: "agrees with the agent's B1/B2 limits "
                     "within their uncertainty (independent_solver_result.json)",
                     cost="B1 N=161: 461 s", available=True),
    "Q20_reduced": dict(element="Q20", degree=2, quadrature="2x2x2 (reduced)",
                        formulation="displacement",
                        validity="3D; possible spurious modes; pressure scatter at nu = 0.45",
                        evidence=lambda: "Q20 MMS rates 3/2 at nu 0.3 and 0.45 "
                                         "(mms_q20_result.json)",
                        cost="B3 59,400 el: 2,739 s, 10.3 GB", available=True),
    "Q20_full": dict(element="Q20", degree=2, quadrature="3x3x3 (full)",
                     formulation="displacement", validity="3D; stiffer near incompressibility",
                     evidence=lambda: "Q20 MMS full rule; B3 moment +0.22% vs reduced at 2,376 el",
                     cost="~3x reduced", available=True),
    "Q20_fbar": dict(element="Q20/P0", degree=2, quadrature="2x2x2 or 3x3x3",
                     formulation="mean-dilatation F-bar (three-field, constant pressure)",
                     validity="nearly incompressible", evidence=lambda: "mixed_b3_result.json",
                     cost="own assembly, ~2-4x Q20_reduced", available=True),
    "local_refinement": dict(element="Q20 graded toward a feature", degree=2, quadrature="2x2x2",
                             formulation="displacement",
                             validity="resolves finite-radius concentrations (h << rho)",
                             evidence=lambda: "z_beta and radial strategies (b3_extended.json, "
                                              "b3_radial.json); radial grading > 1.3 inverts "
                                              "elements at the groove", cost="hours",
                             available=True),
    "IGA": dict(element="NURBS (isogeometric)", degree="p>=2", quadrature="Gauss",
                formulation="displacement or mixed",
                validity="exact curved geometry, smooth stresses",
                evidence=lambda: "not available in this project", cost="n/a", available=False),
}

# supported diagnosis hypothesis (diagnose.py) -> models one step up that address it
ESCALATION = {
    "M1": ["local_refinement", "singular-field QoI (stress-intensity factor)"],
    "M2": ["local_refinement"],
    "P1": ["mooney_rivlin", "arruda_boyce", "nearly_incompressible_mixed"],
    "D1": ["local_refinement", "IGA"],
    "D2": ["Q20_full"],
    "D3": ["Q20_fbar", "nearly_incompressible_mixed"],
    "D4": ["one finer level of the same family"],
    "S1": ["tighter Newton tolerance"],
    "S2": ["direct linear solver"],
    "S3": ["MMS of the element and energy"],
    "Q1": ["smooth (Gaussian-weighted) region average"],
    "Q2": ["signed stress components"],
    "Q3": ["nodal extrapolation to the surface (surface_study.py)"],
}


def escalate(hypotheses):
    out = []
    for h in hypotheses:
        for m in ESCALATION.get(h, []):
            entry = NUMERICAL.get(m) or PHYSICAL.get(m)
            out.append({"hypothesis": h, "model": m,
                        "available": entry["available"] if entry else True,
                        "cost": entry.get("cost") if entry else None})
    return out


def admissible(facts):
    """facts: dict with nu, max_grad_u (or min_detF), dim. Returns
    {model: (ok, reason)} for the physical and numerical models."""
    nu, g, dim = facts.get("nu"), facts.get("max_grad_u"), facts.get("dim", 2)
    res = {}
    if g is not None:
        res["linear_elastic"] = (g < 0.01, f"max |grad u| = {g:.3g} "
                                 + ("< 1%" if g < 0.01 else ">= 1%: geometric nonlinearity"))
    res["neo_hookean_compressible"] = (nu is None or nu < 0.45,
                                       f"nu = {nu}" + ("" if nu is None or nu < 0.45 else
                                                       ": near-incompressible, use a mixed "
                                                       "formulation to cross-check"))
    if dim == 2:
        res["Q4_disp"] = (nu is None or nu < 0.45, "locking risk" if nu and nu >= 0.45 else "ok")
    else:
        res["Q20_reduced"] = (True, "ok; pressure scatter expected" if nu and nu >= 0.45 else "ok")
    return res


# ---- measured physical-model discrepancy on B1 --------------------------------------------
def b1_model_discrepancy(N=81):
    """B1 with small-strain linear elasticity vs compressible neo-Hookean
    (scikit-fem, Q4, 2x2 Gauss, material at element centroids, same
    traction): relative QoI differences and max |grad u|."""
    import scipy.sparse.linalg as spla
    import skfem as fem
    from omar_pfem.verification_agent import independent_solver as S
    m, ib, mu, lam, fext, fixed = S.setup_b1(N, "same")

    @fem.BilinearForm
    def lin(u, v, w):
        eu = 0.5 * (u.grad + np.einsum("ij...->ji...", u.grad))
        ev = 0.5 * (v.grad + np.einsum("ij...->ji...", v.grad))
        tr = eu[0, 0] + eu[1, 1]
        return 2 * w["mu"] * np.einsum("ij...,ij...->...", eu, ev) + w["lam"] * tr * (ev[0, 0] + ev[1, 1])
    K = fem.asm(lin, ib, mu=mu, lam=lam)
    u_lin = fem.solve(*fem.condense(K, fext, D=fixed), solver=spla.spsolve)
    q_lin = S.qois("b1", N, "same", m, ib, u_lin, mu, lam)        # stresses via NH P(F)
    q_nh = S.qois("b1", N, "same", *S.solve("b1", N, "same")[:5])
    gu = ib.interpolate(u_lin).grad
    out = {"N": N, "max_grad_u_linear": float(np.abs(gu).max())}
    for k in ("max_disp_mag", "v_top_off_center", "u_top_off_center", "v_mid_domain",
              "u_mid_domain", "strain_energy"):
        if k == "strain_energy":
            lin_e = 0.5 * float(u_lin @ (K @ u_lin))
            out[k] = {"linear": lin_e, "neo_hookean": q_nh[k],
                      "rel_diff": abs(lin_e - q_nh[k]) / abs(q_nh[k])}
        else:
            out[k] = {"linear": q_lin[k], "neo_hookean": q_nh[k],
                      "rel_diff": abs(q_lin[k] - q_nh[k]) / abs(q_nh[k])}
    return out


def main():
    res = {"b1_linear_vs_neo_hookean": b1_model_discrepancy(),
           "numerical_evidence": {k: v["evidence"]() for k, v in NUMERICAL.items()}}
    json.dump(res, open(OUT, "w"), indent=1, default=str)
    d = res["b1_linear_vs_neo_hookean"]
    print(f"B1 N={d['N']}: max |grad u| (linear) = {d['max_grad_u_linear']:.4f}")
    for k, v in d.items():
        if isinstance(v, dict):
            print(f"  {k:<20} linear {v['linear']:.8g}  neo-Hookean {v['neo_hookean']:.8g}  "
                  f"diff {v['rel_diff']:.3%}")


if __name__ == "__main__":
    main()
