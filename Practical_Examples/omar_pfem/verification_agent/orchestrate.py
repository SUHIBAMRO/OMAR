"""Orchestrated verification run: LLM roles around deterministic gates.

    formalize -> validate spec -> retrieve nearest solved cases
      -> propose plan -> validate plan (code) -> critic -> (one revision)
      -> execute catalogued checks (code) -> record verdicts (code)
      -> advisor explanation -> audit advisor against verdicts (code)

The data a check runs on comes from an adapter keyed by geometry; adapters
expose a fixed QoI vocabulary, so the formalizer cannot invent observables.
"""
import json
import os

import numpy as np

from omar_pfem.verification_agent import gates, repository, trees
from omar_pfem.verification_agent import llm_roles as R
from omar_pfem.verification_agent.spec import ProblemSpec

HERE = os.path.dirname(os.path.abspath(__file__))


# ----------------------------------------------------------------------------
# Data adapters: geometry -> mesh family + QoI series (all precomputed or
# computed by deterministic solvers; no LLM involvement)
# ----------------------------------------------------------------------------


def _b1_uniform_Ns(material="neo_hookean", tol=1e-7):
    """Uniform (grading 0) levels present in the B1 ladder cache."""
    cache = os.path.join(HERE, f"b1_ladder_{material}.json")
    if not os.path.exists(cache):
        return (11, 21, 41, 81)
    rows = json.load(open(cache))
    return tuple(sorted(r["N"] for r in rows if r.get("grading", 0.0) == 0.0
                        and r.get("newton_tol", 1e-7) == tol))


class B1Adapter:
    geometry = "unit_square"
    description = (
        "B1: 2D unit square [0,1]^2, plane strain, compressible neo-Hookean "
        "psi = mu/2 (I1 - 2 - 2 ln J) + lam/2 (ln J)^2, total Lagrangian. "
        "E(x,y) = 1000 + 200 sin(pi x) cos(pi y) (800-1200), nu = 0.3 everywhere (no "
        "near-incompressibility). Bottom edge y=0 clamped (u=v=0); vertical edges "
        "traction-free; top edge y=1 loaded by a DEAD (reference-configuration) normal "
        "traction t_y = -5 - 2 sin(pi x) (compressive, 5-7), applied in 10 equal load "
        "increments, Newton tol 1e-7 (residual ~3e-11 reached; at N=41 tightening to 1e-10 "
        "changes no QoI); min det F 0.9917 at N=41, 0.9876 at N=161 (small strain). Q4 elements, full 2x2 Gauss "
        "integration, E/nu evaluated at element centroids. Uniform meshes N x N nodes, "
        "N=" + ",".join(map(str, _b1_uniform_Ns())) + " (h = 1/(N-1), ratio 2); "
        "strategies at N=81: uniform and corner grading a=0.5, 0.8 (x -> s - a sin(2 pi s)/"
        "(2 pi), y -> t - a sin(pi t)/pi; corner elements 2x and 5x smaller, same node "
        "count). QoIs: max_disp_mag = max nodal |u|; u/v_top_off_center at (0.7, 1.0) and "
        "u/v_mid_domain at (0.3, 0.6), linear interpolation of nodal values (both points are "
        "nodes of every uniform mesh); strain_energy = stored energy, "
        "integral of psi over the body (2x2 Gauss; not the total potential with the load "
        "work); "
        "interior_avg_PK1_frob = detJ-weighted mean of |P|_F over the Gauss points inside "
        "the circle centre (0.5,0.5), radius 0.15 (about 0.56 from the clamped corners); "
        "interior_gk_PK1_frob = the same |P|_F averaged with the smooth weight "
        "exp(-(d/0.1)^2) around (0.5,0.5) instead of the sharp circle (Gauss-point dumps "
        "in b1_gauss_interior/); "
        "peak_PK1_frob = max of |P|_F over all Gauss points (corner singularity where the "
        "clamped edge meets the free edges). Corner theory (singularity.py, Williams "
        "eigenproblem, 90-degree clamped/free corner, plane strain, nu = 0.3): smallest "
        "exponent lam = 0.7112, so energy converges at order 2 lam = 1.42, point values "
        "at >= 1.42 (pollution) and the peak grows like h^(lam-1) = h^-0.289. "
        "Surrogate errors these references must "
        "resolve: best PI displacement relative L2 4.10%; for energy and stress QoIs no "
        "number is recorded, so the class ceilings apply. Tolerances are set by code "
        "(requirements.py), not by the plan.")
    qois = ("max_disp_mag", "u_top_off_center", "v_top_off_center", "u_mid_domain",
            "v_mid_domain", "strain_energy", "interior_avg_PK1_frob", "interior_gk_PK1_frob",
            "peak_PK1_frob")

    def __init__(self, material="neo_hookean", Ns=None, strategy_N=81):
        """Ns=None: every uniform level already in the cache (no new solves)."""
        from omar_pfem.verification_agent.b1_ladder import run_ladder
        cache = os.path.join(HERE, f"b1_ladder_{material}.json")
        Ns = _b1_uniform_Ns(material) if Ns is None else Ns
        self.rows = run_ladder(list(Ns), material, cache)
        self.hs = [r["h"] for r in self.rows]
        # same node count, three refinement strategies: uniform and corner
        # grading a=0.5, 0.8 (b1_ladder.grade_nodes), read from the cache only
        all_rows = json.load(open(cache))
        self.strategy_rows = sorted(
            (r for r in all_rows if r["N"] == strategy_N and r.get("newton_tol", 1e-7) == 1e-7),
            key=lambda r: r.get("grading", 0.0))
        # smooth interior average from the Gauss-point dumps (b1_gauss_interior/)
        from omar_pfem.verification_agent.b1_ladder import kernel_interior_average
        for r in self.rows + self.strategy_rows:
            f = os.path.join(HERE, "b1_gauss_interior", f"N{r['N']}_a{r.get('grading', 0.0)}.npz")
            if os.path.exists(f) and "interior_gk_PK1_frob" not in r:
                r["interior_gk_PK1_frob"], _ = kernel_interior_average(f)

    def series(self, qoi):
        return self.hs, [r[qoi] for r in self.rows], None

    def singular_exponent(self):
        """Clamped/free 90-degree corner, plane strain, nu = 0.3."""
        from omar_pfem.verification_agent.singularity import b1_corner_exponent
        return b1_corner_exponent(0.3)

    def strategies(self, qoi):
        if len(self.strategy_rows) < 2:
            return None
        return [r[qoi] for r in self.strategy_rows], None


class B2Adapter:
    """HELD-OUT problem: the agent's rules, gates and tolerances were designed
    on B1 and B3 and are applied to B2 unchanged (b2_ladder.py)."""
    geometry = "quarter_ring"
    description = (
        "B2: 2D quarter ring, inner radius 1, outer radius 2, plane strain, compressible "
        "neo-Hookean (same psi as B1), total Lagrangian. E(theta) = 1000 + 200 sin(2 theta), "
        "nu = 0.3 everywhere, evaluated at each Gauss point. Internal pressure "
        "p(theta) = 5 + 2 sin(2 theta) on the inner arc (follower direction: local chord "
        "normal of the reference edge), 10 equal load increments, Newton tol 1e-8; "
        "symmetry on both straight edges (u_y = 0 on y = 0, u_x = 0 on x = 0), outer arc "
        "traction-free. By symmetry the problem is a smooth sector of a full ring: no corner "
        "singularity. Q4 elements, 2x2 Gauss, curved arcs approximated by straight chords. "
        "Structured (theta, r) meshes N x N nodes, N = 7, 13, 25, 49, 97, 193 (h = 1/(N-1), "
        "ratio 2); strategies at N = 49: uniform and radial grading toward the inner arc, "
        "r_grading = 1.5, 2.0 (same node count). QoIs: max_disp_mag = max nodal |u|; "
        "u/v_inner at (theta = pi/3, r = 1) and u/v_mid at (theta = pi/3, r = 1.5), both mesh "
        "nodes on every uniform level; strain_energy = stored energy; interior_avg_PK1_frob "
        "= detJ-weighted mean of |P|_F over the Gauss points inside the disc of radius 0.15 "
        "centred at (theta = pi/4, r = 1.5); interior_gk_PK1_frob = the same with the smooth "
        "weight exp(-(d/0.1)^2); peak_PK1_frob = max |P|_F over all Gauss points. No "
        "surrogate errors are recorded for this case, so the class ceilings apply.")
    qois = ("max_disp_mag", "u_inner", "v_inner", "u_mid", "v_mid", "strain_energy",
            "interior_avg_PK1_frob", "interior_gk_PK1_frob", "peak_PK1_frob")

    def __init__(self, strategy_N=49):
        from omar_pfem.verification_agent.b2_ladder import CACHE
        rows = json.load(open(CACHE)) if os.path.exists(CACHE) else []
        rows = [r for r in rows if r["newton_tol"] == 1e-8]
        self.rows = sorted((r for r in rows if r["r_grading"] == 1.0), key=lambda r: r["N"])
        self.hs = [r["h"] for r in self.rows]
        self.strategy_rows = sorted((r for r in rows if r["N"] == strategy_N),
                                    key=lambda r: r["r_grading"])

    def series(self, qoi):
        return self.hs, [r[qoi] for r in self.rows], None

    def strategies(self, qoi):
        if len(self.strategy_rows) < 2:
            return None
        return [r[qoi] for r in self.strategy_rows], None


class CookAdapter:
    """HELD-OUT problem WITH a corner singularity (cook_ladder.py). Spec,
    description and rule predictions were fixed and committed before any
    Cook solve (cook_preregistration.json); rules unchanged."""
    geometry = "cook_membrane"
    description = (
        "Cook's membrane: 2D tapered panel with vertices (0,0), (48,44), (48,60), (0,44), "
        "plane strain, compressible neo-Hookean (same psi as B1), E = 1000, nu = 0.3 "
        "(constant), total Lagrangian. Left edge x = 0 clamped; right edge x = 48 loaded by a "
        "uniform dead shear traction t_y = 5 (total 80) in 10 equal increments; top and "
        "bottom edges traction-free. Q4 elements, 2x2 Gauss, on an N x N mapped mesh of the "
        "parametric square, N = 4, 8, ..., 256 (h = 1/N, ratio 2). Strategies at N = 64: "
        "uniform and grading toward the clamped edge and both left corners (a = 0.5, 0.8; "
        "corner elements 2x and 5x smaller, same element count). QoIs: u_tip, v_tip = "
        "displacement of the corner (48,60), a mesh node on every level; max_disp_mag over "
        "the nodes; strain_energy = stored energy; interior_avg_PK1_frob = detJ-weighted mean "
        "of |P|_F over the Gauss points inside the disc of radius 4 centred at (24,40); "
        "interior_gk_PK1_frob = the same with the smooth weight exp(-(d/3)^2); peak_PK1_frob "
        "= max |P|_F over all Gauss points. Corner theory (singularity.py, Williams "
        "eigenproblem, clamped/free, plane strain, nu = 0.3): the 47.5-degree corner at (0,0) "
        "is regular; the 108.4-degree corner at (0,44) has lam = 0.6442, so energy converges "
        "at order 2 lam = 1.29, point values at >= 1.29 and the peak grows like h^(lam-1) = "
        "h^-0.356. Solver: scikit-fem with the residual/tangent of independent_solver.py. "
        "No surrogate errors are recorded, so the class ceilings apply.")
    qois = ("u_tip", "v_tip", "max_disp_mag", "strain_energy", "interior_avg_PK1_frob",
            "interior_gk_PK1_frob", "peak_PK1_frob")

    def __init__(self, strategy_N=64):
        from omar_pfem.verification_agent.cook_ladder import CACHE
        rows = json.load(open(CACHE)) if os.path.exists(CACHE) else []
        self.rows = sorted((r for r in rows if r["grading"] == 0.0), key=lambda r: r["N"])
        self.hs = [r["h"] for r in self.rows]
        self.strategy_rows = sorted((r for r in rows if r["N"] == strategy_N),
                                    key=lambda r: r["grading"])

    def series(self, qoi):
        return self.hs, [r[qoi] for r in self.rows], None

    def singular_exponent(self):
        from omar_pfem.verification_agent.cook_ladder import cook_corner_exponent
        return cook_corner_exponent()

    def strategies(self, qoi):
        if len(self.strategy_rows) < 2:
            return None
        return [r[qoi] for r in self.strategy_rows], None


class PlateHoleAdapter:
    """HELD-OUT problem from the OpenPINN/PINO paper (plate_hole.py); spec,
    description and predictions committed before any solve
    (plate_hole_preregistration.json); rules unchanged."""
    geometry = "plate_hole"
    description = (
        "Plate with a hole: 2 m x 2 m plate, circular hole of radius 0.25 m centred at (1,1), "
        "plane stress, small-strain linear elasticity, E = 210 GPa, nu = 0.3. Left edge x = 0 "
        "fixed; right edge x = 2 loaded by a uniform traction t_x = 1 MPa; top, bottom and hole "
        "traction-free. Q2 (9-node) elements on a structured polar mesh around the hole "
        "(8m elements around, 3m radially, m = 2, 4, ..., 64, h = 1/m), mid-side nodes on the "
        "exact circle; strategies at m = 16: radial grading toward the hole g = 1.5, 2.0. QoIs: "
        "u_right_mid = u_x at (2,1); v_top_right = u_y at (2,2); max_disp_mag over the nodes; "
        "strain_energy = 1/2 u.K u; hole_avg_sigma_xx = weighted mean of sigma_xx over the "
        "Gauss points within 0.1 of the hole top (1,1.25); hole_gk_avg_sigma_xx = the same with "
        "the weight exp(-(d/0.05)^2); hole_peak_sigma_xx = max sigma_xx within 0.1 of the hole "
        "top (a smooth finite-radius concentration); peak_von_mises = max von Mises over all "
        "Gauss points. The fixed left edge meets the free top and bottom edges at 90-degree "
        "clamped/free corners (singular in plane stress); the hole is far from them. No "
        "surrogate errors are recorded for this case; the OpenPINN surrogate errors (relative "
        "L2 of displacement 3.5-6.3%) are against a single FEM mesh.")
    qois = ("u_right_mid", "v_top_right", "max_disp_mag", "strain_energy", "hole_avg_sigma_xx",
            "hole_gk_avg_sigma_xx", "hole_peak_sigma_xx", "peak_von_mises")

    def __init__(self, strategy_m=16):
        from omar_pfem.verification_agent.plate_hole import CACHE
        rows = json.load(open(CACHE)) if os.path.exists(CACHE) else []
        self.rows = sorted((r for r in rows if r["g"] == 1.0), key=lambda r: r["m"])
        self.hs = [r["h"] for r in self.rows]
        self.strategy_rows = sorted((r for r in rows if r["m"] == strategy_m),
                                    key=lambda r: r["g"])

    def series(self, qoi):
        return self.hs, [r[qoi] for r in self.rows], None

    @property
    def family(self):
        # for the diagnosis engine (post hoc): element counts and det F per level
        return [{"n_elem": r["n_elem"], "min_detF": r.get("posthoc_min_detF")}
                for r in self.rows]

    def singular_exponent(self):
        from omar_pfem.verification_agent.plate_hole import corner_exponent
        return corner_exponent()

    def strategies(self, qoi):
        if len(self.strategy_rows) < 2:
            return None
        return [r[qoi] for r in self.strategy_rows], None


class B3Adapter:
    geometry = "bushing_groove"
    description = (
        "B3: 3D bushing, inner radius 0.5, outer radius 1.0, length 1.0 (z in [0,1]); half "
        "model y >= 0 with symmetry plane y=0. A circumferential groove is cut into the INNER "
        "surface: r_in(z) = 0.5 - 0.1 (1 + cos(pi (z - 0.5)/0.15)) for |z - 0.5| < 0.15, "
        "depth 0.20, half-width 0.15, smooth cosine profile (C1 at the shoulders, NOT a "
        "sharp notch); root curvature radius rho = 0.0228. Loading: the whole inner surface "
        "(groove included) is RIGIDLY ROTATED by phi = 0.05 rad about the y axis through "
        "(0,0,0.5) (a tilting rotation, not torsion about the bushing axis z), applied in 10 "
        "equal increments; outer surface fixed; end faces z=0 and z=1 traction-free, so "
        "the fixed/rotated surfaces meet free end faces along four circular edges "
        "(possible edge singularities, which can pollute global convergence rates). The "
        "groove root, being part of the rotated inner surface, lies on a prescribed-"
        "displacement surface. "
        "Compressible neo-Hookean (same psi as the 2D "
        "form, 3D invariants), E=1000, nu=0.45. Elements: 20-node serendipity hexahedra "
        "(Q20) with curved-surface midside nodes, integrated with 2x2x2 = 8 Gauss points "
        "(the reduced rule for Q20, not the full 27); checked against the full 27-point "
        "rule at 2,376 / 10,200 el: moment and energy differ by 0.22% / 0.08%, gk2 "
        "normal averages by <= 1.3% / 0.33% of the stress scale (shrinking with h); the "
        "Gauss-point pressure scatter is NOT reduced by full integration (std 4.0 vs 4.0, "
        "0.61 vs 0.43), so it is not a reduced-integration spurious mode. Newton-CG, rtol=atol=1e-8 "
        "(tightening to 1e-11 changes QoIs by <= 4e-14); min det F 0.911-0.948. Global "
        "family of 8 levels, 2,376-59,400 elements, refinement ratios (n_elem ratio)^(1/3) "
        "= 1.34, 1.21, 1.16, 1.14, 1.12, 1.11, 1.10 (NOT constant; the Celik order uses the two "
        "actual ratios r21, r32). Strategies at 10,200 elements: uniform and AXIAL-only "
        "clustering of the z layers toward z=0.5 by sinh grading, z_beta = 1, 2, 3 "
        "(theta and r divisions unchanged, same element count), and RADIAL-only "
        "clustering of the r layers toward the inner (groove) surface, r_grading = 1.1, "
        "1.2, 1.3 (t -> t^r_grading; stronger grading inverts elements at the groove), "
        "solved with a direct linear solver (same Newton tolerance; reproduces the CG "
        "result to 1e-12). "
        "Region QoIs: Cauchy stress "
        "(global Cartesian, or cylindrical about z), averaged with Gauss-point volume "
        "weights over the Gauss points inside a sphere of FIXED physical radius "
        "R0 = 2 rho = 0.0456 centred at (0.3, 0, 0.5) = the groove root on the inner "
        "surface, on the symmetry plane (theta = 0), so the sphere is cut by both the "
        "groove surface and y=0 (about a quarter sphere); 2-50 Gauss points inside from "
        "the coarsest to the finest level. At theta ~ 0, cyl_rr ~ sigma_xx, cyl_tt ~ "
        "sigma_yy, cyl_rz ~ sigma_xz and cyl_zz = sigma_zz exactly. *_max = max |component| "
        "over the Gauss points in the sphere. Region QoIs are judged RELATIVE TO A FIXED "
        "STRESS SCALE (the finest-level region average of the Frobenius norm, 0.909), not "
        "to their own magnitude, so near-zero components are not amplified; global QoIs "
        "(reaction_moment_y about the rotation axis, strain_energy; half model) are judged "
        "relative to themselves; strain_energy and reaction_moment_y are linked "
        "(W ~ 1/2 M phi, 0.06% at 2,376 el), so they count as ONE piece of evidence. "
        "Surrogate errors these references must resolve (best recorded, per QoI): "
        "reaction moment 1.38%, strain energy 4.13%, groove-region stresses >= 600% "
        "(every component, every checkpoint); tolerances are set by code from these "
        "(requirements.py), not by the plan. "
        "gk2_* QoIs: the same stress measures averaged with the SMOOTH weight "
        "exp(-(d/l)^2), l = 2 R0 = 0.0912, d = distance to the groove-root point, over "
        "all Gauss points within 0.25 (weight there < 6e-4) instead of the sharp sphere: "
        "a mesh-independent functional (effective ~35-916 Gauss points from coarsest to "
        "finest); judged relative to its own Frobenius-average scale 3.098. A proposed "
        "alternative to the sharp sphere, NOT the definition used in the manuscript.")
    CYL = ("rr", "tt", "zz", "rt", "rz", "tz")
    KERNEL_PREFIX = "gk2_"
    KERNEL_QOIS = ("sigma_xx_avg", "sigma_yy_avg", "sigma_zz_avg", "pressure_avg",
                   "cyl_rr_avg", "cyl_tt_avg", "cyl_zz_avg", "cyl_rz_avg", "sigma_xz_avg",
                   "sigma_frob_avg", "von_mises_avg")
    GLOBAL = ("reaction_moment_y", "strain_energy")

    def __init__(self):
        ext = os.path.join(HERE, "b3_extended.json")
        rows = json.load(open(ext)) if os.path.exists(ext) else []
        fam = sorted((r for r in rows if r["z_beta"] is None and r["rtol"] == 1e-8),
                     key=lambda r: r["n_elem"])
        if len(fam) >= 7:
            # extended re-solve, checked to reproduce the stored data to <=5e-13
            self.family = fam
            base = next(r for r in fam if r["n_elem"] == 10200)
            self.zbeta = [("uniform", base)] + [
                (f"z_beta={r['z_beta']}", r) for r in sorted(
                    (r for r in rows if r["z_beta"] is not None), key=lambda r: r["z_beta"])]
            # radial strategy at the same resolution (b3_global.py radial), if solved
            rad = os.path.join(HERE, "b3_radial.json")
            if os.path.exists(rad):
                self.zbeta += [(f"r_grading={r['r_grading']}", r) for r in sorted(
                    json.load(open(rad)), key=lambda r: r["r_grading"])]
            self.extended = True
        else:
            from omar_pfem.verification_agent.agent import _load_b3_family, _load_b3_zbeta_10200
            self.family = _load_b3_family()
            self.zbeta = _load_b3_zbeta_10200(self.family)
            self.extended = False
        self.hs = [g["n_elem"] ** (-1.0 / 3.0) for g in self.family]
        self.scale = self.family[-1]["region_avg_sigma_frob"]
        self.get = {"sigma_frob_avg": lambda g: g["region_avg_sigma_frob"]}
        for c in ("xx", "yy", "zz", "xy", "xz", "yz"):
            for st in ("avg", "max"):
                self.get[f"sigma_{c}_{st}"] = lambda g, c=c, st=st: g["per_component"][c][st]
        if self.extended:
            for k in self.GLOBAL:
                self.get[k] = lambda g, k=k: g[k]
            self.get["von_mises_avg"] = lambda g: g["region_avg_von_mises"]
            self.get["von_mises_max"] = lambda g: g["region_max_von_mises"]
            self.get["pressure_avg"] = lambda g: g["region_avg_pressure"]
            for c in self.CYL:
                for st in ("avg", "max"):
                    self.get[f"cyl_{c}_{st}"] = lambda g, c=c, st=st: g["per_cylindrical"][c][st]
            self._add_kernel_qois()
        self.qois = tuple(self.get)

    def _add_kernel_qois(self):
        """Gaussian-kernel region averages (l = 2 R0) from the Gauss-point dumps
        of the same solutions (b3_gauss_near_groove/, region_study.py)."""
        from omar_pfem.verification_agent import region_study as RS
        try:
            fam, strat = RS.load()
        except Exception:
            return
        import glob
        rad = []
        for f in sorted(glob.glob(os.path.join(HERE, "b3_gauss_radial", "*_rg*.npz"))):
            if f.endswith("_surface.npz"):
                continue
            d = np.load(f)
            rec = {k: d[k] for k in d.files}
            rec["n_elem"], rec["z_beta"] = int(d["n_elem"]), None
            rec["r_grading"] = float(os.path.basename(f).split("_rg")[1][:-4])
            rad.append(rec)
        key = lambda r: (int(r["n_elem"]), r["z_beta"], float(r.get("r_grading", 1.0)))
        R0 = float(fam[0]["radius"])
        vals = {}
        for rec in fam + strat + rad:
            w, _ = RS.weights(rec, "gauss", 2 * R0)
            q = RS.qois(rec, w)
            q["sigma_frob_avg"] = q.pop("frob_avg")
            vals[key(rec)] = q
        if not all(key(g) in vals for g in self.family + [z for _, z in self.zbeta]):
            return
        self.kernel_values = vals
        for name in self.KERNEL_QOIS:
            self.get[self.KERNEL_PREFIX + name] = (
                lambda g, name=name: self.kernel_values[key(g)][name])
        self.kernel_scale = vals[key(self.family[-1])]["sigma_frob_avg"]

    def _scale(self, qoi):
        if qoi in self.GLOBAL:
            return None                       # globals: self-relative
        if qoi.startswith(self.KERNEL_PREFIX):
            return self.kernel_scale
        return self.scale

    def series(self, qoi):
        return self.hs, [self.get[qoi](g) for g in self.family], self._scale(qoi)

    def strategies(self, qoi):
        return [self.get[qoi](z) for _, z in self.zbeta], self._scale(qoi)

    def admissibility(self):
        rows = self.family + [z for _, z in self.zbeta]
        return {"min_detF": min(r.get("min_detF", float("nan")) for r in rows)}


ADAPTERS = {"unit_square": B1Adapter, "quarter_ring": B2Adapter, "bushing_groove": B3Adapter,
            "cook_membrane": CookAdapter, "plate_hole": PlateHoleAdapter}
ADAPTER_VOCAB = {
    "cook_membrane": (CookAdapter.description, CookAdapter.qois),
    "plate_hole": (PlateHoleAdapter.description, PlateHoleAdapter.qois),
    "unit_square": (B1Adapter.description, B1Adapter.qois),
    "quarter_ring": (B2Adapter.description, B2Adapter.qois),
    "bushing_groove": (B3Adapter.description,
                       ("sigma_frob_avg",) + tuple(f"sigma_{c}_{s}" for c in
                                                   ("xx", "yy", "zz", "xy", "xz", "yz")
                                                   for s in ("avg", "max"))
                       + B3Adapter.GLOBAL + ("von_mises_avg", "von_mises_max", "pressure_avg")
                       + tuple(f"cyl_{c}_{s}" for c in B3Adapter.CYL for s in ("avg", "max"))
                       + tuple(B3Adapter.KERNEL_PREFIX + q for q in B3Adapter.KERNEL_QOIS)),
}


# ----------------------------------------------------------------------------
# Execution of a validated plan (deterministic)
# ----------------------------------------------------------------------------


CONV_CHECKS = gates.CONV_CHECKS
combine = gates.combine_verdicts   # single policy, shared with adaptive.py


def convergence_params(plan):
    """{qoi: {check: params}} for the convergence checks of a plan; used to
    anchor anchored_robustness on the same tolerances."""
    out = {}
    for c in plan["checks"]:
        if c["check"] in CONV_CHECKS:
            for q in c["qois"]:
                out.setdefault(q, {})[c["check"]] = c["params"]
    return out


def set_code_tolerances(plan, spec_d):
    """Replace the tolerances of every (check, QoI) by trees.check_params and
    split checks so each carries its own QoI's values. Returns the new plan
    and the list of pairs whose LLM-chosen values differed."""
    geometry = spec_d.get("geometry")
    checks, changed = [], []
    for c in plan["checks"]:
        if c["check"] not in R.CHECK_CATALOGUE:
            checks.append(c)          # left for validate_plan to reject
            continue
        groups = {}
        for q in c["qois"]:
            p = trees.check_params(c["check"], q, geometry)
            if c["params"] != p:
                changed.append({"check": c["check"], "qoi": q, "llm": dict(c["params"]),
                                "code": p})
            groups.setdefault(tuple(sorted(p.items())), []).append(q)
        for p, qs in groups.items():
            checks.append({**c, "qois": qs, "params": dict(p)})
    return {**plan, "checks": checks}, changed


def run_check(adapter, check, qoi, params, element=None, conv_params=None):
    """Execute one catalogued check. The order window of the convergence
    checks comes from the element (trees.order_window), never from the plan.
    conv_params: {check: params} of the QoI's convergence checks (for the
    anchor of anchored_robustness)."""
    window = trees.order_window(element)
    if check == "qoi_convergence":
        hs, vals, scale = adapter.series(qoi)
        return gates.qoi_convergence_gate(hs, vals, scale=scale, order_window=window, **params)
    if check == "qoi_convergence_multilevel":
        hs, vals, scale = adapter.series(qoi)
        return gates.multilevel_convergence_gate(hs, vals, scale=scale, order_window=window,
                                                 **params)
    if check == "singular_order":
        hs, vals, _ = adapter.series(qoi)
        lam = getattr(adapter, "singular_exponent", lambda: None)()
        k = trees.ELEMENT_DEGREE.get(element, 1) if element is not None else 1
        # POST HOC switch (default off; see gates.singular_order_gate): degree k
        # with lam < k (same rates: the corner limits them), pre-asymptotic label
        posthoc = os.environ.get("VERIF_POSTHOC_SINGULAR") == "1"
        if k != 1 and not (posthoc and lam is not None and lam < k):
            return {"gate": "singular_order", "verdict": "INCONCLUSIVE",
                    "reason": "theory implemented for degree-1 elements only"}
        return gates.singular_order_gate(hs, vals, lam, trees.classify_qoi(qoi), degree=k,
                                         preasymptotic=posthoc)
    if check == "anchored_robustness":
        s = adapter.strategies(qoi)
        if s is None:
            return {"gate": "anchored_robustness", "verdict": "INCONCLUSIVE",
                    "reason": "no alternative refinement strategy computed for this problem"}
        hs, vals, scale = adapter.series(qoi)
        cp = conv_params or {}
        conv = gates.qoi_convergence_gate(hs, vals, scale=scale, order_window=window,
                                          **cp.get("qoi_convergence", {}))
        ml = gates.multilevel_convergence_gate(hs, vals, scale=scale, order_window=window,
                                               **cp.get("qoi_convergence_multilevel", {}))
        if conv["verdict"] != "PASS" and ml["verdict"] != "PASS":
            # a family that did not converge has no limit: comparing the
            # strategies with its finest value would say nothing either way
            return {"gate": "anchored_robustness", "verdict": "INCONCLUSIVE",
                    "anchor_kind": "none",
                    "reason": "refinement family not converged (3-level: "
                              f"{conv['verdict']}, 5-level: {ml['verdict']}), so there is no "
                              "limit to anchor the strategies to; uninformative"}
        # anchor: 3-level Richardson value if that has a rate, else the
        # multilevel fit limit if that has a rate, else the finest value
        anchor, kind = gates.family_anchor(conv)
        if conv.get("evidence") != "rate" and ml.get("evidence") == "rate":
            anchor, kind = gates.family_anchor(ml)
        return gates.anchored_robustness_gate(s[0], anchor, scale=s[1], anchor_kind=kind,
                                              **params)
    if check == "strategy_robustness":
        s = adapter.strategies(qoi)
        if s is None:
            return {"gate": "strategy_robustness", "verdict": "INCONCLUSIVE",
                    "reason": "no alternative refinement strategy computed for this problem"}
        vals, scale = s
        return gates.strategy_robustness_gate(vals, scale=scale, **params)
    if check == "mms_rates":
        return {"gate": "mms_rates", "verdict": "INCONCLUSIVE",
                "reason": "run `agent mms` for code verification; not a per-QoI check"}
    raise ValueError(check)


def orchestrate(backend, problem_text, log=print, trace=None):
    """`trace` (a dict) is filled in place, so a caller can save it even if
    a later step raises."""
    trace = {} if trace is None else trace
    trace["problem_text"] = problem_text

    # 1. formalize
    vocab = "\n".join(f"- geometry '{g}': {d}\n  available QoIs: {list(q)}"
                      for g, (d, q) in ADAPTER_VOCAB.items())
    spec_d, errs = R.formalize(backend, problem_text + "\n\nAVAILABLE DATA ADAPTERS "
                               "(geometry must be one of these; QoIs must come from its "
                               "list):\n" + vocab)
    if spec_d["geometry"] not in ADAPTERS:
        errs.append(f"no data adapter for geometry '{spec_d['geometry']}'")
    else:
        allowed = ADAPTER_VOCAB[spec_d["geometry"]][1]
        errs += [f"QoI '{q}' not available for {spec_d['geometry']}"
                 for q in spec_d["qois"] if q not in allowed]
    trace["spec"], trace["spec_errors"] = spec_d, errs
    log(f"[formalizer] {spec_d['name']}: {len(spec_d['qois'])} QoIs, "
        f"regularity={spec_d['regularity']}")
    for w in spec_d["well_posedness"]:
        log(f"   well-posedness: {w}")
    for q in spec_d["open_questions"]:
        log(f"   open question:  {q}")
    if errs:
        log(f"[guard] spec rejected: {errs}")
        return trace
    spec = ProblemSpec(**{k: (tuple(v) if k == "qois" else v) for k, v in spec_d.items()
                          if k in ProblemSpec.__dataclass_fields__})

    # 2. retrieve
    neigh = [{"similarity": round(s, 3), "problem": r["problem"]["name"],
              "verdicts": {k: v["verdict"] for k, v in r["verdicts"].items()}}
             for s, r in repository.nearest(spec, k=5)]
    trace["neighbours"] = neigh
    log(f"[repository] nearest: {[(n['problem'], n['similarity']) for n in neigh]}")

    # 3. propose / validate / critique, at most one revision
    baseline = trees.rule_plan(spec_d)
    trace["rule_plan"], trace["rule_analysis"] = baseline, trees.apply_rules(spec_d)
    log("[rules] baseline requires: " + "; ".join(
        f"{c['check']} on {len(c['qois'])} QoIs" for c in baseline["checks"])
        + f"; predicts FAIL: {[e['qoi'] for e in baseline['expected_failures']]}")
    def proposed(extra):
        """Propose, then replace every tolerance by the code-derived one
        (trees.check_params) -- tolerances are never the LLM's choice -- and
        validate what remains: catalogue, QoIs, rule-required checks."""
        p, _ = R.propose(backend, spec_d, extra)
        p, changed = set_code_tolerances(p, spec_d)
        errs = R.validate_plan(p, spec_d["qois"]) + trees.plan_rule_errors(p, spec_d)
        return p, errs, changed

    plan, perrs, overridden = proposed(neigh)
    crit = R.critique(backend, spec_d, plan) if not perrs else None
    if perrs or not crit["approve"]:
        feedback = {"validation_errors": perrs, "critic": crit}
        log(f"[guard/critic] revision requested: {json.dumps(feedback)}")
        plan, perrs, overridden = proposed(neigh + [{"revise_previous_plan": feedback}])
        crit = R.critique(backend, spec_d, plan) if not perrs else None
    trace["tolerances_overridden"] = overridden
    if overridden:
        log(f"[code] tolerances set by code, LLM values replaced for {len(overridden)} "
            "check/QoI pairs")
    # Rules are authoritative: if the only remaining problems are checks the
    # rules require but the LLM left out, add them (code tolerances) and
    # record the augmentation, instead of discarding the whole plan.
    rule_only = perrs and all(e.startswith("rules ") for e in perrs)
    if rule_only:
        need = trees.required_checks(plan, spec_d)   # R1-R6 plus plan-level R7
        added = []
        for q, r in need.items():
            have = {c["check"] for c in plan["checks"] if q in c["qois"]}
            for chk in sorted(r["required"]):
                if chk not in have:
                    plan["checks"].append({"check": chk, "qois": [q],
                                           "params": trees.check_params(
                                               chk, q, spec_d.get("geometry")),
                                           "rationale": f"added by rules {r['rules']}"})
                    added.append(f"{chk}:{q}")
        trace["rule_augmented"] = added
        log(f"[rules] plan augmented with required checks the LLM omitted: {added}")
        perrs = R.validate_plan(plan, spec_d["qois"]) + trees.plan_rule_errors(plan, spec_d)
    if not perrs and crit is None:
        # the final plan has not been reviewed yet (e.g. after augmentation)
        crit = R.critique(backend, spec_d, plan)
    trace.update(plan=plan, plan_errors=perrs, critique=crit)
    if perrs:
        log(f"[guard] plan rejected after revision: {perrs}")
        return trace
    if not crit["approve"]:
        log(f"[critic] plan still not approved: {crit['objections']}; executing anyway, "
            "objections stored with the record")
    log(f"[proposer] {len(plan['checks'])} checks; expected failures: "
        f"{[e['qoi'] for e in plan['expected_failures']]}")

    # 4. execute (code only)
    adapter = ADAPTERS[spec.geometry]()
    per_qoi = {}
    conv_params = convergence_params(plan)
    for c in plan["checks"]:
        for q in c["qois"]:
            g = run_check(adapter, c["check"], q, c["params"], element=spec.element,
                          conv_params=conv_params.get(q))
            per_qoi.setdefault(q, {})[c["check"]] = g
    gate_summary = {}
    for q, verdicts in per_qoi.items():
        sub = ProblemSpec(**{**spec.to_dict(), "qois": (q,), "name": f"{spec.name} {q}"})
        rec = repository.record(sub, method={"orchestrated": True, "plan": plan,
                                             "critic": crit},
                                observables={}, verdicts=verdicts)
        overall, evidence, ev_src = combine(verdicts)
        role = trace["rule_analysis"].get(q, {}).get("role", "candidate")
        gate_summary[q] = {"verdict": overall, "convergence_evidence": evidence, "role": role,
                           "evidence_from": ev_src,
                           "gates": {k: {kk: vv for kk, vv in v.items()
                                         if kk in ("verdict", "reason", "relative_spread",
                                                   "observed_order", "gci_fine", "evidence",
                                                   "finest_three", "anchor_kind", "n_levels",
                                                   "fit_order", "fit_limit",
                                                   "uncertainty_fine", "failed_criteria",
                                                   "leave_one_out_orders")}
                                     for k, v in verdicts.items()}}
        log(f"[gates] {q:<24} {overall:<13} evidence={evidence:<5} accepted={rec['accepted']}"
            + ("  (diagnostic, not a reference candidate)" if role == "diagnostic" else ""))
    trace["gate_summary"] = gate_summary

    # prediction score: did the proposer anticipate the failures?
    failed = {q for q, g in gate_summary.items() if g["verdict"] == "FAIL"}
    trace["prediction"] = {}
    for who, exp in (("llm", plan["expected_failures"]),
                     ("rules", baseline["expected_failures"])):
        predicted = {e["qoi"] for e in exp}
        trace["prediction"][who] = {"predicted_fail": sorted(predicted),
                                    "actual_fail": sorted(failed),
                                    "missed": sorted(failed - predicted),
                                    "false_alarm": sorted(predicted - failed)}
        log(f"[score:{who}] predicted FAIL {sorted(predicted)} | actual FAIL {sorted(failed)}"
            f" | missed {sorted(failed - predicted)} | false alarm {sorted(predicted - failed)}")

    # 5. advise and audit
    advice, flags = R.advise(backend, spec_d, gate_summary)
    trace.update(advice=advice, advisor_flags=flags)
    log(f"[advisor] {advice['summary']}")
    log(f"[audit] {'advisor consistent with gates' if not flags else flags}")
    return trace
