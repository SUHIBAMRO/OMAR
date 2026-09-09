"""Method of manufactured solutions: Q4, Q9 and the operator against one
exact analytic solution. The advisor's point 9 (round 5), and round 6's
"this is the last thing to do".

His design, quoted: "We can compare Q4, Q9 and the physics-informed
Transolver against exactly the same analytical solution in L2, H1 and energy
norms and also examine stress errors."

--------------------------------------------------------------------------
THE FORK HE LEFT OPEN, AND WHY THIS MODULE RESOLVES IT THE WAY IT DOES
--------------------------------------------------------------------------
Omar asked whether to drive the study with a body force or to look for a
homogeneous (body-force-free) exact solution. Timon did not answer; he said
a parametrised family is the ideal and that it needs more discussion. The
question has to be settled before any code can run, so here is the reasoning,
stated openly so it can be overruled:

A body-force-free exact solution of finite-strain elasticity on a simple
domain is, in practice, a homogeneous deformation -- a constant deformation
gradient. That is exactly the case a bilinear Q4 element reproduces to
machine precision. The study would measure round-off, both orders would
"converge" instantly, and it would distinguish nothing. Non-trivial
homogeneous solutions exist (inflation of a cylinder, for example) but only
for specific geometry/material pairs, which would mean abandoning either B1
or Neo-Hookean, both of which Timon named.

Manufacturing the solution instead -- choose u*, substitute it into the
governing equations, and apply whatever body force the residual demands --
keeps the geometry, the material and the discretization exactly as they are
in the rest of the report, and it is the standard method for precisely this
purpose. So: MMS with a body force.

The chosen u* vanishes on the entire boundary. That is not cosmetic: the
matrix-free solver constrains fixed DOFs to zero, so a u* that vanishes on
the boundary makes homogeneous Dirichlet conditions the EXACT boundary
condition for the manufactured problem, and no inhomogeneous-Dirichlet
support has to be added to a solver that the rest of the report depends on.

--------------------------------------------------------------------------
WHAT VALIDATES THIS FILE
--------------------------------------------------------------------------
An MMS study is self-validating, which is its main virtue: if the body force
is wrong by so much as a sign or a factor, the discrete solution converges to
the wrong function and the observed convergence RATES collapse. Correct rates
are strong evidence that every piece -- the derivation, the assembly, the
solver, the error norms -- is right. Expected for a smooth solution:

    Q4:  L2 ~ h^2,  H1 semi-norm ~ h^1
    Q9:  L2 ~ h^3,  H1 semi-norm ~ h^2

The body force is derived by nested automatic differentiation rather than by
hand: u* -> F = I + grad u* -> P = dpsi/dF -> b = -Div P. Hand-deriving the
divergence of a Neo-Hookean first Piola-Kirchhoff stress is several lines of
tensor algebra and an easy place to make a silent error. `--verify` checks
the autodiff divergence against a central finite difference before solving.

Usage
-----
    python -m omar_pfem.mms_study --verify            # checks only, seconds
    python -m omar_pfem.mms_study --orders Q4,Q9 --Ns 5,9,17,33 \
        --material neo_hookean --out_json mms_B1_neo_hookean.json
"""
import argparse
import json
import math
import os
import time

import numpy as np
import torch

from omar_pfem.materials_torch import get_material_fns as get_material_fns_torch
from omar_pfem.matrix_free_solver import (
    precompute_shape_data, element_energy_order_agnostic, solve_matrix_free)
from omar_pfem.high_dof_convergence_study import (
    shape_Q4_batched, shape_Q9_batched, fit_convergence_rate)
from omar_pfem.data.q9_element import generate_grid_Q9
from omar_pfem.run_manifest import write_manifest


# ----------------------------------------------------------------------
# The manufactured solution family
# ----------------------------------------------------------------------
# u_x(x,y) = alpha     * sum_i w_i sin(m_i pi x) sin(n_i pi y)   (modes[0])
# u_y(x,y) = alpha*beta * sum_j w_j sin(p_j pi x) sin(q_j pi y)  (modes[1])
#
# MODES_SINGLE is the ORIGINAL field every existing result (Tables 22-24,
# the (alpha,beta) family mms_operator.py samples, mms_family_fem.py) was
# built from, and stays the default for every function below -- nothing
# already published silently changes shape. MODES_RICHER is a NEW, EXTRA
# option (Timon's round-8, point 6: "a richer family of MS containing the
# sum of several sine/cosine spatial modes while preserving the boundary
# conditions"), selected explicitly via --richer_family, never the
# implicit default -- see main()'s own comment on why.
#
# Every term sin(k pi x) with k a positive integer vanishes at x=0 and
# x=1 -- same for y -- so ANY sum of such terms vanishes exactly on all
# four edges of the unit square: homogeneous Dirichlet stays the EXACT
# boundary condition for MODES_RICHER too, no inhomogeneous-Dirichlet
# support needed in the solver, same as MODES_SINGLE.
#
# Both mode sets use DIFFERENT terms for the two components (not a
# beta-scaled copy of one field), for the same reason beta != 1 does:
# a bug that treats the two components alike, or that only gets one
# mode's derivative right while the others coincidentally cancel, cannot
# hide behind a single mode's correct-looking result.
MODES_SINGLE = (((1, 1, 1.0),), ((1, 1, 1.0),))
MODES_RICHER = (
    ((1, 1, 1.0), (2, 1, 0.5), (1, 3, 0.3)),
    ((1, 1, 1.0), (3, 2, -0.4), (2, 2, 0.2)),
)
DEFAULT_MODES = MODES_SINGLE
DEFAULT_ALPHA, DEFAULT_BETA = 0.05, 0.7


def _mode_sum(x, y, modes):
    s = torch.zeros_like(x)
    for m, n, w in modes:
        s = s + w * torch.sin(m * math.pi * x) * torch.sin(n * math.pi * y)
    return s


def _mode_sum_grad(x, y, modes):
    """d/dx and d/dy of _mode_sum(x, y, modes), term by term."""
    pi = math.pi
    dsdx = torch.zeros_like(x)
    dsdy = torch.zeros_like(x)
    for m, n, w in modes:
        dsdx = dsdx + w * m * pi * torch.cos(m * pi * x) * torch.sin(n * pi * y)
        dsdy = dsdy + w * n * pi * torch.sin(m * pi * x) * torch.cos(n * pi * y)
    return dsdx, dsdy


def _mode_sum_str(modes):
    terms = [f"{w:+.2g}*sin({m} pi x)sin({n} pi y)" for m, n, w in modes]
    return " ".join(terms).lstrip("+")


def manufactured_solution_str(alpha, beta, modes=DEFAULT_MODES):
    """Human-readable u* description for logging and the JSON report --
    kept in one place so the printed formula and the saved metadata can
    never drift apart from each other or from `modes`."""
    return (f"u_x = {alpha}*[{_mode_sum_str(modes[0])}]; "
            f"u_y = {alpha}*{beta}*[{_mode_sum_str(modes[1])}]; "
            f"vanishes on the boundary")


def u_exact(xy, alpha=DEFAULT_ALPHA, beta=DEFAULT_BETA, modes=DEFAULT_MODES):
    """(...,2) -> (...,2). Differentiable in xy; used both to evaluate the
    solution and, through autodiff, to derive the body force."""
    x, y = xy[..., 0], xy[..., 1]
    sx = _mode_sum(x, y, modes[0])
    sy = _mode_sum(x, y, modes[1])
    return torch.stack([alpha * sx, alpha * beta * sy], dim=-1)


def _grad_u_single(x, alpha, beta, modes=DEFAULT_MODES):
    """grad u at ONE point: (2,2) with [i][j] = du_i/dx_j."""
    return torch.autograd.functional.jacobian(
        lambda p: u_exact(p, alpha, beta, modes), x, create_graph=True, vectorize=False)


def grad_u_exact(xy, alpha=DEFAULT_ALPHA, beta=DEFAULT_BETA, modes=DEFAULT_MODES):
    """Analytic gradient, (...,2,2) with [...,i,j] = du_i/dx_j.

    Written out rather than autodiffed because it is needed at every Gauss
    point of every mesh in the sweep and closed form is far cheaper. It is
    checked against autodiff in verify_derivation()."""
    x, y = xy[..., 0], xy[..., 1]
    dsx_dx, dsx_dy = _mode_sum_grad(x, y, modes[0])
    dsy_dx, dsy_dy = _mode_sum_grad(x, y, modes[1])
    g = torch.stack([
        torch.stack([alpha * dsx_dx, alpha * dsx_dy], dim=-1),
        torch.stack([alpha * beta * dsy_dx, alpha * beta * dsy_dy], dim=-1),
    ], dim=-2)
    return g


def _psi_and_P(F, params, material, dtype):
    """psi(F) and P = dpsi/dF, batched over the leading dimension.

    `params` is a tuple of per-point tensors, one per material parameter
    (2 for Neo-Hookean's (mu, lam), 4 for Mooney-Rivlin's (c, c1, c2, d),
    3 for Arruda-Boyce's (mu_ab, N_ab, kappa_ab)) -- NOT hardcoded to
    exactly two, since only Neo-Hookean's energy density happens to take
    two parameters; the others do not and would raise a TypeError if
    called with a hardcoded (mu, lam) pair (caught when item #11 first
    tried Mooney-Rivlin/Arruda-Boyce through this file).

    torch.enable_grad() is not decoration: this builds its own little
    autograd graph to get P, and callers legitimately evaluate errors inside
    torch.no_grad() -- scoring a trained network is the obvious case. Without
    it the differentiation silently has no graph to work with and raises."""
    energy_density_fn, _ = get_material_fns_torch(material)
    with torch.enable_grad():
        F = F.detach().requires_grad_(True)
        psi = energy_density_fn(F, *params, dtype=dtype)
        P, = torch.autograd.grad(psi.sum(), F, create_graph=False)
    return psi.detach(), P.detach()


def tangent_modulus_batched(F, params, material, dtype):
    """C[q,i,j,k,l] = d^2 psi / dF_ij dF_kl, batched over the leading
    dimension -- the fourth-order tangent modulus at each point's own F.
    Same nested-jacrev-then-vmap pattern body_force_exact already uses for
    its own second derivative, applied here to psi instead of P: one
    jacrev gives P = dpsi/dF (2,2), a second gives dP/dF (2,2,2,2), and
    vmap batches both over the leading (point) dimension without ever
    forming psi's full batched Hessian at once.

    `params` is a tuple of per-point tensors -- see _psi_and_P's docstring
    for why this is not hardcoded to exactly two (mu, lam)."""
    energy_density_fn, _ = get_material_fns_torch(material)

    def psi_of_F(Fm, *params_i):
        params_exp = tuple(p[None] for p in params_i)
        return energy_density_fn(Fm[None], *params_exp, dtype=dtype).squeeze(0)

    def P_of_F(Fm, *params_i):
        return torch.func.jacrev(psi_of_F)(Fm, *params_i)      # (2,2)

    def C_of_F(Fm, *params_i):
        return torch.func.jacrev(P_of_F)(Fm, *params_i)        # (2,2,2,2)

    return torch.func.vmap(C_of_F)(F, *params)


def P_exact(xy, params, material, alpha=DEFAULT_ALPHA, beta=DEFAULT_BETA,
            dtype=torch.float64, modes=DEFAULT_MODES):
    """First Piola-Kirchhoff stress of the manufactured solution, (...,2,2).
    `params` is a tuple of per-point tensors -- see _psi_and_P's docstring."""
    F = torch.eye(2, dtype=dtype, device=xy.device).expand(
        xy.shape[:-1] + (2, 2)).clone()
    F = F + grad_u_exact(xy, alpha, beta, modes)
    params_flat = tuple(p.reshape(-1) for p in params)
    _, P = _psi_and_P(F.reshape(-1, 2, 2), params_flat, material, dtype)
    return P.reshape(xy.shape[:-1] + (2, 2))


def body_force_exact(xy, params, material, alpha=DEFAULT_ALPHA,
                     beta=DEFAULT_BETA, dtype=torch.float64, modes=DEFAULT_MODES):
    """b = -Div P, with (Div P)_i = sum_j dP_ij / dx_j. `params` is a tuple
    of per-point tensors -- see _psi_and_P's docstring.

    Nested autodiff: P already involves one derivative of psi with respect to
    F, and this takes a second derivative with respect to position. Done
    point-by-point through jacrev over a single-point function, then vmapped,
    because the divergence needs the derivative of P at a point with respect
    to THAT point's coordinates -- a batched jacobian would build the full
    (Npoints x 2 x 2) x (Npoints x 2) cross-derivative, almost all of it
    structurally zero."""
    energy_density_fn, _ = get_material_fns_torch(material)

    def P_at(p, *params_i):
        F = torch.eye(2, dtype=p.dtype, device=p.device) + \
            grad_u_exact(p, alpha, beta, modes)

        def psi_of_F(Fm):
            params_exp = tuple(pi[None] for pi in params_i)
            return energy_density_fn(Fm[None], *params_exp,
                                     dtype=p.dtype).squeeze(0)

        return torch.func.jacrev(psi_of_F)(F)      # (2,2) = P

    def div_P_at(p, *params_i):
        # dP[i,j] / dx[k]  -> contract j == k
        dP = torch.func.jacrev(P_at, argnums=0)(p, *params_i)   # (2,2,2)
        return torch.einsum('ijj->i', dP)

    flat = xy.reshape(-1, 2)
    params_flat = tuple(p.reshape(-1) for p in params)
    div = torch.func.vmap(div_P_at)(flat, *params_flat)
    return (-div).reshape(xy.shape)


# ----------------------------------------------------------------------
# Meshes, quadrature and assembly
# ----------------------------------------------------------------------
def _shape_batched(order):
    return shape_Q4_batched if order == "Q4" else shape_Q9_batched


def _gauss(order):
    if order == "Q4":
        g = 1.0 / math.sqrt(3.0)
        return [(a, b, 1.0) for a in (-g, g) for b in (-g, g)]
    g = math.sqrt(3.0 / 5.0)
    w = {0.0: 8.0 / 9.0, g: 5.0 / 9.0, -g: 5.0 / 9.0}
    return [(a, b, w[a] * w[b]) for a in (-g, 0.0, g) for b in (-g, 0.0, g)]


def build_mesh(order, N, Lx=1.0, Ly=1.0):
    if order == "Q4":
        from omar_pfem.data.data_generate_B1 import generate_grid_Q4
        return generate_grid_Q4(Lx, Ly, N, N)
    return generate_grid_Q9(Lx, Ly, N, N)


def shape_at_gauss(order, dtype=torch.float64):
    """N (G, n_local) and dN/dxi (G, n_local, 2) at every Gauss point.

    shape_Q4_batched / shape_Q9_batched are documented to take (Q,) ARRAYS of
    quadrature coordinates, not scalars -- handed scalars they silently return
    a transposed dN_dxi. All G points are therefore evaluated in one call,
    which is both the documented contract and faster."""
    g = _gauss(order)
    xi = np.array([p[0] for p in g], dtype=np.float64)
    eta = np.array([p[1] for p in g], dtype=np.float64)
    w = torch.tensor([p[2] for p in g], dtype=dtype)
    Nv, dN_dxi = _shape_batched(order)(xi, eta)
    Nv = torch.tensor(np.asarray(Nv), dtype=dtype)
    dN_dxi = torch.tensor(np.asarray(dN_dxi), dtype=dtype)
    n_local = 4 if order == "Q4" else 9
    assert Nv.shape == (len(g), n_local), Nv.shape
    assert dN_dxi.shape == (len(g), n_local, 2), dN_dxi.shape
    return Nv, dN_dxi, w


def quadrature_data(nodes, elements, order, dtype=torch.float64):
    """Per-element, per-Gauss-point physical coordinates, dN/dx, and dV0.

    Returns xg (Q,G,2), dNdx (Q,G,n_local,2), dV (Q,G). The same data serves
    the body-force assembly and every error integral, so the two cannot end
    up using different quadrature."""
    Xe = torch.tensor(nodes[elements], dtype=dtype)          # (Q, n_local, 2)
    Nv, dN_dxi, w = shape_at_gauss(order, dtype)

    # J[q,g,a,b] = dx_a/dxi_b = sum_l Xe[q,l,a] dN_dxi[g,l,b]
    J = torch.einsum('qla,glb->qgab', Xe, dN_dxi)
    detJ = J[..., 0, 0] * J[..., 1, 1] - J[..., 0, 1] * J[..., 1, 0]
    assert (detJ > 0).all(), "an element has a non-positive Jacobian"
    Jinv = torch.stack([
        torch.stack([J[..., 1, 1], -J[..., 0, 1]], dim=-1),
        torch.stack([-J[..., 1, 0], J[..., 0, 0]], dim=-1),
    ], dim=-2) / detJ[..., None, None]                        # dxi_b/dx_a

    dNdx = torch.einsum('glb,qgba->qgla', dN_dxi, Jinv)
    xg = torch.einsum('gl,qla->qga', Nv, Xe)
    dV = detJ * w[None, :]
    return xg, dNdx, dV


def shape_values(order, dtype=torch.float64):
    """N at every Gauss point: (G, n_local)."""
    return shape_at_gauss(order, dtype)[0]


def assemble_body_force(nodes, elements, order, params_e, material,
                        alpha, beta, dtype=torch.float64, modes=DEFAULT_MODES):
    """f_a = integral of N_a * b over the domain, as a (2*n_nodes,) vector.

    params_e is a tuple of per-element tensors, one per material parameter
    (see _psi_and_P's docstring) -- matching the solver's own per-element
    convention."""
    xg, _, dV = quadrature_data(nodes, elements, order, dtype)
    Ng = shape_values(order, dtype)                    # (G, n_local)
    Q, G = dV.shape
    params_g = tuple(p[:, None].expand(Q, G) for p in params_e)
    b = body_force_exact(xg, params_g, material, alpha, beta, dtype, modes)  # (Q,G,2)

    contrib = torch.einsum('gl,qgi,qg->qli', Ng, b, dV)   # (Q, n_local, 2)
    f = torch.zeros(len(nodes), 2, dtype=dtype)
    idx = torch.tensor(elements, dtype=torch.long)
    f.index_add_(0, idx.reshape(-1), contrib.reshape(-1, 2))
    return f.reshape(-1)


def boundary_nodes(nodes, Lx=1.0, Ly=1.0, tol=1e-9):
    x, y = nodes[:, 0], nodes[:, 1]
    return np.where((np.abs(x) < tol) | (np.abs(x - Lx) < tol) |
                    (np.abs(y) < tol) | (np.abs(y - Ly) < tol))[0]


# ----------------------------------------------------------------------
# Errors
# ----------------------------------------------------------------------
def compute_errors(nodes, elements, order, u_h, params_e, material,
                   alpha, beta, dtype=torch.float64, modes=DEFAULT_MODES):
    """L2, H1 semi-norm, energy (value AND norm) and stress errors of u_h
    against u*, all as relative quantities and all integrated on the FE
    mesh's own quadrature.

    energy_rel vs. energy_norm_rel -- these are deliberately DIFFERENT
    quantities, not two names for one thing:
      - energy_rel compares the scalar total energy VALUES U(u_h) and
        U(u*). By classical Galerkin superconvergence (Cea's lemma plus
        the energy functional's own orthogonality property) this
        converges at DOUBLE the H1 rate -- e.g. rate ~2 for Q4 rather
        than H1's ~1 -- which is a real, textbook property of this
        specific functional, not evidence the comparison is broken.
      - energy_norm_rel is the actual NORM of the discretization error in
        the energy inner product, ||e||_E = sqrt(a(e,e)) where
        a(v,w) = int (grad v) : C(F*) : (grad w) dV and C = d^2 psi/dF^2
        is the fourth-order tangent modulus at F* = I + grad(u*) (the
        advisor-confirmed linearization, mirroring the tangent/incremental
        energy norm high_dof_convergence_study.py's
        compute_tangent_energy_error already uses for Table 6a, here
        expressed as a direct quadrature integral against the continuous
        exact solution instead of a matrix-free Hessian-vector product
        against a second discrete field). This is expected to converge at
        the SAME rate as H1_semi_rel (Cea's lemma), not double it -- if it
        did superconverge like energy_rel, that would indicate a bug, not
        a feature.

    An earlier version of energy_norm_rel evaluated u* at the mesh's own
    NODES and reused the matrix-free Hessian-vector machinery from
    high_dof_convergence_study.py (built for comparing two already-
    discrete FE fields). That silently computed ||u_h - I_h(u*)||_E
    (against the NODAL INTERPOLANT of u*) rather than ||u_h - u*||_E
    (against the true continuous solution) -- u_h converging to I_h(u*)
    faster than to u* itself is itself a well-documented superconvergence
    effect on uniform meshes (see e.g. Wahlbin's "Superconvergence in
    Galerkin Finite Element Methods"), which is exactly why that version
    also measured rate ~2 for Q4, defeating the entire point of adding a
    norm that should NOT superconverge. Caught by checking the measured
    rate against Cea's-lemma theory rather than trusting a plausible-
    looking number. Integrating directly at Gauss points (matching
    L2_rel/H1_semi_rel's own convention exactly, with no nodal
    interpolation step) avoids the artifact."""
    xg, dNdx, dV = quadrature_data(nodes, elements, order, dtype)
    Ng = shape_values(order, dtype)
    Q, G = dV.shape
    ue = torch.tensor(u_h[elements], dtype=dtype)            # (Q, n_local, 2)

    uh_g = torch.einsum('gl,qli->qgi', Ng, ue)               # (Q,G,2)
    grad_uh = torch.einsum('qgla,qli->qgia', dNdx, ue)       # (Q,G,2,2) [i][a]

    u_star = u_exact(xg, alpha, beta, modes)
    grad_star = grad_u_exact(xg, alpha, beta, modes)

    def integ(v):
        return float(torch.einsum('qg...,qg->', v, dV).item())

    l2_err = integ(((uh_g - u_star) ** 2).sum(-1))
    l2_ref = integ((u_star ** 2).sum(-1))
    h1_err = integ(((grad_uh - grad_star) ** 2).sum((-1, -2)))
    h1_ref = integ((grad_star ** 2).sum((-1, -2)))

    eye = torch.eye(2, dtype=dtype).expand(Q, G, 2, 2)
    params_g = tuple(p[:, None].expand(Q, G).reshape(-1) for p in params_e)
    F_star = (eye + grad_star).reshape(-1, 2, 2)
    psi_h, P_h = _psi_and_P((eye + grad_uh).reshape(-1, 2, 2), params_g,
                            material, dtype)
    psi_s, P_s = _psi_and_P(F_star, params_g, material, dtype)
    P_h, P_s = P_h.reshape(Q, G, 2, 2), P_s.reshape(Q, G, 2, 2)
    psi_h, psi_s = psi_h.reshape(Q, G), psi_s.reshape(Q, G)

    stress_err = integ(((P_h - P_s) ** 2).sum((-1, -2)))
    stress_ref = integ((P_s ** 2).sum((-1, -2)))
    U_h, U_s = integ(psi_h), integ(psi_s)

    # Energy NORM: a(v,w) = int grad(v) : C(F*) : grad(w) dV, C the tangent
    # modulus at F* (the exact solution's own deformation gradient) -- same
    # linearization point high_dof_convergence_study.py uses (there, the
    # fine reference field; here, the continuous exact solution).
    C = tangent_modulus_batched(F_star, params_g, material, dtype).reshape(Q, G, 2, 2, 2, 2)
    grad_e = grad_uh - grad_star
    e_energy_sq = integ(torch.clamp(
        torch.einsum('qgij,qgijkl,qgkl->qg', grad_e, C, grad_e), min=0.0))
    u_energy_sq = integ(torch.clamp(
        torch.einsum('qgij,qgijkl,qgkl->qg', grad_star, C, grad_star), min=0.0))
    energy_norm_abs = math.sqrt(e_energy_sq)
    energy_norm_rel = energy_norm_abs / (math.sqrt(u_energy_sq) + 1e-30)

    return {
        "L2_rel": math.sqrt(l2_err / l2_ref),
        "H1_semi_rel": math.sqrt(h1_err / h1_ref),
        "stress_rel_L2": math.sqrt(stress_err / stress_ref),
        "energy_rel": abs(U_h - U_s) / abs(U_s),
        "energy_fe": U_h, "energy_exact": U_s,
        "energy_norm_abs": energy_norm_abs, "energy_norm_rel": energy_norm_rel,
    }


# ----------------------------------------------------------------------
# Verification of the derivation itself
# ----------------------------------------------------------------------
def verify_derivation(material="neo_hookean", alpha=DEFAULT_ALPHA,
                      beta=DEFAULT_BETA, dtype=torch.float64, modes=DEFAULT_MODES):
    """Three independent checks, before any mesh is built. If the body force
    is wrong every number this module produces is wrong, and a convergence
    table alone would not say which piece failed."""
    torch.manual_seed(0)
    pts = torch.rand(6, 2, dtype=dtype) * 0.8 + 0.1     # interior points
    # Same E=1000, nu=0.3 (plane strain) solve_mms uses -- generic over
    # however many parameters this material's energy density actually takes
    # (2 for Neo-Hookean, 4 for Mooney-Rivlin, 3 for Arruda-Boyce), rather
    # than hardcoding a (mu, lam) pair that only means something for
    # Neo-Hookean. (385.0, 577.0 were themselves just this same E, nu pair's
    # Neo-Hookean (mu, lam) rounded to 3 figures -- computing it here instead
    # is strictly more precise, not a behavior change.)
    from omar_pfem.materials_torch import get_material_fns
    _, E_nu_to_params = get_material_fns(material)
    E, nu = 1000.0, 0.3
    base_params = E_nu_to_params(
        torch.tensor(E, dtype=dtype), torch.tensor(nu, dtype=dtype),
        mode="plane_strain") if material == "neo_hookean" else \
        E_nu_to_params(torch.tensor(E, dtype=dtype), torch.tensor(nu, dtype=dtype))
    params = tuple(torch.full((6,), float(p), dtype=dtype) for p in base_params)
    ok = True

    # 1) the closed-form gradient against autodiff
    ad = torch.stack([_grad_u_single(p, alpha, beta, modes) for p in pts])
    cf = grad_u_exact(pts, alpha, beta, modes)
    e = (ad - cf).abs().max().item()
    print(f"  grad u*: closed form vs autodiff, max |diff| = {e:.3e}")
    ok &= e < 1e-10

    # 2) u* vanishes on the boundary -- otherwise homogeneous Dirichlet is
    #    the wrong boundary condition and the whole study is invalid
    t = torch.linspace(0, 1, 21, dtype=dtype)
    edge = torch.cat([torch.stack([t, torch.zeros_like(t)], -1),
                      torch.stack([t, torch.ones_like(t)], -1),
                      torch.stack([torch.zeros_like(t), t], -1),
                      torch.stack([torch.ones_like(t), t], -1)])
    e = u_exact(edge, alpha, beta, modes).abs().max().item()
    print(f"  u* on the boundary, max |u*| = {e:.3e}")
    ok &= e < 1e-14

    # 3) the divergence, by autodiff against a central finite difference
    h = 1e-5
    b_ad = body_force_exact(pts, params, material, alpha, beta, dtype, modes)
    div_fd = torch.zeros_like(pts)
    for j in range(2):
        off = torch.zeros(2, dtype=dtype)
        off[j] = h
        Pp = P_exact(pts + off, params, material, alpha, beta, dtype, modes)
        Pm = P_exact(pts - off, params, material, alpha, beta, dtype, modes)
        div_fd += (Pp[..., :, j] - Pm[..., :, j]) / (2 * h)
    b_fd = -div_fd
    rel = ((b_ad - b_fd).norm() / b_fd.norm()).item()
    print(f"  b = -Div P: autodiff vs central difference, rel. diff = {rel:.3e}")
    ok &= rel < 1e-6

    print("  " + ("all derivation checks PASSED" if ok
                  else "DERIVATION CHECK FAILED -- do not trust any result below"))
    return ok


# ----------------------------------------------------------------------
def solve_mms(order, N, material, alpha, beta, device, dtype=torch.float64,
              newton_tol=1e-10, cg_tol=1e-8, verbose=False, modes=DEFAULT_MODES):
    nodes, elements = build_mesh(order, N)
    from omar_pfem.materials_torch import get_material_fns
    _, E_nu_to_params = get_material_fns(material)
    # A uniform material: the point of this study is discretization error, so
    # a spatially varying E would add an interpolation error of its own and
    # confound the convergence rates being measured.
    E, nu = 1000.0, 0.3
    params = E_nu_to_params(torch.tensor(E, dtype=dtype),
                            torch.tensor(nu, dtype=dtype), mode="plane_strain") \
        if material == "neo_hookean" else \
        E_nu_to_params(torch.tensor(E, dtype=dtype), torch.tensor(nu, dtype=dtype))
    n_el = len(elements)
    # One tensor per material parameter (2 for Neo-Hookean, 4 for
    # Mooney-Rivlin, 3 for Arruda-Boyce) -- NOT hardcoded to exactly two
    # (mu, lam), which only ever meant something for Neo-Hookean and
    # silently mislabeled Mooney-Rivlin's/Arruda-Boyce's own parameters as
    # "mu, lam" before this was generalized (item #11: extending this
    # study to another material caught it immediately with a TypeError
    # from the material's own energy-density function).
    elem_params_t = tuple(
        torch.full((n_el,), float(p), dtype=dtype, device=device) for p in params)

    # assemble_body_force builds its own quadrature-point tensors from the
    # numpy `nodes`/`elements` arrays with no explicit device= (torch.tensor
    # defaults to CPU), while elem_params_t above was just built directly on
    # `device` -- on a CPU run these silently match by accident, which is why
    # this was never caught until an actual `device="cuda"` run (Arruda-
    # Boyce's richer-family study, 2026-09-09) hit a real cross-device error
    # inside its energy-density function's clamp. `with torch.device(device):`
    # makes every device-less tensor created inside this call (and any
    # nested torch.func.vmap/jacrev calls it makes) default to the right
    # device automatically, the same fix already used for torch-fem's own
    # analogous device bugs (item #13).
    with torch.device(device):
        fext = assemble_body_force(nodes, elements, order, elem_params_t, material,
                                   alpha, beta, dtype, modes)

    fixed = boundary_nodes(nodes)
    fixed_dofs = np.concatenate([2 * fixed, 2 * fixed + 1])
    free_dofs = np.setdiff1d(np.arange(2 * len(nodes)), fixed_dofs)

    xy_t = torch.tensor(nodes, dtype=dtype, device=device)
    quad_t = torch.tensor(elements, dtype=torch.long, device=device)
    free_t = torch.tensor(free_dofs, dtype=torch.long, device=device)
    fext_free = fext[free_dofs].to(device)

    # CG iteration cap. solve_matrix_free's CG stops on ||r||/||b|| < cg_tol,
    # where b is the Newton right-hand side. On the LAST Newton iteration of
    # each load step b is the already-converged residual -- here around
    # 1e-13 -- so the relative target becomes unreachable in float64 and CG
    # runs to its cap no matter how well it is actually doing. That is a
    # property of the shared solver, not of this study, and the cheap defence
    # is a cap proportional to the problem size: in exact arithmetic CG
    # terminates in at most n_free steps, so 2*n_free is generous while
    # keeping the wasted final iteration bounded. Without this the tiny N=5
    # case alone took minutes.
    cg_cap = max(50, min(2000, 2 * len(free_dofs)))

    t0 = time.time()
    u_free, stats = solve_matrix_free(
        xy_t, quad_t, free_t, elem_params_t, fext_free, n_free=len(free_dofs),
        material=material, order=order, nsteps=5, newton_max=40,
        newton_tol=newton_tol, cg_tol=cg_tol, cg_max_iter=cg_cap,
        use_jacobi=True, device=device, dtype=dtype, verbose=verbose)
    wall = time.time() - t0

    u_full = torch.zeros(2 * len(nodes), dtype=dtype, device=device)
    u_full = u_full.index_copy(0, free_t, u_free)
    u_h = u_full.reshape(len(nodes), 2).cpu().numpy()

    # Same device-mismatch class as assemble_body_force above: compute_errors
    # also builds its quadrature-point tensors from numpy nodes/elements with
    # no explicit device=, while elem_params_t (params_e here) is already on
    # `device` -- a second real instance of the same bug, hit right after
    # fixing the first one, at u_h's own energy-density evaluation.
    with torch.device(device):
        err = compute_errors(nodes, elements, order, u_h, elem_params_t, material,
                             alpha, beta, dtype, modes)
    err.update({"order": order, "N": N, "h": 1.0 / (N - 1),
                "n_nodes": len(nodes), "n_dof": 2 * len(nodes),
                "n_elements": n_el, "wall_clock_s": wall,
                "newton_iters": stats.get("newton_iters_total") if isinstance(stats, dict) else None})
    return err, u_h, nodes, elements


def main():
    p = argparse.ArgumentParser("MMS: Q4 vs Q9 vs the operator, one analytic solution")
    p.add_argument("--material", default="neo_hookean",
                   choices=["neo_hookean", "mooney_rivlin", "arruda_boyce"])
    p.add_argument("--orders", default="Q4,Q9")
    p.add_argument("--Ns", default="5,9,17,33")
    p.add_argument("--alpha", type=float, default=DEFAULT_ALPHA)
    p.add_argument("--beta", type=float, default=DEFAULT_BETA)
    p.add_argument("--verify", action="store_true",
                   help="run the derivation checks and stop")
    p.add_argument("--richer_family", action="store_true",
                   help="use MODES_RICHER (a sum of several sine/cosine spatial "
                        "modes, Timon's round-8 point 6) instead of the default "
                        "MODES_SINGLE. NOT the default: Tables 22-24 and every "
                        "other existing result (mms_operator.py's (alpha,beta) "
                        "family, mms_family_fem.py) were built assuming "
                        "MODES_SINGLE, so this stays deliberately opt-in rather "
                        "than silently changing the shape those already depend "
                        "on. Also changes the default --out_json name so a run "
                        "with this flag can never overwrite the reference file.")
    p.add_argument("--out_json", default=None)
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args()
    modes = MODES_RICHER if args.richer_family else MODES_SINGLE

    print("Verifying the manufactured solution and its body force:")
    if not verify_derivation(args.material, args.alpha, args.beta, modes=modes):
        raise SystemExit("derivation checks failed")
    if args.verify:
        return

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    dtype = torch.float64
    orders = [o.strip() for o in args.orders.split(",") if o.strip()]
    Ns = [int(n) for n in args.Ns.split(",") if n.strip()]
    default_name = (f"mms_richer_B1_{args.material}.json" if args.richer_family
                     else f"mms_B1_{args.material}.json")
    out_json = args.out_json or default_name

    rows = []
    if os.path.exists(out_json):
        try:
            rows = json.load(open(out_json)).get("rows", [])
            print(f"[resume] {len(rows)} rows already done")
        except Exception:
            rows = []
    # A row saved before the energy-NORM field existed lacks
    # "energy_norm_rel" -- treating it as done would silently skip
    # recomputing it and leave that field permanently missing (the same
    # kind of stale-artifact bug the DD-NO coarse-vs-fine notebook hit).
    # Requiring the new field too forces a clean recompute of any old row.
    done = {(r["order"], r["N"]) for r in rows if "energy_norm_rel" in r}

    print(f"\nDevice: {device}, dtype float64, alpha={args.alpha}, beta={args.beta}")
    print(f"{manufactured_solution_str(args.alpha, args.beta, modes)}\n")
    hdr = (f"{'order':<6}{'N':>5}{'DOF':>9}{'L2':>12}{'H1 semi':>12}{'stress':>12}"
           f"{'energy':>12}{'E-norm':>12}{'s':>8}")
    print(hdr)
    print("-" * len(hdr))
    for r in sorted(rows, key=lambda r: (r["order"], r["N"])):
        if "energy_norm_rel" not in r:
            continue
        print(f"{r['order']:<6}{r['N']:>5}{r['n_dof']:>9,}{r['L2_rel']:>12.3e}"
              f"{r['H1_semi_rel']:>12.3e}{r['stress_rel_L2']:>12.3e}"
              f"{r['energy_rel']:>12.3e}{r['energy_norm_rel']:>12.3e}{r['wall_clock_s']:>8.1f}")

    for order in orders:
        for N in Ns:
            if (order, N) in done:
                continue
            err, *_ = solve_mms(order, N, args.material, args.alpha, args.beta,
                                device, dtype, verbose=args.verbose, modes=modes)
            rows.append(err)
            print(f"{err['order']:<6}{err['N']:>5}{err['n_dof']:>9,}"
                  f"{err['L2_rel']:>12.3e}{err['H1_semi_rel']:>12.3e}"
                  f"{err['stress_rel_L2']:>12.3e}{err['energy_rel']:>12.3e}"
                  f"{err['energy_norm_rel']:>12.3e}{err['wall_clock_s']:>8.1f}", flush=True)
            rep = {"study": "method of manufactured solutions",
                   "geometry": "B1 (unit square)", "material": args.material,
                   "manufactured_solution": manufactured_solution_str(args.alpha, args.beta, modes),
                   "richer_family": args.richer_family,
                   "alpha": args.alpha, "beta": args.beta,
                   "body_force": "b = -Div P(F*), by nested autodiff, checked "
                                 "against a central finite difference",
                   "boundary_conditions": "homogeneous Dirichlet on all four "
                                          "edges, exact for this u*",
                   "material_field": "uniform E=1000, nu=0.3, plane strain",
                   "error_definitions": {
                       "L2_rel": "sqrt(int |u_h - u*|^2 dV / int |u*|^2 dV)",
                       "H1_semi_rel": "sqrt(int |grad u_h - grad u*|^2 dV / "
                                      "int |grad u*|^2 dV); the semi-norm, not "
                                      "the full H1 norm",
                       "stress_rel_L2": "sqrt(int |P_h - P*|^2 dV / int |P*|^2 dV), "
                                        "first Piola-Kirchhoff, Frobenius norm",
                       "energy_rel": "|U_h - U*| / |U*| where U = int psi(F) dV is "
                                     "the INTERNAL strain energy, not the total "
                                     "potential Pi = U - W. Both are integrated on "
                                     "the FE mesh's own quadrature, so U* is the "
                                     "exact solution's energy evaluated with the "
                                     "same rule and the comparison is not polluted "
                                     "by quadrature error. This is a VALUE "
                                     "comparison and superconverges (fitted rate "
                                     "is double the H1 rate) -- see energy_norm_rel "
                                     "for the NORM the advisor's point 6 asked for",
                       "energy_norm_rel": "sqrt(int grad(e):C(F*):grad(e) dV / "
                                          "int grad(u*):C(F*):grad(u*) dV), "
                                          "e = u_h - u*, C = d^2 psi/dF^2 the "
                                          "fourth-order tangent modulus at "
                                          "F* = I + grad(u*) (the advisor-"
                                          "confirmed tangent/incremental energy "
                                          "norm, same linearization Table 6a "
                                          "uses, expressed as a direct quadrature "
                                          "integral against the continuous exact "
                                          "solution rather than a matrix-free "
                                          "Hessian-vector product against a "
                                          "second discrete field). A norm, not a "
                                          "value, so expected to converge at the "
                                          "SAME rate as H1_semi_rel (Cea's "
                                          "lemma), not double it like energy_rel",
                       "note": "all quantities are integrated on the same "
                               "quadrature as the body-force assembly, so no two "
                               "of them can disagree about the integration rule",
                   },
                   "solver": {
                       "newton_tol": 1e-10, "cg_tol": 1e-8, "load_steps": 5,
                       "tolerance_independence_checked": "at Q4 N=9 the L2 and H1 "
                           "errors are identical to 12 significant digits across "
                           "cg_tol 1e-6, 1e-8 and 1e-10, so what is reported is "
                           "discretization error and not algebraic error",
                   },
                   "dtype": "float64", "device": device.type,
                   "rows": sorted(rows, key=lambda r: (r["order"], r["N"]))}
            tmp = out_json + ".tmp"
            with open(tmp, "w") as f:
                json.dump(rep, f, indent=2)
            os.replace(tmp, out_json)

    # ---- convergence rates ------------------------------------------
    # fit_convergence_rate returns (least-squares p, per-consecutive-pair p).
    # The pairwise list matters as much as the fit: a rate that is still
    # drifting across the finest interval has not settled, and quoting the
    # fit alone would hide that -- the same caveat Table 20 carries.
    print("\nObserved convergence rates (least squares on log h):")
    hdr2 = f"{'order':<6}{'norm':<10}{'rate':>8}{'expected':>10}   pairwise"
    print(hdr2)
    print("-" * (len(hdr2) + 12))
    rates = {}
    for order in orders:
        rs = sorted([r for r in rows if r["order"] == order], key=lambda r: r["N"])
        if len(rs) < 2:
            continue
        hs = [r["h"] for r in rs]
        exp = {"L2": 2 if order == "Q4" else 3,
               "H1_semi": 1 if order == "Q4" else 2,
               "stress": 1 if order == "Q4" else 2,
               "energy": 2 if order == "Q4" else 4,
               # A NORM, not a value -- expected to track H1_semi's rate,
               # not superconverge like the value comparison above it does.
               "energy_norm": 1 if order == "Q4" else 2}
        got = {}
        for norm, key in (("L2", "L2_rel"), ("H1_semi", "H1_semi_rel"),
                          ("stress", "stress_rel_L2"), ("energy", "energy_rel"),
                          ("energy_norm", "energy_norm_rel")):
            p, pw = fit_convergence_rate(hs, [r[key] for r in rs])
            got[norm] = {"rate": p, "pairwise": pw, "expected": exp[norm]}
            ps = ", ".join(f"{x:.2f}" for x in pw)
            print(f"{order:<6}{norm:<10}{p:>8.2f}{exp[norm]:>10}   [{ps}]")
        rates[order] = got

    if rates:
        rep = json.load(open(out_json))
        rep["convergence_rates"] = rates
        # Only L2 and the H1 semi-norm carry textbook rates worth asserting.
        # The energy rate for a nonlinear functional and the stress rate are
        # reported but not used as a pass/fail gate.
        rep["rate_check"] = {
            o: ("as expected" if all(abs(v[n]["rate"] - v[n]["expected"]) < 0.4
                                     for n in ("L2", "H1_semi"))
                else "OFF -- the manufactured problem or the assembly is wrong")
            for o, v in rates.items()}
        print()
        for o, verdict in rep["rate_check"].items():
            print(f"  {o}: {verdict}")
        with open(out_json, "w") as f:
            json.dump(rep, f, indent=2)
    write_manifest(os.path.dirname(os.path.abspath(out_json)) or ".",
                   {"study": "mms", "material": args.material})
    print(f"\nwrote {out_json}")


if __name__ == "__main__":
    main()
