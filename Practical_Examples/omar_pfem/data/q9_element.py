"""
Q9 (9-node, biquadratic Lagrange) quadrilateral element -- shape functions,
their derivatives, 3x3 Gauss quadrature, and a structured mesh generator --
for the polynomial-order comparison the advisor asked for ("test in this
context also the polynomial order of the FE simulation, bi-linear versus
quadratic").

Q9 (full biquadratic Lagrange) rather than Q8 (8-node serendipity): both
are "quadratic" quadrilaterals and give the same asymptotic convergence
rate; Q9's shape functions are plain tensor products of 1D quadratic
Lagrange polynomials, which is simpler to get right than Q8's serendipity
basis, and this study's structured mesh has no reason to omit the
mid-element node.

Local node ordering (natural coordinates xi, eta in [-1, 1]):
  1:(-1,-1) 2:(1,-1) 3:(1,1) 4:(-1,1)   -- corners, same order as Q4
  5:(0,-1)  6:(1,0)  7:(0,1)  8:(-1,0)  -- edge midpoints
  9:(0,0)                                -- element center
"""
import numpy as np
import torch


def _lagrange1d(t, node):
    """1D quadratic Lagrange basis value + derivative at node in {-1,0,1}."""
    if node == -1:
        return 0.5 * t * (t - 1.0), t - 0.5
    elif node == 0:
        return 1.0 - t * t, -2.0 * t
    elif node == 1:
        return 0.5 * t * (t + 1.0), t + 0.5
    raise ValueError(node)


_NODE_XI_ETA = [(-1, -1), (1, -1), (1, 1), (-1, 1), (0, -1), (1, 0), (0, 1), (-1, 0), (0, 0)]


def shape_Q9(xi, eta):
    """NumPy version, matching fem_core.shape_Q4's calling convention:
    returns (N (9,), dN_dxi (9,2))."""
    N = np.zeros(9)
    dN_dxi = np.zeros((9, 2))
    for k, (nx, ny) in enumerate(_NODE_XI_ETA):
        Lx, dLx = _lagrange1d(xi, nx)
        Ly, dLy = _lagrange1d(eta, ny)
        N[k] = Lx * Ly
        dN_dxi[k, 0] = dLx * Ly
        dN_dxi[k, 1] = Lx * dLy
    return N, dN_dxi


def shape_Q9_torch(xi, eta, device, dtype):
    """torch version: xi, eta are 0-d tensors; returns (N (9,), dN_dxi (9,2))
    as torch tensors, matching train_B1.shape_Q4_torch's convention."""
    Ns, dNs = [], []
    for (nx, ny) in _NODE_XI_ETA:
        Lx, dLx = _lagrange1d(xi, nx)
        Ly, dLy = _lagrange1d(eta, ny)
        Ns.append(Lx * Ly)
        dNs.append(torch.stack([dLx * Ly, Lx * dLy]))
    return torch.stack(Ns), torch.stack(dNs)


def gauss_3x3():
    """3x3 full Gauss quadrature (exact for the biquadratic Jacobian this
    element produces on a non-degenerate mesh), matching the (point, weight)
    convention used by fem_core.element_K_and_fint_TL's 2x2 rule."""
    g = np.sqrt(3.0 / 5.0)
    pts_1d = [-g, 0.0, g]
    wts_1d = [5.0 / 9.0, 8.0 / 9.0, 5.0 / 9.0]
    gauss = []
    for i, xi in enumerate(pts_1d):
        for j, eta in enumerate(pts_1d):
            gauss.append((xi, eta, wts_1d[i] * wts_1d[j]))
    return gauss


def generate_grid_Q9(Lx, Ly, Nx, Ny):
    """Structured Q9 mesh over [0,Lx]x[0,Ly]. Nx, Ny are the number of
    CORNER nodes per side (same meaning as generate_grid_Q4's Nx, Ny) --
    (Nx-1) x (Ny-1) elements, each needing 2 extra nodes per direction
    beyond its corners, so the full node grid has (2*Nx-1) x (2*Ny-1)
    nodes total (corners + edge-midpoints + centers all lie on this one
    refined structured grid). Returns (nodes, elements) with elements'
    local node order matching _NODE_XI_ETA above."""
    fine_Nx, fine_Ny = 2 * Nx - 1, 2 * Ny - 1
    xs = np.linspace(0.0, Lx, fine_Nx)
    ys = np.linspace(0.0, Ly, fine_Ny)
    X, Y = np.meshgrid(xs, ys, indexing="xy")
    nodes = np.vstack([X.ravel(), Y.ravel()]).T

    def idx(i, j):
        return j * fine_Nx + i

    elements = []
    for je in range(Ny - 1):
        for ie in range(Nx - 1):
            i0, j0 = 2 * ie, 2 * je
            n1 = idx(i0, j0)
            n2 = idx(i0 + 2, j0)
            n3 = idx(i0 + 2, j0 + 2)
            n4 = idx(i0, j0 + 2)
            n5 = idx(i0 + 1, j0)
            n6 = idx(i0 + 2, j0 + 1)
            n7 = idx(i0 + 1, j0 + 2)
            n8 = idx(i0, j0 + 1)
            n9 = idx(i0 + 1, j0 + 1)
            elements.append([n1, n2, n3, n4, n5, n6, n7, n8, n9])
    return nodes, np.array(elements, dtype=int)
