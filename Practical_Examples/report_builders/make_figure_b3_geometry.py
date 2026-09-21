"""B3 geometry figure -- Timon's follow-up on round 14 explicitly asked
to "show also a geometry of the problem." Three panels: undeformed
geometry (outer housing fixed, inner core boundary), the deformed
configuration under the true rigid-body rocking rotation (phi=0.05 rad,
the single deterministic configuration the mesh-convergence study
used), and a zoomed meridian cross-section showing the groove -- the
one explicit, disclosed finite-radius stress-concentration feature this
whole benchmark is built around.

Uses a moderate resolution (25,10,23) chosen for visual clarity, not the
fine convergence mesh -- this is a geometry figure, not a mesh-density
demonstration.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

from omar_pfem.data.data_generate_B3 import (
    generate_grid_hex8_bushing, boundary_node_sets, rigid_rotation_displacement,
    groove_R_in, groove_radius_of_curvature)

R_IN0, R_OUT, LZ = 0.5, 1.0, 1.0
GROOVE_DEPTH, GROOVE_HALF_WIDTH = 0.05, 0.15
PHI = 0.05  # the single deterministic rocking angle the convergence study used

OUT = '/home/user/OMAR/Practical_Examples/report_builders/figures/B3_geometry.png'


def main():
    Ntheta, Nr, Nz = 25, 10, 23
    nodes, elements = generate_grid_hex8_bushing(
        R_IN0, R_OUT, LZ, Ntheta, Nr, Nz, GROOVE_DEPTH, GROOVE_HALF_WIDTH)
    inner, outer, sym = boundary_node_sets(
        nodes, R_IN0, R_OUT, LZ, GROOVE_DEPTH, GROOVE_HALF_WIDTH)

    ux, uy, uz = rigid_rotation_displacement(nodes[inner], LZ, PHI)
    nodes_def = nodes.copy()
    nodes_def[inner, 0] += ux
    nodes_def[inner, 1] += uy
    nodes_def[inner, 2] += uz

    fig = plt.figure(figsize=(15, 6))

    ax1 = fig.add_subplot(1, 3, 1, projection="3d")
    ax1.scatter(*nodes[outer].T, s=2, c="tab:blue", label="Outer housing (fixed)", alpha=0.5)
    ax1.scatter(*nodes[inner].T, s=3, c="tab:red", label="Inner core (rocking)", alpha=0.7)
    ax1.set_title("B3 bushing: undeformed geometry\n(half-cylinder, mirror symmetry at y=0)")
    ax1.set_xlabel("x"); ax1.set_ylabel("y"); ax1.set_zlabel("z")
    ax1.legend(loc="upper left", fontsize=8)
    ax1.set_box_aspect([1, 0.55, 1])

    ax2 = fig.add_subplot(1, 3, 2, projection="3d")
    ax2.scatter(*nodes[outer].T, s=2, c="tab:blue", alpha=0.4, label="Outer housing (fixed)")
    ax2.scatter(*nodes_def[inner].T, s=3, c="tab:red", alpha=0.8,
                label=f"Inner core (rocked, phi={PHI} rad)")
    ax2.set_title("B3 bushing: deformed configuration\n(true rigid-body rotation of the core)")
    ax2.set_xlabel("x"); ax2.set_ylabel("y"); ax2.set_zlabel("z")
    ax2.legend(loc="upper left", fontsize=8)
    ax2.set_box_aspect([1, 0.55, 1])

    ax3 = fig.add_subplot(1, 3, 3)
    z_line = np.linspace(0, LZ, 400)
    r_in_line = groove_R_in(z_line, LZ, R_IN0, GROOVE_DEPTH, GROOVE_HALF_WIDTH)
    ax3.plot(z_line, r_in_line, "r-", lw=2.5, label="Inner (bonded) surface, R_in(z)")
    ax3.axhline(R_OUT, color="tab:blue", lw=2, label="Outer housing, R_out (fixed)")
    ax3.fill_between(z_line, r_in_line, R_OUT, color="lightgray", alpha=0.5, label="Rubber annulus")
    rho = groove_radius_of_curvature(GROOVE_DEPTH, GROOVE_HALF_WIDTH)
    z_mid = LZ / 2.0
    ax3.annotate(f"groove (depth={GROOVE_DEPTH}, rho={rho:.4f})",
                 xy=(z_mid, R_IN0 - GROOVE_DEPTH),
                 xytext=(0.05, R_IN0 - GROOVE_DEPTH - 0.06),
                 arrowprops=dict(arrowstyle="->"), fontsize=9)
    ax3.set_xlabel("z (axial)"); ax3.set_ylabel("radius")
    ax3.set_ylim(0.40, 1.05)
    ax3.set_title("Meridian cross-section (theta=0), zoomed\nshowing the groove -- the one explicit\n"
                  "finite-radius stress-concentration feature")
    ax3.legend(loc="upper center", fontsize=8, framealpha=0.9)

    fig.suptitle("B3: rocking elastomeric rubber-mount bushing -- geometry", fontsize=13)
    fig.tight_layout()
    fig.savefig(OUT, dpi=150)
    print("saved", OUT)


if __name__ == "__main__":
    main()
