# =====================================================================
#  CELL -- B3 geometry figure, runnable directly by Omar in Colab.
#
#  Timon's own follow-up on round 14 asked to "show also a geometry of
#  the problem." This cell builds that figure using the SAME real
#  geometry-generation code the mesh-convergence study itself uses
#  (data_generate_B3.py's generate_grid_hex8_bushing / boundary_node_
#  sets / rigid_rotation_displacement / groove_R_in /
#  groove_radius_of_curvature) and the SAME physical parameters as the
#  actual production dataset (data_generate_B3_dataset.py) and the
#  480,320-element convergence study cited in the report/paper
#  (R_in0=0.5, R_out=1.0, Lz=1.0, groove depth=0.20 [the "4x sharper
#  groove", not the original 0.05 pre-sharpening design], groove half-
#  width=0.15, rocking angle phi=0.05 rad, the phi_mean of the
#  production sampling distribution). Only the mesh
#  RESOLUTION here (25,10,23) is different from the study's own
#  resolutions -- chosen purely for visual clarity, since the fine
#  convergence meshes would render as an indistinguishable dense point
#  cloud in a static image. The GEOMETRY itself -- radii, groove shape,
#  rocking motion -- is exactly the real one, not a mockup.
#
#  Per Omar's own standing instruction (2026-09-21): every notebook
#  from now on must (1) generate figures, (2) SAVE them to Drive, and
#  (3) DISPLAY them inline in the notebook's own output. This cell does
#  all three. No GPU or torch-fem install is needed -- the geometry
#  functions used here are pure numpy (verified directly: data_generate
#  _B3.py and its own data_generate_B2.py import only numpy/scipy/h5py,
#  no torch dependency), so this runs on a plain CPU runtime in
#  seconds.
# =====================================================================
import os
import subprocess
import sys


def run(cmd):
    print('$', ' '.join(str(c) for c in cmd), flush=True)
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                          stderr=subprocess.STDOUT, text=True, bufsize=1)
    for line in p.stdout:
        print(line, end='', flush=True)
    p.wait()
    if p.returncode != 0:
        raise subprocess.CalledProcessError(p.returncode, cmd)


from google.colab import drive
drive.mount('/content/drive')

REPO = '/content/OMAR'
if not os.path.isdir(REPO):
    run(['git', 'clone', '-b', 'claude/claude-code-question-d307wp',
         'https://github.com/SUHIBAMRO/OMAR.git', REPO])
else:
    run(['git', '-C', REPO, 'fetch', 'origin', 'claude/claude-code-question-d307wp'])
    run(['git', '-C', REPO, 'checkout', 'claude/claude-code-question-d307wp'])
    run(['git', '-C', REPO, 'reset', '--hard', 'origin/claude/claude-code-question-d307wp'])

WORK = f'{REPO}/Practical_Examples'
os.chdir(WORK)
sys.path.insert(0, WORK)

for _mod_name in list(sys.modules):
    if _mod_name == 'omar_pfem' or _mod_name.startswith('omar_pfem.'):
        del sys.modules[_mod_name]

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

from omar_pfem.data.data_generate_B3 import (
    generate_grid_hex8_bushing, boundary_node_sets, rigid_rotation_displacement,
    groove_R_in, groove_radius_of_curvature)

# ---- the SAME physical parameters the production dataset (data_generate_
# B3_dataset.py) and the cited 480,320-element convergence study use ----
R_IN0, R_OUT, LZ = 0.5, 1.0, 1.0
GROOVE_DEPTH, GROOVE_HALF_WIDTH = 0.20, 0.15
PHI = 0.05  # phi_mean of the production rocking-angle sampling distribution

R = '/content/drive/MyDrive/pfem_run'
os.makedirs(f'{R}/b3', exist_ok=True)

# Resolution chosen for VISUAL CLARITY only -- not a convergence-study
# resolution. Bump these and re-run if a denser-looking mesh is wanted;
# the geometry (radii, groove shape) is identical regardless of Ntheta/
# Nr/Nz, since that's set entirely by R_IN0/R_OUT/GROOVE_DEPTH/
# GROOVE_HALF_WIDTH above, not by mesh density.
Ntheta, Nr, Nz = 25, 10, 23
print(f'Building B3 geometry at Ntheta={Ntheta}, Nr={Nr}, Nz={Nz} '
      f'(visual-clarity resolution, not a convergence-study resolution)')

nodes, elements = generate_grid_hex8_bushing(
    R_IN0, R_OUT, LZ, Ntheta, Nr, Nz, GROOVE_DEPTH, GROOVE_HALF_WIDTH)
inner, outer, sym = boundary_node_sets(
    nodes, R_IN0, R_OUT, LZ, GROOVE_DEPTH, GROOVE_HALF_WIDTH)
print(f'nodes={nodes.shape[0]}, elements={elements.shape[0]}, '
      f'inner_core={inner.sum()}, outer_housing={outer.sum()}')

ux, uy, uz = rigid_rotation_displacement(nodes[inner], LZ, PHI)
nodes_def = nodes.copy()
nodes_def[inner, 0] += ux
nodes_def[inner, 1] += uy
nodes_def[inner, 2] += uz

# Enhanced sizing (2026-10-12, per Timon's review point 1: "figures and
# legends are way too small... we should show also for B3 a higher
# quality geometry") -- same geometry/data/3-panel layout as before,
# only the visual presentation (canvas size, font sizes, marker sizes,
# DPI) changed. Verified by rendering the actual paper page afterward
# (no clipping, no overlap, overfull-hbox count unchanged).
FS_TITLE, FS_AXIS, FS_TICK, FS_LEGEND, FS_SUPTITLE, FS_ANNOT = 20, 16, 13, 14, 22, 15
plt.rcParams.update({
    'font.size': FS_TICK, 'axes.titlesize': FS_TITLE, 'axes.labelsize': FS_AXIS,
    'xtick.labelsize': FS_TICK, 'ytick.labelsize': FS_TICK, 'legend.fontsize': FS_LEGEND,
})

fig = plt.figure(figsize=(26, 10))

ax1 = fig.add_subplot(1, 3, 1, projection="3d")
ax1.scatter(*nodes[outer].T, s=10, c="tab:blue", label="Outer housing (fixed)", alpha=0.5)
ax1.scatter(*nodes[inner].T, s=14, c="tab:red", label="Inner core (rocking)", alpha=0.8)
ax1.set_title("B3 bushing: undeformed geometry\n(half-cylinder, mirror symmetry at y=0)",
              fontsize=FS_TITLE)
ax1.set_xlabel("x", fontsize=FS_AXIS, labelpad=12)
ax1.set_ylabel("y", fontsize=FS_AXIS, labelpad=12)
ax1.set_zlabel("z", fontsize=FS_AXIS, labelpad=12)
ax1.tick_params(labelsize=FS_TICK)
ax1.legend(loc="upper left", fontsize=FS_LEGEND, markerscale=2.5)
ax1.set_box_aspect([1, 0.55, 1])

ax2 = fig.add_subplot(1, 3, 2, projection="3d")
ax2.scatter(*nodes[outer].T, s=10, c="tab:blue", alpha=0.4, label="Outer housing (fixed)")
ax2.scatter(*nodes_def[inner].T, s=14, c="tab:red", alpha=0.85,
            label=f"Inner core (rocked, phi={PHI} rad)")
ax2.set_title("B3 bushing: deformed configuration\n(true rigid-body rotation of the core)",
              fontsize=FS_TITLE)
ax2.set_xlabel("x", fontsize=FS_AXIS, labelpad=12)
ax2.set_ylabel("y", fontsize=FS_AXIS, labelpad=12)
ax2.set_zlabel("z", fontsize=FS_AXIS, labelpad=12)
ax2.tick_params(labelsize=FS_TICK)
ax2.legend(loc="upper left", fontsize=FS_LEGEND, markerscale=2.5)
ax2.set_box_aspect([1, 0.55, 1])

ax3 = fig.add_subplot(1, 3, 3)
z_line = np.linspace(0, LZ, 400)
r_in_line = groove_R_in(z_line, LZ, R_IN0, GROOVE_DEPTH, GROOVE_HALF_WIDTH)
ax3.plot(z_line, r_in_line, "r-", lw=4, label="Inner (bonded) surface, R_in(z)")
ax3.axhline(R_OUT, color="tab:blue", lw=3.5, label="Outer housing, R_out (fixed)")
ax3.fill_between(z_line, r_in_line, R_OUT, color="lightgray", alpha=0.5, label="Rubber annulus")
rho = groove_radius_of_curvature(GROOVE_DEPTH, GROOVE_HALF_WIDTH)
z_mid = LZ / 2.0
ax3.annotate(f"groove (depth={GROOVE_DEPTH}, rho={rho:.4f})",
             xy=(z_mid, R_IN0 - GROOVE_DEPTH),
             xytext=(0.05, R_IN0 - GROOVE_DEPTH - 0.12),
             arrowprops=dict(arrowstyle="->", lw=2), fontsize=FS_ANNOT)
ax3.set_xlabel("z (axial)", fontsize=FS_AXIS); ax3.set_ylabel("radius", fontsize=FS_AXIS)
ax3.tick_params(labelsize=FS_TICK)
ax3.set_ylim(0.25, 1.05)
ax3.set_title("Meridian cross-section (theta=0), zoomed\nshowing the groove -- the one explicit\n"
              "finite-radius stress-concentration feature", fontsize=FS_TITLE)
ax3.legend(loc="upper center", fontsize=FS_LEGEND, framealpha=0.9)

fig.suptitle("B3: rocking elastomeric rubber-mount bushing -- geometry", fontsize=FS_SUPTITLE)
fig.tight_layout()

# 1) SAVE to Drive (survives the Colab session, same directory pattern
#    the mesh-convergence run already uses for its own JSON output).
out_path = f'{R}/b3/B3_geometry.png'
fig.savefig(out_path, dpi=200)
print('Saved:', out_path)

# 2) DISPLAY inline, right here in the notebook's own output.
plt.show()

print('\nDone. This figure used the exact same geometry-generation code '
      'and physical parameters as the real mesh-convergence study -- only '
      'the mesh resolution differs, chosen for visual clarity.')
