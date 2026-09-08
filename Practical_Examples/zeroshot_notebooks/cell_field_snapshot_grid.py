# =====================================================================
#  CELL -- Item #12: a side-by-side grid of converged displacement-field
#  snapshots across mesh resolutions (B1, Neo-Hookean), in the figure
#  style Omar pointed to from Timon's own group's papers (VINO Figs.
#  7a/8d/9d; the PFEM/NOWS screenshots: one panel per resolution/case,
#  arranged in a row, labeled by point count).
#
#  NO NEW SOLVING HAPPENS HERE. This reuses item #4's own already-
#  converged checkpoints on Drive (coarse_B1_neo_hookean_Q4_N{N}.pt,
#  the same ~14.86h of real GPU work already committed and written into
#  the report) -- each one already holds the converged solution's
#  free-DOF displacement. Loading + reshaping that into an image costs
#  no new GPU time; the only "new" work here is generating the figure
#  itself, which is why this cell needs no GPU and finishes in seconds.
#
#  WHAT THIS PRODUCES: one PNG with 7 panels (N=51/101/201/401/701/
#  1001/1401), each showing |u| (displacement magnitude) over the B1
#  domain, side by side -- the same panels visually confirm the
#  convergence story Table 6a's numbers already tell (the field pattern
#  barely changes past N=201, exactly where Table 6a's own L2 error
#  gets small), but as a picture instead of a table of numbers.
# =====================================================================
import os, sys

from google.colab import drive
drive.mount('/content/drive')

REPO = '/content/OMAR'
import subprocess
def run(cmd):
    print('$', ' '.join(str(c) for c in cmd), flush=True)
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    for line in p.stdout:
        print(line, end='', flush=True)
    p.wait()
    if p.returncode != 0:
        raise subprocess.CalledProcessError(p.returncode, cmd)

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

# Same stale-in-kernel-module-cache class of issue as item #13's own
# notebook -- force a clean re-import every run of this cell regardless
# of what an earlier cell in this same long-lived kernel already
# imported.
for _mod_name in list(sys.modules):
    if _mod_name == 'omar_pfem' or _mod_name.startswith('omar_pfem.'):
        del sys.modules[_mod_name]

CHECKPOINT_DIR = '/content/drive/MyDrive/pfem_ckpt'
OUT_DIR = '/content/drive/MyDrive/pfem_run/figures'

from omar_pfem.field_snapshot_grid import make_resolution_grid_figure

RESOLUTIONS = [51, 101, 201, 401, 701, 1001, 1401]
out_path = make_resolution_grid_figure(
    RESOLUTIONS, CHECKPOINT_DIR, os.path.join(OUT_DIR, 'fig_B1_resolution_grid.png'))

print('\nDone. Figure saved to Drive at:', out_path)
print('Fetch it from Drive (search_files/download_file_content) to bring it back for the report.')
