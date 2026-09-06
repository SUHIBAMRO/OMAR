# =====================================================================
#  CELL -- Re-run the point-1 tolerance-vs-cost study with the new
#  peak-stress QoI (Timon round-8, point 1, sharpened by Omar): Table 6a
#  already shows B1 x Neo-Hookean needs tens of minutes to hours of FEM
#  time to reach a reasonable accuracy against a ~10M-DOF reference, but
#  its L2/H1/energy columns are all norms of the error FIELD, not the
#  literal QoI example Timon's email named ("maximum stresses or
#  similar"). high_dof_convergence_study.py now also reports a peak PK1
#  stress QoI (compute_peak_stress_error) alongside those three norms --
#  this cell re-runs the SAME sweep Table 6a used (same geometry,
#  material, resolutions, fine reference) so the new stress numbers land
#  in a directly comparable table, not a different study.
#
#  CHECKPOINT_DIR below points at the ORIGINAL checkpoint folder from
#  whenever Table 6a itself was generated -- found 2026-09-06 by
#  searching Drive directly: /content/drive/MyDrive/pfem_ckpt contains
#  fine_B1_neo_hookean_Q4_N2236.pt (the ~10M-DOF reference itself, the
#  single most expensive solve in the whole study) plus
#  coarse_B1_neo_hookean_Q4_N1001.pt and _N1401.pt (the two next most
#  expensive). solve_one() returns an already-solved field IMMEDIATELY
#  (no re-solve at all) when a checkpoint shows that solve already
#  finished, so those three resolutions cost seconds here, not hours.
#
#  Checked and NOT found anywhere on Drive: checkpoints for N=51, 101,
#  201, 401, 701 -- those five must be solved fresh. From this project's
#  own previously-recorded Table 6a timings (wall-clock in seconds):
#  N=51 227, N=101 445, N=201 884, N=401 1802, N=701 5940 -- summing to
#  ~2.6 h. That is the real expected cost of this run with the
#  checkpoints below, NOT the 8-15+ h a fully-fresh run would need.
#
#  If CHECKPOINT_DIR below turns out to be wrong (moved/renamed since),
#  the run simply falls back to solving everything fresh -- resumable at
#  both the outer solve level and CG's own internal level (via
#  --cg_checkpoint_every), so a disconnect anywhere loses at most a few
#  thousand CG iterations, never a whole solve.
#
#  Only the peak-stress QoI is new; L2/H1/energy are recomputed too
#  (cheap, seconds) simply because they come from the same function call
#  -- no separate flag needed to get them again.
# =====================================================================
import os, subprocess, sys, time

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
    run(['git', '-C', REPO, 'fetch', 'origin',
                    'claude/claude-code-question-d307wp'])
    run(['git', '-C', REPO, 'checkout',
                    'claude/claude-code-question-d307wp'])
    run(['git', '-C', REPO, 'reset', '--hard',
                    'origin/claude/claude-code-question-d307wp'])

WORK = f'{REPO}/Practical_Examples'
os.chdir(WORK)
sys.path.insert(0, WORK)

import torch
print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available()
      else 'NONE - Runtime > Change runtime type > GPU strongly recommended, '
           'this study is expensive even on GPU')

R = '/content/drive/MyDrive/pfem_run'

# Found on Drive 2026-09-06: the original Table 6a checkpoints live in
# pfem_ckpt (a sibling of pfem_run, not inside it). If this path has since
# moved, edit it here -- everything still works, just falls back to a
# fresh multi-hour run instead of reusing the ~10M-DOF reference solve.
CHECKPOINT_DIR = '/content/drive/MyDrive/pfem_ckpt'
OUT_JSON = f'{R}/high_dof_stress_qoi_B1_neo_hookean.json'

os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# Matches Table 6a exactly: same geometry, material, resolutions and
# fine reference, so the new peak-stress column is directly comparable
# to the existing L2/H1/energy columns, not a different study.
GEOMETRY = 'B1'
MATERIAL = 'neo_hookean'
RESOLUTIONS = '51,101,201,401,701,1001,1401'
FINE_N = 2236
ORDER = 'Q4'

t0 = time.time()
run([
    sys.executable, '-u', '-m', 'omar_pfem.high_dof_convergence_study',
    '--geometry', GEOMETRY, '--material', MATERIAL,
    '--resolutions', RESOLUTIONS, '--fine_N', str(FINE_N),
    '--orders', ORDER,
    '--checkpoint_dir', CHECKPOINT_DIR,
    '--out_json', OUT_JSON,
    '--cg_progress_every', '500',
    '--cg_checkpoint_every', '2000',
])
elapsed = time.time() - t0

print(f'\nDone in {elapsed/3600:.2f} h. Results: {OUT_JSON}')
print('Compare the new peak_stress_rel_err / stress_field_l2_rel columns '
      'against the existing l2/h1_semi/energy columns at the same '
      'resolutions -- this is the QoI Timon\'s email named as an example '
      '("maximum stresses or similar") for the point-1 tolerance-vs-cost '
      'check, which the original Table 6a did not directly report.')
print('If this was a fresh run (no pre-existing checkpoints), re-run this '
      'cell again after any disconnect -- both the outer solve and CG\'s '
      'own internal state resume automatically from CHECKPOINT_DIR.')
