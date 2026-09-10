# =====================================================================
#  CELL -- torch-fem's own ACCURACY and MESH CONVERGENCE against "ours"
#  own fine ~10M-DOF reference (Timon round-9, items 2+6, 2026-09-10).
#
#  WHY THIS EXISTS: Timon's reply to the standalone torch-fem-comparison
#  question was explicit: "For the paper, timing should only be
#  compared after the methods demonstrate comparable accuracy and mesh
#  convergence," plus a direct question, "Did you compare the accuracy
#  of your FEM implementation with torch-FEM?" The existing
#  _correctness_check only ever ran at N=11 with a loose float32
#  tolerance (a sanity check, not a real accuracy/convergence answer).
#
#  METHODOLOGY (Omar's own correction after re-reading Timon's email
#  line by line): rather than a simpler ours-vs-torchfem pointwise
#  displacement diff, this runs torch-fem through the EXACT SAME
#  study "ours" own Table 6a/6b/6c already uses
#  (high_dof_convergence_study.py: evaluate against a shared fine
#  ~10M-DOF reference via exact FE point location, fit a convergence
#  rate across several N). This answers accuracy AND mesh convergence
#  together, with numbers directly comparable -- same units, same
#  reference, same convention -- to "ours" own already-published L2/H1
#  numbers at the same N, not a separate ad hoc check.
#
#  PREREQUISITE APPLIED FIRST: torch-fem now runs at matched FP64/1e-8
#  precision (torchfem_comparison.py's solve_theirs), not the old
#  float32/1e-3 -- see that file's own docstring for the real bug this
#  fixed (near_null_space()/skew() hardcoding torch.eye(3) at float32).
#
#  COST/RISK, per PROJECT_STATUS.md's own estimate: torch-fem's peak
#  GPU memory at float32 was already 3.3/9.7/19.5/38.1 GB at
#  N=401/701/1001/1401 -- float64 roughly doubles that. This cell
#  deliberately STOPS at N=701 (~19-20GB expected, should fit most
#  Colab GPUs) rather than also attempting N=1001/1401 in the same run,
#  so a possible OOM at the larger sizes doesn't lose progress on the
#  cheap resolutions. A SEPARATE cell/notebook should attempt
#  N=1001/1401 once this one's memory usage at N=701 is known.
#
#  "ours" OWN NUMBERS AT THE SAME RESOLUTIONS ARE ALREADY COMMITTED --
#  no need to re-solve "ours" here at all:
#  omar_pfem/highdof_stress_qoi_results/high_dof_stress_qoi_B1_neo_hookean_mgv_N701_1001_1401.json
#  has "ours" l2_rel_error/h1_semi_rel_error at N=51/101/201/401/701/
#  1001/1401, all against this SAME fine_N=2236 reference. This cell's
#  own printed summary puts torch-fem's numbers side by side with those
#  already-committed "ours" values for a direct comparison.
#
#  RESUMABLE: run_convergence_study checks its own out_json and skips
#  resolutions already present, so a disconnect loses at most the
#  resolution in progress. The fine reference itself RESUMES from
#  "ours" own already-converged checkpoint (pfem_ckpt/fine_B1_
#  neo_hookean_Q4_N2236.pt) rather than re-solving the single most
#  expensive problem in the whole study.
# =====================================================================
import json
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

run([sys.executable, '-m', 'pip', 'install', '-q', 'torch-fem'])

WORK = f'{REPO}/Practical_Examples'
os.chdir(WORK)
sys.path.insert(0, WORK)

for _mod_name in list(sys.modules):
    if _mod_name == 'omar_pfem' or _mod_name.startswith('omar_pfem.'):
        del sys.modules[_mod_name]

import torch
print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available()
      else 'NONE -- Runtime > Change runtime type > GPU required for a fair, comparable memory number')

# Cheap CPU-scale sanity check before spending any real GPU time -- same
# discipline as every other GPU run in this project.
run([sys.executable, '-m', 'omar_pfem.torchfem_comparison', '11', '1e-8'])

R = '/content/drive/MyDrive/pfem_run'
OUT_JSON = f'{R}/torchfem_convergence_vs_fine_reference.json'
os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)

# "ours" own already-converged fine ~10M-DOF reference -- resumed, not
# re-solved (see this file's own header comment).
CHECKPOINT_DIR = '/content/drive/MyDrive/pfem_ckpt'

# Deliberately stops at 701 this run -- see header comment on the
# N=1001/1401 memory risk.
RESOLUTIONS = [51, 101, 201, 401, 701]

from omar_pfem.torchfem_comparison import run_convergence_study
rows = run_convergence_study(RESOLUTIONS, OUT_JSON, checkpoint_dir=CHECKPOINT_DIR,
                              fine_N=2236, tol=1e-8)

print('\nDone. Results:', OUT_JSON)

# Side-by-side against "ours" own already-committed numbers at the same
# N and fine reference -- no re-solve of "ours" needed for this.
OURS_JSON = f'{REPO}/Practical_Examples/omar_pfem/highdof_stress_qoi_results/high_dof_stress_qoi_B1_neo_hookean_mgv_N701_1001_1401.json'
if os.path.exists(OURS_JSON):
    with open(OURS_JSON) as f:
        ours_rows = {r['N']: r for r in json.load(f)['orders']['Q4']['rows']}
    print('\nN      ours L2_rel      torchfem L2_rel   ours H1_rel      torchfem H1_rel   torchfem wall_s  torchfem peak_MB')
    for row in rows:
        N = row['N']
        o = ours_rows.get(N)
        if o:
            print(f"{N:<6} {o['l2_rel_error']:<14.3e} {row['l2_rel']:<17.3e} "
                  f"{o['h1_semi_rel_error']:<16.3e} {row['h1_semi_rel']:<17.3e} "
                  f"{row['torchfem_wall_clock_s']:<16.2f} {row['torchfem_peak_mem_mb']}")
        else:
            print(f"{N:<6} (no 'ours' row at this N)")
else:
    print(f"\n'ours' comparison file not found at {OURS_JSON} -- printing torch-fem's own rows only:")
    for row in rows:
        print(row)

print('\nWatch torchfem_peak_mem_mb at N=701 before attempting N=1001/1401 in a separate run --')
print('float32 was 9.7GB at N=701 and 19.5/38.1GB at N=1001/1401; float64 roughly doubles that.')
