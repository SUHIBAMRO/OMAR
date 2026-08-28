# =====================================================================
#  CELL — Method of manufactured solutions (Timon round 5, point 9)
#  "We can compare Q4, Q9 and the physics-informed Transolver against
#   exactly the same analytical solution in L2, H1 and energy norms and
#   also examine stress errors."
#
#  THIS CELL IS THE FEM HALF: Q4 and Q9 against the analytic solution.
#  The operator half needs a body-force term in the energy functional and
#  a body-force input channel, which is a separate piece of work -- see
#  omar_pfem/mms_study.py's docstring and PROJECT_STATUS.md.
#
#  Why a manufactured solution and not a body-force-free exact one: a
#  body-force-free exact solution on this geometry is a homogeneous
#  deformation, which Q4 reproduces to machine precision, so the study
#  would measure round-off and distinguish nothing. The reasoning is
#  written out in full in the module docstring.
#
#  The study validates itself: if the body force were wrong, the observed
#  convergence rates would collapse. Expected Q4 L2~h^2 / H1~h^1,
#  Q9 L2~h^3 / H1~h^2. The cell prints the rates and a pass/fail verdict.
#
#  GPU: helpful but not required. FP64, so on a T4/L4 the GPU is barely
#  faster than the CPU -- if you only have those, --cpu is fine.
#  Time: minutes for N up to 33; N=65 Q9 is the expensive one.
#  Resumable: each (order, N) row is appended to the JSON as it finishes.
# =====================================================================
import os, subprocess, sys, json

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
BRANCH = 'claude/claude-code-question-d307wp'
if not os.path.isdir(REPO):
    run(['git', 'clone', '-b', BRANCH, 'https://github.com/SUHIBAMRO/OMAR.git', REPO])
else:
    run(['git', '-C', REPO, 'fetch', 'origin', BRANCH])
    run(['git', '-C', REPO, 'checkout', BRANCH])
    run(['git', '-C', REPO, 'reset', '--hard', f'origin/{BRANCH}'])

WORK = f'{REPO}/Practical_Examples'
os.chdir(WORK)
sys.path.insert(0, WORK)

import torch
HAS_GPU = torch.cuda.is_available()
print('GPU:', torch.cuda.get_device_name(0) if HAS_GPU else 'NONE (fine for this)')

OUT = '/content/drive/MyDrive/pfem_run/mms'
os.makedirs(OUT, exist_ok=True)

MATERIAL = 'neo_hookean'      # Timon: "preferably Neo-Hookean"
NS = '5,9,17,33'              # add 65 for a fourth point if time allows
OUT_JSON = f'{OUT}/mms_B1_{MATERIAL}.json'

# Checks first: the manufactured solution must vanish on the boundary and
# the body force must match a finite-difference divergence. If either fails
# nothing below is meaningful, and the script stops on its own.
run([sys.executable, '-u', '-m', 'omar_pfem.mms_study', '--verify',
     '--material', MATERIAL])

cmd = [sys.executable, '-u', '-m', 'omar_pfem.mms_study',
       '--material', MATERIAL, '--orders', 'Q4,Q9', '--Ns', NS,
       '--out_json', OUT_JSON]
if not HAS_GPU:
    cmd.append('--cpu')
run(cmd)

# ------------------------------------------------------------------ read
rep = json.load(open(OUT_JSON))
print('\n' + '=' * 74)
print('MMS RESULT —', rep['manufactured_solution'])
print('=' * 74)
hdr = f"{'order':<6}{'N':>5}{'DOF':>9}{'L2':>12}{'H1 semi':>12}{'stress':>12}{'energy':>12}"
print(hdr); print('-' * len(hdr))
for r in rep['rows']:
    print(f"{r['order']:<6}{r['N']:>5}{r['n_dof']:>9,}{r['L2_rel']:>12.3e}"
          f"{r['H1_semi_rel']:>12.3e}{r['stress_rel_L2']:>12.3e}{r['energy_rel']:>12.3e}")

if 'convergence_rates' in rep:
    print('\nConvergence rates (expected in brackets):')
    for order, d in rep['convergence_rates'].items():
        bits = ', '.join(f"{n} {v['rate']:.2f} [{v['expected']}]" for n, v in d.items())
        print(f'  {order}: {bits}')
    for order, verdict in rep.get('rate_check', {}).items():
        print(f'  {order}: {verdict}')
    if all(v == 'as expected' for v in rep.get('rate_check', {}).values()):
        print('\nBoth orders converge at their theoretical rates, which means the')
        print('manufactured problem, the body-force assembly, the solver and the')
        print('error norms are all consistent. The FEM half of point 9 is done.')
    else:
        print('\nA rate is off. Do NOT report these numbers -- something in the')
        print('manufactured problem or its assembly is wrong. Check --verify first.')

print(f'\nJSON at {OUT_JSON} -- commit it into '
      f'Practical_Examples/omar_pfem/point9_results/')
