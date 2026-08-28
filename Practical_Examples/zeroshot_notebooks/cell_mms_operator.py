# =====================================================================
#  CELL — MMS, the operator third (Timon round 5, point 9)
#  Completes the three-way comparison. Run Round6_MMS.ipynb first for the
#  Q4/Q9 half; this cell trains the physics-informed operator on the same
#  manufactured family and scores it with the SAME error routine, so all
#  three numbers are directly comparable.
#
#  WHY A NEW MODEL AND NOT AN EXISTING CHECKPOINT: none of the six trained
#  models can run on a manufactured problem. Their energy functional has no
#  body-force term, their inputs have no body-force channel, and they fix
#  one edge instead of four. So this is a new training run.
#
#  READ THIS BEFORE INTERPRETING THE RESULT — the operator minimizes the
#  SAME discrete functional over the SAME Q4 space as the Q4 solver, and the
#  minimizer of that functional IS the Q4 solution. So the operator CANNOT
#  beat Q4 at this mesh; that is arithmetic, not a finding. The number to
#  look at is the ratio operator/Q4: 1.0 would mean the network has fully
#  solved the variational problem. Anything below 1.0 means something is
#  wrong, and the cell says so.
#
#  Labels are free here (u* is analytic), but they are NOT used in
#  training -- the loss is the energy. They are only the scoring truth.
#
#  GPU: yes, strongly preferred. FP32 training, so a T4/L4 is fine.
#  Time: ~20-40 min at N=17 with the defaults below.
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
print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NONE')

OUT = '/content/drive/MyDrive/pfem_run/mms'
os.makedirs(OUT, exist_ok=True)

MATERIAL = 'neo_hookean'
N = 17            # same mesh as the Q4/Q9 rows it is compared against
OUT_JSON = f'{OUT}/mms_operator_B1_{MATERIAL}.json'
OUT_DIR = f'{OUT}/operator_N{N}'

# The energy functional is checked before training: the Q4 FEM solution must
# be the exact minimizer of the Pi the network is about to minimize. If the
# work term were mis-scaled the network would train happily toward the wrong
# field and nothing in the loss curve would look wrong.
print('\n--- verifying the energy functional against the FEM solver ---')
run([sys.executable, '-u', '-m', 'omar_pfem.test_mms_operator'])

cmd = [sys.executable, '-u', '-m', 'omar_pfem.mms_operator',
       '--material', MATERIAL, '--N', str(N),
       '--ntrain', '64', '--ntest', '16',
       '--epochs', '2000', '--batch_size', '8',
       '--validate_every', '25',
       '--out_dir', OUT_DIR, '--out_json', OUT_JSON]
if not torch.cuda.is_available():
    cmd.append('--cpu')
run(cmd)

# ------------------------------------------------------------------ read
R = json.load(open(OUT_JSON))
op, ref = R['operator_on_the_reference_member'], R['fem_reference_same_mesh']
print('\n' + '=' * 74)
print(f"THREE-WAY at N={R['N']} ({R['n_dof']} DOF) — all against the same u*")
print('=' * 74)
hdr = f"{'method':<24}{'L2':>12}{'H1 semi':>12}{'stress':>12}{'energy':>12}"
print(hdr); print('-' * len(hdr))
for name, d in (('Q4 (same mesh)', ref['Q4']), ('Q9 (same N)', ref['Q9']),
                ('operator', op)):
    print(f"{name:<24}{d['L2_rel']:>12.3e}{d['H1_semi_rel']:>12.3e}"
          f"{d['stress_rel_L2']:>12.3e}{d['energy_rel']:>12.3e}")

r = R['operator_over_Q4_L2']
print(f"\noperator / Q4 in L2: {r:.2f}x")
print('\nCeiling, which belongs in any sentence quoting these numbers:')
print(' ', R['ceiling'])
print(f"\nTraining: {R['training']['opt_steps']:,} steps, "
      f"{R['training']['train_wall_clock_s']/60:.1f} min, "
      f"label cost {R['training']['label_cost']}")
print(f"\nJSON at {OUT_JSON} — commit it into "
      f"Practical_Examples/omar_pfem/point9_results/")
