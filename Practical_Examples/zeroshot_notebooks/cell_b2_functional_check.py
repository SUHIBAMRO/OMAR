# =====================================================================
#  DIAGNOSTIC — is the cached FEM solution a minimum of the trainer's Pi?
#
#  Two candidates have now been tested and neither explains the failure:
#  loss_force_norm (necessary, not sufficient) and batch size (0.9888 vs
#  0.9444 at matched steps, which is noise on curves swinging 0.94-1.45).
#
#  The clue nobody had used: a relative error of 1.0 is not an arbitrary
#  bad number. The metric is 0.5*(rms(e_u)/rms(u) + rms(e_v)/rms(v)), and
#  putting uv_pred = 0 into it gives exactly 1. The three B2 runs all sat
#  at 0.94-1.23. So the network is probably predicting nothing -- and it
#  may be doing so CORRECTLY, because a network that minimizes Pi goes
#  where Pi's minimum is.
#
#  So the question is not about the optimizer. It is whether Pi's
#  minimum, built from this cache's node_forces, is anywhere near this
#  cache's uv_exact. No training needed to find out.
#
#  The test scans Pi(s * uv_exact) for s from 0 to 1.5. If the true
#  solution is the minimizer, s=1 wins. If s is near 0, the work term is
#  too weak, the trainer is correctly finding zero, and no optimizer
#  setting will help -- the fault is in the data.
#
#  It also asserts that its stand-in field actually reproduces uv_exact
#  through the soft-Dirichlet ramp before reporting anything, so a wrong
#  assumption about the mask cannot masquerade as a physics result.
#
#  SECONDS. No GPU needed. Reads the cache, writes nothing.
# =====================================================================
import os
import subprocess
import sys

REPO = '/content/OMAR'
BRANCH = 'claude/claude-code-question-d307wp'
R = '/content/drive/MyDrive/pfem_run'

from google.colab import drive
drive.mount('/content/drive')


def run(cmd):
    print('$', ' '.join(str(c) for c in cmd), flush=True)
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                         stderr=subprocess.STDOUT, text=True, bufsize=1)
    for line in p.stdout:
        print(line, end='', flush=True)
    p.wait()
    return p.returncode


if not os.path.isdir(REPO):
    run(['git', 'clone', '-b', BRANCH,
         'https://github.com/SUHIBAMRO/OMAR.git', REPO])
else:
    run(['git', '-C', REPO, 'fetch', 'origin', BRANCH])
    run(['git', '-C', REPO, 'reset', '--hard', f'origin/{BRANCH}'])
run(['git', '-C', REPO, 'log', '--oneline', '-1'])

WORK = f'{REPO}/Practical_Examples'
os.chdir(WORK)
if WORK not in sys.path:
    sys.path.insert(0, WORK)

cache = f'{R}/zeroshot_B2_neo_hookean/samples_cache.pt'
assert os.path.exists(cache), f'no cache at {cache}'
rc = run([sys.executable, '-u', '-m', 'omar_pfem.test_b2_zeroshot_functional',
          '--cache', cache, '--material', 'neo_hookean', '--cpu'])
if rc != 0:
    print('\nThe test stopped. If it stopped on its own assertion about the '
          'stand-in not reproducing uv_exact, that is itself the finding -- '
          'send it over.')
