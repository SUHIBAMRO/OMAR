# =====================================================================
#  DIAGNOSTIC — what is the trained B2 model doing, and what is it fed?
#
#  The previous test settled the previous question. Pi(s * uv_exact) was
#  scanned over s on six samples at both training resolutions and the
#  minimum landed at s = 1.000 every single time, with |W|/U = 2.00 to
#  three decimals. So:
#
#      the cache is fine, the work term is fine, and the functional the
#      trainer minimizes really is minimized by the FEM solution.
#
#  Which rules out the data and leaves the training path. This looks at
#  that path directly, using the checkpoint already on Drive. It trains
#  nothing, writes nothing, and takes seconds on CPU.
#
#  Four measurements:
#
#    1. THE INPUT CHANNELS as the model receives them. fun_material is
#       (E, nu, f_x, f_y) fed RAW -- there is no normalization anywhere in
#       this path. For B2 the load is an inner-edge traction, so f is
#       exactly zero on every node off that boundary, roughly 95% of them
#       at N=21, and after the load repair what remains is 13-21x smaller
#       than it was. E is around 1000. The ratio rms(f)/rms(E) is printed.
#
#    2. WHAT IT PREDICTS -- rms(pred)/rms(uv_exact) and the correlation.
#       An error of 1.0 is what predicting zero scores, but it is also
#       roughly what predicting noise of the right size scores. Those are
#       different failures and this separates them.
#
#    3. WHERE IT SITS ON Pi -- Pi(pred) against Pi(0) = 0 and the
#       Pi(uv_exact) the previous test measured. That says how much of the
#       descent actually happened.
#
#    4. DOES IT USE ITS INPUT -- four different samples on ONE mesh. If
#       the predictions barely differ while the targets differ a lot, the
#       model has collapsed to a function of the coordinates and is
#       ignoring the fields entirely. That would explain an error flat in
#       N, flat across materials, and stuck at 1.0.
#
#  CPU is fine. No GPU needed.
# =====================================================================
import os
import subprocess
import sys

REPO = '/content/OMAR'
BRANCH = 'claude/claude-code-question-d307wp'
R = '/content/drive/MyDrive/pfem_run'
CASE = f'{R}/zeroshot_B2_neo_hookean'

from google.colab import drive
drive.mount('/content/drive')


def run(cmd):
    print('$', ' '.join(str(c) for c in cmd), flush=True)
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                         stderr=subprocess.STDOUT, text=True, bufsize=1)
    for line in p.stdout:
        print(line, end='', flush=True)
    p.wait()
    if p.returncode != 0:
        raise subprocess.CalledProcessError(p.returncode, cmd)


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

cache = f'{CASE}/samples_cache.pt'
ckpt = f'{CASE}/model_best.pt'
for f in (cache, ckpt):
    assert os.path.exists(f), f'missing: {f}'
# the sentinel the retrain notebook writes; without it the checkpoint may
# predate loss_force_norm and probing it would say nothing about the runs
# actually reported
sentinel = f'{CASE}/.trained_with_loss_force_norm'
print('checkpoint trained with the force normalisation:',
      'YES' if os.path.exists(sentinel) else 'NO -- this probe is meaningless')

run([sys.executable, '-u', '-m', 'omar_pfem.test_b2_zeroshot_model',
     '--cache', cache, '--checkpoint', ckpt,
     '--material', 'neo_hookean', '--cpu'])

print('\nNothing was written and nothing trained. Send the whole block over.')
