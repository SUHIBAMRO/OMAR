# =====================================================================
#  DIAGNOSTIC — the trained model, B2 against B1, side by side
#
#  The first run of this probe returned real information and also showed
#  that two things in the probe itself were built wrong. Both are fixed
#  here, and both are worth naming because either would have produced a
#  confident wrong reading:
#
#    * it printed Pi(pred) with no Pi(uv_exact) for the SAME sample. The
#      functional test's Pi numbers are on train_samples and this reads
#      val_samples, which are different problems, so "Pi(pred) = -2.7e-02"
#      could not be compared with anything at all. Both are computed here
#      now, per sample, and what gets printed is the fraction of the
#      available descent the model actually captured.
#
#    * it put the input-channel scales at N=21 and N=33 next to each other
#      as if the difference were a mesh effect. It is not separable that
#      way -- the cache seeds each resolution differently (seed_base =
#      10_000 * N), so those are different DRAWS as well as different
#      meshes. This rebuilds ONE fixed seed on BOTH meshes, which is the
#      controlled version, and prints the mesh-independent load total
#      beside the per-node scale.
#
#  AND IT ADDS THE CONTROL THAT WAS MISSING. B1 reaches 0.066 on the same
#  trainer, the same architecture and the same protocol. Any account of
#  B2's failure that would apply equally to B1 explains nothing. So this
#  runs the identical measurement on both and prints them for comparison.
#
#  What the first run already established, and this re-checks:
#    - the model is NOT predicting zero: rms(pred) 2.5-3.3e-03 against
#      targets 4.1e-03 to 1.28e-02;
#    - its amplitude is 2.5-4x too small, mean ratio 0.375;
#    - W/U at the prediction is about 2, so the prediction is already
#      stationary under rescaling and a stalled slide down a ray is NOT
#      the explanation;
#    - U(pred) sits at 1.6-2.3e-02 on every sample and both meshes, a
#      1.4x spread, while the targets span 3x -- the model emits a field
#      of nearly fixed strain energy whatever it is shown;
#    - it does respond to its input, but about five times too weakly
#      (variability 0.13 and 0.10 against the targets' 0.64 and 0.31).
#
#  WHAT IS NEW IN THIS RUN. Input normalisation has since been TESTED and
#  FALSIFIED -- 0.9910 against the 0.9986 baseline, a 0.8% move where B1
#  sits at 0.066. So the input scaling is not the obstacle, and two
#  structural candidates are left: the Dirichlet ramp and the parametric
#  family. This run measures the first one.
#
#  THE RAMP, AND WHY IT IS NOT THE OBVIOUS TEST. The network's output is
#  mask * raw. B2 uses TWO ramps -- x/R_out on u, y/R_out on v -- which
#  vanish on DIFFERENT edges; B1 uses one y/Ly on both components. The
#  tempting test is "can the mask represent uv_exact at all", and the
#  answer is already yes: the probe's own stand-in assertion reconstructs
#  uv_exact through the mask to machine precision on both geometries, and
#  it would have failed loudly otherwise. So representability is NOT the
#  question and a training run spent on that premise would be wasted.
#
#  The real question is how HARD a field the mask asks for. To output
#  uv_exact the network must emit uv_exact/mask, and near an edge where
#  the mask goes to zero that quotient grows. This run prints, per sample
#  and for BOTH geometries:
#
#      rms(raw demanded) / rms(the output)   -- how much bigger
#      peak/rms of the raw demanded          -- how uneven
#
#  If B2's numbers dwarf B1's, the ramp is a real obstacle and the fix is
#  a mask that does not vanish linearly. If they are comparable, the ramp
#  is exonerated and the last candidate is the parametric family
#  (ParametricFieldB2 varies with theta only, never with r).
#
#  CPU, a couple of minutes for both arms, writes nothing, trains nothing.
# =====================================================================
import os
import subprocess
import sys

REPO = '/content/OMAR'
BRANCH = 'claude/claude-code-question-d307wp'
R = '/content/drive/MyDrive/pfem_run'

# the failing case, and the working one it has to be read against
ARMS = [('B2', 'neo_hookean'), ('B1', 'neo_hookean')]

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

print('\n' + '=' * 78)
print('PRE-FLIGHT')
print('=' * 78)
PLAN = []
for geom, mat in ARMS:
    d = f'{R}/zeroshot_{geom}_{mat}'
    cache, ckpt = f'{d}/samples_cache.pt', f'{d}/model_best.pt'
    ok = os.path.exists(cache) and os.path.exists(ckpt)
    print(f'  {geom} x {mat:<14} cache {"YES" if os.path.exists(cache) else "no "}'
          f'   model {"YES" if os.path.exists(ckpt) else "no "}')
    if ok:
        PLAN.append((geom, mat, cache, ckpt))
    else:
        print(f'    -> skipped, and the comparison is weaker without it')
assert PLAN, 'neither arm has both a cache and a checkpoint'

for geom, mat, cache, ckpt in PLAN:
    print('\n\n' + '#' * 78)
    print(f'#  {geom} x {mat}')
    print('#' * 78)
    run([sys.executable, '-u', '-m', 'omar_pfem.test_b2_zeroshot_model',
         '--geometry', geom, '--material', mat,
         '--cache', cache, '--checkpoint', ckpt, '--cpu'])

print('\n' + '=' * 78)
print('Send BOTH blocks over together. The B1 arm is the control: anything')
print('that looks the same in both printouts is not what breaks B2.')
print('=' * 78)
print('Nothing was written and nothing trained.')
