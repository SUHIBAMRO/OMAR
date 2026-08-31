# =====================================================================
#  THE TWO METRICS DISAGREE ABOUT WHICH MODEL IS BETTER — resolve it
#
#  The energy-vs-error run answered its question and raised a bigger one.
#  Pi fell in both arms:
#
#      arm     epoch 50        epoch 450       descent   roughness
#      N=21    -2.4788e-02  -> -4.3379e-02     44% -> 76%   2.32x -> 1.74x
#      N=33    -3.2643e-02  -> -4.3995e-02     59% -> 80%   2.04x -> 1.73x
#
#  So training IS working on its own objective, and the field is getting
#  SMOOTHER as well as lower in energy. The cell concluded from the rising
#  validation error that the functional must prefer some field other than
#  the FEM solution. That conclusion does not survive the per-sample
#  numbers, and it is withdrawn.
#
#  BECAUSE THE PROBE'S OWN ERROR WENT DOWN, on every sample it looked at.
#  Same samples, same two checkpoints, epoch 50 -> epoch 450:
#
#      N=21 arm, on N=21   0.9763 -> 0.9023   0.7287 -> 0.4474
#                          0.9969 -> 0.9330   0.5877 -> 0.0930
#      N=33 arm, on N=33   0.4826 -> 0.4832   0.6554 -> 0.6250
#                          0.7798 -> 0.6734   0.7687 -> 0.7175
#
#  Eight samples, better or level on all eight, and one of them landing at
#  0.0930 -- which is B1 territory. The correlations move the same way:
#  the N=21 arm's epoch-50 model had correlations of -0.32 and -0.41 on two
#  samples, and at epoch 450 every correlation is positive.
#
#  Meanwhile the trainer says epoch 450 is 1.27x WORSE (0.9622 -> 1.2255).
#
#  THE TWO NUMBERS ARE NOT THE SAME METRIC.
#
#      the trainer   0.5 * ( rms(e_u)/rms(u) + rms(e_v)/rms(v) )
#      the probe     rms(e) / rms(uv_exact)        both components at once
#
#  The trainer's is an average of two per-component ratios, each divided by
#  its OWN component's size. If one displacement component is much smaller
#  than the other, its ratio dominates the average, and the metric mostly
#  reports the weaker component regardless of how well the field as a whole
#  is predicted. The probe's weights each component by how large it
#  actually is.
#
#  WHY THIS MATTERS MORE THAN THE ORIGINAL QUESTION. That trainer metric is
#  what early stopping used. If it ranks these models backwards, then every
#  B2 run in this study stopped at its FIRST validation event and kept the
#  worse model -- which is exactly the pattern on record, in every single
#  run -- and the ~1.0 numbers in the report's B2 zero-shot row are a
#  metric artefact sitting on top of a model that was still improving.
#
#  This does not yet say the report is wrong. Four samples are four
#  samples. That is what this run is for.
#
#  WHAT IT DOES. Both metrics, on ALL 100 val samples of each resolution,
#  for both checkpoints of both arms -- plus the two per-component ratios
#  separately and the size of each component, which is what would expose a
#  small-component effect if there is one.
#
#      the trainer's metric goes UP while the combined one goes DOWN
#          -> the metric ranks the models backwards. Early stopping has
#             been keeping the worse model on every B2 run, and the B2
#             zero-shot numbers need re-reading before they go in the
#             report as they stand.
#
#      both go UP
#          -> the four probe samples were unrepresentative, the trainer is
#             right, and the energy-vs-error reading stands as the cell
#             printed it.
#
#  CPU, a few minutes, trains nothing, writes one JSON per checkpoint.
# =====================================================================
import json
import os
import subprocess
import sys

REPO = '/content/OMAR'
BRANCH = 'claude/claude-code-question-d307wp'
R = '/content/drive/MyDrive/pfem_run'
WORKDIR = f'{R}/b2_single_resolution'
ARMS = ['21', '33']
CKPTS = ['model_best.pt', 'model_final.pt']

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

for res in ARMS:
    d = f'{WORKDIR}/N{res}'
    for ck in CKPTS:
        path = f'{d}/{ck}'
        if not os.path.exists(path):
            print(f'\n[N={res}] {ck} missing -- skipped')
            continue
        print('\n\n' + '#' * 78)
        print(f'#  N={res} alone   {ck}')
        print('#' * 78)
        run([sys.executable, '-u', '-m', 'omar_pfem.compare_val_metrics',
             '--geometry', 'B2', '--material', 'neo_hookean',
             '--cache', f'{d}/samples_cache.pt', '--checkpoint', path,
             '--out_json', f'{d}/val_metrics_{ck.replace(".pt", "")}.json',
             '--cpu'])

# ---- the comparison ---------------------------------------------------
print('\n' + '=' * 78)
print('THE SAME TWO MODELS, THE SAME 100 SAMPLES, TWO METRICS')
print('=' * 78)
print(f"  {'arm':<8}{'checkpoint':<16}{'trainer':>10}{'combined':>10}"
      f"{'rel_u':>9}{'rel_v':>9}{'rms(v)/rms(u)':>15}")
tbl = {}
for res in ARMS:
    for ck in CKPTS:
        jp = f'{WORKDIR}/N{res}/val_metrics_{ck.replace(".pt", "")}.json'
        if not os.path.exists(jp):
            continue
        j = json.load(open(jp))
        # the arm's OWN resolution is the row the trainer validated on
        own = [r for r in j['rows'] if r['N'] == int(res)]
        r = own[0] if own else j['rows'][0]
        tbl[(res, ck)] = r
        print(f"  N={res:<6}{ck:<16}{r['trainer_metric']:>10.4f}"
              f"{r['combined_rel_L2']:>10.4f}{r['rel_u']:>9.4f}"
              f"{r['rel_v']:>9.4f}{r['v_over_u']:>15.4f}")

print()
verdict = []
for res in ARMS:
    a, b = tbl.get((res, 'model_best.pt')), tbl.get((res, 'model_final.pt'))
    if not (a and b):
        continue
    t_up = b['trainer_metric'] > a['trainer_metric']
    c_dn = b['combined_rel_L2'] < a['combined_rel_L2']
    verdict.append((t_up, c_dn))
    print(f"  N={res}: trainer {a['trainer_metric']:.4f} -> "
          f"{b['trainer_metric']:.4f} ({'UP' if t_up else 'DOWN'})   "
          f"combined {a['combined_rel_L2']:.4f} -> "
          f"{b['combined_rel_L2']:.4f} "
          f"({'DOWN' if c_dn else 'UP'})")

print('\n' + '=' * 78)
if verdict and all(t and c for t, c in verdict):
    print('THE METRIC RANKS THE MODELS BACKWARDS. The trainer\'s number rises')
    print('while the error over both components falls, on the full 100')
    print('samples and in every arm. Early stopping used the rising number,')
    print('so every B2 run in this study stopped at its first validation and')
    print('kept the worse model -- which is the pattern in every B2 run on')
    print('record. Look at rel_u against rel_v above: the component with the')
    print('larger ratio is the one driving the average.')
    print()
    print('CONSEQUENCES, in order:')
    print('  1. the B2 zero-shot numbers in the report are this metric and')
    print('     need re-reading before they stand as a result;')
    print('  2. B1\'s numbers are the SAME metric and must be re-checked the')
    print('     same way -- if B1 is unaffected, that is itself the reason')
    print('     the two geometries looked so different;')
    print('  3. only then is it worth retraining B2 with early stopping on a')
    print('     metric that does not invert.')
elif verdict and all(t and not c for t, c in verdict):
    print('BOTH METRICS AGREE the epoch-450 model is worse. The four probe')
    print('samples were unrepresentative, and the energy-vs-error reading')
    print('stands: Pi falls while the error rises.')
else:
    print('MIXED across the arms -- read the two lines above separately.')
print('=' * 78)
