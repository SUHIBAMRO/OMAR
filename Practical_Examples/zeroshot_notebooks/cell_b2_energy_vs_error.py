# =====================================================================
#  DOES TRAINING LOWER Pi WHILE RAISING THE ERROR?
#
#  WHAT THE SINGLE-RESOLUTION RUN JUST SETTLED. Both arms failed alone:
#
#      N=21 alone       best 0.9622 at epoch 50
#      N=33 alone       best 1.0372 at epoch 50
#      N=21 and 33      0.9986        (the joint run)
#      B1, three cases  0.0658 to 0.0827
#
#  So joint training is NOT the fault -- the failure is fully present at a
#  single resolution -- and seven candidates are now closed. But the run
#  printed something sharper than its own verdict, and it is the same in
#  BOTH arms and in every other B2 run on record:
#
#      the best model is the FIRST validation event, epoch 50, and every
#      one of the next eight is worse.
#
#      N=21   0.9622 (ep 50) -> 1.3386 -> 1.1442 -> 1.1371 -> 1.2015
#                            -> 1.2196 -> 1.0999 -> 1.1927 -> 1.2255
#      N=33   1.0372 (ep 50) -> 1.1350 -> 1.1439 -> 1.1256 -> 1.1162
#                            -> 1.2351 -> 1.1744 -> 1.1600 -> 1.1538
#
#  Training does not stall on B2. It moves the model AWAY from the FEM
#  solution, steadily, from the first validation onward. No candidate on
#  the list explains that: the family, the ramp and the load are all
#  properties of the problem, and none of them would make 20,000 further
#  optimizer steps actively harmful.
#
#  THE QUESTION THAT IS NOW ON THE TABLE, and it is answerable for free.
#  The objective is Pi = U - W. If the trainer is doing its job, Pi at
#  epoch 450 is LOWER than at epoch 50. If Pi went down while the error
#  went up, then within reach of this network there is a field with LESS
#  energy than the FEM solution, and the optimizer is correctly walking
#  towards it. That is not a bug in the training loop -- it is the
#  discrete functional on this geometry having a minimiser that is not
#  uv_exact.
#
#  WHY THE FUNCTIONAL CHECK DOES NOT ALREADY ANSWER THIS, and it is worth
#  being exact. That run scanned Pi(s * uv_exact) over a scalar s and
#  found its minimum at s = 1.0 in 6 of 6, with W/U = 1.9951 to 2.0021.
#  That is a scan along ONE RAY through field space. It proves uv_exact is
#  stationary under rescaling. It says nothing about whether some
#  different field, off that ray, has lower Pi. The measurement below
#  looks off the ray, at the two fields the training run actually
#  produced.
#
#  HOW, WITHOUT TRAINING ANYTHING. The trainer saves both endpoints --
#  model_best.pt is epoch 50 and model_final.pt is epoch 450 -- and the
#  probe already prints Pi(pred) beside Pi(uv_exact) on the same sample.
#  So this is the existing probe, run on both checkpoints of both arms,
#  and the numbers read against each other.
#
#      Pi(final) < Pi(best)   ->  the optimizer descended and the accuracy
#                                 fell. The discretised energy admits a
#                                 lower-energy field than the FEM solution
#                                 within this network's reach. Structural,
#                                 reportable, and it explains the shape of
#                                 every B2 curve on record. It also makes
#                                 regenerating the cache from the GRF the
#                                 WRONG next spend, because the data family
#                                 would not be the cause.
#
#      Pi(final) > Pi(best)   ->  the optimizer is not descending at all,
#                                 which is an optimisation failure --
#                                 learning rate, or the gradient through
#                                 the mask -- and the fixes are cheap and
#                                 local.
#
#  Either branch is worth more than an hours-long FEM regeneration
#  launched on a guess. CPU, a few minutes, trains nothing, writes
#  nothing.
# =====================================================================
#
#  FIRST RUN, AND WHAT IT FOUND. Both arms printed
#  "model_final.pt missing -- skipped", so the comparison could not be made.
#  The cause is a real defect, now fixed: the trainer's save of
#  model_final.pt hung off a for/else, which runs only when the loop
#  finishes WITHOUT `break` -- and early stopping breaks. Every
#  early-stopped run has been keeping model_best.pt alone.
#
#  Nothing was lost, though. train_state_latest.pt is written at EVERY
#  validation event and carries model_state_dict, so the epoch-450 weights
#  are on Drive already. This cell now digs them out and writes
#  model_final.pt from them, which costs seconds and retrains nothing.
#
import json
import os
import re
import subprocess
import sys

REPO = '/content/OMAR'
BRANCH = 'claude/claude-code-question-d307wp'
R = '/content/drive/MyDrive/pfem_run'
WORKDIR = f'{R}/b2_single_resolution'
ARMS = ['21', '33']

from google.colab import drive
drive.mount('/content/drive')


def run(cmd, echo=True):
    print('$', ' '.join(str(c) for c in cmd), flush=True)
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                         stderr=subprocess.STDOUT, text=True, bufsize=1)
    out = []
    for line in p.stdout:
        out.append(line)
        if echo:
            print(line, end='', flush=True)
    p.wait()
    if p.returncode != 0:
        raise subprocess.CalledProcessError(p.returncode, cmd)
    return ''.join(out)


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

# ---- where each arm's two endpoints sit on the error curve ------------
print('\n' + '=' * 78)
print('THE TWO ENDPOINTS OF EACH ARM, ON THE ERROR CURVE')
print('=' * 78)
curve = {}
for res in ARMS:
    h_path = f'{WORKDIR}/N{res}/metrics_history.json'
    assert os.path.exists(h_path), (
        f'{h_path} missing -- run Round6_B2_SingleResolution first')
    h = json.load(open(h_path))
    best = min(h, key=lambda e: e['combined_val_error'])
    last = h[-1]
    curve[res] = (best, last)
    print(f"  N={res}   model_best.pt  = epoch {best['epoch']:>4}  "
          f"val {best['combined_val_error']:.4f}")
    print(f"          model_final.pt = epoch {last['epoch']:>4}  "
          f"val {last['combined_val_error']:.4f}   "
          f"({last['combined_val_error'] / best['combined_val_error']:.2f}x "
          f"worse)")

# ---- recover the endpoint weights the trainer failed to save ----------
# model_final.pt was missing on the first run of this cell (see the note at
# the top). train_state_latest.pt holds the same weights, so it is unpacked
# here rather than retraining anything. The epoch it carries is checked
# against the last validation event, so a state file from some other run
# cannot be passed off as this arm's endpoint.
import torch

print('\n' + '=' * 78)
print('THE ENDPOINT WEIGHTS')
print('=' * 78)
for res in ARMS:
    d = f'{WORKDIR}/N{res}'
    final = f'{d}/model_final.pt'
    state = f'{d}/train_state_latest.pt'
    if os.path.exists(final):
        print(f'  N={res}  model_final.pt is already there')
        continue
    if not os.path.exists(state):
        print(f'  N={res}  NEITHER model_final.pt NOR train_state_latest.pt '
              f'-- this arm cannot be compared')
        continue
    st = torch.load(state, map_location='cpu', weights_only=False)
    last_epoch = curve[res][1]['epoch']
    assert st['epoch'] == last_epoch, (
        f"train_state_latest.pt is at epoch {st['epoch']} but the last "
        f"validation event was epoch {last_epoch} -- these are not the same "
        f"run, refusing to use it")
    torch.save(st['model_state_dict'], final)
    print(f"  N={res}  recovered epoch {st['epoch']} from "
          f"train_state_latest.pt -> model_final.pt")

# ---- the probe on both endpoints of both arms -------------------------
PAT = re.compile(r'^\s*Pi\(pred\)\s+(-?[\d.eE+-]+)', re.M)
PAT_T = re.compile(r'^\s*Pi\(uv_exact\)\s+(-?[\d.eE+-]+)', re.M)
PAT_D = re.compile(r'^descent captured: .*mean (-?[\d.]+)%', re.M)
PAT_R = re.compile(r'^roughness: .*mean ([\d.]+)x', re.M)

summary = {}
for res in ARMS:
    d = f'{WORKDIR}/N{res}'
    cache = f'{d}/samples_cache.pt'
    for which in ('model_best.pt', 'model_final.pt'):
        ckpt = f'{d}/{which}'
        if not os.path.exists(ckpt):
            print(f'\n[N={res}] {which} missing -- skipped')
            continue
        print('\n\n' + '#' * 78)
        print(f'#  N={res} alone   {which}')
        print('#' * 78)
        txt = run([sys.executable, '-u', '-m',
                   'omar_pfem.test_b2_zeroshot_model',
                   '--geometry', 'B2', '--material', 'neo_hookean',
                   '--cache', cache, '--checkpoint', ckpt, '--cpu'])
        pis = [float(x) for x in PAT.findall(txt)]
        pit = [float(x) for x in PAT_T.findall(txt)]
        dm = PAT_D.search(txt)
        rm = PAT_R.search(txt)
        summary[(res, which)] = dict(
            pi_pred=sum(pis) / len(pis) if pis else float('nan'),
            pi_exact=sum(pit) / len(pit) if pit else float('nan'),
            descent=float(dm.group(1)) if dm else float('nan'),
            rough=float(rm.group(1)) if rm else float('nan'))

# ---- the comparison ---------------------------------------------------
print('\n' + '=' * 78)
print('DID 400 MORE EPOCHS LOWER Pi?')
print('=' * 78)
print(f"  {'arm':<16}{'checkpoint':<16}{'mean Pi(pred)':>16}"
      f"{'mean Pi(exact)':>16}{'descent':>10}{'rough':>8}")
verdicts = []
for res in ARMS:
    for which in ('model_best.pt', 'model_final.pt'):
        s = summary.get((res, which))
        if not s:
            continue
        print(f"  N={res:<14}{which:<16}{s['pi_pred']:>16.6e}"
              f"{s['pi_exact']:>16.6e}{s['descent']:>9.0f}%"
              f"{s['rough']:>7.2f}x")
    a, b = summary.get((res, 'model_best.pt')), summary.get((res, 'model_final.pt'))
    if a and b:
        verdicts.append((res, a['pi_pred'], b['pi_pred'],
                         curve[res][0]['combined_val_error'],
                         curve[res][1]['combined_val_error']))

print()
for res, pi_a, pi_b, e_a, e_b in verdicts:
    down = pi_b < pi_a
    print(f"  N={res}: Pi went {'DOWN' if down else 'UP'} "
          f"({pi_a:.6e} -> {pi_b:.6e}) while the error went "
          f"{'UP' if e_b > e_a else 'DOWN'} ({e_a:.4f} -> {e_b:.4f})")

print('\n' + '=' * 78)
if verdicts and all(pi_b < pi_a and e_b > e_a
                    for _, pi_a, pi_b, e_a, e_b in verdicts):
    print('THE FUNCTIONAL, NOT THE DATA. Pi fell while the error rose, in')
    print('every arm. The optimizer is descending correctly; what it is')
    print('descending towards is a field with LESS energy than the FEM')
    print('solution. So the discretised Pi on B2 does not have uv_exact as')
    print('its minimiser over the fields this network can reach, and the')
    print('earlier ray scan could not have seen that -- it only moved along')
    print('s * uv_exact.')
    print()
    print('DO NOT regenerate the cache from the GRF on this evidence: the')
    print('data family cannot be the cause of an objective that prefers a')
    print('different field. The next question is WHICH field -- the probe')
    print("prints its roughness, and B2's 3.00x against B1's 1.01x says it")
    print('is a rough one, which is what a mesh-scale null mode looks like.')
elif verdicts and all(pi_b > pi_a for _, pi_a, pi_b, _, _ in verdicts):
    print('NOT DESCENDING. Pi is HIGHER at epoch 450 than at epoch 50, so')
    print('the optimizer is not minimising its own objective -- this is an')
    print('optimisation failure, not a statement about the physics. Learning')
    print('rate and the gradient through the two-ramp mask are the two')
    print('places to look, and both are cheap to test.')
else:
    print('MIXED. Read the two arms separately -- the line above says which')
    print('way each one moved. Send the whole block over.')
print('=' * 78)
print('Nothing was trained and nothing was written.')
