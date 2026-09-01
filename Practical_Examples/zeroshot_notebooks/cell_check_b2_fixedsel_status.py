# =====================================================================
#  CELL — how far did the B2 fixed-selection training actually get?
#
#  Read-only, changes nothing, needs no GPU (torch.load(..., map_location=
#  'cpu') on train_state_latest.pt is enough to read its epoch). Run this
#  BEFORE (re)launching cell_b2_fixed_selection_all.py to see whether a
#  disconnected session left usable progress on Drive, so a restart does
#  not spend hours re-deriving something already sitting there.
#
#  Checks, per material, in order of how far the pipeline got:
#    zeroshot_eval.json      -- fully done, including the 7-mesh eval
#    EARLY_STOPPED           -- training stopped on its own (converged or
#                               patience exhausted); eval may still be
#                               missing if the run died between the two
#    metrics_history.json    -- at least one validation event happened
#    train_state_latest.pt   -- full resume state exists; its stored
#                               epoch is what a restart will continue from
#  A material with none of these has never started.
# =====================================================================
import json
import os

from google.colab import drive
drive.mount('/content/drive')

import torch

R = '/content/drive/MyDrive/pfem_run'
CASES = ['neo_hookean', 'mooney_rivlin', 'arruda_boyce']

for mat in CASES:
    d = f'{R}/zeroshot_B2_{mat}_fixedsel'
    print('=' * 70)
    print(mat)
    print('=' * 70)
    if not os.path.isdir(d):
        print('  directory does not exist -- never started')
        continue

    ev = f'{d}/zeroshot_eval.json'
    if os.path.exists(ev):
        rows = json.load(open(ev)).get('rows', [])
        print(f'  DONE -- zeroshot_eval.json holds {len(rows)}/7 resolutions')
        for r in rows:
            print(f"    N={r['N']:<4} "
                  f"{r.get('mean_rel_L2_vs_fine_reference', float('nan')):.4f}")
        continue

    stopped = os.path.exists(f'{d}/EARLY_STOPPED')
    hist_path = f'{d}/metrics_history.json'
    if os.path.exists(hist_path):
        hist = json.load(open(hist_path))
        last = hist[-1] if hist else None
        # is_best is set by the trainer using whichever selection_metric was
        # actually in force -- do not re-derive "best" from combined_val_error
        # here, since for a both_components run that is not the metric that
        # chose it. The last entry flagged is_best=True is the current best.
        best_entries = [h for h in hist if h.get('is_best')]
        best = best_entries[-1] if best_entries else None
        print(f'  training in progress or stopped -- {len(hist)} validation events, '
              f'selecting on {hist[0].get("selection_metric") if hist else "?"}')
        if last:
            print(f'    last:  epoch {last["epoch"]}   '
                  f'combined={last.get("combined_val_error")}   '
                  f'both_components={last.get("both_components_val_error")}')
        if best:
            print(f'    best:  epoch {best["epoch"]}   '
                  f'combined={best.get("combined_val_error")}   '
                  f'both_components={best.get("both_components_val_error")}')
        print(f'    EARLY_STOPPED marker present: {stopped}')
    else:
        print('  no metrics_history.json yet -- no validation event has happened')

    state_path = f'{d}/train_state_latest.pt'
    if os.path.exists(state_path):
        state = torch.load(state_path, map_location='cpu', weights_only=False)
        print(f'  train_state_latest.pt: epoch {state["epoch"]}, '
              f'best_epoch {state.get("best_epoch")}, '
              f'patience_counter {state.get("patience_counter")}   '
              f'<- a restart resumes from here')
    else:
        print('  no train_state_latest.pt -- a restart would begin at epoch 1')

print('\n' + '=' * 70)
print('Send Claude this output before deciding whether to (re)launch')
print('cell_b2_fixed_selection_all.py.')
print('=' * 70)
