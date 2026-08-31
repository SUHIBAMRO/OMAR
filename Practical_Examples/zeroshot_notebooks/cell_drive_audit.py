# =====================================================================
#  FULL DRIVE AUDIT — every file, opened and read
#
#  The older cell_check_results.py looks for a fixed list of filenames.
#  That answers "is the file I expected there?" and cannot answer "what is
#  actually on Drive?" -- it is blind to anything nobody thought to
#  pattern-match, which is exactly what an audit is for.
#
#  This walks the whole pfem_run tree instead. Every file is listed with
#  its size and time; every JSON is opened and summarised by what it
#  actually contains; every checkpoint is fingerprinted so a result can be
#  tied to the model that produced it; and every study is checked for
#  COMPLETENESS against what it is supposed to hold, so a half-finished
#  file is never mistaken for a finished one.
#
#  It writes nothing. Large tensor files are measured, not loaded -- a
#  sample cache can be hundreds of MB and loading it would risk the
#  runtime for no information.
#
#  Read-only, a couple of minutes.
# =====================================================================
import datetime
import hashlib
import json
import os

R = '/content/drive/MyDrive/pfem_run'
BIG = 64 * 1024 * 1024          # above this, do not load and do not hash

# what a finished study holds, so partial files are named as partial
EXPECT = {
    'pareto_': (9, 'resolutions'),
    'zeroshot_eval': (7, 'resolutions'),
    'mms_family_fem': (3, 'meshes'),
}

from google.colab import drive
drive.mount('/content/drive')

import torch


def human(n):
    for u in ('B', 'KB', 'MB', 'GB'):
        if n < 1024 or u == 'GB':
            return f'{n:.0f}{u}' if u == 'B' else f'{n:.1f}{u}'
        n /= 1024


def when(p):
    return datetime.datetime.fromtimestamp(
        os.path.getmtime(p)).strftime('%m-%d %H:%M')


def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


findings = []


def flag(msg):
    findings.append(msg)
    print(f'      !! {msg}')


def describe_json(path, d):
    """Summarise by SHAPE, not by filename -- the point is to see what is
    there, including files this audit was never written to expect."""
    name = os.path.basename(path)

    if isinstance(d, list):
        print(f'      a list of {len(d)} records')
        if d and isinstance(d[0], dict) and 'epoch' in d[0]:
            best = min(d, key=lambda e: e.get('combined_val_error', 9e9))
            print(f'      epochs {d[0]["epoch"]}..{d[-1]["epoch"]}, '
                  f'{len(d)} validation events')
            print(f'      best combined_val_error {best.get("combined_val_error")}'
                  f' at epoch {best.get("epoch")}')
            if 'both_components_val_error' in d[0]:
                b2 = min(d, key=lambda e: e['both_components_val_error'])
                print(f'      best both_components {b2["both_components_val_error"]:.4f}'
                      f' at epoch {b2["epoch"]}   '
                      f'[selected on {d[-1].get("selection_metric", "?")}]')
                if best['epoch'] != b2['epoch']:
                    flag(f'{name}: the two metrics disagree about the best '
                         f'epoch ({best["epoch"]} vs {b2["epoch"]})')
            else:
                print('      (no both_components column -- written before the '
                      'metric fix)')
        elif d and isinstance(d[0], dict) and 'kind' in d[0]:
            for r in d:
                print(f'      run: {r.get("kind")}  {r.get("started_at", "")[:16]}')
        return

    if not isinstance(d, dict):
        print(f'      {type(d).__name__}')
        return

    rows = d.get('rows')
    if isinstance(rows, list):
        Ns = [r.get('N') for r in rows if isinstance(r, dict)]
        print(f'      {len(rows)} rows' + (f', N={Ns}' if any(Ns) else ''))
        for key, (n, unit) in EXPECT.items():
            if key in name:
                if len(rows) < n:
                    flag(f'{name}: PARTIAL -- {len(rows)}/{n} {unit}')
                else:
                    print(f'      COMPLETE ({len(rows)}/{n} {unit})')
        fp = d.get('checkpoint_fingerprint')
        if fp:
            print(f'      from checkpoint {fp[:16]}')
        for r in rows[:12]:
            if not isinstance(r, dict):
                continue
            bits = []
            for k in ('mean_rel_L2_vs_fine_reference',
                      'mean_combined_rel_L2_vs_fine_reference',
                      'fem_rel_L2', 'operator_rel_L2',
                      'fem_ms_per_sample', 'operator_ms_per_sample'):
                if k in r:
                    bits.append(f'{k.split("_vs_")[0]}={r[k]:.4e}')
            # Two different row shapes carry Q4/Q9 in this project: the
            # family sweep stores {'L2_rel': {'mean': ...}}, the operator-rate
            # study stores {'L2': ...}. Neither is wrong; the audit must read
            # both rather than assume one and raise KeyError on the other.
            for o in ('Q4', 'Q9', 'operator'):
                v = r.get(o)
                if not isinstance(v, dict):
                    continue
                if isinstance(v.get('L2_rel'), dict):
                    bits.append(f'{o} L2={v["L2_rel"]["mean"]:.4e}')
                elif isinstance(v.get('L2_rel'), (int, float)):
                    bits.append(f'{o} L2={v["L2_rel"]:.4e}')
                elif isinstance(v.get('L2'), (int, float)):
                    bits.append(f'{o} L2={v["L2"]:.4e}')
            if bits:
                print(f'        N={r.get("N")}: ' + '  '.join(bits))
        if len(rows) > 12:
            print(f'        ... and {len(rows) - 12} more')
        return

    # anything else: show the scalar/short fields, which is where the
    # single-number results live
    shown = 0
    for k, v in d.items():
        if isinstance(v, (int, float, str, bool)) and len(str(v)) < 90:
            print(f'      {k}: {v}')
            shown += 1
        elif isinstance(v, dict) and all(
                isinstance(x, (int, float)) for x in v.values()):
            print(f'      {k}: ' + ', '.join(f'{a}={b:.4e}'
                                             for a, b in v.items()))
            shown += 1
        if shown >= 14:
            print('      ...')
            break


print('=' * 78)
print(f'EVERY FILE UNDER {R}')
print('=' * 78)
if not os.path.isdir(R):
    raise SystemExit(f'{R} does not exist -- is Drive mounted at the usual place?')

total_bytes, n_files = 0, 0
ckpts = {}
for dirpath, dirnames, filenames in sorted(os.walk(R)):
    dirnames.sort()
    if not filenames:
        continue
    rel = os.path.relpath(dirpath, R)
    print(f'\n{"-" * 78}\n{rel}/\n{"-" * 78}')
    for fn in sorted(filenames):
        p = os.path.join(dirpath, fn)
        try:
            size = os.path.getsize(p)
        except OSError as e:
            print(f'  {fn}   !! cannot stat ({e.__class__.__name__})')
            continue
        total_bytes += size
        n_files += 1
        print(f'  {fn:<44} {human(size):>9}  {when(p)}')

        if fn.endswith('.tmp'):
            flag(f'{rel}/{fn}: a .tmp left behind -- a write was interrupted')
            continue
        if fn == 'EARLY_STOPPED':
            try:
                print('      ' + open(p).read().strip().replace('\n', '; '))
            except Exception:
                pass
            continue
        if fn.endswith('.json'):
            try:
                describe_json(p, json.load(open(p)))
            except Exception as e:
                flag(f'{rel}/{fn}: unreadable JSON ({e.__class__.__name__})')
            continue
        if fn.endswith('.pt'):
            if size > BIG:
                print(f'      (tensor file, {human(size)} -- measured, not '
                      f'loaded)')
                continue
            try:
                obj = torch.load(p, map_location='cpu', weights_only=False)
            except Exception as e:
                flag(f'{rel}/{fn}: will not load ({e.__class__.__name__})')
                continue
            if isinstance(obj, dict) and all(
                    hasattr(v, 'shape') for v in obj.values()) and obj:
                n = sum(v.numel() for v in obj.values())
                f = sha(p)
                ckpts[p] = f
                print(f'      state_dict: {len(obj)} tensors, {n:,} parameters,'
                      f' fingerprint {f[:16]}')
            elif isinstance(obj, dict):
                print(f'      dict with keys: {sorted(obj)[:8]}')
                if 'epoch' in obj:
                    print(f'      resume state at epoch {obj["epoch"]}, '
                          f'best_epoch {obj.get("best_epoch")}, '
                          f'best_combined {obj.get("best_combined_val")}, '
                          f'best_select {obj.get("best_select_val")}')
            else:
                print(f'      {type(obj).__name__}')

# ---- tie every result to the model that produced it -------------------
print('\n' + '=' * 78)
print('DO THE RESULT FILES MATCH THE CHECKPOINTS BESIDE THEM?')
print('=' * 78)
by_fp = {v: k for k, v in ckpts.items()}
checked = 0
for dirpath, _, filenames in os.walk(R):
    for fn in filenames:
        if not fn.endswith('.json'):
            continue
        p = os.path.join(dirpath, fn)
        try:
            d = json.load(open(p))
        except Exception:
            continue
        if not isinstance(d, dict):
            continue
        fp = d.get('checkpoint_fingerprint')
        if not fp:
            continue
        checked += 1
        src = by_fp.get(fp)
        rel = os.path.relpath(p, R)
        if src:
            print(f'  {rel}\n      <- {os.path.relpath(src, R)}')
        else:
            flag(f'{rel}: claims checkpoint {fp[:16]}, which is not any '
                 f'checkpoint on Drive -- the model it was produced with is '
                 f'gone or was overwritten')
if not checked:
    print('  (no result file carries a checkpoint fingerprint)')

# ---- the verdict ------------------------------------------------------
print('\n' + '=' * 78)
print(f'{n_files} files, {human(total_bytes)} total')
print('=' * 78)
if findings:
    print(f'{len(findings)} thing(s) worth looking at:\n')
    for f in findings:
        print(f'  - {f}')
else:
    print('Nothing partial, nothing orphaned, nothing unreadable.')
print('\nThis run wrote nothing.')
