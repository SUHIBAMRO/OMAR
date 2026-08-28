# =====================================================================
#  CELL — what is actually on Drive?
#  Run this any time to see which results exist and how far each got.
#  Reads only; changes nothing.
# =====================================================================
import os, json, glob, datetime

from google.colab import drive
drive.mount('/content/drive')

R = '/content/drive/MyDrive/pfem_run'


def age(path):
    dt = datetime.datetime.fromtimestamp(os.path.getmtime(path))
    return dt.strftime('%Y-%m-%d %H:%M')


def show(title, pattern, summarize):
    print('=' * 70)
    print(title)
    print('=' * 70)
    # run_manifest.json is a LIST of run records, not a result dict, and
    # lives in the same directories -- summarizing it raised
    # "'list' object has no attribute 'get'". Skip it by name.
    files = sorted(f for f in glob.glob(pattern, recursive=True)
                   if os.path.basename(f) != 'run_manifest.json')
    if not files:
        print('  (nothing yet)\n')
        return
    for f in files:
        print(f'  {os.path.relpath(f, R)}   [{age(f)}]')
        try:
            summarize(json.load(open(f)))
        except Exception as e:
            print(f'    !! could not read: {e}')
    print()


def ood(d):
    rows = d.get('rows', [])
    print(f'    {len(rows)}/19 cells done, {d.get("n_samples")} samples each')
    for r in rows:
        deg = r.get('degradation_vs_baseline')
        print(f'      {r["factor"]:<9} k={r["shift_sigma"]:<4} '
              f'err={r["mean_rel_L2"]:.4f}' + (f'  ({deg:.2f}x)' if deg else ''))


def sweep(d):
    rows = d.get('rows', [])
    print(f'    {len(rows)}/8 resolutions done on {d.get("gpu")}')
    for r in rows:
        b = r.get('cost_breakdown_pct') or {}
        extra = (f'   residual {b["residual"]:.1f}% / precond {b["precond"]:.1f}%'
                 f' / CG {b["cg"]:.1f}%') if b else ''
        print(f'      N={r["N"]:<5} {r["n_dof"]:>10,} DOF  '
              f'{r["solve_s"]/60:>7.1f} min  {r["us_per_dof"]:>8.2f} us/DOF{extra}')


def dd(d):
    print(f'    best val rel-L2 = {d.get("best_val_rel_L2"):.4f} '
          f'at step {d.get("best_step"):,}/{d.get("opt_steps"):,}')
    print(f'    training {d.get("train_wall_clock_s", 0):.0f} s   '
          f'labels {d.get("label_generation_cost_h", 0):.2f} h of CPU')


show('OOD PROGRESSIVE  (target: 19 cells)',
     f'{R}/ood_progressive/*.json', ood)
show('GPU FEM SCALING SWEEP  (target: 8 resolutions)',
     f'{R}/gpu_fem_scaling/*.json', sweep)
show('DATA-DRIVEN OPERATOR',
     f'{R}/data_driven/**/data_driven_*.json', dd)

print('=' * 70)
print('ZERO-SHOT CASES  (target: 6)')
print('=' * 70)
for f in sorted(glob.glob(f'{R}/**/zeroshot_*.json', recursive=True) +
                glob.glob(f'{R}/zeroshot_*/*.json', recursive=True)):
    if 'manifest' in f or 'pareto' in f:
        continue
    print(f'  {os.path.relpath(f, R)}   [{age(f)}]')
