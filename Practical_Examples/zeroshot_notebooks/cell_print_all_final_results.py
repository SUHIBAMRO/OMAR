# =====================================================================
#  Print every final result JSON, plainly, so you can read the actual
#  numbers with your own eyes and compare them to the report/summary
#
#  Read-only. No GPU. Clones the repo, opens every result file that
#  feeds a report table, and prints it -- nothing is recomputed, nothing
#  is written. Seconds to run.
#
#  Each block below is labeled with the report Table it corresponds to,
#  so you can hold this output next to the .docx and check the numbers
#  match, one table at a time.
# =====================================================================
import json
import os
import subprocess
import sys

REPO = '/content/OMAR'
BRANCH = 'claude/claude-code-question-d307wp'


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

PF = f'{REPO}/Practical_Examples/omar_pfem'


def load(*parts):
    path = os.path.join(PF, *parts)
    if not os.path.exists(path):
        print(f'  [missing] {path}')
        return None
    return json.load(open(path))


def hr(title):
    print('\n' + '=' * 78)
    print(title)
    print('=' * 78)


def print_rows(rows, cols):
    """cols: list of (key, header, fmt) tuples."""
    widths = [max(len(h), 10) for _, h, _ in cols]
    print('  ' + ' | '.join(h.ljust(w) for (_, h, _), w in zip(cols, widths)))
    for r in rows:
        cells = []
        for (k, h, fmt), w in zip(cols, widths):
            v = r.get(k)
            cells.append((fmt.format(v) if v is not None else '-').ljust(w))
        print('  ' + ' | '.join(cells))


# --- Point 2: accuracy/cost Pareto, Tables 18 / 18a-e ------------------
hr('POINT 2 -- ACCURACY/COST PARETO (Tables 18, 18a-18e)')
for geom, mat, table in [
    ('B1', 'neo_hookean', '18'), ('B1', 'mooney_rivlin', '18a'),
    ('B1', 'arruda_boyce', '18b'), ('B2', 'neo_hookean', '18c'),
    ('B2', 'mooney_rivlin', '18d'), ('B2', 'arruda_boyce', '18e'),
]:
    d = load('point2_results', f'pareto_{geom}_{mat}.json')
    if d is None:
        continue
    print(f'\n--- Table {table}: {geom} x {mat} ---')
    rows = []
    for r in d['rows']:
        # pareto_B1_neo_hookean.json alone nests timings under "run4" --
        # run3 was measured on a faster Colab instance and disagrees with
        # Table 10a (see pareto_table.py); every other file is flat.
        src = r['run4'] if 'run4' in r else r
        rows.append({**r, 'fem_ms_per_sample': src['fem_ms_per_sample'],
                     'operator_ms_per_sample': src['operator_ms_per_sample'],
                     'speedup': src['fem_ms_per_sample'] / src['operator_ms_per_sample']})
    print_rows(rows, [
        ('N', 'N', '{}'),
        ('n_nodes', 'Nodes', '{:,}'),
        ('fem_rel_L2', 'FEM rel.L2', '{:.4f}'),
        ('operator_rel_L2', 'Op rel.L2', '{:.4f}'),
        ('fem_ms_per_sample', 'FEM ms', '{:,.1f}'),
        ('operator_ms_per_sample', 'Op ms', '{:.3f}'),
        ('speedup', 'Speed-up', '{:,.0f}x'),
    ])

# --- Point 5: physical quantities, Tables 15-17 ------------------------
hr('POINT 5 -- PHYSICAL QUANTITIES BEYOND DISPLACEMENT (Tables 15-17)')
for geom in ('B1', 'B2'):
    for mat in ('neo_hookean', 'mooney_rivlin', 'arruda_boyce'):
        d = load('point5_results', f'physical_quantities_{geom}_{mat}.json')
        if d is None:
            continue
        print(f'\n--- {geom} x {mat} ---')
        print(f'  {json.dumps(d.get("summary", d), indent=2)}')

# --- Point 6: OOD, Tables 19 / 19a --------------------------------------
hr('POINT 6 -- OUT-OF-DISTRIBUTION (Tables 19, 19a)')
d = load('point6_results', 'ood_progressive_B1_neo_hookean.json')
if d is not None:
    print('\n--- Table 19: progressive OOD shift ---')
    print_rows(d['rows'], [
        ('factor', 'factor', '{}'),
        ('shift_sigma', 'shift (sigma)', '{}'),
        ('mean_rel_L2', 'mean rel.L2', '{:.4f}'),
        ('degradation_vs_baseline', 'vs baseline', '{:.2f}x'),
    ]) if 'rows' in d else print(f'  {json.dumps(d, indent=2)}')

d = load('point6_results', 'ood_mitigation_B1_neo_hookean.json')
if d is not None:
    print('\n--- Table 19a: normalization mitigation ---')
    print(f'  {json.dumps(d, indent=2)}')

# --- Point 7a: zero-shot resolution invariance, Tables 12, 12b, 12c ----
hr('POINT 7 -- ZERO-SHOT RESOLUTION INVARIANCE (Tables 12, 12b, 12c)')
for mat in ('neo_hookean', 'mooney_rivlin', 'arruda_boyce'):
    d = load('point7a_results', f'zeroshot_B1_{mat}.json')
    if d is None:
        continue
    print(f'\n--- Table 12: B1 x {mat} ---')
    print(f'  {json.dumps(d.get("results", d), indent=2)}')

d = load('point7a_results', 'B2_zeroshot_fixedselection.json')
if d is not None:
    print('\n--- Table 12b: B2 x Neo-Hookean (fixed selection) ---')
    print(f'  {json.dumps(d, indent=2)}')

for mat, table_note in [('mooney_rivlin', ''), ('arruda_boyce', '')]:
    d = load('point7a_results', f'B2_{mat}_zeroshot_fixedselection.json')
    if d is None:
        continue
    print(f'\n--- Table 12c row: B2 x {mat} (fixed selection) ---')
    print(f'  {json.dumps(d, indent=2)}')

# --- Point 7b: data-driven vs physics-informed, Table 21 ---------------
hr('POINT 7b -- PHYSICS-INFORMED VS DATA-DRIVEN (Table 21)')
d = load('point7b_results', 'comparison_B1_neo_hookean.json')
if d is not None:
    print(f'  {json.dumps(d, indent=2)}')

# --- Point 8: GPU-FEM scaling, Tables 20, 20a, 20b ----------------------
hr('POINT 8 -- GPU-NATIVE FEM SCALING (Tables 20, 20a, 20b)')
d = load('point8_results', 'gpu_fem_scaling_B1_neo_hookean.json')
if d is not None and 'rows' in d:
    print('\n--- Table 20: scaling sweep ---')
    print_rows(d['rows'], [
        ('N', 'N', '{}'),
        ('n_dof', 'DOF', '{:,}'),
        ('solve_s', 'solve (s)', '{:.3f}'),
        ('us_per_dof', 'us/DOF', '{:,.0f}'),
    ]) if all('n_dof' in r for r in d['rows']) else print(f'  {json.dumps(d, indent=2)}')
d = load('point8_results', 'gpu_fem_cg_converged_B1_neo_hookean.json')
if d is not None:
    print('\n--- Table 20b: CG-converged re-run ---')
    print(f'  {json.dumps(d, indent=2)}')

# --- Point 9: MMS, Tables 22-24e -----------------------------------------
hr('POINT 9 -- METHOD OF MANUFACTURED SOLUTIONS (Tables 22-24e)')
for fname, label in [
    ('mms_B1_neo_hookean.json', 'Table 22/23: Q4 and Q9 vs manufactured solution'),
    ('mms_operator_B1_neo_hookean.json', 'Table 24: operator vs manufactured solution'),
    ('mms_operator_rate_B1_neo_hookean.json', 'Table 24a/24b: operator across 3 meshes'),
    ('mms_family_fem_B1_neo_hookean.json', 'Table 24c: Q4/Q9 over 16-member family'),
    ('mms_operator_per_member_B1_neo_hookean.json', 'Table 24d/24e: operator per family member'),
]:
    d = load('point9_results', fname)
    if d is None:
        continue
    print(f'\n--- {label} ---')
    print(f'  {json.dumps(d, indent=2)}')

print('\n' + '=' * 78)
print('Done. Compare each block above against the matching Table number')
print('in the report / summary .docx.')
print('=' * 78)
