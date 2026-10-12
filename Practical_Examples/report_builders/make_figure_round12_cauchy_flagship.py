"""Line chart for the new round-12 point 1 flagship table (B1 x
Neo-Hookean, fixed-region Cauchy-stress error vs. resolution, FEM vs.
operator, both vs. the same fine reference N=201). Real numbers
originally read directly from round12_final_accuracy_cauchy_B1_neo_
hookean.json (2026-09-18 GPU run, fetched from Drive) -- never hand-
transcribed.

That original JSON only ever lived in a past Claude session's own
/tmp scratchpad and is gone (2026-10-12). The exact same 16-row data
survived, however, as a literal table (Table 18-R10p) written into
advisor_feedback/PFEM_Transolver_Report_2026-10-05.docx by this
project's own apply_round12_point1_cauchy.py -- reconstruct_round12_
cauchy.py (2026-10-12) reads that table directly via python-docx and
rewrites it as round12_final_accuracy_cauchy_summary_RECONSTRUCTED.json
alongside this file, so this script's own numbers are still real,
still traceable to the original GPU run, just via the published report
table rather than the lost raw JSON. Run that script first if this
file is missing.

Shows the region-weighted AVERAGE (the primary robust statistic Timon
asked for) and the TRUE MAX (reported separately, exactly as he asked)
for both FEM and the operator, log-x by resolution. The true max lines
sit far above and converge far more slowly than the average lines for
BOTH methods -- a real, data-grounded illustration of why he asked to
avoid the bare pointwise maximum as the primary QoI.
"""
import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

matplotlib.rcParams.update({
    'font.size': 13,
    'axes.titlesize': 15,
    'axes.labelsize': 13,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'legend.fontsize': 11,
})

from plot_style import PRIMARY, SECONDARY

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, 'figures')
os.makedirs(OUT, exist_ok=True)
DATA = os.path.join(HERE, 'round12_final_accuracy_cauchy_summary_RECONSTRUCTED.json')

with open(DATA) as f:
    rows = json.load(f)['cases']['B1_neo_hookean']

N = [r['N'] for r in rows]
fem_avg = [r['fem_cauchy_avg_rel_err'] * 100 for r in rows]
no_avg = [r['no_cauchy_avg_rel_err'] * 100 for r in rows]
fem_max = [r['fem_cauchy_max_rel_err'] * 100 for r in rows]
no_max = [r['no_cauchy_max_rel_err'] * 100 for r in rows]

fig, ax = plt.subplots(figsize=(8, 5), dpi=200)
ax.plot(N, fem_avg, 'o-', color=PRIMARY, label='FEM, region average')
ax.plot(N, no_avg, 's-', color=SECONDARY, label='Operator, region average')
ax.plot(N, fem_max, 'o--', color=PRIMARY, alpha=0.5, label='FEM, true max')
ax.plot(N, no_max, 's--', color=SECONDARY, alpha=0.5, label='Operator, true max')

ax.set_xscale('log')
ax.set_xticks(N)
ax.set_xticklabels([str(n) for n in N], rotation=45, fontsize=13)
# Log-scale minor ticks (matplotlib's own default "2x10^1"/"3x10^1"
# labels) overlap the explicit integer-N major labels now that tick
# fontsize is bigger -- turned off since the explicit N labels above
# already label every data point directly, the minor ones add nothing.
ax.minorticks_off()
ax.set_xlabel('N (finite-element mesh resolution)')
ax.set_ylabel('Region Cauchy-stress relative error (%)')
ax.set_title('B1 x Neo-Hookean: fixed-region Cauchy stress vs. resolution\n(vs. the same fine reference, N=201)')
ax.legend(frameon=False, fontsize=13)
ax.grid(True, alpha=0.25)
fig.tight_layout()

out_path = os.path.join(OUT, 'fig_round12_cauchy_flagship.png')
fig.savefig(out_path, bbox_inches='tight')
print('Saved', out_path)
