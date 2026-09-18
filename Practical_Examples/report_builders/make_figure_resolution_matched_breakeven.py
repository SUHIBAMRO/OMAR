"""Bar chart for Table 18-R10e' / Table R10-4' (resolution-matched
break-even, operator vs. torch-fem, both at N=1401, all six cases).

Updated 2026-09-18 (round-12 point 3): the operator side now uses the
compile+TF32-optimized timing (394 ms/sample, ~case-independent) instead
of the earlier default eager fp32 number, per the advisor's own request
to use the optimized number for the paper. torch-fem's own N=1401
numbers are unchanged (still the clean 2026-09-17 re-run, see
update_resolution_matched_breakeven_confirmed.py) -- only the operator
side and the resulting speedups changed.

Both Arruda-Boyce cases are shown as 0x with a "FAILED" label rather
than omitted, so the chart does not silently imply six successful
comparisons.
"""
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'figures')
os.makedirs(OUT, exist_ok=True)

CASES = ['B1 x\nNeo-Hookean', 'B1 x\nMooney-Rivlin', 'B1 x\nArruda-Boyce',
         'B2 x\nNeo-Hookean', 'B2 x\nMooney-Rivlin', 'B2 x\nArruda-Boyce']
SPEEDUPS = [342.6, 339.7, 0, 520.9, 525.5, 0]
FAILED = [False, False, True, False, False, True]
COLORS = ['#2E86AB' if not f else '#C0392B' for f in FAILED]

x = np.arange(len(CASES))

fig, ax = plt.subplots(figsize=(9, 5.5), dpi=200)
bars = ax.bar(x, SPEEDUPS, color=COLORS, width=0.6)

for xi, (v, f) in zip(x, zip(SPEEDUPS, FAILED)):
    if f:
        ax.annotate('FAILED', xy=(xi, 2), xytext=(0, 3), textcoords='offset points',
                    ha='center', fontsize=9, color='#C0392B', fontweight='bold')
    else:
        ax.annotate(f'{v:.1f}x', xy=(xi, v), xytext=(0, 4), textcoords='offset points',
                    ha='center', fontsize=10)

ax.set_xticks(x)
ax.set_xticklabels(CASES, fontsize=9)
ax.set_ylabel('Speedup vs. torch-fem (operator faster by)')
ax.set_title('Resolution-matched comparison at N=1401 (operator, compile+TF32, vs. torch-fem, same N)')
ax.set_ylim(top=max(SPEEDUPS) * 1.2)
ax.grid(True, axis='y', alpha=0.25)

fig.tight_layout()

out_path = os.path.join(OUT, 'fig_resolution_matched_breakeven.png')
fig.savefig(out_path, bbox_inches='tight')
print('Saved', out_path)
