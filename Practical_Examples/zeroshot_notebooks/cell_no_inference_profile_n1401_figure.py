# =====================================================================
#  CELL -- figure for the already-completed N=1401 profiling result
#  (Round6_NO_Inference_Profile_N1401.ipynb), added after the fact per
#  Omar's own catch that the other new round-10 notebooks were missing
#  the figures this project's own convention normally produces
#  alongside a result.
#
#  Reads the JSON that cell already saved to Drive
#  (no_inference_profile_N1401.json) rather than re-running the
#  expensive profiling GPU job just to get a plot out of it.
# =====================================================================
import json
import os

from google.colab import drive
drive.mount('/content/drive')

R = '/content/drive/MyDrive/pfem_run'
IN_JSON = f'{R}/no_inference_profile_N1401.json'
assert os.path.exists(IN_JSON), (
    f'{IN_JSON} not found -- run Round6_NO_Inference_Profile_N1401.ipynb first, '
    'this cell only plots its already-saved result.')

result = json.load(open(IN_JSON))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# plot_style.py lives in the repo, which this standalone figure cell does not
# clone -- inlined here rather than requiring a clone just for two constants.
PRIMARY, SECONDARY = '#2E86AB', '#E67E22'


def add_bar_labels(ax, bars, fmt='{:.1f}', fontsize=8, pad=3):
    for b in bars:
        h = b.get_height()
        ax.annotate(fmt.format(h), xy=(b.get_x() + b.get_width() / 2, h),
                    xytext=(0, pad), textcoords='offset points',
                    ha='center', va='bottom', fontsize=fontsize)


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5), dpi=200)

labels, values, colors = ['fp32'], [result['inference_ms_per_sample']], [PRIMARY]
if result.get('bf16_autocast_ms_per_sample') is not None:
    labels.append('bf16 (self-consistency\nonly, see accuracy notebook)')
    values.append(result['bf16_autocast_ms_per_sample'])
    colors.append(SECONDARY)
bars1 = ax1.bar(labels, values, color=colors)
ax1.set_ylabel('ms/sample')
ax1.set_title('NO inference time, N=1401')
ax1.grid(True, axis='y', alpha=0.25)
add_bar_labels(ax1, bars1, fmt='{:.1f}')

bars2 = ax2.bar(['peak GPU memory'], [result['peak_memory_mb'] / 1024.0], color=PRIMARY)
ax2.set_ylabel('GB')
ax2.set_title(f'Peak memory during inference\n({result["gpu_name"]})')
ax2.grid(True, axis='y', alpha=0.25)
add_bar_labels(ax2, bars2, fmt='{:.2f}')

fig.tight_layout()
FIG_PATH = f'{R}/fig_no_inference_profile_N1401.png'
fig.savefig(FIG_PATH)
print('Saved figure:', FIG_PATH)
