"""Shared visual style for every make_figure_*.py script in this
directory, built after Omar's own direct feedback on the first pass:
the same colour meant different things in different figures (blue was
"B1" in one and "Neo-Hookean" in another), several charts had no value
labels on bars/points, and some legends sat on top of the data instead
of beside it. One shared module fixes all three, consistently, once.
"""

# One colour per material, used in EVERY figure that breaks out by
# material -- so blue always means Neo-Hookean, everywhere.
MATERIAL_COLOR = {
    'Neo-Hookean': '#2E86AB',
    'Mooney-Rivlin': '#E67E22',
    'Arruda-Boyce': '#27AE60',
}
# One colour per geometry, used in every figure that breaks out by
# geometry only (not material) -- kept visually distinct from the
# material palette above so the two groupings are never confused.
GEOMETRY_COLOR = {'B1': '#2E86AB', 'B2': '#C0392B'}

NEUTRAL = '#7f7f7f'
HIGHLIGHT = '#C0392B'
GOOD = '#27AE60'
# For figures with exactly two non-material, non-geometry series (e.g.
# "ours vs. theirs", "in-distribution vs. OOD", "physics-informed vs.
# data-driven") -- kept visually distinct from both palettes above.
PRIMARY = '#2E86AB'
SECONDARY = '#E67E22'


def add_bar_labels(ax, bars, fmt='{:.1f}', fontsize=8, rotation=0, pad=3):
    """Value label centred above each bar. `pad` is in points, so it
    reads correctly on both linear and log-scaled axes."""
    for b in bars:
        h = b.get_height()
        ax.annotate(fmt.format(h), xy=(b.get_x() + b.get_width() / 2, h),
                    xytext=(0, pad), textcoords='offset points',
                    ha='center', va='bottom', fontsize=fontsize, rotation=rotation)


def add_line_labels(ax, x, y, fmt='{:.2f}', fontsize=7, dy=8, color=None):
    """Value label above each point on a (sparse) line/marker series --
    reserved for series with few enough points that this stays
    readable; dense convergence-style curves skip this deliberately."""
    kwargs = {'color': color} if color else {}
    for xi, yi in zip(x, y):
        ax.annotate(fmt.format(yi), xy=(xi, yi), xytext=(0, dy),
                    textcoords='offset points', ha='center', fontsize=fontsize, **kwargs)


def legend_right(ax, fontsize=8, **kwargs):
    """Legend placed OUTSIDE the axes, to the right -- never on top of
    bars or lines. Caller must leave room via fig.tight_layout(rect=...)
    or subplots_adjust."""
    return ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1.0),
                      frameon=False, fontsize=fontsize, **kwargs)


def legend_below(ax, ncol, fontsize=8, y=-0.15, **kwargs):
    """Legend placed OUTSIDE the axes, below -- for wide figures where
    a right-side legend would not fit."""
    return ax.legend(loc='upper center', bbox_to_anchor=(0.5, y), ncol=ncol,
                      frameon=False, fontsize=fontsize, **kwargs)
