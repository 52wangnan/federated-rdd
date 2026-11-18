# -*- coding: utf-8 -*-
"""Common figure style. Every figure is drawn at the width at which it is printed in the paper, so the
font sizes set here are the printed sizes: tick labels, legends and in-plot text 8 pt; axis labels, colorbar
labels and panel titles 8.5 pt. The font family is matplotlib's default.

    import figstyle
    figstyle.setup()
    fig, ax = plt.subplots(figsize=(figstyle.width('fig_charls_density'), 2.8))
"""
import matplotlib

# printed width in the paper (inches); the 2015 versions (suffix _2015) have the same width
WIDTH = {'fig1_protocol': 6.30,
         'fig_sim_coverage': 6.50, 'fig_sim_rmse': 6.50, 'fig_sim_S3_estimand': 6.50, 'fig_sim_exactness': 6.50,
         'fig_comm': 5.90,
         'fig_charls_rdplot': 6.50, 'fig_charls_forest': 6.50, 'fig_charls_bandwidth': 6.50,
         'fig_charls_partitions': 6.50, 'fig_charls_density': 6.50, 'fig_charls_density_2015_vs_2018': 6.50,
         'fig_value_coalitions': 5.90, 'fig_value_shapley': 6.50}
FS = 8            # tick labels, legends, in-plot text (never smaller)
FS_LABEL = 8.5    # axis labels, colorbar labels, panel titles


def width(name):
    """Printed width (inches) of the figure saved as name.png."""
    return WIDTH[name]


def setup():
    matplotlib.rcParams.update({'font.size': FS, 'xtick.labelsize': FS, 'ytick.labelsize': FS, 'legend.fontsize': FS,
                                'axes.labelsize': FS_LABEL, 'axes.titlesize': FS_LABEL, 'figure.labelsize': FS_LABEL,
                                'figure.titlesize': FS_LABEL,
                                'axes.spines.top': False, 'axes.spines.right': False})


def wrap(text, n):
    """Break a panel title after its commas so that no line is longer than n characters (wording unchanged)."""
    parts = text.split(', ')
    lines, cur = [], parts[0]
    for p in parts[1:]:
        if len(cur) + 2 + len(p) <= n:
            cur += ', ' + p
        else:
            lines.append(cur + ',')
            cur = p
    return '\n'.join(lines + [cur])
