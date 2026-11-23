# -*- coding: utf-8 -*-
"""Vector (PDF) versions of Figures 2-4 of the paper, redrawn from stored aggregate results; TrueType fonts are
embedded. No microdata is read.
    python paper/figures_vector.py

Figure 2  estimates under heterogeneous site composition, two panels: (a) covariate composition across ten
          participants, (b) treated-to-control ratios across participants (results/charls_ik/partitions.csv);
Figure 3  conventional covariate-adjusted IK estimates by number of participants (same file);
Figure 4  coalition sample size and minimum detectable effects (results/composition_coalitions/).
The plotting code follows the PNG versions in paper/charls_ik_analysis.py (Figures 2 and 3) and
paper/figures_charls.py (Figure 4). Figure 1 is a diagram drawn by the authors.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

matplotlib.rcParams['pdf.fonttype'] = 42      # embed TrueType fonts
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results' / 'figures_vector'
CON = ROOT / 'results' / 'charls_ik'
CO = ROOT / 'results' / 'composition_coalitions'


def figure_2():
    """Two panels with a shared y axis and one legend."""
    pdpart = pd.read_csv(CON / 'partitions.csv')
    plt.rcParams.update({'font.family': 'DejaVu Serif', 'font.size': 9})
    fig, axes = plt.subplots(1, 2, figsize=(6.3, 3.0), sharey=True)
    panels = [('covariate', '(a) Covariate composition'), ('allocation', '(b) Treated-to-control ratios')]
    for ax, (kind, title) in zip(axes, panels):
        z = pdpart[pdpart.kind == kind]
        ax.plot(z.rep, z.tau, label='FSRD / pooled', lw=2)
        ax.plot(z.rep, z.ivw, label='Inverse-variance average', lw=1)
        ax.plot(z.rep, z['mean'], label='Simple average', lw=1)
        ax.set_title(title, fontsize=9)
        ax.set_xlabel('Partition replication')
        ax.grid(alpha=.2)
    axes[0].set_ylabel('Estimated health-score jump')
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=3, fontsize=8, frameon=False, bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(OUT / 'Figure_2.pdf'); plt.close(fig)


def figure_3():
    pdpart = pd.read_csv(CON / 'partitions.csv')
    plt.rcParams.update({'font.family': 'DejaVu Serif', 'font.size': 10})
    z = pdpart[pdpart.kind == 'K']; fig, ax = plt.subplots(figsize=(6.3, 3.4))
    ax.plot(z.rep, z.tau, label='FSRD / pooled', lw=2); ax.plot(z.rep, z.ivw, label='Inverse-variance average', lw=1)
    ax.plot(z.rep, z['mean'], label='Simple average', lw=1)
    ax.set(xlabel='Number of participants', ylabel='Estimated health-score jump')
    ax.legend(fontsize=8); ax.grid(alpha=.2); fig.tight_layout()
    fig.savefig(OUT / 'Figure_3.pdf'); plt.close(fig)


def figure_4():
    plt.rcdefaults(); matplotlib.rcParams['pdf.fonttype'] = 42
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
    co = pd.read_csv(CO / 'coalitions_ik.csv'); fit = pd.read_csv(CO / 'coalition_fit.csv')
    d = co[(co.design == 'ITT') & co.ok]; f = fit[fit.design == 'ITT'].iloc[0]
    fig, ax = plt.subplots(figsize=(5.6, 3.7))
    dots = ax.scatter(d.N, d.mde, c=d.n_members, cmap='viridis', vmin=1, vmax=7, s=23, alpha=.8, edgecolors='none')
    grid = np.geomspace(d.N.min(), d.N.max(), 150)
    ax.plot(grid, 2.80 * np.exp(f.c) * grid ** (-f.a), color='#a14c2f', lw=1.5, label=f'Descriptive fit (a={f.a:.2f})')
    grand = d[d.n_members == 7].iloc[0]
    ax.scatter([grand.N], [grand.mde], marker='*', s=130, color='#aa2828', edgecolors='white', linewidths=.5, zorder=4,
               label='All seven regions')
    ax.set_xlabel('Coalition sample size N'); ax.set_ylabel('MDE (health-score points)')
    ax.set_xscale('log'); ax.set_yscale('log'); ax.grid(alpha=.2, which='major'); ax.legend(fontsize=8, frameon=False)
    cb = fig.colorbar(dots, ax=ax, ticks=range(1, 8), pad=.02); cb.set_label('Number of regions')
    fig.tight_layout(); fig.savefig(OUT / 'Figure_4.pdf', bbox_inches='tight'); plt.close(fig)


if __name__ == '__main__':
    OUT.mkdir(parents=True, exist_ok=True)
    figure_2(); figure_3(); figure_4()
    print('written to', OUT)
