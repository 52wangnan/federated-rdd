# -*- coding: utf-8 -*-
"""Vector (PDF) versions of Figures 2-5, redrawn from stored aggregate results with the plotting code that
produced the PNG versions (paper/charls_ik_analysis.py for Figures 2-4, paper/figures_charls.py for Figure 5);
TrueType fonts are embedded. No microdata is read.
    python paper/figures_vector.py
Figure 1 is a diagram drawn by the authors.
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


def figures_2_to_4():
    pdpart = pd.read_csv(CON / 'partitions.csv')
    plt.rcParams.update({'font.family': 'DejaVu Serif', 'font.size': 10})
    for number, kind in [(2, 'covariate'), (3, 'allocation'), (4, 'K')]:
        z = pdpart[pdpart.kind == kind]; fig, ax = plt.subplots(figsize=(6.3, 3.4))
        ax.plot(z.rep, z.tau, label='FSRD / pooled', lw=2); ax.plot(z.rep, z.ivw, label='Inverse-variance average', lw=1)
        ax.plot(z.rep, z['mean'], label='Simple average', lw=1)
        ax.set(xlabel='Number of participants' if kind == 'K' else 'Partition replication', ylabel='Estimated health-score jump')
        ax.legend(fontsize=8); ax.grid(alpha=.2); fig.tight_layout()
        fig.savefig(OUT / f'Figure_{number}.pdf'); plt.close(fig)
        plt.rcParams.update({'font.family': 'DejaVu Serif', 'font.size': 10})


def figure_5():
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
    fig.tight_layout(); fig.savefig(OUT / 'Figure_5.pdf', bbox_inches='tight'); plt.close(fig)


if __name__ == '__main__':
    OUT.mkdir(parents=True, exist_ok=True)
    figures_2_to_4(); figure_5()
    print('written to', OUT)
