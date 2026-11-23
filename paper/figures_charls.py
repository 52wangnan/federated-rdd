# -*- coding: utf-8 -*-
"""Figure 4 (coalition sample size and minimum detectable effects) and Supplementary Figure C.2 (self-rated health
and pension receipt by age), from result CSVs only; no microdata is read.

    python paper/figures_charls.py
"""
import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'charls')]
import figstyle  # noqa: E402
import make_charls_tables as mct  # noqa: E402

OUT = ROOT / 'results' / 'ik_benchmarks'
CO = ROOT / 'results' / 'composition_coalitions'


def coalition(name='coalition_mde_fsrd.png'):
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
    fig.tight_layout(); fig.savefig(OUT / name, dpi=240, bbox_inches='tight'); plt.close(fig)


def rdplot_2018():
    plt.rcdefaults()
    mct.configure(2018)
    bins = pd.read_csv(os.path.join(mct.RES, 'rdplot_bins.csv')); fit = pd.read_csv(os.path.join(mct.RES, 'rdplot_fit.csv'))
    fig, axs = plt.subplots(1, 2, figsize=(figstyle.width('fig_charls_rdplot') * 2 / 3, 2.9), layout='constrained')
    for ax, v in zip(axs, ('srh', 'pension')):
        g = bins[bins.variable == v]
        ax.scatter(g['x_mid'] + 60, g['mean'], s=np.clip(g['n'] / 8, 4, 40), color='#6baed6', edgecolor='none', label='6-month bin means')
        for side in ('-', '+'):
            f = fit[(fit.variable == v) & (fit.side == side)].sort_values('x')
            ax.plot(f['x'] + 60, f['fit'], color='#08306b', lw=1.8, label='local linear fit within h' if side == '+' else None)
        ax.axvline(60, color='grey', lw=0.8, ls='--'); ax.set_xlabel('age (years)')
        ax.set_title(mct.FIG_TITLE_2L.get(v, mct.FIG_TITLE[v]))
    fig.legend(*axs[0].get_legend_handles_labels(), loc='outside lower center', ncol=2, frameon=False)
    fig.savefig(OUT / 'fig_charls_rdplot_srh.png', dpi=240); plt.close(fig)


if __name__ == '__main__':
    OUT.mkdir(parents=True, exist_ok=True)
    coalition()
    rdplot_2018()
    print('written to', OUT)
