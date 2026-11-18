# -*- coding: utf-8 -*-
"""Bandwidth sensitivity of the sharp design (Supplementary Figures C.3 and C.4), from result CSVs only.
    python paper/figures_bandwidth.py
"""
import os
import sys
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'charls')]
import figstyle  # noqa: E402
import make_charls_tables as mct  # noqa: E402

OUT = ROOT / 'results' / 'figures_bandwidth'


def bandwidth_sharp_only():
    for wave in (2018, 2015):
        mct.configure(wave)
        suf = '' if wave == 2018 else '_2015'
        bs = pd_read(os.path.join(mct.RES, 'bandwidth_sensitivity.csv'))
        g = bs[(bs.outcome == 'srh') & (bs.design == 'ITT')].sort_values('factor')
        fig, ax = plt.subplots(figsize=(figstyle.width('fig_charls_bandwidth') * 0.62, 2.6), layout='constrained')
        ax.errorbar(g['factor'], g['tau_bc'], yerr=[g['tau_bc'] - g['ci_rb_lo'], g['ci_rb_hi'] - g['tau_bc']], fmt='o',
                    color='#08306b', capsize=3, label='robust BC, 95% CI')
        ax.plot(g['factor'], g['tau_cl'], 'x', color='#e6550d', label='conventional')
        ax.axhline(0, color='grey', lw=0.8)
        ax.set_xticks(g['factor']); ax.set_xlabel(r'$h\,/\,\hat{h}_{\mathrm{MSE}}$')
        ax.set_title(figstyle.wrap(f"{mct.FIG_TITLE['srh']}, sharp RD (FSRD)", 38))
        fig.legend(*ax.get_legend_handles_labels(), loc='outside lower center', ncol=2, frameon=False)
        fig.savefig(OUT / f'fig_charls_bandwidth_srh{suf}.png', dpi=240); plt.close(fig)


def pd_read(path):
    import pandas as pd
    return pd.read_csv(path)


if __name__ == '__main__':
    OUT.mkdir(parents=True, exist_ok=True)
    bandwidth_sharp_only()
    print('written to', OUT)
