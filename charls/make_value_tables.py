# -*- coding: utf-8 -*-
"""Tables and figures of the decision analysis, computed only from results/charls*/value_*.csv.

    python charls/make_value_tables.py   (after charls/value_of_collaboration.py)
    -> output/tables/table_value_*  and  output/figures/fig_value_*  (suffix _2015 for the 2015 wave)
"""
import os
import sys

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
from reporting import save_table  # noqa: E402
import figstyle  # noqa: E402

TAB = os.path.join(ROOT, 'output', 'tables')
FIG = os.path.join(ROOT, 'output', 'figures')
RES = {2018: os.path.join(ROOT, 'results', 'charls'), 2015: os.path.join(ROOT, 'results', 'charls_2015')}
REGION_EN = {'华北': 'North', '东北': 'Northeast', '华东': 'East', '华中': 'Central', '华南': 'South', '西南': 'Southwest',
             '西北': 'Northwest'}
OUT_LABEL = {'srh_good': 'SRH fair or better (da002 version)', 'srh_good_all': 'SRH fair or better (either version)'}
# panel titles of the figures (the tables keep the labels above)
FIG_TITLE = {'srh_good': 'Self-rated health fair or better', 'srh_good_all': 'Self-rated health fair or better, either version'}
DESIGN_TITLE = {'ITT': 'ITT', 'FRD': 'fuzzy RD'}
Z_MDE = 2.80


def en(members):
    return ' + '.join(REGION_EN.get(m, m) for m in members.split('+'))


def main():
    os.makedirs(FIG, exist_ok=True)
    figstyle.setup()
    fits_all, sizes_all = [], []
    for wave, res in RES.items():
        if not os.path.exists(os.path.join(res, 'value_coalitions.csv')):
            continue
        suf = '' if wave == 2018 else f'_{wave}'
        co = pd.read_csv(os.path.join(res, 'value_coalitions.csv'))
        fits = pd.read_csv(os.path.join(res, 'value_fit.csv')).assign(wave=wave)
        sizes = pd.read_csv(os.path.join(res, 'value_min_size.csv')).assign(wave=wave)
        sh = pd.read_csv(os.path.join(res, 'value_shapley.csv'))
        sc = pd.read_csv(os.path.join(res, 'value_smallcell.csv'))
        fits_all.append(fits)
        sizes_all.append(sizes)
        # single regions and the grand coalition
        sel = co[(co.n_members == 1) | (co.n_members == co.n_members.max())].copy()
        rows = []
        for (y, members), g in sel.groupby(['outcome', 'members'], sort=False):
            it, fr = g[g.design == 'ITT'].iloc[0], g[g.design == 'FRD'].iloc[0]
            rows.append({'outcome': OUT_LABEL.get(y, y),
                         'analysed by': 'all 7 regions, federated' if it.n_members > 1 else en(members),
                         'N': int(it.N), 'ITT robust SE': it.se_rb, 'ITT MDE (pp)': 100 * it.mde,
                         'FRD robust SE': fr.se_rb, 'FRD MDE (pp)': 100 * fr.mde})
        t = pd.DataFrame(rows).sort_values(['outcome', 'N'])
        save_table(t, 'table_value_regions' + suf, TAB, digits=3,
                   caption=f'CHARLS {wave}: minimum detectable effect (MDE = 2.80 x robust SE, 80% power, 5% level) '
                           'of each region alone and of the 7-region federation')
        # Shapley
        t = sh.assign(region=sh.region.map(lambda v: REGION_EN.get(v, v)), outcome=sh.outcome.map(lambda v: OUT_LABEL.get(v, v)))
        t = t[['outcome', 'design', 'region', 'N', 'N_share', 'standalone_precision', 'shapley', 'shapley_share']]
        save_table(t, 'table_value_shapley' + suf, TAB, digits=3,
                   caption=f'CHARLS {wave}: Shapley values of the 7 regions; characteristic function = precision 1/Var '
                           '(robust) of the federated estimate of each coalition; shares compared with sample-size shares')
        # small-cell rule
        t = sc.assign(outcome=sc.outcome.map(lambda v: OUT_LABEL.get(v, v)))
        for c in ('suppressed', 'suppressed_final', 'n_h'):
            if c in t:
                t[c] = t[c].fillna(0).astype(int)
        cols = ['sites', 'outcome', 'design', 'min_cell', 'tau_bc', 'se_rb', 'h', 'n_h', 'suppressed', 'suppressed_final']
        t = t[[c for c in cols if c in t]].rename(columns={'min_cell': 'm', 'tau_bc': 'robust BC estimate', 'se_rb': 'robust SE',
                                                         'n_h': 'N in h', 'suppressed': 'windows suppressed (all rounds)',
                                                         'suppressed_final': 'windows suppressed (final h, b)'})
        save_table(t, 'table_value_smallcell' + suf, TAB, digits=4,
                   caption=f'CHARLS {wave}: provinces or the 7 regions as sites, small-cell rule m '
                           '(a site window with fewer than m records is not uploaded)')
        # figure: coalition curve
        outs = list(dict.fromkeys(co.outcome))
        fig, axs = plt.subplots(len(outs), 2, figsize=(figstyle.width('fig_value_coalitions'), 3.1 * len(outs)), squeeze=False,
                                layout='constrained')
        for i, y in enumerate(outs):
            for j, dg in enumerate(('ITT', 'FRD')):
                ax = axs[i, j]
                g = co[(co.outcome == y) & (co.design == dg) & co.ok]
                f = fits[(fits.outcome == y) & (fits.design == dg)].iloc[0]
                sc_ = ax.scatter(g.N, 100 * g.mde, c=g.n_members, cmap='viridis', s=14, zorder=3)
                nn = np.geomspace(g.N.min() * 0.8, g.N.max() * 4, 100)
                ax.plot(nn, 100 * Z_MDE * np.exp(f.c) * nn ** (-f.a), color='#d62728', lw=1.2,
                        label=f'descriptive fit: SE $\\propto N^{{-{f.a:.2f}}}$')
                full = g[g.n_members == g.n_members.max()].iloc[0]
                ax.plot(nn, 100 * full.mde * (nn / full.N) ** (-0.4), color='grey', ls='--', lw=1,
                        label='$N^{-2/5}$ through the\nseven-region point')
                for tgt in (0.10, 0.05):
                    ax.axhline(100 * tgt, color='k', lw=0.6, ls=':')
                ax.set_xscale('log')
                ax.set_yscale('log')
                ax.set_xlabel('coalition sample size N (records)')
                ax.set_ylabel('MDE (percentage points)')
                ax.set_title(figstyle.wrap(f'{FIG_TITLE.get(y, y)}, {DESIGN_TITLE[dg]}', 33))
                # legend under the panel: inside it would cover coalitions or the target lines
                ax.legend(frameon=False, loc='upper center', bbox_to_anchor=(0.5, -0.24))
        fig.colorbar(sc_, ax=axs, label='regions in coalition', shrink=0.6 if len(outs) > 1 else 0.9)
        fig.savefig(os.path.join(FIG, 'fig_value_coalitions' + suf + '.png'), dpi=300, bbox_inches='tight')
        fig.savefig(os.path.join(FIG, 'fig_value_coalitions' + suf + '.pdf'), bbox_inches='tight')
        plt.close(fig)
        # figure: Shapley shares vs size shares
        keys = list(dict.fromkeys(zip(sh.outcome, sh.design)))
        nc = min(2, len(keys))
        nr = -(-len(keys) // nc)
        fig, axs = plt.subplots(nr, nc, figsize=(figstyle.width('fig_value_shapley'), 0.4 + 2.25 * nr), squeeze=False,
                                layout='constrained')
        for ax in axs.flat[len(keys):]:
            ax.set_visible(False)
        for ax, (y, dg) in zip(axs.flat, keys):
            g = sh[(sh.outcome == y) & (sh.design == dg)].sort_values('N_share')
            pos = np.arange(len(g))
            ax.barh(pos - 0.18, g.N_share, height=0.36, color='#bdbdbd', label='sample-size share')
            ax.barh(pos + 0.18, g.shapley_share, height=0.36, color='#08306b', label='Shapley share of precision')
            ax.set_yticks(pos)
            ax.set_yticklabels([REGION_EN.get(v, v) for v in g.region])
            ax.set_title(figstyle.wrap(f'{FIG_TITLE.get(y, y)}, {DESIGN_TITLE[dg]}', 36))
            ax.set_xlabel('share')
        fig.legend(*axs[0, 0].get_legend_handles_labels(), loc='outside lower center', ncol=2, frameon=False)
        fig.savefig(os.path.join(FIG, 'fig_value_shapley' + suf + '.png'), dpi=300, bbox_inches='tight')
        fig.savefig(os.path.join(FIG, 'fig_value_shapley' + suf + '.pdf'), bbox_inches='tight')
        plt.close(fig)
    if fits_all:
        f = pd.concat(fits_all)
        f['outcome'] = f.outcome.map(lambda v: OUT_LABEL.get(v, v))
        save_table(f[['wave', 'outcome', 'design', 'a', 'a_theory', 'r2', 'coalitions_ok', 'coalitions_failed',
                      'N_all', 'se_all', 'mde_all']], 'table_value_fit', TAB, digits=3,
                   caption='Descriptive coalition curve: OLS of log robust SE on log N (theory: a = 2/5). Coalitions overlap; no independent-sample OLS uncertainty is reported.')
        s = pd.concat(sizes_all)
        s['outcome'] = s.outcome.map(lambda v: OUT_LABEL.get(v, v))
        s['smallest_observed_members'] = s.smallest_observed_members.fillna('').map(lambda v: en(v) if v else '')
        s['mde_target'] = s.mde_target.map(lambda v: f'{v:.2f}')
        s['extrapolated'] = s.extrapolated.map(lambda v: 'yes' if bool(v) else 'no')
        for c in ('smallest_observed_N', 'fewest_regions'):
            s[c] = s[c].map(lambda v: '' if pd.isna(v) else str(int(v)))
        save_table(s[['wave', 'outcome', 'design', 'mde_target', 'smallest_observed_N', 'smallest_observed_members',
                      'fewest_regions', 'N_from_fit', 'N_from_theory', 'N_available', 'extrapolated']], 'table_value_min_size', TAB, digits=0,
                   caption='Smallest coalition with MDE at or below the target (0.10 or 0.05): observed among the 127 coalitions, '
                           'from the fitted curve, and from the theoretical rate N^(-2/5) through the 7-region point '
                           '(extrapolated = beyond the available sample)')


if __name__ == '__main__':
    main()
