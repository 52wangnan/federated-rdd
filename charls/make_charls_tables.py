# -*- coding: utf-8 -*-
"""Tables and figures of the CHARLS case study, computed only from results/charls/*.csv / *.json.

    python charls/make_charls_tables.py               -> output/tables/table_charls_*, output/figures/fig_charls_*
    python charls/make_charls_tables.py --wave 2015   -> the same names with the suffix _2015, from results/charls_2015/
"""
import argparse
import json
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

RES = os.path.join(ROOT, 'results', 'charls')
WAVE, SUF = 2018, ''
TAB = os.path.join(ROOT, 'output', 'tables')
FIG = os.path.join(ROOT, 'output', 'figures')
REGION_EN = {'华北': 'North', '东北': 'Northeast', '华东': 'East', '华中': 'Central', '华南': 'South', '西南': 'Southwest',
             '西北': 'Northwest'}
OUT_LABEL = {'srh_good': 'SRH fair or better (0/1)', 'srh': 'SRH 1-5 (higher = worse)', 'cesd10': 'CES-D 10 (0-30)',
             'adl_any': 'Any ADL difficulty (0/1)', 'pension': 'Receiving residents pension (0/1)',
             'srh_good_all': 'SRH fair or better, either version (0/1)', 'adl_any_all': 'Any ADL difficulty, skipped = 0 (0/1)'}
# panel titles of the figures (the tables keep the short labels above)
FIG_TITLE = {'srh_good': 'Self-rated health fair or better', 'srh': 'Self-rated health 1–5 (higher = worse)',
             'pension': 'Receipt of the pension', 'cesd10': 'CES-D 10 (0–30)', 'adl_any_all': 'Any ADL difficulty'}
# the same titles over two lines for the narrow panels of the RD plot
FIG_TITLE_2L = {'srh_good': 'Self-rated health\nfair or better', 'srh': 'Self-rated health 1–5\n(higher = worse)'}
DESIGN_TITLE = {'ITT': 'ITT', 'FRD': 'fuzzy RD'}


def configure(wave):
    global RES, WAVE, SUF
    WAVE, SUF = wave, ('' if wave == 2018 else f'_{wave}')
    RES = os.path.join(ROOT, 'results', 'charls' if wave == 2018 else f'charls_{wave}')


def save(df, name, caption, **kw):
    """save_table with the wave suffix; captions without a year get 'CHARLS <wave>: ' for waves other than 2018."""
    if WAVE != 2018 and not caption.startswith('CHARLS'):
        caption = f'CHARLS {WAVE}: ' + caption
    save_table(df, name + SUF, TAB, caption=caption, **kw)


def savefig(fig, name):
    fig.savefig(os.path.join(FIG, name + SUF + '.png'), dpi=300, bbox_inches='tight')
    fig.savefig(os.path.join(FIG, name + SUF + '.pdf'), bbox_inches='tight')


def ci(lo, hi, d=3):
    return f'[{lo:.{d}f}, {hi:.{d}f}]'


def est_table(df, keys):
    out = df[keys].copy()
    out['conventional (SE)'] = [f'{a:.3f} ({b:.3f})' for a, b in zip(df['tau_cl'], df['se_cl'])]
    out['bias-corrected'] = df['tau_bc'].map(lambda v: f'{v:.3f}')
    out['robust SE'] = df['se_rb'].map(lambda v: f'{v:.3f}')
    out['robust 95% CI'] = [ci(a, b) for a, b in zip(df['ci_rb_lo'], df['ci_rb_hi'])]
    out['robust p'] = df['pv_rb'].map(lambda v: f'{v:.3f}')
    out['h'] = df['h'].map(lambda v: f'{v:.2f}')
    out['b'] = df['b'].map(lambda v: f'{v:.2f}')
    out['N in h (-/+)'] = [f'{int(a)}/{int(b)}' for a, b in zip(df['n_h_minus'], df['n_h_plus'])]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--wave', type=int, default=2018, choices=(2015, 2018))
    configure(ap.parse_args().wave)
    os.makedirs(FIG, exist_ok=True)
    figstyle.setup()
    m = pd.read_csv(os.path.join(RES, 'main_results.csv'))
    main_rows = m[(m.vce == 'hc1') & (m.bw == 'mserd')].copy()
    main_rows['outcome'] = main_rows['outcome'].map(OUT_LABEL)
    save(est_table(main_rows, ['outcome', 'design']), 'table_charls_main',
         caption=f'CHARLS {WAVE}: effect of reaching age 60 (ITT) and of receiving the residents pension (FRD); '
                       'CCT mserd bandwidth, HC1, 7 regional sites (federated = pooled)')
    var = m[(m.vce != 'hc1') | (m.bw != 'mserd')].copy()
    var['outcome'] = var['outcome'].map(OUT_LABEL)
    save(est_table(var, ['outcome', 'design', 'bw', 'vce']), 'table_charls_variants',
         caption=f'CHARLS {WAVE}: cluster-robust variance, the IK (2012) bandwidth and protocol variants '
                 '(observed min/max instead of the public age support; no mass-point adjustment; rdrobust fuzzy bandwidth)')
    chk = float(m['maxdiff_central'].max())
    # bandwidth sensitivity
    bs = pd.read_csv(os.path.join(RES, 'bandwidth_sensitivity.csv'))
    t = est_table(bs, ['outcome', 'design', 'factor'])
    t['outcome'] = t['outcome'].map(OUT_LABEL)
    save(t, 'table_charls_bandwidth', caption='Bandwidth sensitivity: h = factor x h_mserd, rho = h/b fixed')
    # density
    d = pd.read_csv(os.path.join(RES, 'density_test.csv'))
    save(d, 'table_charls_density', caption='Federated McCrary (2008) test, monthly bins (theta = log f+ - log f-)')
    # balance
    b = pd.read_csv(os.path.join(RES, 'covariate_balance.csv'))
    t = est_table(b, ['covariate'])
    t.insert(1, 'sample mean', b['mean'].map(lambda v: f'{v:.3f}'))
    save(t, 'table_charls_balance', caption='Covariates as outcomes (jump at 60); chronic conditions may be '
                                                     'affected by treatment and are shown for information')
    # placebo and donut
    p = pd.read_csv(os.path.join(RES, 'placebo_cutoffs.csv'))
    t = est_table(p, ['outcome', 'cutoff'])
    t['outcome'] = t['outcome'].map(OUT_LABEL)
    save(t, 'table_charls_placebo', caption='Placebo cutoffs (records on one side of 60 only)')
    dn = pd.read_csv(os.path.join(RES, 'donut.csv'))
    t = est_table(dn, ['outcome', 'design', 'donut_months'])
    t['outcome'] = t['outcome'].map(OUT_LABEL)
    save(t, 'table_charls_donut', caption='Donut RD: records within the given number of months of age 60 removed')
    if os.path.exists(os.path.join(RES, 'famine_excluded.csv')):
        fe = pd.read_csv(os.path.join(RES, 'famine_excluded.csv'))
        t = est_table(fe, ['outcome', 'design', 'sample'])
        t['outcome'] = t['outcome'].map(OUT_LABEL)
        save(t, 'table_charls_famine', caption='Main fits without the 1959-1961 birth cohorts (ID-card birth year); '
                                               'bandwidth re-selected')
    # Dong & Lewbel (2015) threshold derivative: jump in the slope at 60
    if os.path.exists(os.path.join(RES, 'threshold_derivative.csv')):
        td = pd.read_csv(os.path.join(RES, 'threshold_derivative.csv'))
        t = est_table(td, ['outcome', 'design'])
        t['outcome'] = t['outcome'].map(OUT_LABEL)
        t.insert(2, 'level effect (robust BC)', td['level_tau_bc'].map(lambda v: '' if not np.isfinite(v) else f'{v:.3f}'))
        save(t, 'table_charls_derivative', digits=4,
             caption=f'CHARLS {WAVE}: Dong & Lewbel (2015) threshold derivative = jump in the slope of E[y | age] at 60 '
                     '(per year of age; local quadratic p = 2, bias correction q = 3, own mserd bandwidth, HC1)')
    # sites
    s = pd.read_csv(os.path.join(RES, 'site_estimates_region.csv'))
    s['site'] = s['site'].map(lambda v: REGION_EN.get(v, v))
    tt = s[s.kind != 'heterogeneity'].copy()
    out = tt[['outcome', 'design', 'site']].copy()
    out['n'] = tt['n']
    out['n within h'] = tt['n_h']
    out['h'] = tt['h'].map(lambda v: '' if not np.isfinite(v) else f'{v:.2f}')
    out['conventional (SE)'] = [f'{a:.3f} ({b:.3f})' if np.isfinite(a) else 'failed' for a, b in zip(tt['tau_cl'], tt['se_cl'])]
    out['bias-corrected (robust SE)'] = [f'{a:.3f} ({b:.3f})' if np.isfinite(a) else 'failed' for a, b in zip(tt['tau_bc'], tt['se_rb'])]
    het = s[s.kind == 'heterogeneity']
    het_txt = '; '.join(f'{o} {dg}: Q = {q:.2f} (df {int(df_)}), p = {pv:.3f}, I2 = {i2:.2f}' for o, dg, q, df_, pv, i2 in
                        zip(het.outcome, het.design, het.tau_cl, het.se_cl, het.tau_bc, het.se_rb))
    out['outcome'] = out['outcome'].map(OUT_LABEL)
    save(out, 'table_charls_regions', caption='Region-level estimates, the federated (= pooled) estimate and '
                                                        'meta-analytic combinations. Heterogeneity (robust BC): ' + het_txt)
    pr = pd.read_csv(os.path.join(RES, 'province_sites.csv'))
    out = pr[['outcome', 'design', 'estimator']].copy()
    out['bias-corrected'] = pr['tau_bc'].map(lambda v: f'{v:.3f}')
    out['robust SE'] = pr['se_rb'].map(lambda v: f'{v:.3f}')
    out['robust 95% CI'] = [ci(a, b_) for a, b_ in zip(pr['ci_rb_lo'], pr['ci_rb_hi'])]
    out['sites used'] = pr.get('sites_used', pd.Series(np.nan, index=pr.index)).fillna(pr['K'])
    out['outcome'] = out['outcome'].map(OUT_LABEL)
    save(out, 'table_charls_provinces', caption=f'{int(pr["K"].max())} province sites: federated estimate with and without the '
                                                          'small-cell rule (min 10 records per window side), IVW and simple average')
    chk_all = max(float(pd.read_csv(os.path.join(RES, f))['maxdiff_central'].max()) for f in sorted(os.listdir(RES))
                  if f.endswith('.csv') and 'maxdiff_central' in pd.read_csv(os.path.join(RES, f), nrows=0).columns)
    comm_table(json.load(open(os.path.join(RES, 'summary.json'), encoding='utf-8')))
    figures(m, bs, s)
    print('max |federated - pooled| over all CHARLS analyses:', chk_all)


def comm_table(summ):
    cm = summ['comm_main']
    comm = pd.DataFrame([dict(rounds=cm['rounds'], bytes_up_per_site=cm['bytes_up_per_site'], bytes_up_total=cm['bytes_up_total'],
                              bytes_down_total=cm['bytes_down_total'], microdata_bytes_total=summ['microdata_bytes_total'])])
    save(comm, 'table_charls_comm', digits=0, caption='Communication of the main CHARLS fit (7 sites, SRH fair or better, ITT)')


def frd_fs(m, y):
    r = m[(m.outcome == y) & (m.design == 'first stage')]
    return float(r['tau_bc'].iloc[0])


def figures(m, bs, s):
    # RD plot
    bins = pd.read_csv(os.path.join(RES, 'rdplot_bins.csv'))
    fit = pd.read_csv(os.path.join(RES, 'rdplot_fit.csv'))
    fig, axs = plt.subplots(1, 3, figsize=(figstyle.width('fig_charls_rdplot'), 2.9), layout='constrained')
    for ax, v in zip(axs, ('srh_good', 'srh', 'pension')):
        g = bins[bins.variable == v]
        ax.scatter(g['x_mid'] + 60, g['mean'], s=np.clip(g['n'] / 8, 4, 40), color='#6baed6', edgecolor='none',
                   label='6-month bin means')
        for side in ('-', '+'):
            f = fit[(fit.variable == v) & (fit.side == side)].sort_values('x')
            ax.plot(f['x'] + 60, f['fit'], color='#08306b', lw=1.8, label='local linear fit within h' if side == '+' else None)
        ax.axvline(60, color='grey', lw=0.8, ls='--')
        ax.set_xlabel('age (years)')
        ax.set_title(FIG_TITLE_2L.get(v, FIG_TITLE[v]))
    fig.legend(*axs[0].get_legend_handles_labels(), loc='outside lower center', ncol=2, frameon=False)
    savefig(fig, 'fig_charls_rdplot')
    plt.close(fig)
    # bandwidth sensitivity
    fig, axs = plt.subplots(2, 2, figsize=(figstyle.width('fig_charls_bandwidth'), 4.6), layout='constrained')
    for ax, (y, dg) in zip(axs.flat, [('srh_good', 'ITT'), ('srh_good', 'FRD'), ('srh', 'ITT'), ('srh', 'FRD')]):
        g = bs[(bs.outcome == y) & (bs.design == dg)].sort_values('factor')
        ax.errorbar(g['factor'], g['tau_bc'], yerr=[g['tau_bc'] - g['ci_rb_lo'], g['ci_rb_hi'] - g['tau_bc']], fmt='o',
                    color='#08306b', capsize=3, label='robust BC, 95% CI')
        ax.plot(g['factor'], g['tau_cl'], 'x', color='#e6550d', label='conventional')
        ax.axhline(0, color='grey', lw=0.8)
        ax.set_xticks(g['factor'])
        ax.set_xlabel(r'$h\,/\,\hat{h}_{\mathrm{MSE}}$')
        ax.set_title(figstyle.wrap(f'{FIG_TITLE[y]}, {DESIGN_TITLE[dg]}', 38))
    fig.legend(*axs[0, 0].get_legend_handles_labels(), loc='outside lower center', ncol=2, frameon=False)
    savefig(fig, 'fig_charls_bandwidth')
    plt.close(fig)
    # density
    db = pd.read_csv(os.path.join(RES, 'density_bins.csv'))
    dt = pd.read_csv(os.path.join(RES, 'density_test.csv'))
    r = dt[dt.h_factor == 1.0].iloc[0]
    g = db[(db['mid'] > -15) & (db['mid'] < 20)]
    fig, ax = plt.subplots(figsize=(figstyle.width('fig_charls_density'), 2.8))
    ax.bar(g['mid'] + 60, g['count'], width=1 / 12, color=np.where(g['mid'] < 0, '#9ecae1', '#3182bd'))
    ax.axvline(60, color='k', lw=0.8, ls='--')
    if WAVE != 2018:
        lo, hi = json.load(open(os.path.join(RES, 'extraction_log.json'), encoding='utf-8'))['famine_cohort_x_range']
        ax.axvspan(lo + 60, hi + 60, color='#fdae6b', alpha=0.4, lw=0, label='born 1959–1961 (ID card)')
        ax.legend(frameon=False, loc='upper right')
    ax.set_xlabel('age (years, monthly bins)')
    ax.set_ylabel('respondents')
    ptxt = 'p < 0.001' if r['p'] < 0.001 else f"p = {r['p']:.2f}"
    ax.set_title(f"McCrary test: log difference {r['theta']:.3f} (SE {r['se']:.3f}), {ptxt}, h = {r['h']:.1f}")
    fig.tight_layout()
    savefig(fig, 'fig_charls_density')
    plt.close(fig)
    # forest plot (regions)
    fig, axs = plt.subplots(2, 2, figsize=(figstyle.width('fig_charls_forest'), 6.6), sharey=True)
    for ax, (y, dg) in zip(axs.flat, [('srh_good', 'ITT'), ('srh_good', 'FRD'), ('srh', 'ITT'), ('srh', 'FRD')]):
        g = s[(s.outcome == y) & (s.design == dg) & (s.kind != 'heterogeneity')].reset_index(drop=True)
        ypos = np.arange(len(g))[::-1]
        # axis limits: 1.5 interval widths around the federated and IVW intervals; wider
        # intervals are drawn up to the axis edge with an arrow and their numeric value (no data altered)
        ref = g[(g.kind == 'federated') | (g.site == 'IVW')]
        lo_r = (ref['tau_bc'] - 1.96 * ref['se_rb']).min()
        hi_r = (ref['tau_bc'] + 1.96 * ref['se_rb']).max()
        wd = hi_r - lo_r
        L, U = lo_r - 1.5 * wd, hi_r + 1.5 * wd
        for yy, (_, rr) in zip(ypos, g.iterrows()):
            if not np.isfinite(rr['tau_bc']):
                ax.text(0, yy, 'failed', va='center')
                continue
            c_ = '#08306b' if rr['kind'] == 'federated' else ('#e6550d' if rr['kind'] == 'meta' else '#6baed6')
            lo, hi, t = rr['tau_bc'] - 1.96 * rr['se_rb'], rr['tau_bc'] + 1.96 * rr['se_rb'], rr['tau_bc']
            ax.plot([max(lo, L), min(hi, U)], [yy, yy], color=c_, lw=1.2)
            if L <= t <= U:
                ax.plot(t, yy, 's' if rr['kind'] != 'site' else 'o', color=c_)
            if lo < L:
                ax.plot(L, yy, '<', color=c_, ms=5)
            if hi > U:
                ax.plot(U, yy, '>', color=c_, ms=5)
            if lo < L or hi > U:
                ax.text(U, yy + 0.35, f'{t:.2f} [{lo:.1f}, {hi:.1f}]', ha='right', color=c_)
        ax.set_xlim(L, U)
        ax.xaxis.set_major_locator(plt.MaxNLocator(5))
        ax.set_yticks(ypos)
        ax.set_yticklabels([REGION_EN.get(v, v) for v in g['site']])
        ax.axvline(0, color='grey', lw=0.8)
        ax.set_title(figstyle.wrap(f'{FIG_TITLE[y]}, {DESIGN_TITLE[dg]}', 34))
        ax.set_xlabel('robust BC estimate, 95% CI')
    fig.tight_layout()
    savefig(fig, 'fig_charls_forest')
    plt.close(fig)
    # partitions of the same records
    pt = pd.read_csv(os.path.join(RES, 'partition_invariance.csv'))
    fig, axs = plt.subplots(1, 2, figsize=(figstyle.width('fig_charls_partitions'), 3.5), layout='constrained')
    for ax, y in zip(axs, ('srh_good', 'srh')):
        g = pt[pt.outcome == y].sort_values('K')
        ax.fill_between(g['K'], g['fed_ci_lo'], g['fed_ci_hi'], color='#c6dbef', label='federated robust 95% CI')
        ax.plot(g['K'], g['fed_tau_bc'], color='#08306b', lw=2, label='federated (= pooled), robust BC')
        ax.plot(g['K'], g['ivw_tau_bc'], color='#e6550d', marker='.', lw=1, label='IVW of site estimates')
        ax.plot(g['K'], g['mean_tau_bc'], color='#31a354', marker='.', lw=1, label='simple average of site estimates')
        ax.axhline(0, color='grey', lw=0.8)
        ax.set_xlabel('number of sites K\n(random partition of the same records)')
        ax.set_title(figstyle.wrap(FIG_TITLE[y] + ', ITT', 40))
        lo, hi = np.nanpercentile(np.r_[g['ivw_tau_bc'], g['mean_tau_bc'], g['fed_ci_lo'], g['fed_ci_hi']], [1, 99])
        ax.set_ylim(lo - 0.1 * (hi - lo), hi + 0.1 * (hi - lo))
    fig.legend(*axs[0].get_legend_handles_labels(), loc='outside lower center', ncol=2, frameon=False)
    savefig(fig, 'fig_charls_partitions')
    plt.close(fig)


if __name__ == '__main__':
    main()
