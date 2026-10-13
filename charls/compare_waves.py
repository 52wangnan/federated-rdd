# -*- coding: utf-8 -*-
"""CHARLS 2015 versus 2018: estimates, density test and covariate balance side by side, computed only from
results/charls/ (2018) and results/charls_2015/.

    python charls/compare_waves.py   (after run_charls.py for both waves)
    -> output/tables/table_charls_2015_vs_2018.*, output/figures/fig_charls_density_2015_vs_2018.*

The 1959-1961 famine birth cohorts: exact x-range from the 2015 extraction log (ID-card birth year); for 2018
(no birth year in the 2018 analysis file) the range implied by births Jan 1959 - Dec 1961 and interviews in
July - August 2018 (99.7% of the 2018 interviews; a few in September and November).
"""
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

RES = {2018: os.path.join(ROOT, 'results', 'charls'), 2015: os.path.join(ROOT, 'results', 'charls_2015')}
TAB = os.path.join(ROOT, 'output', 'tables')
FIG = os.path.join(ROOT, 'output', 'figures')
WAVES = (2018, 2015)
OUT_LABEL = {'srh_good': 'SRH fair or better (da002 version)', 'srh': 'SRH 1-5 (da002 version)', 'cesd10': 'CES-D 10',
             'adl_any': 'Any ADL difficulty (asked subsample)', 'srh_good_all': 'SRH fair or better (either version)',
             'adl_any_all': 'Any ADL difficulty (skipped = 0)'}
COV_LABEL = {'female': 'female', 'han': 'Han', 'edu_below_primary': 'education: below primary',
             'edu_primary': 'education: primary', 'edu_middle_plus': 'education: middle school or more',
             'married': 'married', 'lunar_birthday': 'lunar-calendar birthday', 'n_chronic': 'number of chronic conditions',
             'chr_hypertension': 'hypertension', 'chr_diabetes': 'diabetes', 'chr_arthritis': 'arthritis',
             'srh_form': 'SRH version da001 (randomised)'}


def famine_range(wave):
    if wave == 2015:
        return json.load(open(os.path.join(RES[2015], 'extraction_log.json'), encoding='utf-8'))['famine_cohort_x_range']
    # births 1959-01 ... 1961-12, interviews 2018-07 ... 2018-08
    return [((2018 * 12 + 7) - (1961 * 12 + 12) - 720) / 12, ((2018 * 12 + 8) - (1959 * 12 + 1) - 720) / 12]


def est(r, d=3):
    return f"{r['tau_bc']:.{d}f} [{r['ci_rb_lo']:.{d}f}, {r['ci_rb_hi']:.{d}f}]"


def main():
    load = lambda w, f: pd.read_csv(os.path.join(RES[w], f))
    logs = {w: json.load(open(os.path.join(RES[w], 'extraction_log.json'), encoding='utf-8')) for w in WAVES}
    rows = []
    add = lambda block, item, v18, v15: rows.append({'block': block, 'item': item, '2018': v18, '2015': v15})
    # ------------------------------------------------------------------ sample
    add('sample', 'cohort N (agricultural hukou, 45+, >= 1 chronic condition)', f"{logs[2018]['cohort_n']:,}", f"{logs[2015]['cohort_n']:,}")
    add('sample', 'age 60 at the interview = birth year', '1958', '1955')
    fr = {w: famine_range(w) for w in WAVES}
    add('sample', 'births 1959-1961: x = age - 60 (years)', f'{fr[2018][0]:.1f} to {fr[2018][1]:.1f} (approx.)',
        f'{fr[2015][0]:.1f} to {fr[2015][1]:.1f}')
    # ------------------------------------------------------------------ estimates
    mains = {w: load(w, 'main_results.csv') for w in WAVES}
    for w in WAVES:
        mains[w] = mains[w][(mains[w].vce == 'hc1') & (mains[w].bw == 'mserd')]
    for y in OUT_LABEL:
        for dg in ('ITT', 'FRD', 'first stage'):
            if dg == 'first stage' and y != 'srh_good':
                continue
            v = {}
            for w in WAVES:
                r = mains[w][(mains[w].outcome == y) & (mains[w].design == dg)]
                v[w] = '' if r.empty else est(r.iloc[0]) + ('' if dg == 'first stage' else f", h = {r.iloc[0]['h']:.2f}")
            n = {w: mains[w][(mains[w].outcome == y) & (mains[w].design == 'ITT')] for w in WAVES}
            item = f'{OUT_LABEL[y]}, {dg}' + ('' if dg != 'ITT' else ' (N: ' + ' / '.join(
                f"{int(n[w].iloc[0]['N']):,}" if not n[w].empty else '-' for w in WAVES) + ')')
            add('robust BC estimate [95% CI]', item, v[2018] or 'not built', v[2015])
    # ------------------------------------------------------------------ density
    dens = {w: load(w, 'density_test.csv') for w in WAVES}
    for fct in (0.5, 1.0, 2.0):
        v = {}
        for w in WAVES:
            r = dens[w][dens[w].h_factor == fct].iloc[0]
            v[w] = f"{r['theta']:.3f} ({r['se']:.3f}), p = {r['p']:.3f}, h = {r['h']:.1f}"
        add('McCrary theta = log f+ - log f- (SE)', f'h = {fct:g} x default', v[2018], v[2015])
    # ------------------------------------------------------------------ balance
    bal = {w: load(w, 'covariate_balance.csv').set_index('covariate') for w in WAVES}
    for c in COV_LABEL:
        v = {}
        for w in WAVES:
            v[w] = f"{bal[w].loc[c, 'tau_bc']:.3f}, p = {bal[w].loc[c, 'pv_rb']:.3f}" if c in bal[w].index else 'not built'
        add('balance: jump at 60 (robust BC), robust p', COV_LABEL[c], v[2018], v[2015])
    nrej = {w: f"{int((bal[w]['pv_rb'] < 0.05).sum())} of {len(bal[w])}" for w in WAVES}
    add('balance: jump at 60 (robust BC), robust p', 'tests with p < 0.05', nrej[2018], nrej[2015])
    # ------------------------------------------------------------------ placebo, donut, exactness
    pl = {w: load(w, 'placebo_cutoffs.csv') for w in WAVES}
    add('placebo cutoffs 55, 57, 63, 65', 'tests with robust p < 0.05 (SRH fair or better, SRH 1-5)',
        *(f"{int((pl[w]['pv_rb'] < 0.05).sum())} of {len(pl[w])}" for w in WAVES))
    dn = {w: load(w, 'donut.csv') for w in WAVES}
    rng = lambda d: f"{d['tau_bc'].min():.3f} to {d['tau_bc'].max():.3f}"
    add('donut 1, 2, 3, 6 months', 'SRH fair or better ITT, range of robust BC estimates',
        *(rng(dn[w][(dn[w].outcome == 'srh_good') & (dn[w].design == 'ITT')]) for w in WAVES))
    fe = load(2015, 'famine_excluded.csv')
    r = fe[(fe.outcome == 'srh_good') & (fe.design == 'ITT')].iloc[0]
    add('without 1959-1961 births', 'SRH fair or better ITT', 'not possible (cohort at the cutoff)', est(r) + f", h = {r['h']:.2f}")
    chk = {}
    for w in WAVES:
        chk[w] = max(float(pd.read_csv(os.path.join(RES[w], f))['maxdiff_central'].max()) for f in sorted(os.listdir(RES[w]))
                     if f.endswith('.csv') and 'maxdiff_central' in pd.read_csv(os.path.join(RES[w], f), nrows=0).columns)
    add('exactness', 'max |federated - pooled| over all analyses', f'{chk[2018]:.1e}', f'{chk[2015]:.1e}')
    save_table(pd.DataFrame(rows), 'table_charls_2015_vs_2018', TAB,
               caption='CHARLS 2015 versus 2018: age-60 RD for rural, 45+, chronically ill respondents (7 regional sites, '
                       'CCT mserd, HC1; federated = pooled). In 2015 the SRH question was asked in one of two randomly '
                       'assigned versions; srh and srh_good use the 2018 version (da002) only')
    figure(fr)
    print(pd.DataFrame(rows).to_string())


def figure(fr):
    plt.rcParams.update({'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False})
    fig, axs = plt.subplots(1, 2, figsize=(11, 3.2), sharey=False)
    for ax, w in zip(axs, WAVES):
        db = pd.read_csv(os.path.join(RES[w], 'density_bins.csv'))
        dt = pd.read_csv(os.path.join(RES[w], 'density_test.csv'))
        r = dt[dt.h_factor == 1.0].iloc[0]
        g = db[(db['mid'] > -15) & (db['mid'] < 20)]
        ax.bar(g['mid'] + 60, g['count'], width=1 / 12, color=np.where(g['mid'] < 0, '#9ecae1', '#3182bd'))
        lo, hi = fr[w]
        ax.axvspan(lo + 60, hi + 60, color='#fdae6b', alpha=0.4, lw=0,
                   label='born 1959-1961' + (' (approx.)' if w == 2018 else ''))
        ax.axvspan(60 - r['h'], 60 + r['h'], ymin=0, ymax=0.03, color='k', alpha=0.5, lw=0, label='McCrary window (default h)')
        ax.axvline(60, color='k', lw=0.8, ls='--')
        ax.set_xlabel('age at interview (years, monthly bins)')
        ax.set_ylabel('respondents')
        ptxt = 'p < 0.001' if r['p'] < 0.001 else f"p = {r['p']:.3f}"
        ax.set_title(f"CHARLS {w}: theta = {r['theta']:.3f} (SE {r['se']:.3f}), {ptxt}, h = {r['h']:.1f}", fontsize=8)
        ax.legend(frameon=False, fontsize=7, loc='upper right')
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, 'fig_charls_density_2015_vs_2018.png'), dpi=300, bbox_inches='tight')
    fig.savefig(os.path.join(FIG, 'fig_charls_density_2015_vs_2018.pdf'), bbox_inches='tight')
    plt.close(fig)


if __name__ == '__main__':
    main()
