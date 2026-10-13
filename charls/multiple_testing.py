# -*- coding: utf-8 -*-
"""Holm (1979) adjusted p-values for all main-specification estimates of both waves.

    python charls/multiple_testing.py   (after run_charls.py for 2018 and 2015)
    -> results/charls_multiple_testing.csv, output/tables/table_charls_holm.*

Main specification: CCT mserd bandwidth, HC1, no covariates, public support and mass-point adjustment;
robust bias-corrected p-values.  Every outcome x design (ITT: reaching 60; FRD: receiving the residents'
pension) is one hypothesis.  Holm's step-down adjustment is valid under any dependence between the tests;
it is applied within each wave (2018: 5 outcomes x 2 designs; 2015: 6 x 2) and over both waves together.
"""
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
from reporting import save_table  # noqa: E402

LABEL = {'srh_good': 'SRH fair or better (da002)', 'srh': 'SRH 1-5, higher = worse (da002)',
         'srh_good_all': 'SRH fair or better, either version', 'cesd10': 'CES-D 10',
         'adl_any': 'Any ADL difficulty, asked subsample', 'adl_any_all': 'Any ADL difficulty, skipped = 0'}


def holm(p):
    p = np.asarray(p, float)
    order = np.argsort(p)
    m = len(p)
    adj = np.empty(m)
    run = 0.0
    for rank, i in enumerate(order):
        run = max(run, min(1.0, (m - rank) * p[i]))
        adj[i] = run
    return adj


def main():
    rows = []
    for wave, folder in ((2018, 'charls'), (2015, 'charls_2015')):
        m = pd.read_csv(os.path.join(ROOT, 'results', folder, 'main_results.csv'))
        m = m[(m.bw == 'mserd') & (m.vce == 'hc1') & m.design.isin(['ITT', 'FRD']) & m.outcome.isin(list(LABEL))]
        for _, r in m.iterrows():
            rows.append(dict(wave=wave, outcome=r.outcome, design=r.design, tau_bc=r.tau_bc, se_rb=r.se_rb,
                             ci_rb_lo=r.ci_rb_lo, ci_rb_hi=r.ci_rb_hi, p_raw=r.pv_rb))
    d = pd.DataFrame(rows)
    d['p_holm_within_wave'] = np.nan
    for wave, g in d.groupby('wave'):
        d.loc[g.index, 'p_holm_within_wave'] = holm(g.p_raw)
        d.loc[g.index, 'tests_in_wave'] = len(g)
    d['p_holm_both_waves'] = holm(d.p_raw)
    d = d.sort_values(['wave', 'p_raw'], ascending=[False, True])
    d.to_csv(os.path.join(ROOT, 'results', 'charls_multiple_testing.csv'), index=False)
    t = pd.DataFrame({'wave': d.wave, 'outcome': d.outcome.map(LABEL), 'design': d.design,
                      'robust BC estimate [95% CI]': [f'{a:.3f} [{lo:.3f}, {hi:.3f}]' for a, lo, hi in zip(d.tau_bc, d.ci_rb_lo, d.ci_rb_hi)],
                      'robust p': d.p_raw.map(lambda v: f'{v:.3f}'),
                      'Holm p, within wave': d.p_holm_within_wave.map(lambda v: f'{v:.3f}'),
                      'Holm p, both waves': d.p_holm_both_waves.map(lambda v: f'{v:.3f}')})
    n18, n15 = int((d.wave == 2018).sum()), int((d.wave == 2015).sum())
    save_table(t, 'table_charls_holm', os.path.join(ROOT, 'output', 'tables'),
               caption=f'Holm-adjusted robust p-values, main specification (2018: {n18} tests, 2015: {n15} tests, '
                       f'both waves: {n18 + n15})')
    print(t.to_string())
    print('significant at 5% after Holm (within wave):', int((d.p_holm_within_wave < 0.05).sum()),
          '| raw p < 0.05:', int((d.p_raw < 0.05).sum()))


if __name__ == '__main__':
    main()
