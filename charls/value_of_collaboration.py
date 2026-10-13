# -*- coding: utf-8 -*-
"""Decision analysis: what collaboration is worth.

    python charls/value_of_collaboration.py                 # both waves (2015 is the main wave)
    python charls/value_of_collaboration.py --wave 2015     # one wave
    -> results/charls_{wave}/value_coalitions.csv, value_fit.csv, value_min_size.csv, value_shapley.csv,
       value_smallcell.csv  (tables / figures: charls/make_value_tables.py)

A1  coalition curve.  For every non-empty subset S of the 7 regional sites (127 coalitions), the federated
    robust bias-corrected estimate of the effect on self-rated health 'fair or better' (ITT: reaching 60;
    FRD: receiving the residents' pension) with its own CCT bandwidth; robust SE, minimum detectable effect
    MDE = 2.80 x SE (80% power, two-sided 5% test) and coalition size N.  log SE = c - a log N is fitted by OLS
    over all coalitions; the MSE-optimal bandwidth h ~ N^(-1/5) implies SE ~ (N h)^(-1/2) ~ N^(-2/5).  The
    smallest coalition reaching MDE <= 0.10 / 0.05 is read off the coalitions themselves where possible and
    otherwise extrapolated from the fit (marked as such).
A2  value sharing.  Characteristic function v(S) = 1 / Var_robust(S) (precision; v = 0 for a coalition whose
    estimate cannot be computed and for the empty set); exact Shapley values of the 7 regions, compared with
    shares proportional to sample size.
A3  privacy versus precision.  Provinces (and, for comparison, the 7 regions) as sites with the small-cell
    rule m in {0, 5, 10, 20, 50}: estimate, robust SE and the number of site windows the rule suppressed.

Outcomes: srh_good in both waves (2015: the version comparable with 2018) and, for 2015, srh_good_all
(either randomly assigned SRH version).  Every estimate uses the protocol settings of run_charls.py (public
support 45-120 years, monthly grid with mass-point adjustment).
"""
import argparse
import itertools
import math
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)
from fsrd import fsrd  # noqa: E402
import run_charls as RC  # noqa: E402

BASE = (dict(RC.OUTCOMES), list(RC.BALANCE))          # run_charls.configure() extends these per wave

Z_MDE = 2.80                       # 1.96 + 0.84: 80% power at the two-sided 5% level
TARGETS = (0.10, 0.05)
SMALL_CELL = (0, 5, 10, 20, 50)


def outcomes_for(wave):
    return ['srh_good', 'srh_good_all'] if wave == 2015 else ['srh_good']


def est(sites, fuzzy, **kw):
    try:
        r = fsrd(sites, **{**RC.OPT, **kw})
        return dict(ok=True, tau_bc=r['tau_bc'], se_rb=r['se_rb'], tau_cl=r['tau_cl'], se_cl=r['se_cl'], h=r['h'],
                    n_h=sum(r['n_h']), **({k: r[k] for k in ('suppressed', 'suppressed_final')} if 'suppressed' in r else {}))
    except (np.linalg.LinAlgError, ValueError, ZeroDivisionError, FloatingPointError):
        return dict(ok=False, tau_bc=np.nan, se_rb=np.nan, tau_cl=np.nan, se_cl=np.nan, h=np.nan, n_h=np.nan)


def coalitions(df, wave):
    rows = []
    for y in outcomes_for(wave):
        for fuzzy in (False, True):
            sites = RC.sites_of(df, y, fuzzy=fuzzy)
            names = [s.name for s in sites]
            for k in range(1, len(sites) + 1):
                for idx in itertools.combinations(range(len(sites)), k):
                    S = [sites[i] for i in idx]
                    r = est(S, fuzzy)
                    rows.append(dict(wave=wave, outcome=y, design='FRD' if fuzzy else 'ITT', members='+'.join(names[i] for i in idx),
                                     n_members=k, mask=sum(1 << i for i in idx), N=int(sum(s.n for s in S)),
                                     mde=Z_MDE * r['se_rb'], **r))
            print(f'  {wave} {y} {"FRD" if fuzzy else "ITT"}: {2 ** len(sites) - 1} coalitions', flush=True)
    return pd.DataFrame(rows)


def fit_and_sizes(co):
    fits, sizes = [], []
    for (y, dg), g in co.groupby(['outcome', 'design']):
        ok = g[g.ok & np.isfinite(g.se_rb) & (g.se_rb > 0)]
        X = np.column_stack([np.ones(len(ok)), np.log(ok.N)])
        beta, *_ = np.linalg.lstsq(X, np.log(ok.se_rb), rcond=None)
        resid = np.log(ok.se_rb) - X @ beta
        s2 = resid @ resid / (len(ok) - 2)
        cov = s2 * np.linalg.inv(X.T @ X)
        a, c = -beta[1], beta[0]
        r2 = 1 - resid @ resid / np.sum((np.log(ok.se_rb) - np.log(ok.se_rb).mean()) ** 2)
        full = g[g.n_members == g.n_members.max()].iloc[0]
        fits.append(dict(outcome=y, design=dg, a=a, a_se=float(np.sqrt(cov[1, 1])), a_theory=0.4, c=c, r2=r2,
                         coalitions_ok=int(len(ok)), coalitions_failed=int((~g.ok).sum()),
                         N_all=int(full.N), se_all=float(full.se_rb), mde_all=float(full.mde)))
        for tgt in TARGETS:
            hit = ok[ok.mde <= tgt]
            n_star = float(np.exp((np.log(tgt / Z_MDE) - c) / (-a)))
            n_theory = float(full.N * (full.mde / tgt) ** (1 / 0.4))       # N^(-2/5) through the grand coalition
            sizes.append(dict(outcome=y, design=dg, mde_target=tgt,
                              smallest_observed_N=int(hit.N.min()) if len(hit) else np.nan,
                              smallest_observed_members=(hit.sort_values('N').iloc[0].members if len(hit) else ''),
                              fewest_regions=int(hit.n_members.min()) if len(hit) else np.nan,
                              N_from_fit=n_star, N_from_theory=n_theory, N_available=int(full.N),
                              extrapolated=bool(min(n_star, n_theory) > full.N)))
    return pd.DataFrame(fits), pd.DataFrame(sizes)


def shapley(co):
    rows = []
    for (y, dg), g in co.groupby(['outcome', 'design']):
        names = g[g.n_members == 1].sort_values('mask').members.tolist()
        n = len(names)
        v = {0: 0.0}
        for _, r in g.iterrows():
            v[int(r['mask'])] = (1.0 / r.se_rb ** 2) if (r.ok and np.isfinite(r.se_rb) and r.se_rb > 0) else 0.0
        size = {int(r['mask']): int(r.N) for _, r in g.iterrows()}
        full = (1 << n) - 1
        phi = np.zeros(n)
        for i in range(n):
            for m in range(1 << n):
                if m & (1 << i):
                    continue
                s = bin(m).count('1')
                wgt = math.factorial(s) * math.factorial(n - s - 1) / math.factorial(n)
                phi[i] += wgt * (v[m | (1 << i)] - v[m])
        for i in range(n):
            rows.append(dict(outcome=y, design=dg, region=names[i], N=size[1 << i], N_share=size[1 << i] / size[full],
                             standalone_precision=v[1 << i], shapley=phi[i], shapley_share=phi[i] / v[full],
                             grand_precision=v[full], efficiency_gap=float(phi.sum() - v[full])))
    return pd.DataFrame(rows)


def small_cell(df, wave):
    rows = []
    for by in ('province', 'region'):
        for y in outcomes_for(wave):
            for fuzzy in (False, True):
                S = RC.sites_of(df, y, by=by, fuzzy=fuzzy)
                for m in SMALL_CELL:
                    r = est(S, fuzzy, min_cell=m)
                    rows.append(dict(wave=wave, sites=by, outcome=y, design='FRD' if fuzzy else 'ITT', min_cell=m, K=len(S),
                                     N=int(sum(s.n for s in S)), **r))
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--wave', default='both', choices=('both', '2015', '2018'))
    a = ap.parse_args()
    waves = (2015, 2018) if a.wave == 'both' else (int(a.wave),)
    for wave in waves:
        RC.OUTCOMES, RC.BALANCE = dict(BASE[0]), list(BASE[1])
        RC.configure(wave)
        df = RC.load()
        out = RC.OUT
        os.makedirs(out, exist_ok=True)
        co = coalitions(df, wave)
        co.to_csv(os.path.join(out, 'value_coalitions.csv'), index=False)
        fits, sizes = fit_and_sizes(co)
        fits.to_csv(os.path.join(out, 'value_fit.csv'), index=False)
        sizes.to_csv(os.path.join(out, 'value_min_size.csv'), index=False)
        shapley(co).to_csv(os.path.join(out, 'value_shapley.csv'), index=False)
        small_cell(df, wave).to_csv(os.path.join(out, 'value_smallcell.csv'), index=False)
        print(wave, fits.round(3).to_string(), sizes.to_string(), sep='\n')


if __name__ == '__main__':
    main()
