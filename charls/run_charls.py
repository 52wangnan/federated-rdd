# -*- coding: utf-8 -*-
"""CHARLS case study: the age-60 eligibility threshold of the urban-rural residents' pension.

    python charls/run_charls.py               (after charls/extract.py)       -> results/charls/*.csv, *.json
    python charls/run_charls.py --wave 2015   (after charls/extract_2015.py)  -> results/charls_2015/*.csv, *.json

The 2015 wave adds the outcomes srh_good_all (both randomly assigned SRH versions) and adl_any_all (ADL with
respondents skipped past the items counted as no difficulty), the balance check srh_form, and 'famine': the
main ITT / FRD fits without the 1959-1961 birth cohorts (famine_excluded.csv).

Sites are the seven geographic regions (main analysis) or the 27 provinces.  Every estimate is computed
by the federated protocol (sites upload sufficient statistics only) and checked against the same
computation on the pooled records ('maxdiff_central').  Analyses:
  main           SRD / intention-to-treat effect of reaching 60 and fuzzy RD (take-up = currently receiving
                 the pension), CCT mserd bandwidth, conventional and robust bias-corrected inference, HC1;
                 with and without the predetermined covariates (CCFT 2019); cluster-robust variants
                 (community; month-of-age mass points); IK (2012) bandwidth with conventional inference
  sensitivity    h in {0.5, 0.75, 1, 1.25, 1.5, 2} x h_mserd (rho = h/b kept fixed)
  density        federated McCrary (2008) test on monthly bins
  balance        predetermined covariates as outcomes
  placebo        placebo cutoffs 55, 57, 63, 65 (one side of the true cutoff only)
  donut          records within 1, 2, 3, 6 months of the cutoff removed
  sites          region-level estimates, inverse-variance (IVW) and simple averages, Cochran Q / I^2
  partitions     random partitions into K = 2..60 sites (seed 20261011 + K): federated vs IVW vs average
  provinces      27 province sites with and without the small-cell rule (min 10 records per window side)
  rdplot         binned means (federated sums) and the fitted local linear lines
  derivative     Dong & Lewbel (2015) threshold derivative: jump in the slope at 60 (deriv = 1, p = 2, q = 3)

Protocol settings (OPT, used by every estimate): the public support of age, 45-120 years (x = -15 ... 60),
replaces the sites' smallest / largest ages; age in months is a discrete running variable on the public
grid of 1/12 year, so the bandwidth selector uses exact quantiles and rdrobust's default mass-point
adjustment (masspoints = 'adjust').  The 'variants' rows show the main fits under the sites' observed
range (= what rdrobust uses), without the mass-point adjustment, and with rdrobust's fuzzy bandwidth.
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import chi2

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
from fsrd import Site, Federation, fsrd, centralized, site_estimates, meta_ivw, meta_mean, mccrary  # noqa: E402
from fsrd.federation import dd, dd_exact  # noqa: E402

OUT = os.path.join(ROOT, 'results', 'charls')
WAVE = 2018
SEED = 20261011
COVS = ['female', 'han', 'edu_below_primary', 'edu_primary', 'edu_middle_plus', 'married']
OUTCOMES = {'srh_good': 'Self-rated health fair or better (0/1)',
            'srh': 'Self-rated health, 1 very good - 5 very poor',
            'cesd10': 'CES-D 10 depressive symptoms (0-30)',
            'adl_any': 'Any basic ADL difficulty (0/1)'}
BALANCE = COVS + ['lunar_birthday', 'n_chronic', 'chr_hypertension', 'chr_diabetes', 'chr_arthritis']
EXTRA_OUTCOMES = {2015: {'srh_good_all': 'Self-rated health fair or better, either SRH version (0/1)',
                         'adl_any_all': 'Any basic ADL difficulty, skipped respondents = 0 (0/1)'},
                  2018: {'adl_any_all': 'Any basic ADL difficulty, skipped respondents = 0 (0/1)'}}
SUPPORT = (45.0 - 60.0, 120.0 - 60.0)        # public age range 45-120 in centred years
MASS_GRID = 1.0 / 12.0                       # age in months: public grid of the running variable
OPT = dict(support=SUPPORT, mass_grid=MASS_GRID)
EXTRA_BALANCE = {2015: ['srh_form']}
CMP = ('tau_cl', 'tau_bc', 'se_cl', 'se_rb', 'h', 'b')


def configure(wave):
    """Point OUT / OUTCOMES / BALANCE at one CHARLS wave (2018 keeps the original paths and lists)."""
    global OUT, WAVE, OUTCOMES, BALANCE
    WAVE = wave
    OUT = os.path.join(ROOT, 'results', 'charls' if wave == 2018 else f'charls_{wave}')
    OUTCOMES = dict(OUTCOMES, **EXTRA_OUTCOMES.get(wave, {}))
    BALANCE = BALANCE + EXTRA_BALANCE.get(wave, [])


def load():
    df = pd.read_csv(os.path.join(ROOT, 'data', f'charls{WAVE}_rd.csv'), dtype={'ID': str, 'householdID': str, 'communityID': str})
    df['comm_id'] = pd.factorize(df['communityID'])[0]
    df['age_m'] = df['age_months'].astype(int)
    return df


def sites_of(df, y, by='region', fuzzy=False, covs=False, cluster=None, labels=None):
    need = ['x', y] + (['pension'] if fuzzy else []) + (COVS if covs else [])
    d = df.dropna(subset=need)
    groups = d.groupby(by, sort=True) if labels is None else d.groupby(labels, sort=True)
    out = []
    for name, g in groups:
        out.append(Site(g['x'].to_numpy(float), g[y].to_numpy(float),
                        d=g['pension'].to_numpy(float) if fuzzy else None,
                        w=g[COVS].to_numpy(float) if covs else None,
                        cl=None if cluster is None else g[cluster].to_numpy(), name=str(name)))
    return out


def row_of(r, **tags):
    row = dict(tags)
    for k in ('tau_cl', 'se_cl', 'tau_bc', 'se_rb', 'pv_cl', 'pv_rb', 'h', 'b'):
        row[k] = r[k]
    row.update(ci_cl_lo=r['ci_cl'][0], ci_cl_hi=r['ci_cl'][1], ci_rb_lo=r['ci_rb'][0], ci_rb_hi=r['ci_rb'][1],
               n_h_minus=r['n_h'][0], n_h_plus=r['n_h'][1], N=r['N'], K=r['K'],
               rounds=r['comm']['rounds'], bytes_up_site=r['comm']['bytes_up_per_site'])
    return row


def run(sites, **kw):
    """Federated estimate plus the pooled-data check (protocol options OPT unless overridden)."""
    kw = {**OPT, **kw}
    r = fsrd(sites, **kw)
    c = centralized(sites, **kw)
    r['maxdiff_central'] = max(abs(r[k] - c[k]) for k in CMP)
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--wave', type=int, default=2018, choices=(2015, 2018))
    configure(ap.parse_args().wave)
    os.makedirs(OUT, exist_ok=True)
    df = load()
    summary = dict(N_cohort=len(df), sites='7 geographic regions', covariates=COVS)
    grid = np.unique(df['age_m'].to_numpy())
    # ------------------------------------------------------------------ main results
    rows, mains = [], {}
    for y in OUTCOMES:
        for fuzzy in (False, True):
            for covs in (False, True):
                S = sites_of(df, y, fuzzy=fuzzy, covs=covs)
                r = run(S, covs=covs)
                design = ('FRD' if fuzzy else 'ITT') + ('+cov' if covs else '')
                rows.append(dict(row_of(r, outcome=y, design=design, bw='mserd', vce='hc1'), maxdiff_central=r['maxdiff_central']))
                mains[(y, fuzzy, covs)] = r
                if fuzzy:
                    fs = r['fs']
                    rows.append(dict(outcome=y, design='first stage' + ('+cov' if covs else ''), bw='mserd', vce='hc1',
                                     tau_cl=fs['tau_cl'], se_cl=fs['se_cl'], tau_bc=fs['tau_bc'], se_rb=fs['se_rb'],
                                     pv_cl=fs['pv_cl'], pv_rb=fs['pv_rb'], h=r['h'], b=r['b'],
                                     ci_cl_lo=fs['ci_cl'][0], ci_cl_hi=fs['ci_cl'][1], ci_rb_lo=fs['ci_rb'][0], ci_rb_hi=fs['ci_rb'][1],
                                     n_h_minus=r['n_h'][0], n_h_plus=r['n_h'][1], N=r['N'], K=r['K']))
            # cluster-robust variants and the IK pipeline (no covariates)
            S = sites_of(df, y, fuzzy=fuzzy, cluster='comm_id')
            r = run(S, vce='cluster')
            rows.append(dict(row_of(r, outcome=y, design='FRD' if fuzzy else 'ITT', bw='mserd', vce='cluster: community'),
                             maxdiff_central=r['maxdiff_central']))
            S = sites_of(df, y, fuzzy=fuzzy, cluster='age_m')
            r = run(S, vce='cluster_grid', grid=grid)
            rows.append(dict(row_of(r, outcome=y, design='FRD' if fuzzy else 'ITT', bw='mserd', vce='cluster: month of age'),
                             maxdiff_central=r['maxdiff_central']))
            S = sites_of(df, y, fuzzy=fuzzy)
            rik = run(S, bw='ik')
            rows.append(dict(row_of(rik, outcome=y, design='FRD' if fuzzy else 'ITT', bw='ik (b = h)', vce='hc1'),
                             maxdiff_central=rik['maxdiff_central']))
            if y == 'srh' and not fuzzy:
                summary['ik_info_srh'] = {k: (float(v) if np.ndim(v) == 0 else [float(t) for t in v]) for k, v in rik['bw_info'].items()}
            # protocol variants: observed range instead of the public support (rdrobust's default), no
            # mass-point adjustment, and (FRD) rdrobust's fuzzy bandwidth
            r = run(S, support=None)
            rows.append(dict(row_of(r, outcome=y, design='FRD' if fuzzy else 'ITT', bw='mserd, observed min/max', vce='hc1'),
                             maxdiff_central=r['maxdiff_central']))
            r = run(S, mass_grid=None)
            rows.append(dict(row_of(r, outcome=y, design='FRD' if fuzzy else 'ITT', bw='mserd, no mass-point adjustment', vce='hc1'),
                             maxdiff_central=r['maxdiff_central']))
            if fuzzy:
                r = run(S, fuzzy_bw='rdrobust')
                rows.append(dict(row_of(r, outcome=y, design='FRD', bw='mserd, rdrobust fuzzy bandwidth', vce='hc1'),
                                 maxdiff_central=r['maxdiff_central']))
    main_df = pd.DataFrame(rows)
    main_df.to_csv(os.path.join(OUT, 'main_results.csv'), index=False)
    print(main_df[['outcome', 'design', 'bw', 'vce', 'tau_cl', 'tau_bc', 'se_rb', 'ci_rb_lo', 'ci_rb_hi', 'h', 'maxdiff_central']].round(4).to_string())
    r0 = mains[('srh_good', False, False)]
    summary['main_srh_good_itt'] = {k: r0[k] for k in ('tau_cl', 'se_cl', 'tau_bc', 'se_rb', 'ci_rb', 'h', 'b', 'n_h')}
    summary['comm_main'] = r0['comm']
    summary['comm_rounds_main'] = r0['comm_rounds']
    nrec = sum(s.n for s in sites_of(df, 'srh_good'))
    summary['microdata_bytes_total'] = float(nrec * 2 * 8)
    # ------------------------------------------------------------------ bandwidth sensitivity
    rows = []
    for y in ('srh_good', 'srh'):
        for fuzzy in (False, True):
            base = mains[(y, fuzzy, False)]
            rho = base['h'] / base['b']
            S = sites_of(df, y, fuzzy=fuzzy)
            for fct in (0.5, 0.75, 1.0, 1.25, 1.5, 2.0):
                r = run(S, bw=(fct * base['h'], fct * base['h'] / rho))
                rows.append(dict(row_of(r, outcome=y, design='FRD' if fuzzy else 'ITT', factor=fct), maxdiff_central=r['maxdiff_central']))
    pd.DataFrame(rows).to_csv(os.path.join(OUT, 'bandwidth_sensitivity.csv'), index=False)
    # ------------------------------------------------------------------ density test
    S = [Site(g['x'].to_numpy(float), np.zeros(len(g)), name=n) for n, g in df.groupby('region')]
    dens = []
    m0 = mccrary(S, 1 / 12, *SUPPORT)
    for fct in (0.5, 1.0, 2.0):
        m = m0 if fct == 1.0 else mccrary(S, 1 / 12, *SUPPORT, h=fct * m0['h'])
        dens.append(dict(h_factor=fct, h=m['h'], theta=m['theta'], se=m['se'], z=m['z'], p=m['p'],
                         f_minus=m['f_minus'], f_plus=m['f_plus'], N=m['N'], rounds=m['comm']['rounds']))
    pd.DataFrame(dens).to_csv(os.path.join(OUT, 'density_test.csv'), index=False)
    pd.DataFrame(m0['bins']).to_csv(os.path.join(OUT, 'density_bins.csv'), index=False)
    summary['density'] = dens[1]
    # ------------------------------------------------------------------ covariate balance
    rows = []
    for v in BALANCE:
        r = run(sites_of(df, v))
        rows.append(dict(row_of(r, covariate=v), mean=float(df[v].mean()), maxdiff_central=r['maxdiff_central']))
    bal = pd.DataFrame(rows)
    bal.to_csv(os.path.join(OUT, 'covariate_balance.csv'), index=False)
    # ------------------------------------------------------------------ placebo cutoffs and donut
    rows = []
    for y in ('srh_good', 'srh'):
        for c in (55, 57, 63, 65):
            d = df[df['x'] < 0] if c < 60 else df[df['x'] >= 0]
            d = d.assign(x=d['x'] - (c - 60))
            # public support of the one-sided subsample, re-centred at the placebo cutoff (ages 45-60 or 60-120)
            sup = (SUPPORT[0] - (c - 60), -(c - 60)) if c < 60 else (-(c - 60), SUPPORT[1] - (c - 60))
            r = run(sites_of(d, y), support=sup)
            rows.append(dict(row_of(r, outcome=y, cutoff=c), maxdiff_central=r['maxdiff_central']))
    pd.DataFrame(rows).to_csv(os.path.join(OUT, 'placebo_cutoffs.csv'), index=False)
    rows = []
    for y in ('srh_good', 'srh'):
        for fuzzy in (False, True):
            for dm in (1, 2, 3, 6):
                d = df[np.abs(df['age_m'] - 720) >= dm]
                r = run(sites_of(d, y, fuzzy=fuzzy))
                rows.append(dict(row_of(r, outcome=y, design='FRD' if fuzzy else 'ITT', donut_months=dm),
                                 maxdiff_central=r['maxdiff_central']))
    pd.DataFrame(rows).to_csv(os.path.join(OUT, 'donut.csv'), index=False)
    if 'famine_cohort' in df:
        # 1959-1961 birth cohorts (ID-card birth year) removed; bandwidth re-selected on the remaining records
        rows = []
        for y in ('srh_good', 'srh', 'srh_good_all'):
            for fuzzy in (False, True):
                d = df[df['famine_cohort'] != 1]
                r = run(sites_of(d, y, fuzzy=fuzzy))
                rows.append(dict(row_of(r, outcome=y, design='FRD' if fuzzy else 'ITT', sample='without 1959-1961 births'),
                                 maxdiff_central=r['maxdiff_central']))
        pd.DataFrame(rows).to_csv(os.path.join(OUT, 'famine_excluded.csv'), index=False)
    # ------------------------------------------------------------------ region-level estimates and meta-analysis
    rows = []
    for y in ('srh_good', 'srh'):
        for fuzzy in (False, True):
            S = sites_of(df, y, fuzzy=fuzzy)
            se = site_estimates(S, **OPT)
            for s_, e in zip(S, se):
                rows.append(dict({k: v for k, v in e.items() if k != 'site'}, outcome=y, design='FRD' if fuzzy else 'ITT', site=s_.name, kind='site'))
            fed = mains[(y, fuzzy, False)]
            rows.append(dict(outcome=y, design='FRD' if fuzzy else 'ITT', site='federated (= pooled)', kind='federated',
                             tau_cl=fed['tau_cl'], se_cl=fed['se_cl'], tau_bc=fed['tau_bc'], se_rb=fed['se_rb'], h=fed['h'],
                             n=fed['N'], n_h=sum(fed['n_h']), ok=True))
            for nm, fn in (('IVW', meta_ivw), ('simple average', meta_mean)):
                mc, mr = fn(se, 'cl'), fn(se, 'rb')
                rows.append(dict(outcome=y, design='FRD' if fuzzy else 'ITT', site=nm, kind='meta', tau_cl=mc['tau'], se_cl=mc['se'],
                                 tau_bc=mr['tau'], se_rb=mr['se'], ok=True, n=np.nan, n_h=np.nan, h=np.nan))
            a = np.array([e['tau_bc'] for e in se], float)
            v = np.array([e['se_rb'] for e in se], float) ** 2
            ok = np.isfinite(a) & np.isfinite(v)
            w = 1 / v[ok]
            Q = float(np.sum(w * (a[ok] - np.sum(w * a[ok]) / np.sum(w)) ** 2))
            dfq = int(ok.sum() - 1)
            rows.append(dict(outcome=y, design='FRD' if fuzzy else 'ITT', site='Cochran Q', kind='heterogeneity', tau_cl=Q,
                             se_cl=dfq, tau_bc=float(chi2.sf(Q, dfq)), se_rb=max(0.0, (Q - dfq) / Q) if Q > 0 else 0.0, ok=True))
    pd.DataFrame(rows).to_csv(os.path.join(OUT, 'site_estimates_region.csv'), index=False)
    # ------------------------------------------------------------------ random partitions
    rows = []
    for y in ('srh_good', 'srh'):
        d = df.dropna(subset=['x', y]).reset_index(drop=True)
        for K in range(2, 61):
            rng = np.random.default_rng(SEED + K)
            lab = rng.permutation(np.arange(len(d)) % K)
            S = sites_of(d.assign(part=lab), y, by='part')
            r = fsrd(S, **OPT)
            se = site_estimates(S, **OPT)
            iv, mn = meta_ivw(se, 'rb'), meta_mean(se, 'rb')
            ivc, mnc = meta_ivw(se, 'cl'), meta_mean(se, 'cl')
            rows.append(dict(outcome=y, K=K, fed_tau_bc=r['tau_bc'], fed_ci_lo=r['ci_rb'][0], fed_ci_hi=r['ci_rb'][1],
                             fed_tau_cl=r['tau_cl'], fed_h=r['h'], ivw_tau_bc=iv['tau'], ivw_se_rb=iv['se'], mean_tau_bc=mn['tau'],
                             mean_se_rb=mn['se'], ivw_tau_cl=ivc['tau'], mean_tau_cl=mnc['tau'],
                             sites_failed=int(sum(not e['ok'] for e in se)), bytes_up_site=r['comm']['bytes_up_per_site']))
        print('partitions done', y, flush=True)
    pd.DataFrame(rows).to_csv(os.path.join(OUT, 'partition_invariance.csv'), index=False)
    # ------------------------------------------------------------------ provinces and the small-cell rule
    rows = []
    for y in ('srh_good', 'srh'):
        for fuzzy in (False, True):
            S = sites_of(df, y, by='province', fuzzy=fuzzy)
            for mc in (0, 10):
                r = fsrd(S, min_cell=mc, **OPT)
                rows.append(row_of(r, outcome=y, design='FRD' if fuzzy else 'ITT', estimator=f'federated, min_cell={mc}'))
            se = site_estimates(S, **OPT)
            for nm, fn in (('IVW', meta_ivw), ('simple average', meta_mean)):
                m = fn(se, 'rb')
                rows.append(dict(outcome=y, design='FRD' if fuzzy else 'ITT', estimator=nm, tau_bc=m['tau'], se_rb=m['se'],
                                 ci_rb_lo=m['ci'][0], ci_rb_hi=m['ci'][1], K=len(S), sites_used=m['k_used']))
    pd.DataFrame(rows).to_csv(os.path.join(OUT, 'province_sites.csv'), index=False)
    # ------------------------------------------------------------------ RD plot data (federated bin sums)
    edges = np.arange(-15, 20.0001, 0.5)
    rows = []
    for y in ('srh_good', 'srh', 'pension'):
        S = sites_of(df, y)
        agg = Federation(S).round('rdplot:bins', _bin_sums, edges=edges)
        mid = (edges[:-1] + edges[1:]) / 2
        for j in range(len(mid)):
            if agg['cnt'][j] > 0:
                rows.append(dict(variable=y, x_mid=mid[j], n=agg['cnt'][j], mean=agg['sy'][j] / agg['cnt'][j]))
    pd.DataFrame(rows).to_csv(os.path.join(OUT, 'rdplot_bins.csv'), index=False)
    rows = []
    for y in ('srh_good', 'srh'):
        r = mains[(y, False, False)]
        for s, sg in (('-', -1), ('+', 1)):
            c = r['side_coef'][s][:, 0]
            for xv in np.linspace(0, sg * r['h'], 50):
                rows.append(dict(variable=y, side=s, x=xv, fit=c[0] + c[1] * xv, h=r['h']))
    fs = mains[('srh_good', True, False)]
    for s, sg in (('-', -1), ('+', 1)):
        c = fs['side_coef'][s][:, 1]
        for xv in np.linspace(0, sg * fs['h'], 50):
            rows.append(dict(variable='pension', side=s, x=xv, fit=c[0] + c[1] * xv, h=fs['h']))
    pd.DataFrame(rows).to_csv(os.path.join(OUT, 'rdplot_fit.csv'), index=False)
    # ------------------------------------------------------------------ Dong & Lewbel (2015) threshold derivative
    # jump in the slope of E[y | age] at 60 (per year of age): d tau(c) / d c under their local policy invariance;
    # local quadratic fits (p = 2), bias correction with q = 3, own mserd bandwidths for deriv = 1
    rows = []
    for y in [v for v in ('srh_good', 'srh', 'srh_good_all', 'cesd10') if v in OUTCOMES] + ['pension']:
        r = run(sites_of(df, y), deriv=1, p=2, q=3)
        rows.append(dict(row_of(r, outcome=y, design='slope change at 60 (ITT)' if y != 'pension' else 'slope change at 60 (take-up)'),
                         level_tau_bc=mains[(y, False, False)]['tau_bc'] if (y, False, False) in mains else np.nan,
                         maxdiff_central=r['maxdiff_central']))
    pd.DataFrame(rows).to_csv(os.path.join(OUT, 'threshold_derivative.csv'), index=False)
    with open(os.path.join(OUT, 'summary.json'), 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=1, default=_json)
    print('done')


def _bin_sums(site, edges, min_cell=0):
    """Per-bin counts and outcome sums (double-double, like every other federated sum)."""
    idx = np.searchsorted(edges, site.x, side='right') - 1
    ok = (idx >= 0) & (idx < len(edges) - 1)
    T = np.zeros((ok.sum(), len(edges) - 1))
    T[np.arange(ok.sum()), idx[ok]] = site.y[ok]
    return dict(cnt=dd_exact(np.bincount(idx[ok], minlength=len(edges) - 1)), sy=dd(T))


def _json(o):
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    return str(o)


if __name__ == '__main__':
    main()
