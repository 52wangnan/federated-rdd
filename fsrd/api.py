# -*- coding: utf-8 -*-
"""High-level entry points.

fsrd(sites, ...)        federated RD (sharp, or fuzzy if the sites carry a take-up variable d)
centralized(sites, ...) the same computation on the pooled records (one site): the benchmark
site_estimates(...)     every site analysed alone (input to the meta-analytic benchmarks)
meta_ivw(...), meta_mean(...)
"""
import time
import numpy as np
from scipy.stats import norm
from .federation import Federation, Site, pool_sites
from .bandwidth import bw_cct, bw_ik
from .lprd import estimate, GramCache


def strip(site, covs=True, fuzzy=True):
    return Site(site.x, site.y, d=site.d if fuzzy else None, w=site.w if covs else None, cl=site.cl, name=site.name)


def add_site_indicators(sites):
    """Site fixed effects: append K - 1 site indicators (site 1 = reference) to every site's covariates.
    Each site knows its own index, so the indicators are created locally; they enter the common-slope
    covariate block (CCFT 2019), and the estimator stays an exact function of summed statistics.
    Note: with indicators, the aggregated Gram blocks contain site-specific sums (each indicator singles
    out one site), so secure aggregation no longer hides site-level window totals from the sponsor."""
    K = len(sites)
    out = []
    for k, s in enumerate(sites):
        D = np.zeros((s.n, K - 1))
        if k > 0:
            D[:, k - 1] = 1.0
        w = D if s.w is None else np.column_stack([s.w, D])
        out.append(Site(s.x, s.y, d=s.d, w=w, cl=s.cl, name=s.name))
    return out


def prepare(sites, covs=True, fuzzy=True, site_fe=False):
    S = [strip(s, covs, fuzzy) for s in sites]
    if site_fe and len(S) > 1:
        S = add_site_indicators(S)
    return S


def fsrd(sites, bw='mserd', p=1, q=2, deriv=0, kernel='triangular', vce='hc1', covs=True, fuzzy=True, level=0.95,
         rho=None, min_cell=0, scaleregul=1.0, bwrestrict=True, grid=None, nbins=128, bw_covs=None,
         support=None, mass_grid=None, masspoints='adjust', bwcheck=None, fuzzy_bw='reduced_form', site_fe=False):
    """Federated local polynomial RD.

    bw : 'mserd' (CCT MSE-optimal h and b), 'ik' (IK 2012 h; b = h, i.e. rho = 1, as rdrobust does when
         only h is given), a number h (b = h / rho, rho = 1 by default) or a pair (h, b).
    deriv : 0 = jump in the level; 1 = jump in the slope (Dong & Lewbel 2015 threshold derivative; use p = 2, q = 3).
    covs : use the sites' covariates (common slope on both sides, CCFT 2019).
    bw_covs : covariates in the bandwidth selector (default: same as covs, as in rdrobust).
    fuzzy : if False, a take-up variable is ignored (sharp / intention-to-treat analysis).
    min_cell : small-cell rule (see Federation); 0 = off, then the result equals the pooled one.
    support : public range (lo, hi) of the centred running variable; None = sites report min / max.
    mass_grid, masspoints, bwcheck : discrete running variable on the public grid {k * mass_grid}
         (exact quantiles, rdrobust's mass-point adjustment); see bandwidth.bw_cct.
    fuzzy_bw : 'reduced_form' or 'rdrobust' (bandwidth for the fuzzy ratio as rdrobust_bw computes it).
    site_fe : add site indicators to the common-slope covariates (needed when site composition jumps at c).
    Returns the estimate dict plus 'bw_info', the communication summary 'comm' and, with min_cell > 0,
    the number of windows the small-cell rule suppressed ('suppressed', 'suppressed_final')."""
    t0 = time.perf_counter()
    if vce == 'cluster' and len(sites) > 1:
        # vce='cluster' needs every cluster inside one site (e.g. communities within provinces); in a real
        # deployment this is part of the analysis plan; here the harness verifies it on the labels
        seen = set()
        for s in sites:
            labs = set(np.unique(s.cl).tolist())
            if seen & labs:
                raise ValueError("vce='cluster' requires clusters nested within sites; use 'cluster_grid'")
            seen |= labs
    covs_eff = covs or site_fe
    S = prepare(sites, covs, fuzzy, site_fe)
    fed = Federation(S, min_cell=min_cell)
    cache = GramCache()
    info = {}
    bwopt = dict(support=support, mass_grid=mass_grid, masspoints=masspoints, bwcheck=bwcheck, fuzzy_bw=fuzzy_bw)
    bvce = vce if vce in ('hc0', 'hc1', 'hc2', 'hc3') else 'hc1'
    if isinstance(bw, str) and bw == 'mserd':
        bw_c = covs if bw_covs is None else bw_covs
        if bw_c == covs:
            h, b, info = bw_cct(fed, p, q, deriv, kernel, bvce, scaleregul, bwrestrict, nbins, cache, **bwopt)
        else:
            fb = Federation(prepare(sites, bw_c, fuzzy, site_fe), min_cell=min_cell, ledger=fed.ledger)
            h, b, info = bw_cct(fb, p, q, deriv, kernel, bvce, scaleregul, bwrestrict, nbins, **bwopt)
        if rho is not None:
            b = h / rho
    elif isinstance(bw, str) and bw == 'ik':
        h, info = bw_ik(Federation([strip(s, False, False) for s in sites], min_cell=min_cell, ledger=fed.ledger),
                        kernel, nbins, support=support)
        b = h if rho is None else h / rho
    elif np.ndim(bw) == 0:
        h = float(bw)
        b = h if rho is None else h / rho
    else:
        h, b = map(float, bw)
    est = estimate(fed, h, b, p, q, kernel, vce, level, cache=cache, grid=grid, deriv=deriv)
    est['bw'] = bw if isinstance(bw, str) else 'fixed'
    est['bw_info'] = info
    est['K'] = len(sites)
    est['N'] = int(sum(s.n for s in sites))
    est['covs_used'] = bool(covs_eff)
    est['comm'] = fed.ledger.summary()
    est['comm_rounds'] = fed.ledger.table()
    if min_cell > 0:
        est['suppressed'] = int(sum(len(s.suppressed) for s in S))
        est['suppressed_final'] = int(sum(1 for s in S for (sd, hh) in s.suppressed if hh in (float(h), float(b))))
    est['time'] = time.perf_counter() - t0
    return est


def centralized(sites, **kw):
    """The same computation on the pooled records.  Site indicators (site_fe) are attached before
    pooling, so the pooled fit uses exactly the same regressors."""
    if kw.pop('site_fe', False) and len(sites) > 1:
        sites = add_site_indicators([strip(s, kw.get('covs', True), kw.get('fuzzy', True)) for s in sites])
        kw['covs'] = True
    return fsrd([pool_sites(sites)], **kw)


def site_estimates(sites, **kw):
    """Each site on its own (own bandwidth); failures (too few records near the cutoff, singular
    designs) are recorded as NaN."""
    rows = []
    for k, s in enumerate(sites):
        try:
            r = fsrd([s], **kw)
            rows.append(dict(site=k, ok=True, n=s.n, n_h=sum(r['n_h']), h=r['h'], tau_cl=r['tau_cl'], se_cl=r['se_cl'],
                             tau_bc=r['tau_bc'], se_rb=r['se_rb']))
        except (np.linalg.LinAlgError, ValueError, ZeroDivisionError, FloatingPointError):
            rows.append(dict(site=k, ok=False, n=s.n, n_h=np.nan, h=np.nan, tau_cl=np.nan, se_cl=np.nan,
                             tau_bc=np.nan, se_rb=np.nan))
    return rows


def _valid(rows, t, s):
    a = np.array([r[t] for r in rows], float)
    e = np.array([r[s] for r in rows], float)
    ok = np.isfinite(a) & np.isfinite(e) & (e > 0)
    return a[ok], e[ok]


def meta_ivw(rows, kind='cl', level=0.95):
    """Fixed-effect inverse-variance-weighted combination of the site estimates."""
    t, s = ('tau_cl', 'se_cl') if kind == 'cl' else ('tau_bc', 'se_rb')
    a, e = _valid(rows, t, s)
    if len(a) == 0:
        return dict(tau=np.nan, se=np.nan, ci=[np.nan, np.nan], k_used=0)
    w = 1.0 / e ** 2
    tau = float(np.sum(w * a) / np.sum(w))
    se = float(np.sqrt(1.0 / np.sum(w)))
    z = norm.ppf(0.5 + level / 2)
    return dict(tau=tau, se=se, ci=[tau - z * se, tau + z * se], k_used=int(len(a)))


def meta_mean(rows, kind='cl', level=0.95):
    """Unweighted average of the site estimates, SE = sqrt(sum se_k^2) / K_used."""
    t, s = ('tau_cl', 'se_cl') if kind == 'cl' else ('tau_bc', 'se_rb')
    a, e = _valid(rows, t, s)
    if len(a) == 0:
        return dict(tau=np.nan, se=np.nan, ci=[np.nan, np.nan], k_used=0)
    tau = float(np.mean(a))
    se = float(np.sqrt(np.sum(e ** 2)) / len(a))
    z = norm.ppf(0.5 + level / 2)
    return dict(tau=tau, se=se, ci=[tau - z * se, tau + z * se], k_used=int(len(a)))
