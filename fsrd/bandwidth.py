# -*- coding: utf-8 -*-
"""Federated bandwidth selectors, computed by the sponsor from summed site statistics.

bw_cct : MSE-optimal common bandwidth 'mserd' of Calonico, Cattaneo & Titiunik (2014) /
         Calonico, Cattaneo & Farrell (2020), transcribed from rdbwselect() of rdrobust 2.1.1
         (pilot c_bw -> d -> b -> h; d and b capped at the support range before they are used; optional
         mass-point adjustment), with heteroskedasticity-robust (HC) instead of nearest-neighbour variance
         estimates: NN residuals need each record's nearest neighbours, which may sit at another site, while
         HC meats are plain sums and can be federated exactly.
bw_ik  : Imbens & Kalyanaraman (2012) plug-in bandwidth for the local linear estimator.

Support of the running variable.  `support=(lo, hi)` (centred units) gives a public range, e.g. the age
range covered by the survey; then no site reveals its smallest or largest value.  Without it the
sites report their own min / max (combined by min / max), which matches rdrobust but discloses the
extreme values.

Quantiles.  With `mass_grid=step` (a discrete running variable on the public grid {k * step}) the sites
send their counts on the grid points and the sponsor gets exact type-2 quantiles (as rdrobust), the
number of distinct support points (mass-point adjustment) and their positions (bwcheck).  Otherwise the
IQR (CCT pilot) and the side medians (IK) come from a federated histogram with `nbins` equal-width bins
per side and linear interpolation inside a bin.  The same routines are used for the centralised (K = 1)
computation, so federated and centralised bandwidths coincide.
"""
from math import factorial

import numpy as np
from . import local as L
from .kernels import C_PILOT, boundary_constants
from .lprd import side_fit, gamma_fwl, collect_meat, fetch_grams, GramCache

EPS_RANGE = 1 + np.sqrt(np.finfo(float).eps)          # rdbwselect: range * (1 + sqrt(eps))


# ----------------------------------------------------------------------------- shared rounds
def describe(fed, tag, support=None):
    d = fed.round(f'{tag}:describe', L.describe, minmax=support is None)
    N = float(d['n'])
    mean = d['sx'] / N
    var = (d['sxx'] - N * mean ** 2) / (N - 1)
    xmin, xmax = (float(d['min_x']), float(d['max_x'])) if support is None else map(float, support)
    return dict(N=N, sd=float(np.sqrt(var)), xmin=xmin, xmax=xmax,
                N_l=float(d['n_side'][0]), N_r=float(d['n_side'][1]))


def hist_edges(ds, nbins):
    return np.linspace(ds['xmin'], 0.0, nbins + 1), np.linspace(0.0, ds['xmax'], nbins + 1)


def grid_points(lo, hi, step):
    k = np.arange(int(np.floor(lo / step + 1e-9)), int(np.ceil(hi / step - 1e-9)) + 1)
    return k * step, (np.append(k, k[-1] + 1) - 0.5) * step


def quantile_from_hist(edges, counts, prob):
    """Quantile of binned data, linear interpolation within the bin containing the target rank."""
    cum = np.cumsum(counts)
    t = prob * cum[-1]
    i = int(np.searchsorted(cum, t))
    i = min(i, len(counts) - 1)
    prev = cum[i - 1] if i > 0 else 0.0
    frac = (t - prev) / counts[i] if counts[i] > 0 else 0.5
    return float(edges[i] + frac * (edges[i + 1] - edges[i]))


def quantile_type2_grid(points, counts, prob):
    """Hyndman-Fan type-2 quantile (R quantile(type = 2), rdrobust's quantile_type2) of data that sit
    on the grid points with the given counts: exact, from the counts alone."""
    cum = np.cumsum(counts)
    n = int(cum[-1])
    stat = lambda j: float(points[int(np.searchsorted(cum, j))])      # j-th order statistic (1-based)
    hh = n * prob
    j = int(np.floor(hh))
    return 0.5 * (stat(j) + stat(j + 1)) if hh == j else stat(j + 1)


def _check(bw, name):
    if not np.isfinite(bw) or bw <= 0:
        raise ValueError(f'bandwidth step {name} failed ({bw})')
    return bw


# ----------------------------------------------------------------------------- CCT (mserd)
def bw_cct(fed, p=1, q=2, deriv=0, kernel='triangular', vce='hc1', scaleregul=1.0, bwrestrict=True, nbins=128,
           cache=None, tag='cct', support=None, mass_grid=None, masspoints='adjust', bwcheck=None,
           fuzzy_bw='reduced_form'):
    """Federated MSE-optimal bandwidths (h, b) following rdbwselect(bwselect = 'mserd') of rdrobust 2.1.1.

    mass_grid  : step of the public grid of a discrete running variable (enables exact quantiles and,
                 with masspoints = 'adjust', rdrobust's mass-point adjustment); None = continuous.
    fuzzy_bw   : 'reduced_form' chooses the bandwidths for the outcome alone; 'rdrobust' combines the
                 outcome and take-up columns as rdrobust_bw() does for a fuzzy design."""
    cache = GramCache() if cache is None else cache
    my = 2 if fed.fuzzy else 1
    covs = fed.p > 0
    sides = ('-', '+')
    ds = describe(fed, tag, support)
    N = ds['N']
    M = N
    bw_min = 0.0
    if mass_grid is not None:
        pts, edges = grid_points(ds['xmin'], ds['xmax'], mass_grid)
        cnt = fed.round(f'{tag}:grid', L.grid_counts, edges=edges)['cnt']
        iqr = quantile_type2_grid(pts, cnt, 0.75) - quantile_type2_grid(pts, cnt, 0.25)
        left, right = (pts < 0) & (cnt > 0), (pts >= 0) & (cnt > 0)
        M_l, M_r = int(left.sum()), int(right.sum())
        if masspoints == 'adjust':
            M = M_l + M_r
            if (1 - M_l / ds['N_l'] >= 0.2 or 1 - M_r / ds['N_r'] >= 0.2) and bwcheck is None:
                bwcheck = 10
        if bwcheck is not None:
            ul = np.sort(-pts[left])                                    # distances, closest first
            ur = np.sort(pts[right])
            bw_min = max(ul[min(bwcheck, M_l) - 1], ur[min(bwcheck, M_r) - 1]) * EPS_RANGE
    else:
        el, er = hist_edges(ds, nbins)
        hst = fed.round(f'{tag}:hist', L.histogram, edges_l=el, edges_r=er)
        iqr = quantile_from_hist(np.concatenate([el, er[1:]]), np.concatenate([hst['cnt_l'], hst['cnt_r']]), 0.75) - \
            quantile_from_hist(np.concatenate([el, er[1:]]), np.concatenate([hst['cnt_l'], hst['cnt_r']]), 0.25)
    c_bw = C_PILOT[kernel] * min(ds['sd'], iqr / 1.349) * M ** (-0.2)
    rng = {'-': -ds['xmin'], '+': ds['xmax']}
    bw_max = max(rng.values())
    if bwrestrict:
        c_bw = min(c_bw, bw_max)
    c_bw = max(c_bw, bw_min)
    hB_d = {s: rng[s] * EPS_RANGE for s in sides}
    g = fetch_grams(fed, [(s, c_bw, q + 2, True) for s in sides] + [(s, hB_d[s], q + 2, False) for s in sides],
                    kernel, cache, f'{tag}:gram-pilot')
    # V parts (all at the pilot bandwidth c_bw): d-step (o=q+1, nu=q+1), b-step (q, p+1), h-step (p, deriv)
    steps = dict(d=(q + 1, q + 1), b=(q, p + 1), h=(p, deriv))
    Vf = {}
    for st, (o, nu) in steps.items():
        for s in sides:
            gram = g[(s, c_bw, q + 2)]
            invG, beta = side_fit(gram, o)
            gamma = gamma_fwl([(gram, o)], my) if covs else np.zeros((0, my))
            sv = np.zeros(my + fed.p)
            if my == 2 and fuzzy_bw == 'rdrobust':
                # rdrobust_bw(): delta-method weights built from this side's own nu-th coefficients
                bz = beta[nu] * factorial(nu) / c_bw ** nu
                tY = bz[0] - gamma[:, 0] @ bz[my:]
                tT = bz[1] - gamma[:, 1] @ bz[my:]
                sv[0], sv[1] = 1 / tT, -tY / tT ** 2
                sv[my:] = -(1 / tT) * gamma[:, 0] + (tY / tT ** 2) * gamma[:, 1]
            else:
                sv[0] = 1.0
                sv[my:] = -gamma[:, 0]
            Vf[st, s] = dict(o=o, nu=nu, invG=invG, beta=beta, s=sv,
                             BConst=float((invG @ gram['G'][:o + 1, o + 1])[nu]))
    reqs = [dict(side=s, type='R', h=c_bw, p=Vf[st, s]['o'], res_h=c_bw, res_o=Vf[st, s]['o'],
                 beta_s=Vf[st, s]['beta'] @ Vf[st, s]['s'], s=Vf[st, s]['s'], win=[c_bw], vce=vce,
                 **({'lev_invG': Vf[st, s]['invG']} if vce in ('hc2', 'hc3') else {}))
            for st in steps for s in sides]
    mts = fed.round(f'{tag}:meat-pilot', L.meats, reqs=reqs, kernel=kernel)
    V = {}
    for (st, s), o in zip([(st, s) for st in steps for s in sides], mts):
        f = Vf[st, s]
        Mt = collect_meat(o, vce, f['o'] + 1)
        VV = (f['invG'] @ Mt @ f['invG'])[f['nu'], f['nu']] / c_bw ** (2 * f['nu'])
        V[st, s] = (2 * f['nu'] + 1) * c_bw ** (2 * f['nu'] + 1) * VV
    # d-step bias (order q+2 fit over the whole side, no regularisation)
    B = {}
    o, nu = steps['d']
    for s in sides:
        _, betaB = side_fit(g[(s, hB_d[s], q + 2)], o + 1)
        B['d', s] = np.sqrt(2 * (o + 1 - nu)) * Vf['d', s]['BConst'] * (betaB[o + 1] @ Vf['d', s]['s']) / hB_d[s] ** (o + 1)
    d_bw = _check(float(((V['d', '-'] + V['d', '+']) / (B['d', '+'] - B['d', '-']) ** 2) ** (1.0 / (2 * o + 3))), 'd')
    if bwrestrict:
        d_bw = min(d_bw, bw_max)
    d_bw = max(d_bw, bw_min)

    def regularised_step(st, hB, oB, tag2):
        o, nu = steps[st]
        gg = fetch_grams(fed, [(s, hB, oB, False) for s in sides], kernel, cache, f'{tag}:gram-{tag2}')
        fits = {s: side_fit(gg[(s, hB, oB)], oB) for s in sides}
        rq = [dict(side=s, type='R', h=hB, p=oB, res_h=hB, res_o=oB, beta_s=fits[s][1] @ Vf[st, s]['s'],
                   s=Vf[st, s]['s'], win=[hB], vce=vce, **({'lev_invG': fits[s][0]} if vce in ('hc2', 'hc3') else {}))
              for s in sides]
        mm = fed.round(f'{tag}:meat-{tag2}', L.meats, reqs=rq, kernel=kernel)
        Bs, Rs = {}, {}
        for s, out in zip(sides, mm):
            f = Vf[st, s]
            invGB, betaB = fits[s]
            Mt = collect_meat(out, vce, oB + 1)
            VB = (invGB @ Mt @ invGB)[o + 1, o + 1] / hB ** (2 * (o + 1))
            Bs[s] = np.sqrt(2 * (o + 1 - nu)) * f['BConst'] * (betaB[o + 1] @ f['s']) / hB ** (o + 1)
            Rs[s] = scaleregul * 2 * (o + 1 - nu) * 3 * f['BConst'] ** 2 * VB
        num = V[st, '-'] + V[st, '+']
        den = (Bs['+'] - Bs['-']) ** 2 + scaleregul * (Rs['-'] + Rs['+'])
        return _check(float((num / den) ** (1.0 / (2 * o + 3))), st), dict(B=Bs, R=Rs, V=num)

    b_bw, info_b = regularised_step('b', d_bw, q + 1, 'b')
    if bwrestrict:
        b_bw = min(b_bw, bw_max)                     # rdbwselect caps b before the h-step uses it
    h_bw, info_h = regularised_step('h', b_bw, q, 'h')
    if bwrestrict:
        h_bw = min(h_bw, bw_max)
    info = dict(N=N, M=M, sd=ds['sd'], iqr=iqr, c_bw=c_bw, d_bw=d_bw, b=b_bw, h=h_bw, bw_max=bw_max, bw_min=bw_min,
                bwcheck=bwcheck, support=[ds['xmin'], ds['xmax']], B_h=info_h['B'], R_h=info_h['R'], V_h=info_h['V'])
    return h_bw, b_bw, info


# ----------------------------------------------------------------------------- IK (2012)
IK_REG = 2160.0      # regularisation r = 3 x var(m''), var(m'') = 720 sigma^2 / (N_2 h_2^4) (uniform kernel)


def bw_ik(fed, kernel='triangular', nbins=128, reg_const=IK_REG, tag='ik', support=None):
    """Imbens & Kalyanaraman (2012) bandwidth, steps 1-3 of their Section 4.2.
      1. h1 = 1.84 S_X N^(-1/5); f(c) and sigma^2_-(c), sigma^2_+(c) from [c-h1, c) and [c, c+h1].
      2. third derivative from a global cubic with a jump, fitted on the records between the
         medians of X below and above the cutoff;  h2,+- = 3.56 (sigma^2_+- / (f m3^2))^(1/7) N_+-^(-1/7).
      3. second derivatives from local quadratics on [c-h2,-, c) and [c, c+h2,+];
         r_+- = 2160 sigma^2_+- / (N_2,+- h2,+-^4);
         h = C_K ((sigma^2_- + sigma^2_+) / (f ((m2_+ - m2_-)^2 + r_+ + r_-)))^(1/5) N^(-1/5).
    Only the outcome is used (no covariates)."""
    ds = describe(fed, tag, support)
    N = ds['N']
    h1 = 1.84 * ds['sd'] * N ** (-0.2)
    el, er = hist_edges(ds, nbins)
    r2 = fed.round(f'{tag}:window+hist', L.ik_step1, h1=h1, edges_l=el, edges_r=er)
    n, sy, syy = r2['mom'][:, 0], r2['mom'][:, 1], r2['mom'][:, 2]
    ybar = sy / n
    s2 = (syy - n * ybar ** 2) / (n - 1)
    f = (n[0] + n[1]) / (2 * N * h1)
    med_l = quantile_from_hist(el, r2['cnt_l'], 0.5)
    med_r = quantile_from_hist(er, r2['cnt_r'], 0.5)
    scale = max(-med_l, med_r)
    r3 = fed.round(f'{tag}:cubic', L.global_poly, lo=med_l, hi=med_r, scale=scale)
    gam = np.linalg.solve(r3['G'], r3['Ry'])
    m3 = 6.0 * gam[4] / scale ** 3
    Nside = np.array([ds['N_l'], ds['N_r']])
    m3sq = max(m3 ** 2, 1e-300)
    h2 = 3.56 * (s2 / (f * m3sq)) ** (1.0 / 7.0) * Nside ** (-1.0 / 7.0)
    rng = np.array([-ds['xmin'], ds['xmax']])
    win = np.minimum(h2, rng * (1 + 1e-12))             # a window wider than the support = whole side
    r4 = fed.round(f'{tag}:quad', L.grams, reqs=[('-', win[0], 2, False), ('+', win[1], 2, False)], kernel='uniform', y_only=True)
    m2, n2 = np.zeros(2), np.zeros(2)
    for j, g in enumerate(r4):
        _, beta = side_fit(g, 2)
        m2[j] = 2.0 * beta[2, 0] / win[j] ** 2
        n2[j] = g['n']
    r = reg_const * s2 / (n2 * h2 ** 4)
    # IK publish C_K = 3.4375 for the triangular kernel (the exact value is 480^(1/5) = 3.43754)
    CK = 3.4375 if kernel == 'triangular' else boundary_constants(kernel)['C_IK']
    h = CK * ((s2[0] + s2[1]) / (f * ((m2[1] - m2[0]) ** 2 + r[0] + r[1]))) ** 0.2 * N ** (-0.2)
    _check(h, 'ik')
    info = dict(N=N, h1=h1, f=f, s2_minus=s2[0], s2_plus=s2[1], n_h1=n.tolist(), med=[med_l, med_r], m3=m3,
                h2_minus=h2[0], h2_plus=h2[1], m2_minus=m2[0], m2_plus=m2[1], n_h2=n2.tolist(),
                r_minus=r[0], r_plus=r[1], C_K=CK, h=h)
    return float(h), info
