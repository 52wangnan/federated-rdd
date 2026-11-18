# -*- coding: utf-8 -*-
"""Sponsor-side local polynomial RD estimation from aggregated sufficient statistics.

For bandwidths (h, b) and orders (p, q) the sponsor reproduces rdrobust's estimation step
(Calonico, Cattaneo & Titiunik, 2014; Calonico, Cattaneo, Farrell & Titiunik, 2019):

  round 1  sites -> sum kw_h R R', sum kw_h R D', (covariates) sum kw_h w w', sum kw_h w [y,d]'
           at (h, p+1) and (b, q) on each side of the cutoff;
  sponsor  solves the normal equations: side intercepts, common covariate slope gamma (CCFT 2019),
           tau_conventional, the bias-corrected tau_bc;
  round 2  sponsor broadcasts the global coefficients; sites return the sandwich 'meats'
           sum (X_i r_i)(X_i r_i)' (HC) or cluster score sums;
  sponsor  conventional SE and robust bias-corrected SE (CCT 2014), CIs.

All quantities are exact functions of the summed statistics, and the sums are carried in double-double
precision (federation.acc_sum), so the federated result is bitwise identical to the estimate computed
on the pooled records.
"""
from math import factorial

import numpy as np
from scipy.stats import norm
from . import local as L


def inv(G):
    return np.linalg.inv(G)


def side_fit(gram, order):
    """Scaled-unit WLS of every data column on R(u), u = x/h, from one side's Gram statistics."""
    if gram is None or gram['n'] < order + 2:
        raise ValueError('too few observations in the kernel window for a polynomial of order %d' % order)
    invG = inv(gram['G'][:order + 1, :order + 1])
    return invG, invG @ gram['RD'][:order + 1]


def gamma_fwl(fits, my, return_info=False):
    """Covariate slopes common to the listed (gram, order) polynomial fits (Frisch-Waugh-Lovell:
    the side polynomials are partialled out).  Returns p x my."""
    ZWZ = ZWY = 0.0
    for gram, order in fits:
        invG = inv(gram['G'][:order + 1, :order + 1])
        U = gram['RD'][:order + 1, my:]
        Ry = gram['RD'][:order + 1, :my]
        ZWZ = ZWZ + gram['ZZ'] - U.T @ invG @ U
        ZWY = ZWY + gram['ZD'] - U.T @ invG @ Ry
    keep = independent_columns(ZWZ)
    gamma = np.zeros((ZWZ.shape[0], my))
    if keep.any():
        gamma[keep] = np.linalg.solve(ZWZ[np.ix_(keep, keep)], ZWY[keep])
    info = dict(keep=keep, residual_gram=ZWZ)
    return (gamma, info) if return_info else gamma


class InvalidFit(ValueError):
    """A prespecified design/variance feasibility failure, not an outcome-size filter."""


def check_final_design(gh, fp, fq, info, my, p, q, nu, h, b, corr):
    """Check identifiability and residual degrees of freedom using existing Gram blocks.

    A redundant covariate may be dropped only when both reported RD contrasts are
    invariant to the unidentified coefficient direction. The effective joint design
    must also leave positive residual degrees of freedom in the h window. These
    checks apply equally to pooled, federated and single-site fits, without an SE floor.
    """
    n = sum(int(gh[s]['n']) for s in ('-', '+'))
    rank_w = 0 if info is None else int(info['keep'].sum())
    rank = 2 * (p + 1) + rank_w
    if n <= rank:
        raise InvalidFit('saturated_joint_design: n_h=%d, effective_rank=%d, residual_df=%d' % (n, rank, n-rank))
    if info is not None and not info['keep'].all():
        keep = info['keep']; G = info['residual_gram']
        omitted = np.flatnonzero(~keep)
        A = np.linalg.solve(G[np.ix_(keep, keep)], G[np.ix_(keep, ~keep)]) if keep.any() else np.zeros((0, len(omitted)))
        null = np.zeros((len(keep), len(omitted)))
        null[keep] = -A
        null[omitted, np.arange(len(omitted))] = 1
        cw = fp['+'][1][nu, my:] - fp['-'][1][nu, my:]
        qw = cw - (h / b)**(p+1) * (corr['+'] * fq['+'][1][p+1, my:] - corr['-'] * fq['-'][1][p+1, my:])
        for label, weights in [('conventional', cw), ('bias_corrected', qw)]:
            sensitivity = weights @ null
            tolerance = 1e-8 * np.maximum(1., np.sum(np.abs(weights[:, None] * null), axis=0))
            if np.any(np.abs(sensitivity) > tolerance):
                raise InvalidFit('unidentified_rd_contrast: %s contrast depends on dropped covariate coefficients' % label)
    return dict(joint_n_h=n, effective_rank=rank, residual_df=n-rank,
                covariates_kept=rank_w, covariates_dropped=0 if info is None else int((~info['keep']).sum()))


def independent_columns(ZWZ, tol=1e-10):
    """Covariates kept in the window: a column with no variation after partialling out the side
    polynomials (e.g. the indicator of a site without records there) is dropped, and so is a column that
    is a linear combination of the columns kept before it (e.g. the last site indicator when the reference
    site has no records in the window, so that the other indicators sum to the intercept).  Dropped
    slopes are set to 0.  Columns are screened in order by their residual variance given the kept ones
    (Cholesky), a deterministic function of the summed statistics, so federated = pooled is preserved."""
    d = np.diag(ZWZ)
    keep = d > 1e-12 * max(1.0, float(np.max(np.abs(d))))
    sel = []
    Lc = np.zeros((0, 0))
    for j in np.flatnonzero(keep):
        l = np.linalg.solve(Lc, ZWZ[sel, j]) if sel else np.zeros(0)
        r = d[j] - l @ l
        if r > tol * d[j]:
            Lc = np.block([[Lc, np.zeros((len(sel), 1))], [l[None, :], np.sqrt(r)[None, None]]])
            sel.append(j)
        else:
            keep[j] = False
    return keep


def meat_factor(vce, n, k, G=None):
    if n - k <= 0 or (G is not None and G < 2):
        raise ValueError('too few observations (or clusters) for the variance estimator')
    if vce == 'hc1':
        return n / (n - k)
    if vce in ('cluster', 'cluster_grid'):
        return ((n - 1) / (n - k)) * (G / (G - 1))
    return 1.0


def collect_meat(out, vce, k):
    """Turn aggregated meat output into a (factor-corrected) meat matrix."""
    if vce == 'cluster_grid':
        Sg = out['Sg'][out['cnt'] > 0]
        return (Sg.T @ Sg) * meat_factor(vce, out['n'], k, len(Sg))
    return out['M'] * meat_factor(vce, out['n'], k, out.get('G'))


class GramCache(dict):
    """(side, bandwidth, order, zz) -> aggregated Gram statistics; reused when a later step needs the
    same window (a Gram of order o also contains every lower order; zz = covariate blocks included)."""

    def find(self, side, h, order, zz=False):
        for (s, hh, o, z), g in self.items():
            if s == side and hh == h and o >= order and (z or not zz):
                return g
        return None


def fetch_grams(fed, wanted, kernel, cache, tag):
    """One communication round for every (side, h, order, zz) not already in the cache.
    zz: the covariate cross-products are needed (only for the fits that estimate gamma)."""
    need = []
    for side, h, o, z in wanted:
        z = bool(z and fed.p > 0)
        if cache.find(side, h, o, z) is None and (side, h, o, z) not in need:
            need.append((side, h, o, z))
    if need:
        got = fed.round(tag, L.grams, reqs=need, kernel=kernel)
        for key, g in zip(need, got):
            cache[key] = g
    return {(s, h, o): cache.find(s, h, o, bool(z and fed.p > 0)) for s, h, o, z in wanted}


def estimate(fed, h, b, p=1, q=2, kernel='triangular', vce='hc1', level=0.95, cache=None,
             grid=None, tag='est', deriv=0):
    """Local polynomial RD at bandwidths (h, b).  Sharp RD if the sites carry no take-up variable,
    fuzzy RD (ratio of the outcome and take-up jumps, delta method) otherwise.
    deriv = nu > 0 estimates the jump in the nu-th derivative (nu = 1: the change in slope at the cutoff,
    i.e. the derivative of the cutoff effect with respect to the threshold in Dong & Lewbel, 2015);
    use p >= nu + 1.  Bias correction and robust variance follow rdrobust(deriv = nu)."""
    cache = GramCache() if cache is None else cache
    my = 2 if fed.fuzzy else 1
    covs = fed.p > 0
    sides = ('-', '+')
    grams = fetch_grams(fed, [(s, h, p + 1, True) for s in sides] + [(s, b, q, False) for s in sides], kernel, cache,
                        f'{tag}:gram')
    gh = {s: grams[(s, h, p + 1)] for s in sides}
    gb = {s: grams[(s, b, q)] for s in sides}
    gamma, gamma_info = gamma_fwl([(gh[s], p) for s in sides], my, return_info=True) if covs else (np.zeros((0, my)), None)
    S = np.vstack([np.eye(my), -gamma])                       # data columns -> adjusted outcomes
    nu = deriv
    scale = factorial(nu) / h ** nu                           # scaled coefficient -> nu! x d^nu m / dx^nu
    fp, fq, Lt, corr = {}, {}, {}, {}
    tcl, tbc = 0.0, 0.0
    for s in sides:
        sg = 1.0 if s == '+' else -1.0
        fp[s] = side_fit(gh[s], p)
        fq[s] = side_fit(gb[s], q)
        Lt[s] = gh[s]['G'][:p + 1, p + 1]                      # sum kw R_p u^(p+1)
        bp = fp[s][1] @ S
        bq = fq[s][1] @ S
        corr[s] = (fp[s][0] @ Lt[s])[nu]
        tcl = tcl + sg * scale * bp[nu]
        tbc = tbc + sg * scale * (bp[nu] - (h / b) ** (p + 1) * corr[s] * bq[p + 1])
    fit_diagnostics = check_final_design(gh, fp, fq, gamma_info, my, p, q, nu, h, b, corr)
    if my == 1:
        combos = {'tau': np.array([1.0])}
        tau_cl, tau_bc = float(tcl[0]), float(tbc[0])
    else:
        tY, tT = tcl
        slin = np.array([1.0 / tT, -tY / tT ** 2])
        combos = {'tau': slin, 'itt': np.array([1.0, 0.0]), 'fs': np.array([0.0, 1.0])}
        tau_cl = float(tY / tT)
        tau_bc = float(tau_cl - slin @ (tcl - tbc))
    # ---------------------------------------------------------------- variance round
    reqs, keys = [], []
    for name, slin in combos.items():
        sfull = S @ slin
        for s in sides:
            common = dict(side=s, h=h, p=p, s=sfull, win=[h, b], vce=vce)
            if vce == 'cluster_grid':
                common['grid'] = grid
            lev = lambda G: {'lev_invG': G} if vce in ('hc2', 'hc3') else {}
            reqs.append(dict(common, type='R', res_h=h, res_o=p, beta_s=fp[s][1] @ sfull, **lev(fp[s][0])))
            reqs.append(dict(common, type='Q', b=b, q=q, Ltil=Lt[s], gtil=fq[s][0][p + 1],
                             res_h=b, res_o=q, beta_s=fq[s][1] @ sfull, **lev(fq[s][0])))
            keys += [(name, s, 'cl'), (name, s, 'rb')]
    mts = fed.round(f'{tag}:meat', L.meats, reqs=reqs, kernel=kernel)
    V = {}
    for (name, s, kind), o in zip(keys, mts):
        # degrees-of-freedom k: p + 1 for the conventional, q + 1 for the robust variance (HC1 and, as in
        # rdrobust 2.x, the CR1 cluster correction)
        k = (p + 1) if kind == 'cl' else (q + 1)
        M = collect_meat(o, vce, k)
        V[(name, s, kind)] = float((fp[s][0] @ M @ fp[s][0])[nu, nu]) * scale ** 2
    z = norm.ppf(0.5 + level / 2)
    se = {(name, kind): np.sqrt(V[(name, '-', kind)] + V[(name, '+', kind)]) for name in combos for kind in ('cl', 'rb')}
    out = dict(h=float(h), b=float(b), p=p, q=q, deriv=nu, kernel=kernel, vce=vce, level=level,
               n_h=[int(gh[s]['n']) for s in sides], n_b=[int(gb[s]['n']) for s in sides],
               tau_cl=tau_cl, se_cl=float(se[('tau', 'cl')]), tau_bc=tau_bc, se_rb=float(se[('tau', 'rb')]),
               gamma=gamma, fit_diagnostics=fit_diagnostics)
    out.update(_ci(tau_cl, out['se_cl'], tau_bc, out['se_rb'], z))
    # fitted local polynomial on each side (original units of x; covariate-adjusted outcome columns)
    out['side_coef'] = {s: (fp[s][1] @ S) / (h ** np.arange(p + 1))[:, None] for s in sides}
    if my == 2:
        for name, j in (('itt', 0), ('fs', 1)):
            a_cl, a_bc = float(tcl[j]), float(tbc[j])
            out[name] = dict(tau_cl=a_cl, se_cl=float(se[(name, 'cl')]), tau_bc=a_bc, se_rb=float(se[(name, 'rb')]))
            out[name].update(_ci(a_cl, se[(name, 'cl')], a_bc, se[(name, 'rb')], z))
    return out


def _ci(tcl, scl, tbc, srb, z):
    return dict(ci_cl=[float(tcl - z * scl), float(tcl + z * scl)], ci_rb=[float(tbc - z * srb), float(tbc + z * srb)],
                pv_cl=float(2 * norm.sf(abs(tcl / scl))), pv_rb=float(2 * norm.sf(abs(tbc / srb))))
