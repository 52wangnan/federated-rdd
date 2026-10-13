# -*- coding: utf-8 -*-
"""Computations that run INSIDE a site.  Each function takes a Site plus the parameters broadcast
by the sponsor and returns only sums over the site's records (no record-level data leaves it).

Conventions
- side '-' = records with x < 0, side '+' = records with x >= 0 (x is centred at the cutoff).
- polynomial bases are built in scaled units u = x / h (h = bandwidth of that fit), which keeps
  the Gram matrices well conditioned; the sponsor converts coefficients back (beta_j = beta~_j / h^j).
- data columns D = [y, (d), w_1..w_p]: outcome, take-up (fuzzy only), covariates.
- every sum is formed from per-record terms that are computed element-wise (so a record contributes
  bit-for-bit the same term wherever it is stored) and is returned as a double-double pair (see
  federation.acc_sum); counts are exact.
- small-cell rule: if a site's window on one side has fewer than min_cell records, the site
  contributes zeros for that window (min_cell = 0 disables the rule).
"""
import numpy as np
from .kernels import kweight
from .federation import dd, dd_exact, acc_sum


def _side(x, side):
    return x < 0 if side == '-' else x >= 0


def _vander(u, order):
    return np.vander(u, order + 1, increasing=True)


def _outer(A, B):
    """Per-record outer products, shape (n, a, b)."""
    return A[:, :, None] * B[:, None, :]


def _outer_sym(A, B):
    """Per-record upper-triangle products A_i B_j (i <= j) of a symmetric outer product, shape (n, k(k+1)/2)."""
    iu, ju = np.triu_indices(A.shape[1])
    return A[:, iu] * B[:, ju]


def _rowdot(A, v):
    """Per-record dot products computed element-wise (independent of how many records are stored)."""
    return (A * v[None, :]).sum(axis=1)


def kw_side(site, side, h, kernel, min_cell=0):
    """Kernel weights for one side (zero off-side and outside the window).  Windows zeroed by the
    small-cell rule are logged locally in site.suppressed (side, bandwidth) for the audit tables."""
    x = site.x
    kw = np.where(_side(x, side), kweight(x, h, kernel), 0.0)
    if min_cell > 0 and 0 < np.count_nonzero(kw) < min_cell:
        kw[:] = 0.0
        site.suppressed.add((side, float(h)))
    return kw


# ----------------------------------------------------------------------------- round types
def describe(site, minmax=True, min_cell=0):
    """Counts and first two moments of the running variable; with minmax=True also its range
    (combined by min / max, so it reveals the extreme values; use a public support instead)."""
    x = site.x
    out = dict(n=dd_exact(len(x)), sx=dd(x), sxx=dd(x * x),
               n_side=dd_exact([np.sum(x < 0), np.sum(x >= 0)]))
    if minmax:
        out.update(min_x=x.min(), max_x=x.max())
    return out


def grid_counts(site, edges, min_cell=0):
    """Counts on a public grid of support points (bins centred on the points), for mass-point
    adjustments and exact quantiles of a discrete running variable."""
    return dict(cnt=dd_exact(np.histogram(site.x, bins=edges)[0]))


def histogram(site, edges_l, edges_r, min_cell=0):
    """Counts of the running variable on a public bin grid, separately below/above the cutoff."""
    x = site.x
    return dict(cnt_l=dd_exact(np.histogram(x[x < 0], bins=edges_l)[0]),
                cnt_r=dd_exact(np.histogram(x[x >= 0], bins=edges_r)[0]))


def bin_counts(site, edges, min_cell=0):
    return dict(cnt=dd_exact(np.histogram(site.x, bins=edges)[0]))


def grams(site, reqs, kernel, y_only=False, min_cell=0):
    """Kernel-weighted Gram statistics for a list of (side, h, order, zz[, scale]) windows.
    Returns, per request: n (records with positive weight), G = sum kw R R', RD = sum kw R D',
    and, if zz and the site has covariates, ZZ = sum kw w w', ZD = sum kw w [y, d]'.
    y_only: outcome column only.  G and ZZ travel as upper triangles (keys G_sym, ZZ_sym)."""
    D = site.y[:, None] if y_only else site.data_matrix()
    my = 1 if site.d is None else 2
    out = []
    for req in reqs:
        side, h, order, zz = req[:4]
        scale = req[4] if len(req) > 4 else h            # polynomial scale (default: the bandwidth)
        kw = kw_side(site, side, h, kernel, min_cell)
        m = kw > 0
        R = _vander(site.x[m] / scale, order)
        kR = R * kw[m, None]
        Dm = D[m]
        o = dict(n=dd_exact(m.sum()), G_sym=dd(_outer_sym(kR, R)), RD=dd(_outer(kR, Dm)))
        if zz and site.w is not None and not y_only:
            kW = site.w[m] * kw[m, None]
            o['ZZ_sym'] = dd(_outer_sym(kW, site.w[m]))
            o['ZD'] = dd(_outer(kW, Dm[:, :my]))
        out.append(o)
    return out


def window_moments(site, h, min_cell=0):
    """IK step 1: counts, sum y, sum y^2 in [-h, 0) and [0, h] (outcome column 0 only)."""
    x, y = site.x, site.y
    out = []
    for side in ('-', '+'):
        m = _side(x, side) & (np.abs(x) <= h)
        if min_cell > 0 and m.sum() < min_cell:
            m[:] = False
        out.append(np.column_stack([np.ones(m.sum()), y[m], y[m] ** 2]))
    return dict(mom=np.stack([dd(out[0]), dd(out[1])], axis=1))


def ik_step1(site, h1, edges_l, edges_r, min_cell=0):
    """IK round 2: window moments at h1 together with the histogram used for the side medians."""
    o = window_moments(site, h1, min_cell)
    o.update(histogram(site, edges_l, edges_r))
    return o


def global_poly(site, lo, hi, scale, min_cell=0):
    """IK step 2: OLS sums for y on (1, 1{x>=0}, u, u^2, u^3), u = x/scale, over lo <= x <= hi."""
    x, y = site.x, site.y
    m = (x >= lo) & (x <= hi)
    u = x[m] / scale
    R = np.column_stack([np.ones(m.sum()), (x[m] >= 0).astype(float), u, u ** 2, u ** 3])
    return dict(n=dd_exact(m.sum()), G_sym=dd(_outer_sym(R, R)), Ry=dd(R * y[m, None]))


def meats(site, reqs, kernel, min_cell=0):
    """Sandwich 'meat' sums for a list of requests (all broadcast quantities are global).

    Each request is a dict with
      side, h, p           : the design X_i = kw_h(x_i) * R_p(x_i / h)
      type 'R' or 'Q'      : 'Q' is the CCT bias-corrected design
                             Q_i = kw_h R_p(u_h) - (h/b)^(p+1) * Ltil * (gtil' R_q(u_b)) * kw_b
                             (needs b, q, Ltil, gtil)
      res_h, res_o, beta_s : residual r_i = D_i's - R_res(x_i/res_h)' beta_s  (beta_s = beta~ @ s)
      s                    : combination of the data columns (1, -gamma) / delta-method weights
      win                  : bandwidths whose union defines the window (rdrobust: max(h, b))
      vce                  : 'hc0','hc1','hc2','hc3' (hc1 factor applied by the sponsor),
                             'cluster' (clusters nested in sites) or 'cluster_grid' (public label grid)
      lev_invG             : inverse Gram of the p-fit at h (hc2 / hc3 leverages)
      grid                 : public cluster labels (cluster_grid)
    """
    D = site.data_matrix()
    x = site.x
    out = []
    for r in reqs:
        side, h, p = r['side'], r['h'], r['p']
        kwh = kw_side(site, side, h, kernel, min_cell)
        kws = [kw_side(site, side, hb, kernel, min_cell) for hb in r['win']]
        m = np.zeros(len(x), bool)
        for k in kws:
            m |= k > 0
        xm = x[m]
        Rp = _vander(xm / h, p)
        XR = Rp * kwh[m, None]
        X = XR
        if r['type'] == 'Q':
            b, q = r['b'], r['q']
            kwb = kw_side(site, side, b, kernel, min_cell)[m]
            Rq = _vander(xm / b, q)
            X = X - (h / b) ** (p + 1) * (_rowdot(Rq, r['gtil']) * kwb)[:, None] * r['Ltil'][None, :]
        res = _rowdot(D[m], r['s']) - _rowdot(_vander(xm / r['res_h'], r['res_o']), r['beta_s'])
        vce = r['vce']
        if vce in ('hc2', 'hc3'):
            if r['type'] == 'Q':
                # rdrobust 2.x: the robust residuals (from the q-fit at b) use the q-fit leverage
                hii = np.einsum('ij,jk,ik->i', Rq, r['lev_invG'], Rq * kwb[:, None])
            else:
                hii = np.einsum('ij,jk,ik->i', Rp, r['lev_invG'], XR)
            hii = np.minimum(hii, 1.0 - 1e-8)
            res = res / np.sqrt(1.0 - hii) if vce == 'hc2' else res / (1.0 - hii)
        S = X * res[:, None]
        o = dict(n=dd_exact(m.sum()))
        if vce == 'cluster':
            cl = site.cl[m]
            labs, idx = np.unique(cl, return_inverse=True)
            Sg = np.zeros((len(labs), S.shape[1]))
            np.add.at(Sg, idx, S)                       # record order within a cluster is preserved
            o['M_sym'] = dd(_outer_sym(Sg, Sg))
            o['G'] = dd_exact(len(labs))
        elif vce == 'cluster_grid':
            grid = r['grid']
            lab = site.cl[m]
            idx = np.searchsorted(grid, lab)
            if np.any(idx >= len(grid)) or np.any(grid[np.minimum(idx, len(grid) - 1)] != lab):
                raise ValueError('cluster label outside the public grid')
            Sg = np.zeros((2, len(grid), S.shape[1]))
            for j in np.unique(idx):
                hi, lo = acc_sum(S[idx == j])
                Sg[0, j], Sg[1, j] = hi, lo
            o['Sg'] = Sg
            o['cnt'] = dd_exact(np.bincount(idx, minlength=len(grid)))
        else:
            o['M_sym'] = dd(_outer_sym(S, S))
        out.append(o)
    return out
