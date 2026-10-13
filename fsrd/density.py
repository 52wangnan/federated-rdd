# -*- coding: utf-8 -*-
"""Federated manipulation (density) test of McCrary (2008).

The only statistic the sites send is the count of records in each bin of a public grid, so the
test is exact under federation.  With a discrete running variable (age in months) the bins are
centred on the support points; a bin belongs to the right side if its midpoint is >= 0.

Steps (McCrary 2008, Section III): normalised cell sizes Y_j = N_j / (N b); bandwidth from a global
fourth-order polynomial on each side, h_side = 3.348 [sigma~^2 (range_side) / sum_j f~''(r_j)^2]^(1/5),
h = mean of the two; triangular-kernel local linear fits on each side give f_-, f_+;
theta = ln f_+ - ln f_-,  se = sqrt( (1/(N h)) (24/5) (1/f_+ + 1/f_-) ).
"""
import numpy as np
from scipy.stats import norm
from .federation import Federation
from . import local as L


def mccrary(sites, binwidth, lo, hi, h=None, centred=True, fed=None):
    """lo, hi: support of the running variable (public).  centred=True puts bin midpoints on the grid
    {k * binwidth}; otherwise 0 is a bin edge."""
    fed = Federation(sites) if fed is None else fed
    off = binwidth / 2 if centred else 0.0
    kl = int(np.floor((lo + off) / binwidth)) - 1
    kh = int(np.ceil((hi + off) / binwidth)) + 1
    edges = np.arange(kl, kh + 1) * binwidth - off
    cnt = fed.round('density:bins', L.bin_counts, edges=edges)['cnt']
    mid = (edges[:-1] + edges[1:]) / 2
    N = cnt.sum()
    Yj = cnt / (N * binwidth)
    keep = (mid >= lo - binwidth) & (mid <= hi + binwidth)
    mid, Yj, cnt = mid[keep], Yj[keep], cnt[keep]
    left, right = mid < 0, mid >= 0
    # trim empty tails outside the observed support
    nz = np.nonzero(cnt)[0]
    sup = np.zeros(len(cnt), bool)
    sup[nz.min():nz.max() + 1] = True
    left &= sup
    right &= sup
    if h is None:
        hs = []
        for m in (left, right):
            r = mid[m]
            X = np.vander(r, 5, increasing=True)
            beta = np.linalg.lstsq(X, Yj[m], rcond=None)[0]
            resid = Yj[m] - X @ beta
            s2 = np.mean(resid ** 2)
            f2 = 2 * beta[2] + 6 * beta[3] * r + 12 * beta[4] * r ** 2
            hs.append(3.348 * (s2 * (r.max() - r.min()) / np.sum(f2 ** 2)) ** 0.2)
        h = float(np.mean(hs))
    f = []
    for m in (left, right):
        r, yv = mid[m], Yj[m]
        w = np.clip(1 - np.abs(r) / h, 0, None)
        k = w > 0
        X = np.column_stack([np.ones(k.sum()), r[k]])
        sw = np.sqrt(w[k])
        f.append(np.linalg.lstsq(X * sw[:, None], yv[k] * sw, rcond=None)[0][0])
    fl, fr = f
    theta = np.log(fr) - np.log(fl)
    se = np.sqrt((1.0 / (N * h)) * (24.0 / 5.0) * (1.0 / fr + 1.0 / fl))
    return dict(theta=float(theta), se=float(se), z=float(theta / se), p=float(2 * norm.sf(abs(theta / se))),
                f_minus=float(fl), f_plus=float(fr), h=float(h), binwidth=binwidth, N=int(N),
                bins=dict(mid=mid[sup], density=Yj[sup], count=cnt[sup]), comm=fed.ledger.summary())
