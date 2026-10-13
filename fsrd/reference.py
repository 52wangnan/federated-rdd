# -*- coding: utf-8 -*-
"""Observation-level reference implementation on pooled data (used only by the tests).

A line-by-line transcription of the estimation step of rdrobust() and of rdrobust_bw()/rdbwselect
(bwselect='mserd') from rdrobust 2.1.1 (Calonico, Cattaneo, Farrell & Titiunik), restricted to the
HC0-HC3 / CR1 cluster variance options.  It works in the original units of x, builds the bias-correction
matrix Q_q explicitly and never uses sufficient statistics, so it is an independent check of the
federated code in lprd.py / bandwidth.py when the rdrobust package itself is not installed.
"""
from math import factorial

import numpy as np
from scipy.stats import norm
from .kernels import kweight, C_PILOT

EPS_RANGE = 1 + np.sqrt(np.finfo(float).eps)


def _inv(A):
    return np.linalg.inv(A)


def _vander(z, o):
    return np.vander(z, o + 1, increasing=True)


def _res(D, pred, vce, k, hii=None, cluster=False):
    r = D - pred
    n = len(r)
    if vce == 'hc1' and not cluster:
        r = r * np.sqrt(n / (n - k))
    elif vce == 'hc2':
        r = r / np.sqrt(np.maximum(1 - hii, 1e-8))[:, None]
    elif vce == 'hc3':
        r = r / np.maximum(1 - hii, 1e-8)[:, None]
    return r


def _vce(RX, res, s, C=None, k=None):
    e = res @ s
    if C is None:
        S = RX * e[:, None]
        return S.T @ S
    labs, idx = np.unique(C, return_inverse=True)
    Sg = np.zeros((len(labs), RX.shape[1]))
    np.add.at(Sg, idx, RX * e[:, None])
    n, G = len(C), len(labs)
    return (Sg.T @ Sg) * ((n - 1) / (n - k)) * (G / (G - 1))


def rdrobust_bw(Y, X, T, Z, c, o, nu, o_B, h_V, h_B, scale, vce, kernel):
    w = kweight(X - c, h_V, kernel)
    ind = w > 0
    eY, eX, eW = Y[ind], X[ind], w[ind]
    R_V = _vander(eX - c, o)
    invG_V = _inv((R_V * eW[:, None]).T @ R_V)
    D_V = eY[:, None]
    dT = 0
    if T is not None:
        dT = 1
        D_V = np.column_stack([D_V, T[ind]])
    s = np.array([1.0])
    gamma = None
    if Z is not None:
        eZ = Z[ind]
        D_V = np.column_stack([D_V, eZ])
        U = (R_V * eW[:, None]).T @ D_V
        ZWD = (eZ * eW[:, None]).T @ D_V
        colsZ = slice(1 + dT, None)
        UiGU = U[:, colsZ].T @ invG_V @ U
        ZWZ = ZWD[:, colsZ] - UiGU[:, colsZ]
        ZWY = ZWD[:, :1 + dT] - UiGU[:, :1 + dT]
        gamma = np.linalg.solve(ZWZ, ZWY)
        s = np.concatenate([[1.0], -gamma[:, 0]])
    beta_V = invG_V @ (R_V * eW[:, None]).T @ D_V
    if T is not None and Z is None:
        tY, tT = factorial(nu) * beta_V[nu, 0], factorial(nu) * beta_V[nu, 1]
        s = np.array([1 / tT, -tY / tT ** 2])
    if T is not None and Z is not None:
        tY = factorial(nu) * (beta_V[nu, 0] - gamma[:, 0] @ beta_V[nu, 2:])
        tT = factorial(nu) * (beta_V[nu, 1] - gamma[:, 1] @ beta_V[nu, 2:])
        s = np.concatenate([[1 / tT, -tY / tT ** 2], -(1 / tT) * gamma[:, 0] + (tY / tT ** 2) * gamma[:, 1]])
    hii = None
    if vce in ('hc2', 'hc3'):
        hii = np.einsum('ij,jk,ik->i', R_V, invG_V, R_V * eW[:, None])
    res_V = _res(D_V, R_V @ beta_V, vce, o + 1, hii)
    V_V = (invG_V @ _vce(R_V * eW[:, None], res_V, s) @ invG_V)[nu, nu]
    v = (R_V * eW[:, None]).T @ (((eX - c) / h_V) ** (o + 1))
    Hp = h_V ** np.arange(o + 1)
    BConst = (Hp * (invG_V @ v))[nu]
    w = kweight(X - c, h_B, kernel)
    ind = w > 0
    eY, eX, eW = Y[ind], X[ind], w[ind]
    R_B = _vander(eX - c, o_B)
    invG_B = _inv((R_B * eW[:, None]).T @ R_B)
    D_B = eY[:, None]
    if T is not None:
        D_B = np.column_stack([D_B, T[ind]])
    if Z is not None:
        D_B = np.column_stack([D_B, Z[ind]])
    beta_B = invG_B @ (R_B * eW[:, None]).T @ D_B
    BWreg = 0.0
    if scale > 0:
        hii = None
        if vce in ('hc2', 'hc3'):
            hii = np.einsum('ij,jk,ik->i', R_B, invG_B, R_B * eW[:, None])
        res_B = _res(D_B, R_B @ beta_B, vce, o_B + 1, hii)
        V_B = (invG_B @ _vce(R_B * eW[:, None], res_B, s) @ invG_B)[o + 1, o + 1]
        BWreg = 3 * BConst ** 2 * V_B
    B = np.sqrt(2 * (o + 1 - nu)) * BConst * (s @ beta_B[o + 1])
    V = (2 * nu + 1) * h_V ** (2 * nu + 1) * V_V
    R = scale * (2 * (o + 1 - nu)) * BWreg
    return dict(V=V, B=B, R=R, rate=1.0 / (2 * o + 3))


def _quantile_type2(x, prob):
    x = np.sort(x)
    n = len(x)
    hh = n * prob
    j = int(np.floor(hh))
    return 0.5 * (x[j - 1] + x[j]) if hh == j else x[j]


def rdbwselect_mserd(y, x, c=0.0, covs=None, fuzzy=None, p=1, q=2, deriv=0, kernel='triangular', vce='hc1',
                     scaleregul=1.0, bwrestrict=True, iqr=None, masspoints='off', bwcheck=None, support=None):
    """mserd bandwidths.  iqr=None: exact type-2 quantiles; support=(lo, hi): public range instead of
    the observed min / max; masspoints='adjust' as rdrobust."""
    l, r = x < c, x >= c
    N = len(x)
    x_sd = np.std(x, ddof=1)
    if iqr is None:
        iqr = _quantile_type2(x, 0.75) - _quantile_type2(x, 0.25)
    lo, hi = (x.min(), x.max()) if support is None else (c + support[0], c + support[1])
    range_l, range_r = abs(c - lo), abs(hi - c)
    bw_max = max(range_l, range_r)
    M = N
    bw_min = 0.0
    if masspoints == 'adjust' or bwcheck is not None:
        ul = np.sort(np.unique(x[l]))[::-1]
        ur = np.unique(x[r])
        if masspoints == 'adjust':
            M = len(ul) + len(ur)
            if (1 - len(ul) / l.sum() >= 0.2 or 1 - len(ur) / r.sum() >= 0.2) and bwcheck is None:
                bwcheck = 10
        if bwcheck is not None:
            bw_min = max(abs(ul[min(bwcheck, len(ul)) - 1] - c), abs(ur[min(bwcheck, len(ur)) - 1] - c)) * EPS_RANGE
    c_bw = C_PILOT[kernel] * min(x_sd, iqr / 1.349) * M ** (-0.2)
    if bwrestrict:
        c_bw = min(c_bw, bw_max)
    c_bw = max(c_bw, bw_min)
    Zl = None if covs is None else covs[l]
    Zr = None if covs is None else covs[r]
    Tl = None if fuzzy is None else fuzzy[l]
    Tr = None if fuzzy is None else fuzzy[r]
    Cd_l = rdrobust_bw(y[l], x[l], Tl, Zl, c, q + 1, q + 1, q + 2, c_bw, range_l * EPS_RANGE, 0, vce, kernel)
    Cd_r = rdrobust_bw(y[r], x[r], Tr, Zr, c, q + 1, q + 1, q + 2, c_bw, range_r * EPS_RANGE, 0, vce, kernel)
    d_bw = ((Cd_l['V'] + Cd_r['V']) / (Cd_r['B'] - Cd_l['B']) ** 2) ** Cd_l['rate']
    if bwrestrict:
        d_bw = min(d_bw, bw_max)
    d_bw = max(d_bw, bw_min)
    Cb_l = rdrobust_bw(y[l], x[l], Tl, Zl, c, q, p + 1, q + 1, c_bw, d_bw, scaleregul, vce, kernel)
    Cb_r = rdrobust_bw(y[r], x[r], Tr, Zr, c, q, p + 1, q + 1, c_bw, d_bw, scaleregul, vce, kernel)
    b_bw = ((Cb_l['V'] + Cb_r['V']) / ((Cb_r['B'] - Cb_l['B']) ** 2 + scaleregul * (Cb_r['R'] + Cb_l['R']))) ** Cb_l['rate']
    if bwrestrict:
        b_bw = min(b_bw, bw_max)
    Ch_l = rdrobust_bw(y[l], x[l], Tl, Zl, c, p, deriv, q, c_bw, b_bw, scaleregul, vce, kernel)
    Ch_r = rdrobust_bw(y[r], x[r], Tr, Zr, c, p, deriv, q, c_bw, b_bw, scaleregul, vce, kernel)
    h_bw = ((Ch_l['V'] + Ch_r['V']) / ((Ch_r['B'] - Ch_l['B']) ** 2 + scaleregul * (Ch_r['R'] + Ch_l['R']))) ** Ch_l['rate']
    if bwrestrict:
        h_bw = min(h_bw, bw_max)
    return float(h_bw), float(b_bw), dict(c_bw=c_bw, d_bw=d_bw)


def rdrobust_est(y, x, h, b, c=0.0, fuzzy=None, covs=None, p=1, q=2, deriv=0, kernel='triangular', vce='hc1',
                 cluster=None, level=0.95):
    """Estimation step of rdrobust() at given (h, b) (h != b branch)."""
    out = {}
    side = {}
    dT = 0 if fuzzy is None else 1
    for nm, msk in (('l', x < c), ('r', x >= c)):
        X, Y = x[msk], y[msk]
        W_h = kweight(X - c, h, kernel)
        W_b = kweight(X - c, b, kernel)
        ind = (W_h > 0) if h > b else (W_b > 0)
        eX, eY, W_h, W_b = X[ind], Y[ind], W_h[ind], W_b[ind]
        D = eY[:, None]
        if fuzzy is not None:
            D = np.column_stack([D, fuzzy[msk][ind]])
        if covs is not None:
            D = np.column_stack([D, covs[msk][ind]])
        C = None if cluster is None else cluster[msk][ind]
        u = (eX - c) / h
        R_q = _vander(eX - c, q)
        R_p = R_q[:, :p + 1]
        L = (R_p * W_h[:, None]).T @ u ** (p + 1)
        invG_q = _inv((R_q * W_b[:, None]).T @ R_q)
        invG_p = _inv((R_p * W_h[:, None]).T @ R_p)
        e_p1 = np.zeros(q + 1)
        e_p1[p + 1] = 1.0
        Q_q = ((R_p * W_h[:, None]).T - h ** (p + 1) * np.outer(L, e_p1) @ (invG_q @ R_q.T * W_b[None, :])).T
        beta_p = invG_p @ (R_p * W_h[:, None]).T @ D
        beta_q = invG_q @ (R_q * W_b[:, None]).T @ D
        beta_bc = invG_p @ Q_q.T @ D
        hii_p = hii_q = None
        if vce in ('hc2', 'hc3'):
            hii_p = np.einsum('ij,jk,ik->i', R_p, invG_p, R_p * W_h[:, None])
            hii_q = np.einsum('ij,jk,ik->i', R_q, invG_q, R_q * W_b[:, None])
        res_h = _res(D, R_p @ beta_p, vce, p + 1, hii_p, cluster is not None)
        res_b = _res(D, R_q @ beta_q, vce, q + 1, hii_q, cluster is not None)
        side[nm] = dict(R_p=R_p, W_h=W_h, Q_q=Q_q, invG_p=invG_p, beta_p=beta_p, beta_bc=beta_bc,
                        res_h=res_h, res_b=res_b, C=C, D=D, eZ=None if covs is None else D[:, 1 + dT:])
    bp = side['r']['beta_p'] - side['l']['beta_p']
    bbc = side['r']['beta_bc'] - side['l']['beta_bc']
    if covs is not None:
        ZWZ = ZWY = 0.0
        for sd in side.values():
            U = (sd['R_p'] * sd['W_h'][:, None]).T @ sd['D']
            ZWD = (sd['eZ'] * sd['W_h'][:, None]).T @ sd['D']
            UiGU = U[:, 1 + dT:].T @ sd['invG_p'] @ U
            ZWZ = ZWZ + ZWD[:, 1 + dT:] - UiGU[:, 1 + dT:]
            ZWY = ZWY + ZWD[:, :1 + dT] - UiGU[:, :1 + dT]
        gamma = np.linalg.solve(ZWZ, ZWY)
    else:
        gamma = np.zeros((0, 1 + dT))
    Smat = np.vstack([np.eye(1 + dT), -gamma])
    nu = deriv
    tcl = factorial(nu) * (bp[nu] @ Smat)
    tbc = factorial(nu) * (bbc[nu] @ Smat)
    if fuzzy is None:
        s = Smat[:, 0]
        tau_cl, tau_bc = tcl[0], tbc[0]
    else:
        sl = np.array([1 / tcl[1], -tcl[0] / tcl[1] ** 2])
        s = Smat @ sl
        tau_cl = tcl[0] / tcl[1]
        tau_bc = tau_cl - sl @ (tcl - tbc)
    V_cl = V_rb = 0.0
    for sd in side.values():
        V_cl += (sd['invG_p'] @ _vce(sd['R_p'] * sd['W_h'][:, None], sd['res_h'], s, sd['C'], p + 1) @ sd['invG_p'])[nu, nu]
        V_rb += (sd['invG_p'] @ _vce(sd['Q_q'], sd['res_b'], s, sd['C'], q + 1) @ sd['invG_p'])[nu, nu]
    V_cl *= factorial(nu) ** 2
    V_rb *= factorial(nu) ** 2
    z = norm.ppf(0.5 + level / 2)
    out.update(tau_cl=float(tau_cl), tau_bc=float(tau_bc), se_cl=float(np.sqrt(V_cl)), se_rb=float(np.sqrt(V_rb)),
               gamma=gamma)
    out['ci_rb'] = [out['tau_bc'] - z * out['se_rb'], out['tau_bc'] + z * out['se_rb']]
    return out


def local_linear_lstsq(y, x, h, kernel='triangular', covs=None):
    """Plain weighted least squares on the pooled records (stacked regression with side-specific
    intercepts and slopes and, optionally, common covariate slopes).  Returns the jump at 0."""
    w = kweight(x, h, kernel)
    m = w > 0
    T = (x[m] >= 0).astype(float)
    cols = [1 - T, T, (1 - T) * x[m], T * x[m]]
    if covs is not None:
        cols += [covs[m]]
    Xd = np.column_stack(cols)
    sw = np.sqrt(w[m])
    beta = np.linalg.lstsq(Xd * sw[:, None], y[m] * sw, rcond=None)[0]
    return float(beta[1] - beta[0])
