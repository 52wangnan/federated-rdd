# -*- coding: utf-8 -*-
"""Data-generating processes of the simulation study.

S1  baseline linear design: X ~ U(-1, 1), Y = rho0 + T + rho2 X + e, e ~ N(0, 0.1^2), rho0 = 0.5,
    rho2 = 1; N = 10,000 split into K equal sites; site k's
    support is shifted by s_k ~ U(-0.1, 0.1) (slight shifts in support).  tau = 1.
S2  covariate design: as S1 without shift plus p = 10 covariates,
    Y = eta0 + T + eta2 X + W'eta3 + e, W_k ~ N(0, (1 + lambda k)^2 I_p), lambda in {0, 0.1, 0.2, 0.4},
    eta3 = linspace(0.5, -0.5, p).  tau = 1.
S3  site-specific effects and unequal site sizes.  N_k proportional to k, W_k ~ N(mu_k 1, s_k^2 I_3)
    with mu_k = -0.5 + (k-1)/(K-1), s_k = 2 - (k-1)/(K-1); Y = 0.5 + T + X + W'eta3 + T * W_1 + e.
    Site effect tau_k = 1 + mu_k; pooled cutoff effect tau_pool = sum_k pi_k tau_k with pi_k = N_k / N
    (X has the same density in every site).
S4  curvature at the cutoff (Lee 2008 design used by IK 2012 and CCT 2014, model 1),
    X = 2 Beta(2, 4) - 1, quintic m(.) on each side, e ~ N(0, 0.1295^2), tau = 0.04.
S6  site composition jumps at the cutoff.  Site k has intercept alpha_k = linspace(-0.5, 0.5, K) and keeps
    a record with X >= 0 with probability r_k = linspace(0.2, 0.8, K) and one with X < 0 with probability 1 - r_k
    (sites sample treated and control units at different rates);
    Y = alpha_k + T + X + e, e ~ N(0, 0.1^2), tau = 1 in every site.  P(site | X) jumps at 0, so the pooled RD
    picks up sum_k (pi_k+ - pi_k-) alpha_k unless site indicators are included.
SUPPORT: public support of the centred running variable in each design (used instead of the sites' min / max).
S5  pure quadratic curvature, X ~ U(-1, 1), Y = 0.5 + T + X + 8 X^2 (X < 0) - 6 X^2 (X >= 0) + e,
    e ~ N(0, 0.1^2), tau = 1.  The local quadratic pilot fit is unbiased here, so the robust bias-corrected
    interval should have close to nominal coverage while the conventional one under-covers.
"""
import numpy as np

ETA3_S3 = np.array([1.0, -0.5, 0.25])


def site_sizes(N, K, unequal=False):
    if not unequal:
        return np.full(K, N // K)
    w = np.arange(1, K + 1, dtype=float)
    n = np.floor(N * w / w.sum()).astype(int)
    n[-1] += N - n.sum()
    return n


def s1(K, rng, N=10000, rho0=0.5, rho2=1.0, sigma=0.1, shift=0.1):
    out = []
    for nk in site_sizes(N, K):
        s = rng.uniform(-shift, shift)
        x = rng.uniform(-1 + s, 1 + s, nk)
        y = rho0 + 1.0 * (x >= 0) + rho2 * x + rng.normal(0, sigma, nk)
        out.append(dict(x=x, y=y))
    return out, 1.0


def s2(K, lam, rng, N=10000, p=10, sigma=0.1):
    eta3 = np.linspace(0.5, -0.5, p)
    out = []
    for k, nk in enumerate(site_sizes(N, K), start=1):
        x = rng.uniform(-1, 1, nk)
        w = rng.normal(0, 1 + lam * k, (nk, p))
        y = 0.5 + 1.0 * (x >= 0) + x + w @ eta3 + rng.normal(0, sigma, nk)
        out.append(dict(x=x, y=y, w=w))
    return out, 1.0


def s3_truth(K, N=10000):
    nk = site_sizes(N, K, unequal=True)
    mu = -0.5 + np.arange(K) / (K - 1)
    tau_k = 1.0 + mu
    return float(np.sum(nk / N * tau_k)), float(np.mean(tau_k)), tau_k


def s3(K, rng, N=10000, sigma=0.1):
    nks = site_sizes(N, K, unequal=True)
    out = []
    for j, nk in enumerate(nks):
        mu = -0.5 + j / (K - 1)
        sk = 2.0 - j / (K - 1)
        x = rng.uniform(-1, 1, nk)
        w = rng.normal(mu, sk, (nk, 3))
        T = (x >= 0).astype(float)
        y = 0.5 + T + x + w @ ETA3_S3 + T * w[:, 0] + rng.normal(0, sigma, nk)
        out.append(dict(x=x, y=y, w=w))
    return out, s3_truth(K, N)[0]


def lee_m(x):
    left = 0.48 + 1.27 * x + 7.18 * x ** 2 + 20.21 * x ** 3 + 21.54 * x ** 4 + 7.33 * x ** 5
    right = 0.52 + 0.84 * x - 3.00 * x ** 2 + 7.99 * x ** 3 - 9.01 * x ** 4 + 3.56 * x ** 5
    return np.where(x < 0, left, right)


def s4(K, rng, N=10000, sigma=0.1295):
    out = []
    for nk in site_sizes(N, K):
        x = 2 * rng.beta(2, 4, nk) - 1
        y = lee_m(x) + rng.normal(0, sigma, nk)
        out.append(dict(x=x, y=y))
    return out, 0.04


def s5(K, rng, N=10000, sigma=0.1):
    out = []
    for nk in site_sizes(N, K):
        x = rng.uniform(-1, 1, nk)
        T = x >= 0
        y = 0.5 + 1.0 * T + x + np.where(T, -6.0 * x ** 2, 8.0 * x ** 2) + rng.normal(0, sigma, nk)
        out.append(dict(x=x, y=y))
    return out, 1.0


def s6(K, rng, N=10000, sigma=0.1):
    alpha = np.linspace(-0.5, 0.5, K)
    keep_r = np.linspace(0.2, 0.8, K)
    out = []
    for k, nk in enumerate(site_sizes(N, K)):
        x = rng.uniform(-1, 1, nk)
        u = rng.uniform(size=nk)
        keep = np.where(x >= 0, u < keep_r[k], u < 1 - keep_r[k])
        x = x[keep]
        y = alpha[k] + 1.0 * (x >= 0) + x + rng.normal(0, sigma, keep.sum())
        out.append(dict(x=x, y=y))
    return out, 1.0


SUPPORT = {'S1': (-1.1, 1.1), 'S2': (-1.0, 1.0), 'S3': (-1.0, 1.0), 'S4': (-1.0, 1.0), 'S5': (-1.0, 1.0), 'S6': (-1.0, 1.0)}
