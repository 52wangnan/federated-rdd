# -*- coding: utf-8 -*-
"""Kernels and boundary (one-sided) kernel constants for local polynomial RD.

kweight() follows rdrobust_kweight(): triangular (1-|u|)1(|u|<=1)/h, uniform 0.5*1(|u|<=1)/h,
Epanechnikov 0.75(1-u^2)1(|u|<=1)/h.  The 1/h factor cancels in every estimator and sandwich
variance below; it is kept only so that numbers can be compared with rdrobust line by line.
"""
from functools import lru_cache

import numpy as np
from scipy import integrate

KERNELS = ('triangular', 'uniform', 'epanechnikov')


def kprofile(u, kernel):
    """Kernel profile on u (without the 1/h factor)."""
    a = np.abs(u)
    if kernel == 'triangular':
        return np.where(a <= 1.0, 1.0 - a, 0.0)
    if kernel == 'uniform':
        return np.where(a <= 1.0, 0.5, 0.0)
    if kernel == 'epanechnikov':
        return np.where(a <= 1.0, 0.75 * (1.0 - a ** 2), 0.0)
    raise ValueError(f'unknown kernel {kernel}')


def kweight(z, h, kernel):
    return kprofile(z / h, kernel) / h


@lru_cache(maxsize=None)
def boundary_constants(kernel):
    """One-sided moments mu_j = int_0^1 u^j K(u) du and the local-linear boundary constants.

    B_K  = (mu2^2 - mu1 mu3) / (mu0 mu2 - mu1^2)      (bias:  h^2 * B_K/2 * m''(c) on each side)
    C_V  = int_0^1 (mu2 - mu1 u)^2 K(u)^2 du / (mu0 mu2 - mu1^2)^2   (variance: C_V sigma^2 / (f n h))
    C_IK = (C_V / B_K^2)^(1/5)  -- the constant of the IK (2012) / MSE-optimal bandwidth;
           3.4375 for the triangular kernel, 2.702 uniform, 3.1999 Epanechnikov.
    """
    K = lambda u: float(kprofile(np.array(u), kernel))
    mu = [integrate.quad(lambda u, j=j: u ** j * K(u), 0, 1)[0] for j in range(4)]
    den = mu[0] * mu[2] - mu[1] ** 2
    B = (mu[2] ** 2 - mu[1] * mu[3]) / den
    CV = integrate.quad(lambda u: (mu[2] - mu[1] * u) ** 2 * K(u) ** 2, 0, 1)[0] / den ** 2
    return dict(mu=mu, B_K=B, C_V=CV, C_IK=(CV / B ** 2) ** 0.2)


# pilot-bandwidth constants used by rdbwselect (c_bw = C_c * min(sd, IQR/1.349) * N^(-1/5))
C_PILOT = {'triangular': 2.576, 'uniform': 1.843, 'epanechnikov': 2.34}
