# -*- coding: utf-8 -*-
"""Quick start on simulated data (no real data needed):  python examples/quickstart.py

1. sharp RD: 10 sites, running variable X ~ U(-1, 1), cutoff 0, 10 covariates, true effect 1;
2. fuzzy RD: take-up jumps from 0.2 to 0.7 at the cutoff, true effect of take-up 2.
Each design is estimated three ways: federated (sites upload summed statistics only), centralized
(the same estimator on the pooled records) and the inverse-variance-weighted average of per-site fits.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from fsrd import Site, fsrd, centralized, site_estimates, meta_ivw  # noqa: E402
from simulation.dgp import s2  # noqa: E402


def show(title, fed, cen, ivw):
    print(f'\n{title}')
    print(f'  federated   : tau = {fed["tau_bc"]:.4f}  robust 95% CI [{fed["ci_rb"][0]:.4f}, {fed["ci_rb"][1]:.4f}]'
          f'  h = {fed["h"]:.4f}')
    print(f'  centralized : tau = {cen["tau_bc"]:.4f}  robust 95% CI [{cen["ci_rb"][0]:.4f}, {cen["ci_rb"][1]:.4f}]'
          f'  h = {cen["h"]:.4f}')
    print(f'  IVW of per-site fits: tau = {ivw["tau"]:.4f}  (sites used: {ivw["k_used"]})')
    diff = max(abs(fed[k] - cen[k]) for k in ('tau_cl', 'tau_bc', 'se_cl', 'se_rb', 'h', 'b'))
    print(f'  max |federated - centralized| = {diff:.1e}')
    c = fed['comm']
    print(f'  communication: {c["rounds"]} rounds, at most {c["bytes_up_per_site"] / 1024:.1f} KB uploaded per site')


def main():
    rng = np.random.default_rng(2026)
    K = 10

    # 1. sharp RD with covariates
    data, tau = s2(K, lam=0.2, rng=rng)
    sites = [Site(d['x'], d['y'], w=d['w'], name=f'site{k + 1}') for k, d in enumerate(data)]
    kw = dict(bw='mserd', support=(-1.0, 1.0))
    show(f'Sharp RD, {K} sites, true effect {tau}', fsrd(sites, **kw), centralized(sites, **kw),
         meta_ivw(site_estimates(sites, **kw), kind='rb'))

    # 2. fuzzy RD: take-up probability jumps at the cutoff
    sites = []
    for k in range(K):
        n = 1000
        x = rng.uniform(-1, 1, n)
        d = (rng.uniform(size=n) < np.where(x >= 0, 0.7, 0.2)).astype(float)
        y = 0.5 + 2.0 * d + x + rng.normal(0, 0.5, n)
        sites.append(Site(x, y, d=d, name=f'site{k + 1}'))
    show('Fuzzy RD, 10 sites, true effect of take-up 2.0', fsrd(sites, **kw), centralized(sites, **kw),
         meta_ivw(site_estimates(sites, **kw), kind='rb'))


if __name__ == '__main__':
    main()
