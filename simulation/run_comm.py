# -*- coding: utf-8 -*-
"""Communication and run-time scaling of the federated protocol.

    python simulation/run_comm.py      -> results/comm/comm_scaling.csv, results/comm/comm_rounds.csv

For K in {2, 5, 10, 20, 40, 60} sites and p in {0, 3, 10, 26} covariates (N = 10,000, design S2 with
lambda = 0, seed 20261011) we record the number of rounds, the float64 values uploaded per site
(x 8 bytes), the total upload, the broadcast volume, and the wall-clock time (median of 5 runs,
single process) of the federated and the centralised computation.  'micro_bytes_site' is what a site
would upload if it shipped its records instead (N_k x (p + 2) float64 values).  A network latency of
50 ms per round is added in 't_fed_net50' as a simple model of wide-area communication.
"""
import os
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)
from fsrd import Site, fsrd, centralized  # noqa: E402
import dgp  # noqa: E402

SEED = 20261011
KS = (2, 5, 10, 20, 40, 60)
PS = (0, 3, 10, 26)
LATENCY = 0.05
SUPPORT = (-1.0, 1.0)          # public support of the running variable (no min / max is sent)


def sites_for(K, p):
    rng = np.random.default_rng(np.random.SeedSequence([SEED, 99, K, p]))
    if p == 0:
        data, _ = dgp.s1(K, rng, shift=0.0)
    else:
        data, _ = dgp.s2(K, 0.0, rng, p=p)
    return [Site(d['x'], d['y'], w=d.get('w')) for d in data]


def timed(fn, reps=5):
    ts, out = [], None
    for _ in range(reps):
        t0 = time.perf_counter()
        out = fn()
        ts.append(time.perf_counter() - t0)
    return out, float(np.median(ts))


def main():
    os.makedirs(os.path.join(ROOT, 'results', 'comm'), exist_ok=True)
    rows = []
    for p in PS:
        for K in KS:
            S = sites_for(K, p)
            covs = p > 0
            fed, t_fed = timed(lambda: fsrd(S, covs=covs, support=SUPPORT))
            cen, t_cen = timed(lambda: centralized(S, covs=covs, support=SUPPORT))
            ik = fsrd(S, bw='ik', covs=covs, support=SUPPORT)
            nk = np.array([s.n for s in S])
            c = fed['comm']
            rows.append(dict(K=K, p=p, N=int(nk.sum()), rounds=c['rounds'], floats_up_site=c['bytes_up_per_site'] / 8,
                             bytes_up_site=c['bytes_up_per_site'], bytes_up_total=c['bytes_up_total'],
                             bytes_down_total=c['bytes_down_total'],
                             micro_bytes_site=float(nk.max() * (p + 2) * 8), micro_bytes_total=float(nk.sum() * (p + 2) * 8),
                             ik_rounds=ik['comm']['rounds'], ik_bytes_up_site=ik['comm']['bytes_up_per_site'],
                             t_fed=t_fed, t_cen=t_cen, t_fed_local=c['t_local'],
                             t_fed_net50=t_fed + LATENCY * c['rounds'],
                             absdiff_tau=abs(fed['tau_bc'] - cen['tau_bc']), absdiff_h=abs(fed['h'] - cen['h'])))
            print(rows[-1], flush=True)
    pd.DataFrame(rows).to_csv(os.path.join(ROOT, 'results', 'comm', 'comm_scaling.csv'), index=False)
    S = sites_for(10, 10)
    r = fsrd(S, covs=True, support=SUPPORT)
    tab = pd.DataFrame(r['comm_rounds'])
    tab['bytes_up_per_site'] = tab['floats_up_per_site'] * 8
    tab.to_csv(os.path.join(ROOT, 'results', 'comm', 'comm_rounds_K10_p10.csv'), index=False)
    print(tab)


if __name__ == '__main__':
    main()
