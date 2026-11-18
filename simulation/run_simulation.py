# -*- coding: utf-8 -*-
"""Monte Carlo study (500 replications per configuration).

    python simulation/run_simulation.py               # full run  -> results/sim/sim_raw.csv
    python simulation/run_simulation.py --reps 20     # quick check

Every replication draws its data from np.random.default_rng(SeedSequence([20261011, design, K, 100*lambda, rep])),
so any single replication can be regenerated on its own.  Per replication we record
  fed_*  federated RD, CCT mserd bandwidth, conventional and robust bias-corrected inference
  cen_*  the same estimator on the pooled data (max |fed - cen| over tau, tau_bc, SEs, h, b)
  ik_*   federated RD at the IK (2012) bandwidth (conventional inference, the original pipeline corrected)
  nocov_* (S2 only) federated RD without covariate adjustment
  fe_*    (S6 only) federated RD with site indicators (site fixed effects); maxdiff_fe_cen = |fed - pooled|
Every estimator uses the design's public support (dgp.SUPPORT) instead of the sites' min / max.
  ivw_*, mean_*  site-level CCT estimates combined by inverse-variance weighting / simple averaging
"""
import argparse
import json
import os
import platform
import sys
import time

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)
from fsrd import Site, fsrd, centralized, site_estimates, meta_ivw, meta_mean  # noqa: E402
import dgp  # noqa: E402

BASE_SEED = 20261011
KS = (5, 10, 20, 40)
LAMS = (0.0, 0.1, 0.2, 0.4)
DESIGN_ID = {'S1': 1, 'S2': 2, 'S3': 3, 'S4': 4, 'S5': 5, 'S6': 6}
COVS = {'S1': False, 'S2': True, 'S3': True, 'S4': False, 'S5': False, 'S6': False}
CMP = ('tau_cl', 'tau_bc', 'se_cl', 'se_rb', 'h', 'b')


def configs():
    out = [('S1', K, 0.0) for K in KS]
    out += [('S2', K, lam) for lam in LAMS for K in KS]
    out += [('S3', K, 0.0) for K in KS]
    out += [('S4', K, 0.0) for K in KS]
    out += [('S5', K, 0.0) for K in KS]
    out += [('S6', K, 0.0) for K in KS]
    return out


def make(design, K, lam, rep):
    ss = np.random.SeedSequence([BASE_SEED, DESIGN_ID[design], K, int(round(100 * lam)), rep])
    rng = np.random.default_rng(ss)
    if design == 'S1':
        data, tau = dgp.s1(K, rng)
    elif design == 'S2':
        data, tau = dgp.s2(K, lam, rng)
    elif design == 'S3':
        data, tau = dgp.s3(K, rng)
    elif design == 'S4':
        data, tau = dgp.s4(K, rng)
    elif design == 'S5':
        data, tau = dgp.s5(K, rng)
    else:
        data, tau = dgp.s6(K, rng)
    return [Site(d['x'], d['y'], w=d.get('w')) for d in data], tau


def _put(row, pre, r):
    for k in ('tau_cl', 'se_cl', 'tau_bc', 'se_rb', 'h', 'b'):
        row[f'{pre}_{k}'] = r[k]


def one(design, K, lam, rep):
    sites, tau = make(design, K, lam, rep)
    covs = COVS[design]
    sup = dict(support=dgp.SUPPORT[design])          # public support of the running variable
    row = dict(design=design, K=K, lam=lam, rep=rep, tau=tau, N=sum(s.n for s in sites))
    fed = fsrd(sites, covs=covs, **sup)
    _put(row, 'fed', fed)
    row.update(fed_n_h=sum(fed['n_h']), fed_rounds=fed['comm']['rounds'], fed_bytes_site=fed['comm']['bytes_up_per_site'],
               fed_bytes_total=fed['comm']['bytes_up_total'], fed_time=fed['time'])
    t0 = time.perf_counter()
    cen = centralized(sites, covs=covs, **sup)
    row['cen_time'] = time.perf_counter() - t0
    row['cen_tau_cl'] = cen['tau_cl']
    row['maxdiff_fed_cen'] = max(abs(fed[k] - cen[k]) for k in CMP)
    ik = fsrd(sites, bw='ik', covs=covs, **sup)
    _put(row, 'ik', ik)
    row.update(ik_rounds=ik['comm']['rounds'], ik_bytes_site=ik['comm']['bytes_up_per_site'])
    if design == 'S2':
        nc = fsrd(sites, covs=False, **sup)
        _put(row, 'nocov', nc)
    if design == 'S6':
        fe = fsrd(sites, site_fe=True, **sup)
        _put(row, 'fe', fe)
        cfe = centralized(sites, site_fe=True, **sup)
        row['maxdiff_fe_cen'] = max(abs(fe[k] - cfe[k]) for k in CMP)
    rows = site_estimates(sites, covs=covs, **sup)
    row['site_fail'] = int(sum(not r['ok'] for r in rows))
    for reason in ('saturated_joint_design', 'unidentified_rd_contrast'):
        row['site_fail_' + reason] = sum(reason in r['failure_reason'] for r in rows)
    for kind in ('cl', 'rb'):
        for name, fn in (('ivw', meta_ivw), ('mean', meta_mean)):
            m = fn(rows, kind)
            row[f'{name}_{kind}_tau'] = m['tau']
            row[f'{name}_{kind}_se'] = m['se']
            row[f'{name}_{kind}_k_used'] = m['k_used']
    return row


def chunk(design, K, lam, reps):
    return [one(design, K, lam, r) for r in reps]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--reps', type=int, default=500)
    ap.add_argument('--jobs', type=int, default=max(1, min(20, (os.cpu_count() or 2) - 2)))
    ap.add_argument('--out', default=os.path.join(ROOT, 'results', 'sim', 'sim_raw.csv'))
    a = ap.parse_args()
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    t0 = time.time()
    tasks = []
    step = 25
    for design, K, lam in configs():
        for start in range(0, a.reps, step):
            tasks.append((design, K, lam, list(range(start, min(a.reps, start + step)))))
    res = Parallel(n_jobs=a.jobs, verbose=5)(delayed(chunk)(*t) for t in tasks)
    df = pd.DataFrame([r for part in res for r in part]).sort_values(['design', 'lam', 'K', 'rep'])
    df.to_csv(a.out, index=False)
    meta = dict(base_seed=BASE_SEED, reps=a.reps, configs=len(configs()), wall_seconds=round(time.time() - t0, 1),
                python=sys.version, numpy=np.__version__, pandas=pd.__version__, platform=platform.platform(),
                created=time.strftime('%Y-%m-%d %H:%M:%S'))
    with open(os.path.splitext(a.out)[0] + '_meta.json', 'w', encoding='utf-8') as f:
        json.dump(meta, f, indent=1)
    print(f'wrote {a.out}: {len(df)} rows in {meta["wall_seconds"]} s')


if __name__ == '__main__':
    main()
