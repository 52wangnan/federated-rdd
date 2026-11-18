# -*- coding: utf-8 -*-
"""Site-level benchmarks with local IK bandwidths for designs S1 and S2 (Tables 2 and 4).
    python paper/local_ik_benchmarks.py --jobs 7      # recompute
    python paper/local_ik_benchmarks.py --summary     # summaries only
The pooled/federated IK results are read unchanged from results/sim/sim_raw.csv; this script regenerates the
same Monte Carlo draws (run_simulation.make) and refits every site with its own IK bandwidth (b = h), as
paper/extensions_s3_curvature.py does for S3. No estimate is clipped.
"""
import os
for _v in ('OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'OMP_NUM_THREADS'):
    os.environ[_v] = '1'
import argparse
import hashlib
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import norm
from joblib import Parallel, delayed

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'simulation')]
from run_simulation import make, BASE_SEED, COVS
from fsrd import fsrd, centralized, site_estimates, meta_ivw, meta_mean
import dgp

OUT = ROOT / 'results' / 'ik_benchmarks'
Z = norm.ppf(.975)
FIELDS = ('tau_cl', 'tau_bc', 'se_cl', 'se_rb', 'h', 'b')
CONFIGS = [('S1', 0.0)] + [('S2', lam) for lam in (0.0, 0.1, 0.2, 0.4)]
KS = (5, 10, 20, 40)


def local_one(design, K, lam, rep):
    sites, truth = make(design, K, lam, rep)
    rows = site_estimates(sites, bw='ik', covs=COVS[design], support=dgp.SUPPORT[design])
    row = dict(design=design, K=K, lam=lam, rep=rep, tau=truth, failed_sites=sum(not r['ok'] for r in rows))
    for kind in ('cl', 'rb'):
        for name, fn in [('ivw', meta_ivw), ('mean', meta_mean)]:
            r = fn(rows, kind)
            row.update({f'{name}_{kind}_tau': r['tau'], f'{name}_{kind}_se': r['se'], f'{name}_{kind}_k_used': r['k_used']})
    return row


def chunk(design, K, lam, reps):
    return [local_one(design, K, lam, rep) for rep in reps]


def checks():
    raw = pd.read_csv(ROOT / 'results/sim/sim_raw.csv')
    out = []
    for design, lam in CONFIGS:
        for K in KS:
            for rep in range(3):
                S, truth = make(design, K, lam, rep)
                kw = dict(bw='ik', covs=COVS[design], support=dgp.SUPPORT[design])
                f, c = fsrd(S, **kw), centralized(S, **kw)
                old = raw[(raw.design == design) & (raw.K == K) & (np.isclose(raw.lam, lam)) & (raw.rep == rep)].iloc[0]
                assert abs(truth - old.tau) < 1e-12
                ds = max(abs(f[k] - old['ik_' + k]) for k in FIELDS)
                dc = max(abs(f[k] - c[k]) for k in FIELDS)
                assert ds < 1e-10 and dc < 1e-10, (design, K, lam, rep, ds, dc)
                out.append(dict(design=design, K=K, lam=lam, rep=rep, maxdiff_stored=ds, maxdiff_central=dc))
    pd.DataFrame(out).to_csv(OUT / 'local_ik_regeneration_checks.csv', index=False)


def compute(jobs):
    OUT.mkdir(parents=True, exist_ok=True)
    checks()
    tasks = [(d, K, lam, list(range(s, min(s + 25, 500)))) for d, lam in CONFIGS for K in KS for s in range(0, 500, 25)]
    done = Parallel(n_jobs=jobs, verbose=5)(delayed(chunk)(*a) for a in tasks)
    rows = [x for part in done for x in part]
    pd.DataFrame(rows).sort_values(['design', 'lam', 'K', 'rep']).to_csv(OUT / 'local_ik_s1s2_raw.csv', index=False)
    src = ROOT / 'results/sim/sim_raw.csv'
    (OUT / 'local_ik_provenance.json').write_text(json.dumps(dict(
        seed=BASE_SEED, repetitions=500, configs=CONFIGS, sites='K in (5,10,20,40)',
        bandwidth='each site selects its own IK bandwidth, b = h', sim_raw_sha256=hashlib.sha256(src.read_bytes()).hexdigest()),
        indent=2), encoding='utf-8')


def metrics(t, se, truth):
    t, se, truth = np.asarray(t), np.asarray(se), np.asarray(truth)
    ok = np.isfinite(t) & np.isfinite(se) & (se > 0)
    e = t[ok] - truth[ok]; n = len(e); rm = np.sqrt(np.mean(e ** 2)); cov = np.mean(np.abs(e) <= Z * se[ok])
    return dict(reps=n, bias=e.mean(), sd=np.std(t[ok], ddof=1), rmse=rm, mcse_bias=np.std(e, ddof=1) / np.sqrt(n),
                mcse_rmse=np.std(e ** 2, ddof=1) / (2 * rm * np.sqrt(n)), coverage=cov,
                mcse_coverage=np.sqrt(cov * (1 - cov) / n), length=np.mean(2 * Z * se[ok]))


def summary():
    raw = pd.read_csv(ROOT / 'results/sim/sim_raw.csv')
    loc = pd.read_csv(OUT / 'local_ik_s1s2_raw.csv')
    keys = ['design', 'K', 'lam', 'rep']
    loc = loc.rename(columns={c: 'lik_' + c for c in loc.columns if c not in keys})
    m = raw[raw.design.isin(['S1', 'S2'])].merge(loc.drop(columns='lik_tau'), on=['design', 'K', 'lam', 'rep'], validate='one_to_one')
    assert len(m) == 10000
    rows = []
    for (design, lam, K), g in m.groupby(['design', 'lam', 'K']):
        assert len(g) == 500
        for label, t, s, used in [('FSRD / pooled, IK', 'ik_tau_cl', 'ik_se_cl', None),
                                  ('FSRD / pooled, IK, robust BC', 'ik_tau_bc', 'ik_se_rb', None),
                                  ('IVW, local IK', 'lik_ivw_cl_tau', 'lik_ivw_cl_se', 'lik_ivw_cl_k_used'),
                                  ('IVW, local IK, robust BC', 'lik_ivw_rb_tau', 'lik_ivw_rb_se', 'lik_ivw_rb_k_used'),
                                  ('Simple average, local IK', 'lik_mean_cl_tau', 'lik_mean_cl_se', 'lik_mean_cl_k_used'),
                                  ('Simple average, local IK, robust BC', 'lik_mean_rb_tau', 'lik_mean_rb_se', 'lik_mean_rb_k_used')]:
            rows.append(dict(design=design, lam=lam, K=K, estimator=label,
                             omitted_sites_mean=0. if used is None else float(np.mean(K - g[used])),
                             reps_all_sites=500 if used is None else int((g[used] == K).sum()),
                             **metrics(g[t], g[s], g.tau)))
    pd.DataFrame(rows).to_csv(OUT / 'local_ik_s1s2_summary.csv', index=False)
    print(pd.DataFrame(rows).query("~estimator.str.contains('robust')")[['design', 'lam', 'K', 'estimator', 'bias', 'sd', 'rmse', 'mcse_rmse']].to_string())


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--jobs', type=int, default=7); ap.add_argument('--summary', action='store_true')
    a = ap.parse_args()
    if not a.summary: compute(a.jobs)
    summary()
