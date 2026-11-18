# -*- coding: utf-8 -*-
"""Heterogeneous effects with unequal site sizes (design S3, Table 7), coverage under quadratic and quintic
curvature (Table 5, Supplementary Figure C.1) and regional estimates; every display is CSV-driven.
python paper/extensions_s3_curvature.py --jobs 8
python paper/extensions_s3_curvature.py --tables-only
S4/S5 reuse the 500-replication records. S3 local IK fits regenerate the same draws from run_simulation.make;
they do not overwrite sim_raw.csv.
"""
import os
for _v in ('OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'OMP_NUM_THREADS'):
    os.environ[_v] = '1'
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import norm
from joblib import Parallel, delayed

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'simulation')]
from run_simulation import make, BASE_SEED
from fsrd import fsrd, centralized, site_estimates, meta_ivw, meta_mean
import dgp

OUT = ROOT / 'results' / 'extensions'
Z = norm.ppf(.975)
FIELDS = ('tau_cl', 'tau_bc', 'se_cl', 'se_rb', 'h', 'b')
LABELS = {'IK conventional': ('ik_tau_cl', 'ik_se_cl'),
          'IK robust BC': ('ik_tau_bc', 'ik_se_rb'),
          'CCT conventional': ('fed_tau_cl', 'fed_se_cl'),
          'CCT robust BC': ('fed_tau_bc', 'fed_se_rb')}


def metrics(t, se, truth):
    t, se, truth = np.asarray(t), np.asarray(se), np.asarray(truth)
    ok = np.isfinite(t) & np.isfinite(se) & (se > 0)
    e = t[ok] - truth[ok]
    n = len(e)
    if not n:
        raise ValueError('No valid estimates')
    cover = np.mean(np.abs(e) <= Z * se[ok])
    rmse = np.sqrt(np.mean(e ** 2))
    return dict(reps=n, bias=e.mean(), SD=np.std(t[ok], ddof=1), RMSE=rmse,
                MCSE_bias=np.std(e, ddof=1) / np.sqrt(n),
                MCSE_RMSE=np.std(e ** 2, ddof=1) / (2 * rmse * np.sqrt(n)),
                coverage=cover, MCSE_coverage=np.sqrt(cover * (1 - cover) / n),
                length=np.mean(2 * Z * se[ok]))


def local_one(K, rep):
    sites, truth = make('S3', K, 0., rep)
    rows = site_estimates(sites, bw='ik', covs=True, support=dgp.SUPPORT['S3'])
    row = dict(K=K, rep=rep, tau=truth, failed_sites=sum(not r['ok'] for r in rows))
    for kind in ('cl', 'rb'):
        for name, fn in [('ivw', meta_ivw), ('mean', meta_mean)]:
            r = fn(rows, kind)
            row.update({f'{name}_{kind}_tau': r['tau'], f'{name}_{kind}_se': r['se'],
                        f'{name}_{kind}_k_used': r['k_used']})
    for r in rows:
        r.update(K=K, rep=rep)
    return row, rows


def chunk(K, reps):
    return [local_one(K, rep) for rep in reps]


def compute(jobs):
    OUT.mkdir(parents=True, exist_ok=True)
    raw_path = ROOT / 'results/sim/sim_raw.csv'
    raw = pd.read_csv(raw_path)
    checks = []
    # Check regeneration, stored IK values, and federation against pooling.
    for K in (5, 10, 20, 40):
        for rep in range(5):
            S, truth = make('S3', K, 0., rep)
            opts = dict(bw='ik', covs=True, support=dgp.SUPPORT['S3'])
            f, c = fsrd(S, **opts), centralized(S, **opts)
            old = raw[(raw.design == 'S3') & (raw.K == K) & (raw.rep == rep)].iloc[0]
            assert abs(truth - old.tau) < 1e-12
            ds = max(abs(f[k] - old['ik_' + k]) for k in FIELDS)
            dc = max(abs(f[k] - c[k]) for k in FIELDS)
            assert ds < 1e-10 and dc < 1e-10, (K, rep, ds, dc)
            checks.append(dict(K=K, rep=rep, maxdiff_stored=ds, maxdiff_central=dc))
    pd.DataFrame(checks).to_csv(OUT / 's3_regeneration_checks.csv', index=False)
    tasks = [(K, list(range(s, min(s + 25, 500)))) for K in (5, 10, 20, 40) for s in range(0, 500, 25)]
    done = Parallel(n_jobs=jobs, verbose=10)(delayed(chunk)(*a) for a in tasks)
    rows = [x for part in done for x in part]
    pd.DataFrame([x[0] for x in rows]).sort_values(['K', 'rep']).to_csv(OUT / 's3_local_ik_raw.csv', index=False)
    pd.DataFrame([r for x in rows for r in x[1]]).sort_values(['K', 'rep', 'site']).to_csv(OUT / 's3_site_ik_raw.csv', index=False)
    sources = [raw_path, ROOT / 'results/charls_ik/regions.csv']
    (OUT / 'provenance.json').write_text(json.dumps(dict(seed=BASE_SEED, repetitions=500,
        seed_sequence='[20261011, 3, K, 0, rep]', bandwidth='IK, b=h; local sites select their own h',
        sources={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}), indent=2), encoding='utf-8')


def summarize():
    raw = pd.read_csv(ROOT / 'results/sim/sim_raw.csv')
    curve = []
    for (design, K), g in raw[raw.design.isin(['S4', 'S5'])].groupby(['design', 'K']):
        assert len(g) == 500
        for label, (t, s) in LABELS.items():
            curve.append(dict(design=design, K=K, method=label, **metrics(g[t], g[s], g.tau)))
    pd.DataFrame(curve).to_csv(OUT / 'curvature_summary.csv', index=False)
    old = raw[raw.design == 'S3']
    local = pd.read_csv(OUT / 's3_local_ik_raw.csv')
    merged = old[['K', 'rep', 'tau', *['ik_' + f for f in FIELDS]]].merge(local.drop(columns='tau'), on=['K', 'rep'], validate='one_to_one')
    assert len(merged) == 2000
    data = []
    for K, g in merged.groupby('K'):
        assert len(g) == 500 and np.allclose(g.tau, dgp.s3_truth(K)[0], atol=1e-12, rtol=0)
        for name, t, s, used in [('FSRD / pooled IK', 'ik_tau_bc', 'ik_se_rb', None),
                                ('IVW, local IK', 'ivw_rb_tau', 'ivw_rb_se', 'ivw_rb_k_used'),
                                ('Local Average, local IK', 'mean_rb_tau', 'mean_rb_se', 'mean_rb_k_used')]:
            data.append(dict(K=K, method=name, truth=g.tau.iloc[0],
                omitted_sites_mean=0 if used is None else float(np.mean(K - g[used])),
                reps_all_sites=500 if used is None else int((g[used] == K).sum()),
                **metrics(g[t], g[s], g.tau)))
    pd.DataFrame(data).to_csv(OUT / 's3_summary.csv', index=False)
    region = pd.read_csv(ROOT / 'results/charls_ik/regions.csv')
    # No new estimators or changes of outcome, wave, covariate set, or scale.
    region.to_csv(OUT / 'region_forest.csv', index=False)


def displays():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10, 'axes.spines.top': False,
                         'axes.spines.right': False, 'pdf.fonttype': 42})
    df = pd.read_csv(OUT / 'curvature_summary.csv')
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5), sharey=True)
    styles = [('IK conventional', '#b75b32', 'o'), ('IK robust BC', '#236b9a', 's'),
              ('CCT conventional', '#aa7b11', '^'), ('CCT robust BC', '#51753b', 'D')]
    for ax, design, title in zip(axes, ['S5', 'S4'], ['(a) Quadratic curvature (S5)', '(b) Lee quintic design (S4)']):
        for j, (label, color, marker) in enumerate(styles):
            g = df[(df.design == design) & (df.method == label)].sort_values('K')
            ax.errorbar(np.arange(4) + (j - 1.5) * .07, g.coverage, yerr=1.96 * g.MCSE_coverage,
                        color=color, marker=marker, markersize=4, linewidth=1.1, capsize=2, label=label)
        ax.axhline(.95, color='.3', ls='--', lw=1)
        ax.set_xticks(range(4), [5, 10, 20, 40]); ax.set_xlabel('Number of sites K')
        ax.set_title(title, fontsize=11); ax.set_ylim(.30, 1.01); ax.grid(axis='y', alpha=.2)
    axes[0].set_ylabel('Coverage of nominal 95% interval')
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=4, fontsize=9, frameon=False)
    fig.tight_layout(rect=(0, .11, 1, 1))
    for ext in ['png', 'pdf']: fig.savefig(OUT / ('curvature_coverage.' + ext), dpi=240, bbox_inches='tight')
    plt.close(fig)
    reg = pd.read_csv(OUT / 'region_forest.csv')
    names = {'东北': 'Northeast', '华东': 'East', '华中': 'Central', '华北': 'North',
             '华南': 'South', '西北': 'Northwest', '西南': 'Southwest', 'All seven regions': 'All seven regions'}
    fig, ax = plt.subplots(figsize=(8.5, 4.1))
    for i, r in reg.iterrows():
        pooled = r.site == 'All seven regions'
        ax.errorbar(r.tau_bc, i, xerr=[[r.tau_bc - r.ci_rb_lo], [r.ci_rb_hi - r.tau_bc]],
                    fmt='D' if pooled else 'o', color='#236b9a' if pooled else '#555555', capsize=3, markersize=6 if pooled else 5)
    ax.set_yticks(range(len(reg)), [f'{names[r.site]}   (N={int(r.N):,}; h={r.h:.2f})' for _, r in reg.iterrows()])
    ax.invert_yaxis(); ax.axvline(0, color='.5', ls='--', lw=1)
    ax.set_xlabel('Bias-corrected health-score jump and robust 95% CI')
    ax.set_xlim(-.95, .95); ax.grid(axis='x', alpha=.2)
    fig.tight_layout()
    for ext in ['png', 'pdf']: fig.savefig(OUT / ('region_forest.' + ext), dpi=240, bbox_inches='tight')
    plt.close(fig)
    # Display table consumed by the manuscript builder; full precision retained above.
    s3 = pd.read_csv(OUT / 's3_summary.csv')
    table = []
    for _, r in s3.iterrows():
        table.append({'K': int(r.K), 'Estimator (IK, robust BC)': r.method.replace(', local IK', '').replace(' / pooled IK', ' / pooled'),
            'Bias': f'{r.bias:.3f}', 'SD': f'{r.SD:.3f}',
            'RMSE (MCSE)': f'{r.RMSE:.3f} ({r.MCSE_RMSE:.3f})',
            'Coverage (MCSE)': f'{r.coverage:.3f} ({r.MCSE_coverage:.3f})',
            'Mean CI length': f'{r.length:.3f}'})
    pd.DataFrame(table).to_csv(OUT / 's3_manuscript_table.csv', index=False)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--jobs', type=int, default=8)
    ap.add_argument('--tables-only', action='store_true')
    args = ap.parse_args()
    start = time.perf_counter()
    if not args.tables_only: compute(args.jobs)
    summarize()
    displays()
    print(f'Completed: {OUT} ({time.perf_counter() - start:.1f} s)', flush=True)
