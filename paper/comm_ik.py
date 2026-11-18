# -*- coding: utf-8 -*-
"""Communication and run time of the IK pipeline (Table 6), next to the CCT-type pipeline.
    python paper/comm_ik.py   -> results/ik_benchmarks/comm_ik.csv, results/ik_benchmarks/comm_rounds_ik_K10_p10.csv
Same data and grid as simulation/run_comm.py (N = 10,000; K in {2,5,10,20,40,60}; p in {0,3,10,26}).
Each analysis is timed as the median of 5 single-process runs on the machine that runs this script, for
both pipelines, so that the IK and CCT-type times come from the same machine.
"""
import os
for _v in ('OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'OMP_NUM_THREADS'):
    os.environ[_v] = '1'
import platform
import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'simulation')]
from fsrd import fsrd, centralized  # noqa: E402
from run_comm import sites_for, timed, KS, PS, SUPPORT, LATENCY  # noqa: E402

OUT = ROOT / 'results' / 'ik_benchmarks'


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for p in PS:
        for K in KS:
            S = sites_for(K, p); covs = p > 0
            ik, t_ik = timed(lambda: fsrd(S, bw='ik', covs=covs, support=SUPPORT))
            ikc, t_ikc = timed(lambda: centralized(S, bw='ik', covs=covs, support=SUPPORT))
            cct, t_cct = timed(lambda: fsrd(S, covs=covs, support=SUPPORT))
            cctc, t_cctc = timed(lambda: centralized(S, covs=covs, support=SUPPORT))
            nk = np.array([s.n for s in S]); c = ik['comm']
            rows.append(dict(K=K, p=p, N=int(nk.sum()), ik_rounds=c['rounds'], ik_bytes_up_site=c['bytes_up_per_site'],
                             ik_bytes_up_total=c['bytes_up_total'], ik_bytes_down_total=c['bytes_down_total'],
                             micro_bytes_site=float(nk.max() * (p + 2) * 8), micro_bytes_total=float(nk.sum() * (p + 2) * 8),
                             t_ik_fed=t_ik, t_ik_pooled=t_ikc, t_ik_sites_parallel=c['t_local'],
                             t_ik_fed_net50=t_ik + LATENCY * c['rounds'],
                             cct_rounds=cct['comm']['rounds'], cct_bytes_up_site=cct['comm']['bytes_up_per_site'],
                             t_cct_fed=t_cct, t_cct_pooled=t_cctc,
                             absdiff_ik_tau=abs(ik['tau_bc'] - ikc['tau_bc']), absdiff_ik_se=abs(ik['se_rb'] - ikc['se_rb']),
                             absdiff_ik_h=abs(ik['h'] - ikc['h'])))
            print({k: rows[-1][k] for k in ('K', 'p', 'ik_rounds', 'ik_bytes_up_site', 't_ik_fed', 't_ik_pooled', 'absdiff_ik_tau')}, flush=True)
    df = pd.DataFrame(rows); df['machine'] = f'{platform.system()} {platform.machine()}, Python {platform.python_version()}'
    df.to_csv(OUT / 'comm_ik.csv', index=False)
    r = fsrd(sites_for(10, 10), bw='ik', covs=True, support=SUPPORT)
    tab = pd.DataFrame(r['comm_rounds']); tab['bytes_up_per_site'] = tab['floats_up_per_site'] * 8
    tab.to_csv(OUT / 'comm_rounds_ik_K10_p10.csv', index=False)
    print(tab)


if __name__ == '__main__':
    main()
