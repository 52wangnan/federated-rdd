# -*- coding: utf-8 -*-
"""Re-run everything: tests, simulations, communication benchmark, CHARLS extraction and analysis
(2018 wave, and the 2015 wave as a birth-cohort check), the decision analysis (value of collaboration), the
Holm multiple-testing table, and all tables / figures.

    python run_all.py                  # full run (about 25 minutes on 20 cores)
    python run_all.py --quick          # smoke run: 40 replications -> results/sim_quick/ (main results untouched)
    python run_all.py --tables-only    # only regenerate tables and figures from the result files
    python run_all.py --skip-sim       # tests + CHARLS only
    python run_all.py --raw /path/to/CHARLS                # location of the raw CHARLS files
    python run_all.py --skip-charls                         # no CHARLS access: tests + simulations only
"""
import argparse
import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.abspath(__file__))


def step(args, title):
    print(f'\n=== {title}: python {" ".join(args)}', flush=True)
    t0 = time.time()
    env = dict(os.environ, PYTHONIOENCODING='utf-8', PYTHONWARNINGS='ignore')
    r = subprocess.run([sys.executable] + args, cwd=ROOT, env=env)
    if r.returncode != 0:
        sys.exit(f'step failed: {title}')
    print(f'=== {title} done in {time.time() - t0:.0f} s', flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--quick', action='store_true')
    ap.add_argument('--tables-only', action='store_true')
    ap.add_argument('--skip-sim', action='store_true')
    ap.add_argument('--skip-charls', action='store_true')
    ap.add_argument('--jobs', type=int, default=None)
    ap.add_argument('--raw', default=None, help='folder with the raw CHARLS files (charls.pku.edu.cn)')
    a = ap.parse_args()
    if not a.skip_charls and not a.tables_only and a.raw is None:
        sys.exit('CHARLS data are not distributed with this repository: pass --raw <folder> or use --skip-charls')
    if not a.tables_only:
        step(['tests/run_tests.py'], 'unit tests')
    if not a.skip_sim:
        if a.quick:
            # smoke run: written to results/sim_quick/ so the 500-replication results are never overwritten
            sim = ['simulation/run_simulation.py', '--reps', '40', '--out', 'results/sim_quick/sim_raw.csv']
            if a.jobs:
                sim += ['--jobs', str(a.jobs)]
            step(sim, 'Monte Carlo simulation (quick, 40 replications)')
        else:
            if not a.tables_only:
                sim = ['simulation/run_simulation.py', '--reps', '500']
                if a.jobs:
                    sim += ['--jobs', str(a.jobs)]
                step(sim, 'Monte Carlo simulation')
                step(['simulation/run_comm.py'], 'communication / run-time benchmark')
            step(['simulation/make_sim_tables.py'], 'simulation tables and figures')
    if not a.skip_charls:
        if not a.tables_only:
            step(['charls/extract.py', '--raw', a.raw], 'CHARLS extraction')
            step(['charls/run_charls.py'], 'CHARLS analysis')
            step(['charls/extract_2015.py', '--raw', a.raw], 'CHARLS 2015 extraction')
            step(['charls/run_charls.py', '--wave', '2015'], 'CHARLS 2015 analysis')
            step(['charls/value_of_collaboration.py'], 'decision analysis: coalitions, Shapley values, small-cell rule')
        step(['charls/make_charls_tables.py'], 'CHARLS tables and figures')
        step(['charls/make_charls_tables.py', '--wave', '2015'], 'CHARLS 2015 tables and figures')
        step(['charls/compare_waves.py'], 'CHARLS 2015 versus 2018')
        step(['charls/multiple_testing.py'], 'Holm-adjusted p-values (both waves)')
        step(['charls/make_value_tables.py'], 'decision-analysis tables and figures')
    print('\nall steps finished')


if __name__ == '__main__':
    main()
