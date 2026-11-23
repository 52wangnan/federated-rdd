# -*- coding: utf-8 -*-
"""Re-run everything: tests, simulations, communication benchmark, CHARLS extraction and analysis
(2018 wave, and the 2015 wave as a birth-cohort check), the decision analysis (value of collaboration), the
Holm multiple-testing table, and all tables / figures.

    python run_all.py                  # full run (about 25 minutes on 20 cores)
    python run_all.py --quick          # smoke run: 40 replications -> results/sim_quick/ (main results untouched)
    python run_all.py --tables-only    # only regenerate tables and figures from the result files
    python run_all.py --skip-sim       # tests + CHARLS only
    python run_all.py --raw data/raw   # location of the raw CHARLS files
    python run_all.py --paper          # IK analyses and displays reported in the paper (after a full run)
    python run_all.py --paper --tables-only   # only redraw the paper displays from the stored results
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
    ap.add_argument('--paper', action='store_true', help='regenerate the IK analyses and displays reported in the paper from the '
                    'existing simulation records and the extracted CHARLS data')
    ap.add_argument('--tables-only', action='store_true')
    ap.add_argument('--skip-sim', action='store_true')
    ap.add_argument('--skip-charls', action='store_true')
    ap.add_argument('--jobs', type=int, default=None)
    ap.add_argument('--raw', default='data/raw')
    a = ap.parse_args()
    if a.paper:
        if not a.tables_only:
            step(['tests/run_tests.py'], 'unit tests')
            step(['paper/charls_ik_analysis.py'], 'CHARLS IK analyses (Table 9, Figures 2 and 3)')
            step(['paper/smallcell_rule.py'], 'small-cell rule (Supplementary Table C.9)')
            step(['paper/comm_ik.py'], 'communication and run time of the IK pipeline (Table 6)')
        extension = ['paper/extensions_s3_curvature.py']
        if a.tables_only:
            extension += ['--tables-only']
        elif a.jobs:
            extension += ['--jobs', str(a.jobs)]
        step(extension, 'curvature coverage, heterogeneous effects, regions (Tables 5 and 7)')
        coalition = ['paper/composition_coalitions.py']
        if a.tables_only:
            coalition += ['--tables-only']
        elif a.jobs:
            coalition += ['--jobs', str(a.jobs)]
        step(coalition, 'site composition and coalition precision (Table 8)')
        local = ['paper/local_ik_benchmarks.py'] + (['--summary'] if a.tables_only else (['--jobs', str(a.jobs)] if a.jobs else []))
        step(local, 'local IK benchmarks for designs S1 and S2 (Tables 2 and 4)')
        step(['paper/figures_charls.py'], 'Figure 4 and Supplementary Figure C.2')
        step(['paper/figures_bandwidth.py'], 'Supplementary Figures C.3 and C.4')
        step(['paper/figures_vector.py'], 'vector versions of Figures 2-4')
        return
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
