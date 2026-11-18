# -*- coding: utf-8 -*-
"""Small-cell rule for the 2018 self-rated health score, IK bandwidth with b = h (Supplementary Table C.9).
    python paper/smallcell_rule.py   -> results/ik_benchmarks/smallcell_srh_2018.csv
Same rule as charls/value_of_collaboration.small_cell: a site does not upload a window with fewer than m
records on one side. Reads the local CHARLS analysis file; only aggregate results are written.
"""
import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'charls')]
from fsrd import fsrd  # noqa: E402
import run_charls as RC  # noqa: E402

OUT = ROOT / 'results' / 'ik_benchmarks'
SMALL_CELL = (0, 5, 10, 20, 50)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    RC.configure(2018); df = RC.load(); rows = []
    for by in ('province', 'region'):
        S = RC.sites_of(df, 'srh', by=by, fuzzy=False)
        for m in SMALL_CELL:
            r = fsrd(S, **{**RC.OPT, 'bw': 'ik'}, min_cell=m)
            rows.append(dict(wave=2018, sites=by, outcome='srh', design='sharp', min_cell=m, K=len(S), N=int(sum(s.n for s in S)),
                             tau_bc=r['tau_bc'], se_rb=r['se_rb'], h=r['h'], n_h=int(sum(r['n_h'])),
                             suppressed=int(r.get('suppressed', 0)), suppressed_final=int(r.get('suppressed_final', 0))))
            print(rows[-1], flush=True)
    pd.DataFrame(rows).to_csv(OUT / 'smallcell_srh_2018.csv', index=False)


if __name__ == '__main__':
    main()
