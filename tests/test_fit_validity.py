"""Regression tests for degenerate covariate windows and feasible redundancy."""
from pathlib import Path
import sys
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'simulation')]
from fsrd import Site, fsrd, centralized, site_estimates
from fsrd.lprd import InvalidFit
from run_simulation import make


def test_saturated_covariate_windows_rejected():
    for rep, index in [(384, 1), (296, 2), (292, 1), (306, 2), (464, 2)]:
        sites, _ = make('S3', 40, 0., rep)
        s = sites[index]
        row = site_estimates([s], support=(-1, 1))[0]
        assert not row['ok'], (rep, row)
        assert 'saturated_joint_design' in row['failure_reason']
        assert np.isnan(row['se_cl'])


def test_unidentified_contrast_rejected_and_redundancy_retained():
    rng = np.random.default_rng(20261102)
    x = np.linspace(-.99, .99, 300)
    w = rng.normal(size=len(x))
    y = 1 + (x >= 0) + x + .3*w + rng.normal(0, .1, len(x))
    # A covariate equal to treatment makes the RD contrast unidentified.
    try:
        fsrd([Site(x, y, w=(x >= 0))], bw=.8)
    except InvalidFit as e:
        assert 'unidentified_rd_contrast' in str(e)
    else:
        raise AssertionError('unidentified RD contrast was accepted')
    # Duplicated covariates do not alter the target and can safely be dropped.
    sites = [Site(x[j::4], y[j::4], w=np.column_stack([w[j::4], 2*w[j::4]])) for j in range(4)]
    f, c = fsrd(sites, bw=.8), centralized(sites, bw=.8)
    one = fsrd([Site(x, y, w=w)], bw=.8)
    assert f['fit_diagnostics']['covariates_dropped'] == 1
    for k in ['tau_cl', 'tau_bc', 'se_cl', 'se_rb']:
        assert f[k] == c[k]
        assert abs(f[k] - one[k]) < 1e-10


def test_sd_mcse_handles_heavy_tail_and_missing_replication():
    from make_sim_tables import metrics
    t = np.r_[np.linspace(-1, 1, 499), 80.]
    s = np.ones(500)
    m = metrics(t, s, np.zeros(500))
    assert m['sd_mcse'] > 10 * m['sd'] / np.sqrt(2*499)
    assert abs(metrics(t+7, s, np.full(500, 7.))['sd_mcse']-m['sd_mcse']) < 1e-12
    t[0] = np.nan
    assert metrics(t, s, np.zeros(500))['R'] == 499
