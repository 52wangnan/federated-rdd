# -*- coding: utf-8 -*-
"""Unit tests for fsrd.  Run with  python tests/run_tests.py  (or pytest, if installed)."""
import os
import re
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from fsrd import Site, fsrd, centralized, pool_sites, mccrary, boundary_constants, Federation  # noqa: E402
from fsrd import reference as R  # noqa: E402
from fsrd.bandwidth import bw_ik, quantile_from_hist  # noqa: E402

TOL_FED = 1e-10          # federated vs centralised (task requirement)
TOL_REF = 1e-8           # federated vs observation-level reference in original units
KERNELS = ('triangular', 'uniform', 'epanechnikov')


class Skip(Exception):
    pass


def _split(n, K, rng):
    return rng.integers(0, K, n)


def _sites(x, y, lab, K, d=None, w=None, cl=None):
    return [Site(x[lab == k], y[lab == k], d=None if d is None else d[lab == k],
                 w=None if w is None else w[lab == k], cl=None if cl is None else cl[lab == k]) for k in range(K)]


def _data(seed=0, N=6000, K=7, curved=True):
    rng = np.random.default_rng(seed)
    x = rng.uniform(-1, 1, N)
    W = rng.normal(0, 1, (N, 3))
    m = 0.5 + 0.8 * x + (-1.5 * x ** 2 if curved else 0.0)
    D = ((x >= 0) & (rng.uniform(size=N) < 0.8)) | ((x < 0) & (rng.uniform(size=N) < 0.15))
    y = m + 1.0 * (x >= 0) + W @ np.array([0.5, -0.3, 0.2]) + rng.normal(0, 0.3, N)
    yf = m + 0.7 * D + W @ np.array([0.5, -0.3, 0.2]) + rng.normal(0, 0.3, N)
    cl = rng.integers(0, 200, N)
    lab = _split(N, K, rng)
    cl = cl * K + lab                                  # clusters nested in sites
    return x, y, yf, D.astype(float), W, cl, lab, K


KEYS = ('tau_cl', 'tau_bc', 'se_cl', 'se_rb', 'h', 'b')


def _maxdiff(a, b, keys=KEYS):
    return max(abs(a[k] - b[k]) for k in keys)


# ----------------------------------------------------------------------------- (a) exactness
def test_a_noiseless_linear_exact_any_bandwidth():
    """Noise-free linear DGP: every bandwidth and kernel must return the true jump exactly,
    for the conventional and the bias-corrected estimator, with and without covariates."""
    rng = np.random.default_rng(1)
    N, K = 3000, 4
    x = rng.uniform(-1, 1, N)
    W = rng.normal(0, 1, (N, 2))
    lab = _split(N, K, rng)
    worst = 0.0
    for tau in (1.0, -0.37):
        y = 0.5 + tau * (x >= 0) + 1.3 * x - 0.4 * (x >= 0) * x
        yw = y + W @ np.array([0.7, -1.1])
        for h in (0.05, 0.1, 0.3, 0.7, 1.5, 3.0):
            for kern in KERNELS:
                for covs, yy in ((False, y), (True, yw)):
                    r = fsrd(_sites(x, yy, lab, K, w=W), bw=(h, 1.3 * h), kernel=kern, covs=covs)
                    worst = max(worst, abs(r['tau_cl'] - tau), abs(r['tau_bc'] - tau))
    assert worst < 1e-9, worst


def test_a_noiseless_fuzzy_exact():
    """Y = 1 + 2 D + x exactly: the fuzzy estimator must return 2 at any bandwidth."""
    rng = np.random.default_rng(2)
    N, K = 4000, 5
    x = rng.uniform(-1, 1, N)
    D = (((x >= 0) & (rng.uniform(size=N) < 0.85)) | ((x < 0) & (rng.uniform(size=N) < 0.1))).astype(float)
    y = 1 + 2 * D + x
    lab = _split(N, K, rng)
    for h in (0.1, 0.4, 1.0):
        r = fsrd(_sites(x, y, lab, K, d=D), bw=h)
        assert abs(r['tau_cl'] - 2) < 1e-9 and abs(r['tau_bc'] - 2) < 1e-9, (h, r['tau_cl'], r['tau_bc'])


# ----------------------------------------------------------------------------- (b) federated == centralised
def test_b_federated_equals_centralized_sharp():
    x, y, yf, D, W, cl, lab, K = _data(3)
    for covs in (False, True):
        for bw in ('mserd', 'ik', 0.35):
            f = fsrd(_sites(x, y, lab, K, w=W), bw=bw, covs=covs)
            c = centralized(_sites(x, y, lab, K, w=W), bw=bw, covs=covs)
            assert _maxdiff(f, c) < TOL_FED, (covs, bw, _maxdiff(f, c))


def test_b_federated_equals_centralized_fuzzy():
    x, y, yf, D, W, cl, lab, K = _data(4)
    for covs in (False, True):
        f = fsrd(_sites(x, yf, lab, K, d=D, w=W), covs=covs)
        c = centralized(_sites(x, yf, lab, K, d=D, w=W), covs=covs)
        assert _maxdiff(f, c) < TOL_FED
        for part in ('itt', 'fs'):
            assert max(abs(f[part][k] - c[part][k]) for k in ('tau_cl', 'tau_bc', 'se_cl', 'se_rb')) < TOL_FED


def test_b_federated_equals_centralized_vce_options():
    x, y, yf, D, W, cl, lab, K = _data(5)
    for vce in ('hc0', 'hc1', 'hc2', 'hc3', 'cluster'):
        f = fsrd(_sites(x, y, lab, K, w=W, cl=cl), vce=vce)
        c = centralized(_sites(x, y, lab, K, w=W, cl=cl), vce=vce)
        assert _maxdiff(f, c) < TOL_FED, (vce, _maxdiff(f, c))


def test_b_partition_invariance():
    """The same records split into 1, 2, 5, 20 or 60 sites give the same answer."""
    x, y, yf, D, W, cl, lab, K = _data(6, N=8000)
    rng = np.random.default_rng(7)
    base = fsrd([Site(x, y, w=W)], covs=True)
    for K2 in (2, 5, 20, 60):
        lab2 = _split(len(x), K2, rng)
        r = fsrd(_sites(x, y, lab2, K2, w=W), covs=True)
        assert _maxdiff(r, base) < TOL_FED, (K2, _maxdiff(r, base))


def test_b_bitwise_identical():
    """With double-double sums the federated and pooled computations agree bit for bit
    (sharp/fuzzy, covariates, all variance options, IK and CCT bandwidths, random partitions)."""
    x, y, yf, D, W, cl, lab, K = _data(15)
    rng = np.random.default_rng(16)
    lab2 = _split(len(x), 23, rng)
    cases = [dict(), dict(covs=True), dict(bw='ik'), dict(vce='hc3', covs=True), dict(vce='cluster'),
             dict(vce='cluster_grid', grid=np.unique(cl))]
    for kw in cases:
        for L_, K_ in ((lab, K), (lab2, 23)):
            if kw.get('vce') == 'cluster' and K_ == 23:
                continue                      # clusters are nested in the 7-site partition only
            f = fsrd(_sites(x, y, L_, K_, w=W, cl=cl), **kw)
            c = centralized(_sites(x, y, L_, K_, w=W, cl=cl), **kw)
            assert all(f[k] == c[k] for k in KEYS), (kw, {k: f[k] - c[k] for k in KEYS})
    f = fsrd(_sites(x, yf, lab2, 23, d=D, w=W), covs=True)
    c = centralized(_sites(x, yf, lab2, 23, d=D, w=W), covs=True)
    assert all(f[k] == c[k] for k in KEYS)


def test_b_matches_rdrobust_transcription():
    """Federated estimates equal an observation-level transcription of rdrobust's code
    (estimation step and mserd bandwidth) and plain weighted least squares."""
    x, y, yf, D, W, cl, lab, K = _data(8)
    for covs in (None, W):
        f = fsrd(_sites(x, y, lab, K, w=W), covs=covs is not None)
        h, b, _ = R.rdbwselect_mserd(y, x, covs=covs, iqr=f['bw_info']['iqr'])
        assert abs(h - f['h']) < TOL_REF and abs(b - f['b']) < TOL_REF, (h, f['h'], b, f['b'])
        ref = R.rdrobust_est(y, x, f['h'], f['b'], covs=covs)
        assert _maxdiff(f, ref, ('tau_cl', 'tau_bc', 'se_cl', 'se_rb')) < TOL_REF
        assert abs(R.local_linear_lstsq(y, x, f['h'], covs=covs) - f['tau_cl']) < TOL_REF
    for vce in ('hc0', 'hc2', 'hc3'):
        f = fsrd(_sites(x, y, lab, K, w=W), bw=(0.3, 0.5), vce=vce)
        ref = R.rdrobust_est(y, x, 0.3, 0.5, covs=W, vce=vce)
        assert _maxdiff(f, ref, ('tau_cl', 'tau_bc', 'se_cl', 'se_rb')) < TOL_REF, vce
    f = fsrd(_sites(x, y, lab, K, w=W, cl=cl), bw=(0.3, 0.5), vce='cluster')
    ref = R.rdrobust_est(y, x, 0.3, 0.5, covs=W, vce='cluster', cluster=cl)
    assert _maxdiff(f, ref, ('tau_cl', 'tau_bc', 'se_cl', 'se_rb')) < TOL_REF
    f = fsrd(_sites(x, yf, lab, K, d=D, w=W), bw=(0.3, 0.5))
    ref = R.rdrobust_est(yf, x, 0.3, 0.5, fuzzy=D, covs=W)
    assert _maxdiff(f, ref, ('tau_cl', 'tau_bc', 'se_cl', 'se_rb')) < TOL_REF


def test_b_ik_bandwidth_formula():
    """IK (2012) bandwidth recomputed directly on the pooled records (h2 with the 1/7 power)."""
    x, y, yf, D, W, cl, lab, K = _data(9)
    h, info = bw_ik(Federation(_sites(x, y, lab, K)))
    N = len(x)
    h1 = 1.84 * np.std(x, ddof=1) * N ** (-0.2)
    L, Rr = (x >= -h1) & (x < 0), (x >= 0) & (x <= h1)
    f = (L.sum() + Rr.sum()) / (2 * N * h1)
    s2 = np.array([np.var(y[L], ddof=1), np.var(y[Rr], ddof=1)])
    ml, mr = info['med']                                  # binned side medians (shared routine)
    sel = (x >= ml) & (x <= mr)
    Xc = np.column_stack([np.ones(sel.sum()), x[sel] >= 0, x[sel], x[sel] ** 2, x[sel] ** 3])
    m3 = 6 * np.linalg.lstsq(Xc, y[sel], rcond=None)[0][4]
    Nside = np.array([(x < 0).sum(), (x >= 0).sum()])
    h2 = 3.56 * (s2 / (f * m3 ** 2)) ** (1 / 7) * Nside ** (-1 / 7)
    m2, n2 = [], []
    for j, msk in enumerate(((x >= -h2[0]) & (x < 0), (x >= 0) & (x <= h2[1]))):
        m2.append(2 * np.linalg.lstsq(np.vander(x[msk], 3, increasing=True), y[msk], rcond=None)[0][2])
        n2.append(msk.sum())
    r = 2160 * s2 / (np.array(n2) * h2 ** 4)
    href = 3.4375 * ((s2.sum()) / (f * ((m2[1] - m2[0]) ** 2 + r.sum()))) ** 0.2 * N ** (-0.2)
    assert abs(href - h) < 1e-8 * max(1, h), (href, h)
    assert abs(info['h2_minus'] - h2[0]) < 1e-8 and abs(info['h2_plus'] - h2[1]) < 1e-8
    # side medians from the histogram are within one bin of the exact medians
    assert abs(ml - np.median(x[x < 0])) < 2 / 128 and abs(mr - np.median(x[x >= 0])) < 2 / 128


def test_b_density_test_federated():
    rng = np.random.default_rng(10)
    N = 20000
    xm = rng.integers(-180, 400, N)                     # running variable in months, cutoff at 0
    x = xm / 12.0
    lab = _split(N, 9, rng)
    S = _sites(x, np.zeros(N), lab, 9)
    a = mccrary(S, 1 / 12, x.min(), x.max())
    b = mccrary([pool_sites(S)], 1 / 12, x.min(), x.max())
    assert abs(a['theta'] - b['theta']) < 1e-12 and abs(a['se'] - b['se']) < 1e-12
    assert a['p'] > 0.001          # uniform months: no jump


def test_b_cluster_nesting_is_checked():
    x, y, yf, D, W, cl, lab, K = _data(17)
    lab2 = _split(len(x), 5, np.random.default_rng(18))
    try:
        fsrd(_sites(x, y, lab2, 5, cl=cl), vce='cluster')
    except ValueError:
        return
    raise AssertionError('clusters spanning several sites were not rejected')


def test_b_site_fixed_effects():
    """Site indicators in the common-slope covariates: federated == pooled bit for bit, and in a noise-free
    design whose site composition jumps at the cutoff the FE estimate is exact while the pooled one is not."""
    x, y, yf, D, W, cl, lab, K = _data(19)
    f = fsrd(_sites(x, y, lab, K, w=W), site_fe=True)
    c = centralized(_sites(x, y, lab, K, w=W), site_fe=True)
    assert all(f[k] == c[k] for k in KEYS)
    rng = np.random.default_rng(20)
    K2 = 5
    alpha = np.linspace(-0.5, 0.5, K2)
    r_keep = np.linspace(0.2, 0.8, K2)
    sites = []
    for k in range(K2):
        xk = rng.uniform(-1, 1, 3000)
        keep = np.where(xk >= 0, rng.uniform(size=3000) < r_keep[k], rng.uniform(size=3000) < 1 - r_keep[k])
        xk = xk[keep]
        sites.append(Site(xk, alpha[k] + 1.0 * (xk >= 0) + xk))
    fe = fsrd(sites, bw=(0.4, 0.6), site_fe=True)
    no = fsrd(sites, bw=(0.4, 0.6))
    assert abs(fe['tau_cl'] - 1) < 1e-9 and abs(fe['tau_bc'] - 1) < 1e-9, fe['tau_cl']
    assert abs(no['tau_cl'] - 1) > 0.1


def test_b_site_fixed_effects_reference_site_outside_window():
    """The reference site has no records in the kernel window, so the other site indicators sum to the
    intercept (collinear): the redundant indicator is dropped, federated == pooled bit for bit, and the
    estimate equals the one computed without the reference site (same column space)."""
    rng = np.random.default_rng(22)
    alpha = (0.3, -0.2, 0.1, 0.4)
    sites = []
    for k in range(4):
        xk = rng.uniform(-1, 1, 2000)
        if k == 0:
            xk = xk[np.abs(xk) > 0.6]
        sites.append(Site(xk, alpha[k] + 1.0 * (xk >= 0) + xk - 0.5 * xk ** 2 + rng.normal(0, 0.1, len(xk))))
    f = fsrd(sites, bw=(0.3, 0.45), site_fe=True)
    c = centralized(sites, bw=(0.3, 0.45), site_fe=True)
    assert all(f[k] == c[k] for k in KEYS)
    g = fsrd(sites[1:], bw=(0.3, 0.45), site_fe=True)
    assert _maxdiff(f, g, ('tau_cl', 'tau_bc', 'se_cl', 'se_rb')) < 1e-10
    # the last indicator (site 4) is dropped, i.e. site 4 becomes the reference; same contrasts as g
    gf, gg = f['gamma'][:, 0], g['gamma'][:, 0]
    assert gf[2] == 0.0 and abs((gf[1] - gf[0]) - gg[0]) < 1e-10 and abs(-gf[0] - gg[1]) < 1e-10
    m = fsrd(sites, site_fe=True, support=(-1.0, 1.0))
    assert np.isfinite(m['tau_bc']) and np.isfinite(m['se_rb'])


def test_b_threshold_derivative():
    """deriv = 1 (jump in the slope): exact on a noise-free kink, federated == pooled, matches the
    rdrobust transcription."""
    x, y, yf, D, W, cl, lab, K = _data(21)
    y0 = 1 + 0.5 * x + 0.8 * (x >= 0) * x
    for h in (0.2, 0.5, 1.0):
        r = fsrd(_sites(x, y0, lab, K), bw=(h, 1.5 * h), deriv=1, p=2, q=3)
        assert abs(r['tau_cl'] - 0.8) < 1e-9 and abs(r['tau_bc'] - 0.8) < 1e-9
    f = fsrd(_sites(x, y, lab, K), deriv=1, p=2, q=3)
    c = centralized(_sites(x, y, lab, K), deriv=1, p=2, q=3)
    assert all(f[k] == c[k] for k in KEYS)
    h, b, _ = R.rdbwselect_mserd(y, x, p=2, q=3, deriv=1, iqr=f['bw_info']['iqr'])
    assert abs(h - f['h']) < TOL_REF and abs(b - f['b']) < TOL_REF
    ref = R.rdrobust_est(y, x, f['h'], f['b'], p=2, q=3, deriv=1)
    assert _maxdiff(f, ref, ('tau_cl', 'tau_bc', 'se_cl', 'se_rb')) < TOL_REF


def test_b_public_support_and_mass_points():
    """A public support replaces the sites' min / max (which are then not sent); a public grid gives exact
    quantiles and rdrobust's mass-point adjustment.  Both: federated == pooled, equal to the transcription."""
    x, y, yf, D, W, cl, lab, K = _data(22)
    f = fsrd(_sites(x, y, lab, K), support=(-1.0, 1.0))
    c = centralized(_sites(x, y, lab, K), support=(-1.0, 1.0))
    assert all(f[k] == c[k] for k in KEYS)
    assert f['comm_rounds'][0]['floats_up_per_site'] == 7          # n, sum x (2), sum x^2 (2), side counts (2)
    h, b, _ = R.rdbwselect_mserd(y, x, iqr=f['bw_info']['iqr'], support=(-1.0, 1.0))
    assert abs(h - f['h']) < TOL_REF and abs(b - f['b']) < TOL_REF
    xg, yg, Dg, labg, Kg = _grid_data(23)
    for sup in (None, (-15.0, 40.0)):
        f = fsrd(_sites(xg, yg, labg, Kg), mass_grid=1 / 12, support=sup)
        c = centralized(_sites(xg, yg, labg, Kg), mass_grid=1 / 12, support=sup)
        assert all(f[k] == c[k] for k in KEYS)
        h, b, _ = R.rdbwselect_mserd(yg, xg, masspoints='adjust', support=sup)
        assert abs(h - f['h']) < TOL_REF and abs(b - f['b']) < TOL_REF, (sup, h, f['h'])
    f = fsrd(_sites(xg, yg, labg, Kg, d=Dg), mass_grid=1 / 12, fuzzy_bw='rdrobust')
    h, b, _ = R.rdbwselect_mserd(yg, xg, fuzzy=Dg, masspoints='adjust')
    assert abs(h - f['h']) < TOL_REF and abs(b - f['b']) < TOL_REF


def test_b_small_cell_rule_counts():
    """The small-cell rule zeroes small windows and reports how many it suppressed."""
    x, y, yf, D, W, cl, lab, K = _data(24, N=3000, K=10)
    r = fsrd(_sites(x, y, lab, K), bw=(0.1, 0.15), min_cell=15)
    assert r['suppressed'] > 0 and r['suppressed_final'] > 0
    assert fsrd(_sites(x, y, lab, K), bw=(0.1, 0.15), min_cell=1)['suppressed'] == 0


def test_b_small_cell_rule_off_by_default():
    x, y, yf, D, W, cl, lab, K = _data(11)
    r0 = fsrd(_sites(x, y, lab, K), min_cell=0)
    r1 = fsrd(_sites(x, y, lab, K), min_cell=5)       # every window holds far more than 5 records
    assert _maxdiff(r0, r1) < TOL_FED


def test_b_communication_independent_of_site_size():
    """Bytes uploaded per site do not depend on the number of records (O(q^2) per round)."""
    out = []
    for N in (2000, 20000):
        x, y, yf, D, W, cl, lab, K = _data(12, N=N)
        out.append(fsrd(_sites(x, y, lab, K, w=W), covs=True)['comm'])
    assert out[0]['rounds'] == out[1]['rounds'] == 10
    assert out[0]['bytes_up_per_site'] == out[1]['bytes_up_per_site']


# ----------------------------------------------------------------------------- (c) rdrobust
def test_c_published_rdrobust_senate_example():
    """Senate data shipped with rdrobust (Cattaneo, Frandsen & Titiunik 2015): the published output of
    rdrobust(vote, margin) reports h = 17.754, 360 / 323 effective observations and a conventional
    estimate of 7.414 (Calonico, Cattaneo, Farrell & Titiunik 2017, Stata Journal)."""
    path = os.path.join(HERE, 'fixtures', 'rdrobust_senate.csv')
    d = pd.read_csv(path).dropna(subset=['vote', 'margin'])
    r = fsrd([Site(d['margin'].to_numpy(float), d['vote'].to_numpy(float))], bw=(17.754, 28.028))
    assert r['n_h'] == [360, 323], r['n_h']
    assert abs(r['tau_cl'] - 7.414) < 5e-4, r['tau_cl']


def _rdr(y, x, **kw):
    """rdrobust() output as the fsrd keys (stdout of the package silenced)."""
    import contextlib
    import io
    from rdrobust import rdrobust
    with contextlib.redirect_stdout(io.StringIO()):
        r = rdrobust(y, x, **kw)
    c, s, b = np.asarray(r.coef).ravel(), np.asarray(r.se).ravel(), np.asarray(r.bws)
    return dict(tau_cl=c[0], tau_bc=c[1], se_cl=s[0], se_rb=s[2], h=b[0, 0], b=b[1, 0])


def _grid_data(seed, N=8000, K=6):
    """A discrete running variable on a monthly grid (mass points), sharp and fuzzy outcomes."""
    rng = np.random.default_rng(seed)
    xm = rng.integers(-180, 480, N)
    x = xm / 12.0
    D = (((x >= 0) & (rng.uniform(size=N) < 0.75)) | ((x < 0) & (rng.uniform(size=N) < 0.05))).astype(float)
    y = 0.6 - 0.01 * x + 0.0004 * x ** 2 + 0.3 * D + rng.normal(0, 0.4, N)
    return x, y, D, rng.integers(0, K, N), K


def test_c_rdrobust_python_package():
    """Comparison with the official Python package rdrobust 2.1.1 (vce = hc0-hc3 / cluster; the default
    nearest-neighbour variance cannot be federated and is not compared).  Tolerances are not relaxed:
    1e-10 for estimates and SEs at given (h, b); 1e-8 (relative) for the mserd bandwidths."""
    try:
        import rdrobust  # noqa: F401
    except ImportError:
        raise Skip('Python package rdrobust is not installed (pip install rdrobust)')
    x, y, yf, D, W, cl, lab, K = _data(13)
    est = ('tau_cl', 'tau_bc', 'se_cl', 'se_rb')
    # fixed (h, b): every variance option, covariates, fuzzy, cluster
    for vce in ('hc0', 'hc1', 'hc2', 'hc3'):
        f = fsrd(_sites(x, y, lab, K), bw=(0.3, 0.45), vce=vce)
        assert _maxdiff(f, _rdr(y, x, h=0.3, b=0.45, vce=vce), est) < 1e-10, vce
    f = fsrd(_sites(x, y, lab, K, w=W), bw=(0.3, 0.45), covs=True)
    assert _maxdiff(f, _rdr(y, x, h=0.3, b=0.45, vce='hc1', covs=W), est) < 1e-10
    f = fsrd(_sites(x, yf, lab, K, d=D, w=W), bw=(0.3, 0.45), covs=True)
    assert _maxdiff(f, _rdr(yf, x, h=0.3, b=0.45, vce='hc1', fuzzy=D, covs=W), est) < 1e-10
    f = fsrd(_sites(x, y, lab, K, cl=cl), bw=(0.3, 0.45), vce='cluster')
    assert _maxdiff(f, _rdr(y, x, h=0.3, b=0.45, vce='hc1', cluster=cl), est) < 1e-10
    # change in slope (Dong & Lewbel threshold derivative): deriv = 1, p = 2, q = 3
    f = fsrd(_sites(x, y, lab, K), bw=(0.4, 0.6), deriv=1, p=2, q=3)
    assert _maxdiff(f, _rdr(y, x, h=0.4, b=0.6, deriv=1, p=2, vce='hc1'), est) < 1e-10
    # mserd bandwidths, continuous running variable (sd < IQR / 1.349 here, so the binned IQR is not used)
    for kw, rkw in ((dict(covs=False), dict()), (dict(covs=True), dict(covs=W)),
                    (dict(covs=False, deriv=1, p=2, q=3), dict(deriv=1, p=2))):
        f = fsrd(_sites(x, y, lab, K, w=W), **kw)
        r = _rdr(y, x, vce='hc1', bwselect='mserd', **rkw)
        assert abs(f['h'] - r['h']) < 1e-8 * r['h'] and abs(f['b'] - r['b']) < 1e-8 * r['b'], (kw, f['h'], r['h'])
        assert _maxdiff(f, r, est) < 1e-8, kw
    # discrete running variable: exact grid quantiles + rdrobust's default mass-point adjustment;
    # fuzzy design with rdrobust's fuzzy bandwidth
    xg, yg, Dg, labg, Kg = _grid_data(14)
    f = fsrd(_sites(xg, yg, labg, Kg), mass_grid=1 / 12)
    r = _rdr(yg, xg, vce='hc1', bwselect='mserd')
    assert abs(f['h'] - r['h']) < 1e-8 * r['h'] and _maxdiff(f, r, est) < 1e-8, (f['h'], r['h'])
    f = fsrd(_sites(xg, yg, labg, Kg, d=Dg), mass_grid=1 / 12, fuzzy_bw='rdrobust')
    r = _rdr(yg, xg, fuzzy=Dg, vce='hc1', bwselect='mserd')
    assert abs(f['h'] - r['h']) < 1e-8 * r['h'] and _maxdiff(f, r, est) < 1e-8, (f['h'], r['h'])


# ----------------------------------------------------------------------------- constants
def test_kernel_constants():
    c = boundary_constants('triangular')
    assert abs(c['B_K'] + 0.1) < 1e-10 and abs(c['C_V'] - 4.8) < 1e-8 and abs(c['C_IK'] - 3.4375) < 1e-3
    assert abs(boundary_constants('uniform')['C_IK'] - 2.702) < 1e-3
    assert abs(boundary_constants('epanechnikov')['C_IK'] - 3.1999) < 1e-3


def test_quantile_from_hist():
    rng = np.random.default_rng(14)
    v = rng.uniform(0, 1, 100000)
    e = np.linspace(0, 1, 513)
    c = np.histogram(v, e)[0].astype(float)
    for pr in (0.25, 0.5, 0.75):
        assert abs(quantile_from_hist(e, c, pr) - np.quantile(v, pr)) < 2e-3
