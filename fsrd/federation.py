# -*- coding: utf-8 -*-
"""Sites, the sponsor's aggregation channel, and the communication ledger.

This is an in-process simulation of a sum-aggregation interface, not a cryptographic
secure-aggregation implementation. A deployment could protect this interface using a
separate secure-aggregation protocol under its stated adversary assumptions. This code
provides neither cryptographic confidentiality nor differential privacy. With configured
public support bounds, sites need not report minima/maxima; the optional legacy min_/max_
keys are explicitly combined by min/max and listed in the ledger.

A call Federation.round(name, fn, **params) runs fn(site, **params) at every site, sums the
results and logs one communication round: the number of float64 values each site uploads, the
number of values broadcast by the sponsor, and the local compute time of the slowest site.
"""
import time
import numpy as np


class Site:
    """One data holder.  x: running variable already centred at the cutoff (X - c); y: outcome;
    d: treatment take-up (fuzzy RD, optional); w: covariate matrix n x p (optional);
    cl: integer cluster labels (optional)."""

    def __init__(self, x, y, d=None, w=None, cl=None, name=None):
        self.x = np.asarray(x, float)
        self.y = np.asarray(y, float)
        n = len(self.x)
        if len(self.y) != n:
            raise ValueError('x and y differ in length')
        self.d = None if d is None else np.asarray(d, float)
        if w is not None:
            w = np.asarray(w, float)
            if w.ndim == 1:
                w = w[:, None]
            if w.shape[1] == 0:
                w = None
        self.w = w
        self.cl = None if cl is None else np.asarray(cl)
        self.name = name
        self.suppressed = set()          # windows zeroed by the small-cell rule (local audit log)

    @property
    def n(self):
        return len(self.x)

    @property
    def p(self):
        return 0 if self.w is None else self.w.shape[1]

    def outcomes(self):
        """Outcome columns: [y] (sharp) or [y, d] (fuzzy)."""
        return self.y[:, None] if self.d is None else np.column_stack([self.y, self.d])

    def data_matrix(self):
        """[outcomes | covariates]; the column layout used by every Gram request."""
        D = self.outcomes()
        return D if self.w is None else np.column_stack([D, self.w])


def pool_sites(sites):
    """Centralised benchmark: all records in one site (the K = 1 federation)."""
    cat = lambda a: None if getattr(sites[0], a) is None else np.concatenate([getattr(s, a) for s in sites])
    return Site(cat('x'), cat('y'), d=cat('d'), w=cat('w'), cl=cat('cl'), name='pooled')


def _count_numbers(obj):
    if obj is None or isinstance(obj, (str, bool)):
        return 0
    if isinstance(obj, (int, float, np.integer, np.floating)):
        return 1
    if isinstance(obj, np.ndarray):
        return int(obj.size)
    if isinstance(obj, dict):
        return sum(_count_numbers(v) for v in obj.values())
    if isinstance(obj, (list, tuple)):
        return sum(_count_numbers(v) for v in obj)
    return 0


class Ledger:
    """Communication log: one entry per round."""

    def __init__(self):
        self.rounds = []

    def add(self, name, up_per_site, down, t_local, t_sponsor=0.0):
        self.rounds.append(dict(name=name, up_per_site=up_per_site, down=down, t_local=t_local, t_sponsor=t_sponsor))

    @property
    def n_rounds(self):
        return len(self.rounds)

    def summary(self, latency=0.0):
        """Bytes are counted as 8 bytes per float64 value."""
        if not self.rounds:
            return dict(rounds=0, bytes_up_per_site=0, bytes_up_total=0, bytes_down_total=0, t_local=0.0, t_net=0.0)
        K = len(self.rounds[0]['up_per_site'])
        up_site = np.sum([r['up_per_site'] for r in self.rounds], axis=0) * 8
        return dict(rounds=self.n_rounds,
                    bytes_up_per_site=float(np.max(up_site)),
                    bytes_up_per_site_mean=float(np.mean(up_site)),
                    bytes_up_total=float(np.sum(up_site)),
                    bytes_down_total=float(8 * K * sum(r['down'] for r in self.rounds)),
                    t_local=float(sum(r['t_local'] for r in self.rounds)),
                    t_net=float(latency * self.n_rounds))

    def table(self):
        return [dict(round=i + 1, name=r['name'], floats_up_per_site=int(np.max(r['up_per_site'])),
                     floats_down=int(r['down'])) for i, r in enumerate(self.rounds)]


class Federation:
    """Sponsor-side handle on K sites.  min_cell > 0 switches on the small-cell rule: a site whose
    kernel window on one side of the cutoff contains fewer than min_cell records contributes zeros
    for that window (the rule is applied inside the local functions, see local.py)."""

    def __init__(self, sites, min_cell=0, ledger=None):
        self.sites = list(sites)
        self.min_cell = int(min_cell)
        self.ledger = Ledger() if ledger is None else ledger

    @property
    def K(self):
        return len(self.sites)

    @property
    def p(self):
        return self.sites[0].p

    @property
    def fuzzy(self):
        return self.sites[0].d is not None

    def round(self, name, fn, **params):
        outs, ups, tl = [], [], []
        for s in self.sites:
            t0 = time.perf_counter()
            o = fn(s, min_cell=self.min_cell, **params)
            tl.append(time.perf_counter() - t0)
            outs.append(o)
            ups.append(_count_numbers(o))
        agg = _unpack(_secure_sum(outs))
        self.ledger.add(name, ups, _count_numbers(params), max(tl))
        return agg


def _secure_sum(outs):
    """Element-wise sum over sites (min/max for keys named min_*/max_*); nested dicts allowed."""
    first = outs[0]
    if isinstance(first, dict):
        return {k: _secure_sum([o[k] for o in outs]) if isinstance(first[k], (dict, list))
                else _combine(k, [o[k] for o in outs]) for k in first}
    if isinstance(first, list):
        return [_secure_sum([o[i] for o in outs]) for i in range(len(first))]
    raise TypeError(type(first))


def _combine(key, vals):
    if key.startswith('min_'):
        return np.min(np.array(vals, float), axis=0)
    if key.startswith('max_'):
        return np.max(np.array(vals, float), axis=0)
    if np.issubdtype(np.asarray(vals[0]).dtype, np.integer):          # counts: exact integer sums
        return np.sum(np.stack([np.asarray(v) for v in vals]), axis=0)
    # every other statistic arrives as a double-double pair [hi, lo] (see acc_sum)
    return acc_sum(np.concatenate([np.asarray(v, float) for v in vals], axis=0))[0]


def _unpack(obj):
    """Symmetric matrices travel as their upper triangle under the key '<name>_sym'."""
    if isinstance(obj, list):
        return [_unpack(o) for o in obj]
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            if k.endswith('_sym'):
                out[k[:-4]] = sym_unpack(v)
            else:
                out[k] = _unpack(v) if isinstance(v, (dict, list)) else v
        return out
    return obj


def sym_unpack(v):
    m = v.shape[-1]
    k = int(round((np.sqrt(8 * m + 1) - 1) / 2))
    iu = np.triu_indices(k)
    M = np.zeros(v.shape[:-1] + (k, k))
    M[..., iu[0], iu[1]] = v
    M[..., iu[1], iu[0]] = v
    return M


# ----------------------------------------------------------------------------- accurate summation
def acc_sum(T):
    """Sum over axis 0 in (about) twice the working precision: pairwise error-free TwoSum cascade with
    the rounding errors accumulated separately (Ogita, Rump & Oishi 2005, Sum2).  Returns (hi, lo) with
    hi = fl(sum) and hi + lo representing the sum to ~106 bits.

    Every site reports its sums as such a pair and the sponsor combines the pairs the same way, so the
    federated totals round to the same double as the totals computed on the pooled records: federated
    and pooled results are then bitwise identical (up to the rare case of a sum lying within ~1e-30 of a
    rounding midpoint).  This mirrors secure-aggregation encodings, which carry statistics as wide
    fixed-point integers; the pair is counted as two float64 values in the communication ledger
    (integer counts are sent once, symmetric matrices as their upper triangle)."""
    s = np.asarray(T, float)
    shape = s.shape[1:]
    if s.shape[0] == 0:
        return np.zeros(shape), np.zeros(shape)
    err = np.zeros(shape)
    while s.shape[0] > 1:
        if s.shape[0] % 2:
            s = np.concatenate([s, np.zeros((1,) + shape)], axis=0)
        a, b = s[0::2], s[1::2]
        x = a + b
        z = x - a
        err = err + ((a - (x - z)) + (b - z)).sum(axis=0)
        s = x
    hi = s[0] + err
    lo = err - (hi - s[0])
    return hi, lo


def dd(T):
    """Per-record contributions (axis 0) -> double-double sum, shape (2, ...)."""
    hi, lo = acc_sum(T)
    return np.stack([hi, lo])


def dd_exact(v):
    """Integer counts: sent once as exact integers (summed exactly by the sponsor)."""
    return np.asarray(v, dtype=np.int64)
