# -*- coding: utf-8 -*-
"""Tables and figures of the simulation study, computed only from the result files

    results/sim/sim_raw.csv, results/comm/comm_scaling.csv, results/comm/comm_rounds_K10_p10.csv

    python simulation/make_sim_tables.py  -> results/sim/sim_summary.csv, output/tables/*, output/figures/*

Monte Carlo standard errors: bias sd/sqrt(R); SD sd/sqrt(2(R-1)); RMSE  sd((t-tau)^2)/(2 RMSE sqrt(R));
coverage sqrt(c(1-c)/R); CI length sd(len)/sqrt(R).  Estimates are never rescaled or shifted.
"""
import os
import sys

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
from reporting import save_table, pm  # noqa: E402

RES = os.path.join(ROOT, 'results')
TAB = os.path.join(ROOT, 'output', 'tables')
FIG = os.path.join(ROOT, 'output', 'figures')
Z = 1.959963984540054

# name -> (estimate column, se column, inference type)
EST = {
    'Federated (= pooled), conventional': ('fed_tau_cl', 'fed_se_cl'),
    'Federated (= pooled), robust BC': ('fed_tau_bc', 'fed_se_rb'),
    'Federated, IK bandwidth, conventional': ('ik_tau_cl', 'ik_se_cl'),
    'Federated, IK bandwidth, robust BC (b = h)': ('ik_tau_bc', 'ik_se_rb'),
    'Federated, no covariates, robust BC': ('nocov_tau_bc', 'nocov_se_rb'),
    'IVW meta-analysis, conventional': ('ivw_cl_tau', 'ivw_cl_se'),
    'IVW meta-analysis, robust BC': ('ivw_rb_tau', 'ivw_rb_se'),
    'Simple average, conventional': ('mean_cl_tau', 'mean_cl_se'),
    'Simple average, robust BC': ('mean_rb_tau', 'mean_rb_se'),
    'Federated + site FE, conventional': ('fe_tau_cl', 'fe_se_cl'),
    'Federated + site FE, robust BC': ('fe_tau_bc', 'fe_se_rb'),
}
DESIGN_LABEL = {'S1': 'S1: linear, no covariates',
                'S2': 'S2: linear, p = 10 covariates, variance heterogeneity lambda',
                'S3': 'S3: site-specific effects, unequal sites, target tau_pool',
                'S4': 'S4: Lee (2008) curvature design',
                'S5': 'S5: quadratic curvature',
                'S6': 'S6: site composition jumps at the cutoff (site intercepts, side-specific sampling rates)'}


def metrics(t, se, tau):
    ok = np.isfinite(t) & np.isfinite(se)
    t, se = t[ok], se[ok]
    R = len(t)
    if R < 2:
        return dict(R=R)
    e = t - tau
    sd = t.std(ddof=1)
    mse = np.mean(e ** 2)
    rmse = np.sqrt(mse)
    cov = np.abs(e) <= Z * se
    ln = 2 * Z * se
    return dict(R=R, bias=e.mean(), bias_mcse=sd / np.sqrt(R), sd=sd, sd_mcse=sd / np.sqrt(2 * (R - 1)),
                rmse=rmse, rmse_mcse=np.std(e ** 2, ddof=1) / (2 * rmse * np.sqrt(R)),
                coverage=cov.mean(), coverage_mcse=np.sqrt(cov.mean() * (1 - cov.mean()) / R),
                ci_length=ln.mean(), ci_length_mcse=ln.std(ddof=1) / np.sqrt(R))


def summarise(raw):
    rows = []
    for (design, lam, K), g in raw.groupby(['design', 'lam', 'K']):
        tau = g['tau'].to_numpy()
        base = dict(design=design, lam=lam, K=K, tau=float(np.mean(tau)), reps=len(g),
                    h_mean=g['fed_h'].mean(), b_mean=g['fed_b'].mean(), ik_h_mean=g['ik_h'].mean(),
                    rounds=g['fed_rounds'].mean(), bytes_up_site=g['fed_bytes_site'].mean(),
                    ik_rounds=g['ik_rounds'].mean(), time_fed=g['fed_time'].median(), time_cen=g['cen_time'].median(),
                    maxdiff_fed_cen=max(g['maxdiff_fed_cen'].max(), g['maxdiff_fe_cen'].max() if 'maxdiff_fe_cen' in g and g['maxdiff_fe_cen'].notna().any() else 0.0),
                    site_fail_mean=g['site_fail'].mean(),
                    reps_with_site_fail=int((g['site_fail'] > 0).sum()))
        for name, (tc, sc) in EST.items():
            if tc not in g or g[tc].isna().all():
                continue
            m = metrics(g[tc].to_numpy(float), g[sc].to_numpy(float), tau)
            rows.append(dict(base, estimator=name, **m))
    return pd.DataFrame(rows)


def design_table(s, design, estimators, digits=4):
    d = s[(s.design == design) & s.estimator.isin(estimators)].copy()
    d['order'] = d['estimator'].map({e: i for i, e in enumerate(estimators)})
    d = d.sort_values(['lam', 'K', 'order'])
    out = pd.DataFrame({
        **({'lambda': d['lam'].to_numpy()} if design == 'S2' else {}),
        'K': d['K'].to_numpy(),
        'Estimator': d['estimator'].to_numpy(),
        'Bias (MCSE)': [pm(a, b, digits) for a, b in zip(d['bias'], d['bias_mcse'])],
        'SD (MCSE)': [pm(a, b, digits) for a, b in zip(d['sd'], d['sd_mcse'])],
        'RMSE (MCSE)': [pm(a, b, digits) for a, b in zip(d['rmse'], d['rmse_mcse'])],
        'Coverage 95% (MCSE)': [pm(a, b, 3) for a, b in zip(d['coverage'], d['coverage_mcse'])],
        'CI length': [f'{v:.{digits}f}' for v in d['ci_length']],
        'Valid reps': d['R'].astype(int).to_numpy(),
    })
    return out


def main():
    raw = pd.read_csv(os.path.join(RES, 'sim', 'sim_raw.csv'))
    s = summarise(raw)
    s.to_csv(os.path.join(RES, 'sim', 'sim_summary.csv'), index=False)
    nrep = f' ({raw.groupby(["design", "lam", "K"]).rep.nunique().max()} replications)'
    main_est = ['Federated (= pooled), conventional', 'Federated (= pooled), robust BC', 'Federated, IK bandwidth, conventional',
                'IVW meta-analysis, conventional', 'IVW meta-analysis, robust BC', 'Simple average, conventional',
                'Simple average, robust BC']
    for dsg in ('S1', 'S3', 'S4', 'S5'):
        save_table(design_table(s, dsg, main_est), f'table_sim_{dsg}', TAB, caption=DESIGN_LABEL[dsg] + nrep)
    if (s.design == 'S6').any():
        s6_est = ['Federated (= pooled), robust BC', 'Federated + site FE, robust BC', 'Federated + site FE, conventional',
                  'IVW meta-analysis, robust BC', 'Simple average, robust BC']
        save_table(design_table(s, 'S6', s6_est), 'table_sim_S6', TAB, caption=DESIGN_LABEL['S6'] + nrep)
    s2_est = ['Federated (= pooled), robust BC', 'Federated, no covariates, robust BC', 'Federated, IK bandwidth, conventional',
              'IVW meta-analysis, robust BC', 'Simple average, robust BC']
    save_table(design_table(s, 'S2', s2_est), 'table_sim_S2', TAB, caption=DESIGN_LABEL['S2'] + nrep)
    # bandwidths, communication, timing and the federated-vs-pooled check per configuration
    cfg = s.drop_duplicates(['design', 'lam', 'K'])[['design', 'lam', 'K', 'tau', 'h_mean', 'b_mean', 'ik_h_mean', 'rounds',
                                                     'bytes_up_site', 'ik_rounds', 'time_fed', 'time_cen', 'maxdiff_fed_cen',
                                                     'site_fail_mean', 'reps_with_site_fail']]
    cfg = cfg.rename(columns={'h_mean': 'mean h (mserd)', 'b_mean': 'mean b', 'ik_h_mean': 'mean h (IK)', 'rounds': 'rounds (CCT)',
                              'bytes_up_site': 'bytes up / site', 'ik_rounds': 'rounds (IK)', 'time_fed': 'time fed (s, median)',
                              'time_cen': 'time pooled (s, median)', 'maxdiff_fed_cen': 'max |fed - pooled|',
                              'site_fail_mean': 'failed site fits / rep', 'reps_with_site_fail': 'reps with a failed site'})
    cfg['max |fed - pooled|'] = cfg['max |fed - pooled|'].map(lambda v: f'{v:.1e}')
    save_table(cfg, 'table_sim_configurations', TAB, caption='Bandwidths, communication, run time and exactness per configuration')
    # S3: truth and bias relative to tau_pool vs the unweighted mean of the site effects
    sys.path.insert(0, HERE)
    import dgp
    rows = []
    for K in sorted(raw.loc[raw.design == 'S3', 'K'].unique()):
        tp, tm, _ = dgp.s3_truth(int(K))
        rows.append(dict(K=int(K), tau_pool=tp, mean_of_site_effects=tm))
    save_table(pd.DataFrame(rows), 'table_sim_S3_truth', TAB, caption='S3: pooled cutoff effect and unweighted mean of site effects')
    comm_tables()
    figures(raw, s)
    print(s[s.estimator.str.contains('robust')][['design', 'lam', 'K', 'estimator', 'bias', 'rmse', 'coverage']].round(4).to_string())


def comm_tables():
    p = os.path.join(RES, 'comm', 'comm_scaling.csv')
    if not os.path.exists(p):
        return
    c = pd.read_csv(p)
    t = c[['K', 'p', 'rounds', 'bytes_up_site', 'bytes_up_total', 'bytes_down_total', 'micro_bytes_site', 'micro_bytes_total',
           'ik_rounds', 'ik_bytes_up_site', 't_fed', 't_fed_local', 't_cen', 't_fed_net50', 'absdiff_tau', 'absdiff_h']].copy()
    t['absdiff_tau'] = t['absdiff_tau'].map(lambda v: f'{v:.1e}')
    t['absdiff_h'] = t['absdiff_h'].map(lambda v: f'{v:.1e}')
    t = t.rename(columns={'bytes_up_site': 'upload/site (B)', 'bytes_up_total': 'upload total (B)', 'bytes_down_total': 'broadcast total (B)',
                          'micro_bytes_site': 'raw records/site (B)', 'micro_bytes_total': 'raw records total (B)',
                          'ik_rounds': 'rounds (IK + est.)', 'ik_bytes_up_site': 'IK upload/site (B)',
                          't_fed': 't fed, sites serial (s)', 't_fed_local': 't sites, parallel (s)', 't_cen': 't pooled (s)',
                          't_fed_net50': 't fed + 50 ms/round (s)', 'absdiff_tau': '|fed - pooled| tau_bc', 'absdiff_h': '|fed - pooled| h'})
    save_table(t, 'table_comm_scaling', TAB, digits=3,
               caption='Communication and run time, CCT pipeline (N = 10,000; median of 5 runs; bytes = 8 per value). '
                       "'t fed, sites serial' runs every site in one process; 't sites, parallel' sums over rounds the slowest site's compute time")
    r = pd.read_csv(os.path.join(RES, 'comm', 'comm_rounds_K10_p10.csv'))
    save_table(r, 'table_comm_rounds_K10_p10', TAB, caption='Rounds of the federated CCT pipeline (K = 10, p = 10)')


def figures(raw, s):
    os.makedirs(FIG, exist_ok=True)
    plt.rcParams.update({'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False})
    col = {'Federated (= pooled), conventional': '#1f77b4', 'Federated (= pooled), robust BC': '#08306b',
           'IVW meta-analysis, conventional': '#e6550d', 'IVW meta-analysis, robust BC': '#a63603',
           'Simple average, conventional': '#74c476', 'Simple average, robust BC': '#006d2c',
           'Federated, IK bandwidth, conventional': '#9e9ac8', 'Federated + site FE, robust BC': '#d62728'}
    ls = lambda e: '--' if 'conventional' in e else '-'
    # coverage
    designs = [('S1', 0.0), ('S2', 0.4), ('S3', 0.0), ('S4', 0.0), ('S5', 0.0), ('S6', 0.0)]
    fig, axs = plt.subplots(1, len(designs), figsize=(15.5, 3.2), sharey=True)
    for ax, (dsg, lam) in zip(axs, designs):
        d = s[(s.design == dsg) & (s.lam == lam)]
        for e, c_ in col.items():
            g = d[d.estimator == e].sort_values('K')
            if len(g):
                ax.errorbar(g['K'], g['coverage'], yerr=Z * g['coverage_mcse'], color=c_, ls=ls(e), marker='o', ms=3, capsize=2, label=e)
        ax.axhline(0.95, color='grey', lw=0.8)
        ax.set_xscale('log')
        ax.minorticks_off()
        ax.set_xticks([5, 10, 20, 40])
        ax.set_xticklabels(['5', '10', '20', '40'])
        ax.set_title(dsg + (f' (λ = {lam})' if dsg == 'S2' else ''))
        ax.set_xlabel('number of sites K')
    axs[0].set_ylabel('coverage of nominal 95% CI')
    axs[0].set_ylim(0, 1.02)
    handles, labels = axs[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=4, frameon=False, bbox_to_anchor=(0.5, -0.12))
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, 'fig_sim_coverage.png'), dpi=300, bbox_inches='tight')
    fig.savefig(os.path.join(FIG, 'fig_sim_coverage.pdf'), bbox_inches='tight')
    plt.close(fig)
    # RMSE
    fig, axs = plt.subplots(1, len(designs), figsize=(15.5, 3.2))
    for ax, (dsg, lam) in zip(axs, designs):
        d = s[(s.design == dsg) & (s.lam == lam)]
        for e in ('Federated (= pooled), robust BC', 'IVW meta-analysis, robust BC', 'Simple average, robust BC',
                  'Federated (= pooled), conventional', 'Federated + site FE, robust BC'):
            g = d[d.estimator == e].sort_values('K')
            ax.errorbar(g['K'], g['rmse'], yerr=Z * g['rmse_mcse'], color=col[e], ls=ls(e), marker='o', ms=3, capsize=2, label=e)
        ax.set_xscale('log')
        ax.minorticks_off()
        ax.set_xticks([5, 10, 20, 40])
        ax.set_xticklabels(['5', '10', '20', '40'])
        ax.set_title(dsg + (f' (λ = {lam})' if dsg == 'S2' else ''))
        ax.set_xlabel('number of sites K')
        ax.set_ylim(bottom=0)
    axs[0].set_ylabel('RMSE')
    handles, labels = axs[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=4, frameon=False, bbox_to_anchor=(0.5, -0.1))
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, 'fig_sim_rmse.png'), dpi=300, bbox_inches='tight')
    fig.savefig(os.path.join(FIG, 'fig_sim_rmse.pdf'), bbox_inches='tight')
    plt.close(fig)
    # S3 estimand: distribution of estimates around tau_pool
    d = raw[raw.design == 'S3']
    fig, axs = plt.subplots(1, 4, figsize=(12, 3), sharey=True)
    sys.path.insert(0, HERE)
    import dgp
    for ax, K in zip(axs, sorted(d.K.unique())):
        g = d[d.K == K]
        tp, tm, _ = dgp.s3_truth(int(K))
        data = [g['fed_tau_bc'].dropna(), g['ivw_rb_tau'].dropna(), g['mean_rb_tau'].dropna()]
        ax.boxplot(data, widths=0.5, showfliers=False)
        ax.set_xticks([1, 2, 3])
        ax.set_xticklabels(['federated', 'IVW', 'average'])
        ax.axhline(tp, color='k', lw=1, label=r'$\tau_{\mathrm{ITT}}$ (pooled effect)')
        ax.axhline(tm, color='grey', lw=1, ls=':', label='mean of site effects')
        ax.set_title(f'S3, K = {K}')
    axs[0].set_ylabel('estimate (robust BC)')
    axs[0].legend(frameon=False, fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, 'fig_sim_S3_estimand.png'), dpi=300, bbox_inches='tight')
    fig.savefig(os.path.join(FIG, 'fig_sim_S3_estimand.pdf'), bbox_inches='tight')
    plt.close(fig)
    # exactness: federated vs pooled over all replications
    v = raw['maxdiff_fed_cen'].to_numpy()
    fig, ax = plt.subplots(figsize=(4.5, 3))
    lv = np.log10(np.maximum(v, 1e-17))
    ax.hist(lv, bins=40, color='#08306b')
    ax.axvline(-10, color='red', lw=1, ls='--', label='tolerance 1e-10')
    ax.set_xlabel('log10 max |federated - pooled|  (exact zero shown at -17)')
    ax.set_ylabel('replications')
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, 'fig_sim_exactness.png'), dpi=300, bbox_inches='tight')
    fig.savefig(os.path.join(FIG, 'fig_sim_exactness.pdf'), bbox_inches='tight')
    plt.close(fig)
    # communication
    p = os.path.join(RES, 'comm', 'comm_scaling.csv')
    if os.path.exists(p):
        c = pd.read_csv(p)
        from matplotlib.lines import Line2D
        fig, axs = plt.subplots(1, 2, figsize=(9, 3.6))
        ps = sorted(c['p'].unique())
        colors = {pp: plt.cm.viridis(i / max(1, len(ps))) for i, pp in enumerate(ps)}
        for pp, g in c.groupby('p'):
            g = g.sort_values('K')
            clr = colors[pp]
            axs[0].plot(g['K'], g['bytes_up_site'] / 1024, marker='o', ms=4, color=clr)
            axs[0].plot(g['K'], g['micro_bytes_site'] / 1024, marker='x', ms=4, ls=':', color=clr)
            axs[1].plot(g['K'], g['t_fed_local'] + 0.05 * g['rounds'], marker='o', ms=4, color=clr)
            axs[1].plot(g['K'], g['t_fed'], marker='s', ls='--', color=clr, ms=3)
            axs[1].plot(g['K'], g['t_cen'], marker='x', ms=4, ls=':', color=clr)
        for ax in axs:
            ax.set_xscale('log')
            ax.minorticks_off()
            ax.set_xticks([2, 5, 10, 20, 40, 60])
            ax.set_xticklabels(['2', '5', '10', '20', '40', '60'])
            ax.set_xlabel('number of sites K (N = 10,000)')
        axs[0].set_yscale('log')
        axs[0].set_ylabel('upload per site (KiB)')
        axs[1].set_ylabel('wall-clock time (s)')
        axs[0].set_title('upload per site', fontsize=9)
        axs[1].set_title('time per analysis', fontsize=9)
        # legends below the panels: colour = number of covariates p, line style = quantity
        col_h = [Line2D([], [], color=colors[pp], lw=2) for pp in ps]
        axs[0].legend(col_h + [Line2D([], [], color='k', marker='o', ms=4), Line2D([], [], color='k', marker='x', ms=4, ls=':')],
                      [f'p = {pp} covariates' for pp in ps] + ['federated protocol', 'uploading the raw records'],
                      frameon=False, fontsize=7, ncol=3, loc='upper center', bbox_to_anchor=(0.5, -0.2))
        axs[1].legend([Line2D([], [], color='k', marker='o', ms=4), Line2D([], [], color='k', marker='s', ms=3, ls='--'),
                       Line2D([], [], color='k', marker='x', ms=4, ls=':')],
                      ['federated, sites in parallel + 50 ms per round', 'federated, all sites computed serially', 'pooled data'],
                      frameon=False, fontsize=7, ncol=1, loc='upper center', bbox_to_anchor=(0.5, -0.2))
        fig.tight_layout()
        fig.savefig(os.path.join(FIG, 'fig_comm.png'), dpi=300, bbox_inches='tight')
        fig.savefig(os.path.join(FIG, 'fig_comm.pdf'), bbox_inches='tight')
        plt.close(fig)


if __name__ == '__main__':
    main()
