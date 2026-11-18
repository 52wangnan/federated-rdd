# -*- coding: utf-8 -*-
"""Site composition that changes at the cutoff (design S6, Table 8) and coalition precision (Figure 5).
python paper/composition_coalitions.py --jobs 8
python paper/composition_coalitions.py --tables-only
S6: 500 replications x 4 values of K, IK fits with and without site fixed effects.
Coalitions: all 127 nonempty regional subsets, 2018 health score, IK, b = h, HC1.
The normal-approximation MDE = 2.80 x robust SE is descriptive.
"""
import os
for _v in ('OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'OMP_NUM_THREADS'):
    os.environ[_v] = '1'
import argparse
import hashlib
import itertools
import json
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'simulation'),str(ROOT/'charls'),str(ROOT/'paper')]
from fsrd import fsrd,centralized
from run_simulation import make,BASE_SEED
import run_charls as rc
import dgp
from extensions_s3_curvature import metrics,FIELDS

OUT=ROOT/'results/composition_coalitions'


def s6_one(K,rep):
    S,truth=make('S6',K,0.,rep)
    opts=dict(bw='ik',site_fe=True,covs=True,support=dgp.SUPPORT['S6'])
    r=fsrd(S,**opts)
    row=dict(K=K,rep=rep,N=sum(s.n for s in S),truth=truth,**{k:r[k] for k in FIELDS})
    if rep<5:
        c=centralized(S,**opts)
        row['maxdiff_central']=max(abs(r[k]-c[k]) for k in FIELDS)
        assert row['maxdiff_central']<1e-10
    return row


def s6_chunk(K,reps):
    return [s6_one(K,rep) for rep in reps]


def coalition_one(sites,idx,fuzzy):
    S=[sites[i] for i in idx]
    tags=dict(design='FRD' if fuzzy else 'ITT',members='+'.join(s.name for s in S),
              mask=sum(1<<i for i in idx),n_members=len(idx),N=sum(s.n for s in S),wave=2018,outcome='srh')
    opts=dict(**rc.OPT,bw='ik',covs=False,fuzzy=fuzzy)
    try:
        r=fsrd(S,**opts);c=centralized(S,**opts)
        diff=max(abs(r[k]-c[k]) for k in FIELDS)
        assert diff<1e-10
        fs_fields = {} if not fuzzy else dict(fs_tau_cl=r['fs']['tau_cl'],fs_tau_bc=r['fs']['tau_bc'],
            fs_se_rb=r['fs']['se_rb'],fs_pv_rb=r['fs']['pv_rb'],
            fs_ci_lo=r['fs']['ci_rb'][0],fs_ci_hi=r['fs']['ci_rb'][1],
            fs_z2=(r['fs']['tau_bc']/r['fs']['se_rb'])**2)
        return dict(**tags,ok=True,**{k:r[k] for k in FIELDS},**fs_fields,n_h=sum(r['n_h']),
                    ci_lo=r['ci_rb'][0],ci_hi=r['ci_rb'][1],mde=2.80*r['se_rb'],maxdiff_central=diff)
    except (ValueError,np.linalg.LinAlgError,ZeroDivisionError,FloatingPointError) as e:
        return dict(**tags,ok=False,error=str(e))


def compute(jobs):
    OUT.mkdir(parents=True,exist_ok=True)
    tasks=[(K,list(range(start,start+25))) for K in (5,10,20,40) for start in range(0,500,25)]
    chunks=Parallel(n_jobs=jobs,verbose=10)(delayed(s6_chunk)(*task) for task in tasks)
    pd.DataFrame([r for ch in chunks for r in ch]).sort_values(['K','rep']).to_csv(OUT/'s6_ik_fe_raw.csv',index=False)
    rc.configure(2018);data=rc.load();rows=[]
    for fuzzy in (False,True):
        S=rc.sites_of(data,'srh',fuzzy=fuzzy,covs=False)
        assert len(S)==7
        coalitions=[idx for n in range(1,8) for idx in itertools.combinations(range(7),n)]
        rows+=Parallel(n_jobs=jobs,verbose=5)(delayed(coalition_one)(S,idx,fuzzy) for idx in coalitions)
    pd.DataFrame(rows).to_csv(OUT/'coalitions_ik.csv',index=False)
    sources=[ROOT/'results/sim/sim_raw.csv',ROOT/'data/charls2018_rd.csv']
    (OUT/'provenance.json').write_text(json.dumps(dict(seed=BASE_SEED,reps=500,
        s6_seed='[20261011,6,K,0,rep]',coalitions='complete enumeration; no random seed used',
        wave=2018,outcome='srh (1=very good,5=very poor)',bandwidth='IK, b=h; HC1',
        sources={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}),indent=2),encoding='utf8')


def summarize():
    raw=pd.read_csv(ROOT/'results/sim/sim_raw.csv');old=raw[raw.design=='S6']
    fe=pd.read_csv(OUT/'s6_ik_fe_raw.csv');g=old.merge(fe,on=['K','rep'],suffixes=('_stored',''),validate='one_to_one')
    assert len(g)==2000
    assert np.allclose(g.h,g.ik_h,rtol=0,atol=1e-10)
    assert np.all(g.N==g.N_stored)
    summary=[]
    for K,d in g.groupby('K'):
        assert len(d)==500
        for label,t,se in [('Unadjusted FSRD / pooled','ik_tau_bc','ik_se_rb'),('FSRD / pooled + site FE','tau_bc','se_rb')]:
            summary.append(dict(K=K,method=label,N_min=int(d.N.min()),N_max=int(d.N.max()),N_mean=d.N.mean(),
                population_composition_jump=.2*(K+1)/(K-1),h_mean=d.h.mean(),
                **metrics(d[t],d[se],d.truth)))
    sm=pd.DataFrame(summary);sm.to_csv(OUT/'s6_summary.csv',index=False)
    co=pd.read_csv(OUT/'coalitions_ik.csv');fits=[];counts=[]
    for dg,d in co.groupby('design'):
        assert len(d)==127
        ok=d[d.ok & np.isfinite(d.se_rb) & (d.se_rb>0)]
        X=np.column_stack([np.ones(len(ok)),np.log(ok.N)])
        beta,*_=np.linalg.lstsq(X,np.log(ok.se_rb),rcond=None)
        full=ok[ok.n_members==7].iloc[0]
        # Overlapping coalitions: slope is descriptive, no OLS SE/p-value.
        fits.append(dict(design=dg,c=beta[0],a=-beta[1],coalitions_ok=len(ok),
             N_all=int(full.N),n_h_all=int(full.n_h),se_all=full.se_rb,mde_all=full.mde,
             mde_min=ok.mde.min(),mde_max=ok.mde.max(),single_mde_min=ok[ok.n_members==1].mde.min(),
             single_mde_max=ok[ok.n_members==1].mde.max(),h_all=full.h,
             additions_increasing_mde=sum(float(ok.loc[ok['mask']==(int(r['mask'])|(1<<i)),'mde'].iloc[0])>r.mde
                for _,r in ok.iterrows() for i in range(7) if not (int(r['mask'])&(1<<i))
                and (ok['mask']==(int(r['mask'])|(1<<i))).any())))
        for n,dd in ok.groupby('n_members'):
            counts.append(dict(design=dg,regions=n,coalitions=len(dd),N_min=dd.N.min(),N_max=dd.N.max(),
                               MDE_min=dd.mde.min(),MDE_median=dd.mde.median(),MDE_max=dd.mde.max()))
    pd.DataFrame(fits).to_csv(OUT/'coalition_fit.csv',index=False)
    pd.DataFrame(counts).to_csv(OUT/'coalition_by_size.csv',index=False)
    if 'fs_tau_bc' in co:
        fs = co[co.design=='FRD'][['members','n_members','N','fs_tau_cl','fs_tau_bc','fs_se_rb',
            'fs_pv_rb','fs_ci_lo','fs_ci_hi','fs_z2']]
        fs.to_csv(OUT/'coalition_first_stage.csv',index=False)
        pd.DataFrame([dict(coalitions=len(fs),bc_jump_min=fs.fs_tau_bc.min(),bc_jump_max=fs.fs_tau_bc.max(),
            intervals_including_zero=int(((fs.fs_ci_lo<=0)&(fs.fs_ci_hi>=0)).sum()),
            min_squared_robust_z=fs.fs_z2.min())]).to_csv(OUT/'first_stage_summary.csv',index=False)
    # The full coalition must reproduce the existing main estimate.
    main=pd.read_csv(ROOT/'results/charls_ik/charls_main.csv')
    checks=[]
    for _,r in co[co.n_members==7].iterrows():
        # Filter using the actual column conventions of paper/charls_ik_analysis.py.
        d=main[(main.wave==2018)&(main.outcome=='srh')&(main.bw=='ik')&(main.design==r.design)]
        assert len(d)==1
        err=max(abs(float(d.iloc[0][k])-r[k]) for k in FIELDS)
        assert err<1e-10
        checks.append(dict(design=r.design,maxdiff_main=err))
    pd.DataFrame(checks).to_csv(OUT/'main_checks.csv',index=False)


def displays():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
    co=pd.read_csv(OUT/'coalitions_ik.csv');fit=pd.read_csv(OUT/'coalition_fit.csv')
    fig,axes=plt.subplots(1,2,figsize=(9,3.6),sharex=True)
    for ax,dg in zip(axes,['ITT','FRD']):
        d=co[(co.design==dg)&co.ok];f=fit[fit.design==dg].iloc[0]
        dots=ax.scatter(d.N,d.mde,c=d.n_members,cmap='viridis',vmin=1,vmax=7,s=23,alpha=.8,edgecolors='none')
        grid=np.geomspace(d.N.min(),d.N.max(),150)
        ax.plot(grid,2.80*np.exp(f.c)*grid**(-f.a),color='#a14c2f',lw=1.5,label=f'Descriptive fit (a={f.a:.2f})')
        grand=d[d.n_members==7].iloc[0]
        ax.scatter([grand.N],[grand.mde],marker='*',s=120,color='#aa2828',edgecolors='white',linewidths=.5,zorder=4,label='All seven regions')
        ax.set_title(dg);ax.set_xlabel('Coalition sample size N');ax.set_ylabel('MDE (health-score points)')
        ax.set_xscale('log');ax.set_yscale('log');ax.grid(alpha=.2,which='major');ax.legend(fontsize=8,frameon=False)
    fig.subplots_adjust(left=.08,right=.86,bottom=.20,top=.89,wspace=.32)
    cb=fig.colorbar(dots,cax=fig.add_axes([.89,.23,.02,.60]),ticks=range(1,8));cb.set_label('Number of regions')
    for ext in ['png','pdf']:fig.savefig(OUT/('coalition_mde.'+ext),dpi=240,bbox_inches='tight')
    plt.close(fig)
    s=pd.read_csv(OUT/'s6_summary.csv');table=[]
    for _,r in s.iterrows():
        table.append({'K':int(r.K),'Specification (IK, robust BC)':'No site FE' if r.method.startswith('Unadjusted') else 'With site FE',
         'Bias':f'{r.bias:.4f}','SD':f'{r.SD:.4f}','RMSE (MCSE)':f'{r.RMSE:.4f} ({r.MCSE_RMSE:.4f})',
         'Coverage (MCSE)':f'{r.coverage:.3f} ({r.MCSE_coverage:.3f})','Mean CI length':f'{r.length:.3f}'})
    pd.DataFrame(table).to_csv(OUT/'s6_manuscript_table.csv',index=False)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--jobs',type=int,default=8);ap.add_argument('--tables-only',action='store_true')
    args=ap.parse_args();start=time.perf_counter()
    if not args.tables_only:compute(args.jobs)
    summarize();displays()
    print(f'Done: {OUT} ({time.perf_counter()-start:.1f}s)',flush=True)
