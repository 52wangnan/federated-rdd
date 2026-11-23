"""CHARLS analysis with IK bandwidths (b = h) reported in the paper: main and regional estimates (Table 9),
partitions of the same records across sites (Figures 2 and 3), covariate balance, placebo cutoffs and bandwidth checks,
together with the summary of the existing Monte Carlo draws (reused without transformation).
Run: python paper/charls_ik_analysis.py      (needs the extracted CHARLS analysis file in data/)
"""
from pathlib import Path
import sys,json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'charls'),str(ROOT/'simulation')]
import run_charls as rc
from fsrd import fsrd,centralized,site_estimates,meta_ivw,meta_mean
from run_simulation import make
import dgp
OUT=ROOT/'results'/'charls_ik'; OUT.mkdir(exist_ok=True)
SEED=20261011

def save(df,name): df.to_csv(OUT/(name+'.csv'),index=False)
def protocol_figure():
    from matplotlib.patches import FancyBboxPatch
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10})
    fig,ax=plt.subplots(figsize=(8,3.9));ax.set(xlim=(0,10),ylim=(0,5));ax.axis('off')
    blocks=[(.2,3.25,4.4,1.25,'Stage 1  Federated IK bandwidth','Sites: counts, sums and square sums\nPolynomial Gram matrices and cross-products'),(5.4,3.25,4.4,1.25,'Sponsor','Density and outcome variances\nCurvature pilots and regularized bandwidth'),(.2,.55,4.4,1.4,'Stage 2  Federated effect estimation','Sites: weighted regression cross-products\nLocal linear fit at h and quadratic fit at b\nResidual and influence-weight cross-products'),(5.4,.55,4.4,1.4,'Sponsor','Pooled normal equations\nConventional and bias-corrected estimates\nHC sandwich variance and confidence interval')]
    for x,y,w,h,title,txt in blocks:
        ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=.05',facecolor='#f3f3f3',edgecolor='#444444'))
        ax.text(x+w/2,y+h-.22,title,ha='center',va='center',fontsize=10,fontweight='bold');ax.text(x+w/2,y+.45,txt,ha='center',va='center',fontsize=9)
    for a,b in [((4.65,3.85),(5.3,3.85)),((7.6,3.2),(2.4,2.1)),((4.65,1.25),(5.3,1.25))]:ax.annotate('',xy=b,xytext=a,arrowprops=dict(arrowstyle='->',color='#333333'))
    ax.text(5,2.6,'Common bandwidth and specification',ha='center',fontsize=9,bbox=dict(facecolor='white',edgecolor='none',pad=2))
    ax.text(5,.1,'Individual records remain at each site; the simulator does not provide cryptographic protection.',ha='center',fontsize=8)
    fig.tight_layout();fig.savefig(OUT/'figure1.png',dpi=220);plt.close(fig)
def summary():
    raw=pd.read_csv(ROOT/'results/sim/sim_raw.csv'); rows=[]
    for (design,lam,K),g in raw.groupby(['design','lam','K']):
        for label,t,s in [('FSRD IK','ik_tau_cl','ik_se_cl'),('FSRD IK RBC','ik_tau_bc','ik_se_rb'),('FSRD CCT','fed_tau_cl','fed_se_cl'),('FSRD CCT RBC','fed_tau_bc','fed_se_rb'),('IVW local CCT','ivw_cl_tau','ivw_cl_se'),('Mean local CCT','mean_cl_tau','mean_cl_se')]:
            e=g[t]-g.tau; n=len(e); rm=np.sqrt(np.mean(e**2)); cov=(e.abs()<=1.959963984540054*g[s]).mean()
            rows.append(dict(design=design,lam=lam,K=K,estimator=label,reps=n,bias=e.mean(),sd=e.std(ddof=1),rmse=rm,mcse_bias=e.std(ddof=1)/np.sqrt(n),mcse_rmse=np.std(e**2,ddof=1)/(2*rm*np.sqrt(n)),coverage=cov,mcse_coverage=np.sqrt(cov*(1-cov)/n),length=2*1.959963984540054*g[s].mean(),h=g.ik_h.mean() if 'IK' in label else g.fed_h.mean()))
    save(pd.DataFrame(rows),'simulation')
    checks=[]
    for design in ['S1','S2']:
        for K in [5,10,20,40]:
            for lam in ([0.] if design=='S1' else [0.,.1,.2,.4]):
                S,t=make(design,K,lam,0); kw=dict(bw='ik',covs=design=='S2',support=dgp.SUPPORT[design]); a=fsrd(S,**kw); b=centralized(S,**kw)
                checks.append(dict(design=design,K=K,lam=lam,maxdiff=max(abs(a[k]-b[k]) for k in rc.CMP)))
    save(pd.DataFrame(checks),'ik_equivalence_checks')
    assert max(r['maxdiff'] for r in checks)<1e-10

def charls():
    rc.configure(2018); df=rc.load(); rows=[]
    for wave in [2018,2015]:
        rc.WAVE=wave; d=rc.load()
        for y in ['srh','srh_good']:
            for fuzzy,covs in [(False,False),(False,True),(True,False)]:
                S=rc.sites_of(d,y,fuzzy=fuzzy,covs=covs)
                for bw in ['ik','mserd']:
                    r=rc.run(S,bw=bw,covs=covs); row=rc.row_of(r,wave=wave,outcome=y,design=('FRD' if fuzzy else 'ITT')+('+cov' if covs else ''),bw=bw)
                    row['maxdiff']=r['maxdiff_central']; rows.append(row)
                    if fuzzy:
                        f=r['fs']; rows.append(dict(wave=wave,outcome=y,design='first stage',bw=bw,**{k:f[k] for k in ['tau_cl','tau_bc','se_rb','pv_rb']},ci_rb_lo=f['ci_rb'][0],ci_rb_hi=f['ci_rb'][1],h=r['h'],N=r['N']))
                    if wave==2018 and y=='srh' and not fuzzy and not covs and bw=='ik':
                        (OUT/'ik_info.json').write_text(json.dumps(r['bw_info'],indent=2,default=lambda x:float(x)),encoding='utf8')
    save(pd.DataFrame(rows),'charls_main')
    rc.WAVE=2018; S=rc.sites_of(df,'srh'); main=rc.run(S,bw='ik',covs=False)
    sens=[]
    for mult in [.5,.75,1,1.25,1.5,2]:
        r=rc.run(S,bw=main['h']*mult,covs=False); sens.append(rc.row_of(r,multiplier=mult))
    save(pd.DataFrame(sens),'bandwidth')
    bal=[]
    outcome_sample = df.dropna(subset=['x','srh']).copy()
    for y in ['female','han','edu_below_primary','edu_primary','edu_middle_plus','married']:
        r=rc.run(rc.sites_of(outcome_sample,y),bw=main['h'],covs=False); bal.append(rc.row_of(r,variable=y,sample='observed main health outcome'))
    bd=pd.DataFrame(bal); p=bd.pv_rb.to_numpy(); order=np.argsort(p); adj=np.empty(len(p));adj[order]=np.maximum.accumulate(np.minimum(1,p[order]*(len(p)-np.arange(len(p)))));bd['holm_p']=adj;save(bd,'balance')
    observed=df.copy();observed['srh_observed']=observed.srh.notna().astype(float)
    obs=rc.run(rc.sites_of(observed,'srh_observed'),bw=main['h'],covs=False)
    save(pd.DataFrame([rc.row_of(obs,variable='srh observed',sample='eligible cohort')]),'outcome_observation')
    pl=[]
    for c in [55,57,63,65]:
        d=df[df.x<0].copy() if c<60 else df[df.x>=0].copy();d['x']=d.x-(c-60)
        bound=(-15-(c-60),60-c) if c<60 else (60-c,60-(c-60))
        r=rc.run(rc.sites_of(d,'srh'),bw='ik',covs=False,support=bound);pl.append(rc.row_of(r,cutoff=c))
    save(pd.DataFrame(pl),'placebo')
    # Complete cases are partitioned, never resampled or duplicated.
    d=df.dropna(subset=['x','srh']+rc.COVS).copy().reset_index(drop=True); n=len(d); rng=np.random.default_rng(SEED)
    part=[]
    for kind in ['covariate','allocation','K']:
        reps=range(1,31) if kind!='K' else range(2,61)
        for rep in reps:
            K=10 if kind!='K' else rep
            if kind=='covariate':
                # Different education composition, but identical analysis columns at every site.
                u=rng.uniform(size=n); score=.65*d.edu_middle_plus.to_numpy()+u; rank=np.argsort(np.argsort(score));lab=np.minimum(K-1,(rank*K//n))
            elif kind=='allocation':
                prob=np.linspace(.1,.9,K);left=(1-prob)/(1-prob).sum();right=prob/prob.sum();lab=np.empty(n,int);side=d.x.to_numpy()>=0
                lab[side]=rng.choice(K,side.sum(),p=right);lab[~side]=rng.choice(K,(~side).sum(),p=left)
            else: lab=rng.permutation(n)%K
            sites=rc.sites_of(d,'srh',covs=True,labels=lab);r=rc.run(sites,bw='ik',covs=True)
            loc=site_estimates(sites,bw='ik',covs=True,**rc.OPT);ivw=meta_ivw(loc,'cl');mean=meta_mean(loc,'cl')
            part.append(dict(kind=kind,rep=rep,K=K,N=n,tau=r['tau_cl'],tau_bc=r['tau_bc'],se=r['se_rb'],ivw=ivw['tau'],mean=mean['tau'],local_used=ivw['k_used'],maxdiff=r['maxdiff_central'],h=r['h']))
        print('finished',kind,flush=True)
    pdpart=pd.DataFrame(part)
    assert len(pdpart)==119 and pdpart.maxdiff.max()<1e-10
    save(pdpart,'partitions')
    pdpart=pd.read_csv(OUT/'partitions.csv')
    plt.rcParams.update({'font.family':'DejaVu Serif','font.size':10})
    for kind in ['covariate','allocation','K']:
        z=pdpart[pdpart.kind==kind];fig,ax=plt.subplots(figsize=(6.3,3.4));ax.plot(z.rep,z.tau,label='FSRD / pooled',lw=2);ax.plot(z.rep,z.ivw,label='Inverse-variance average',lw=1);ax.plot(z.rep,z['mean'],label='Simple average',lw=1)
        ax.set(xlabel='Number of participants' if kind=='K' else 'Partition replication',ylabel='Estimated health-score jump');ax.legend(fontsize=8);ax.grid(alpha=.2);fig.tight_layout();fig.savefig(OUT/f'partitions_{kind}.png',dpi=220);plt.close(fig)
    reg=[]
    for s in S:
        try:r=rc.run([s],bw='ik',covs=False);reg.append(rc.row_of(r,site=s.name))
        except ValueError:pass
    reg.append(rc.row_of(main,site='All seven regions'));save(pd.DataFrame(reg),'regions')

if __name__=='__main__':
    summary();charls();protocol_figure();print('DONE',OUT,flush=True)
