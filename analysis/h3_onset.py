"""H3: within-country VR-onset DiD on GBD roughness, permutation inference, leave-one-out, event study."""
import numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
from common import *
D=load_gbd(); CR=pd.read_csv(f'{RES}/country_richness.csv')
Y=vr_years(); cnt=Y[Y.Year<=2021].groupby('iso3').Year.nunique()
treated=CR[(CR.onset>=1998)&(CR.onset<=2013)&(CR.iso3.map(cnt).fillna(0)>=5)].set_index('iso3').onset
always=set(CR[CR.onset<=1990].iso3); never=set(CR[CR.onset.isna()].iso3)
print('treated',len(treated),treated.astype(int).to_dict()); print('always',len(always),'never',len(never))
keep=set(treated.index)|always|never
rng=np.random.default_rng(2026); out=[]; ev_all=[]
def twfe(panel,post):
    """y ~ post + country FE + year FE via alternating projections on a small panel."""
    y=panel.y.values.copy(); p=post.astype(float).copy(); c=panel.CountryCode.values; t=panel.Year.values
    for _ in range(50):
        for grp in (c,t):
            y-=pd.Series(y).groupby(grp).transform('mean').values; p-=pd.Series(p).groupby(grp).transform('mean').values
    return (p@y)/(p@p), y, p
def cluster_se(y,p,b,c):
    e=y-b*p; s=pd.Series(p*e).groupby(c).sum().values; return np.sqrt((s**2).sum())/(p@p)
for kind in ['death','prev']:
    R=roughness_long(D[D.CountryCode.isin(keep)],outcome_cols(D,kind))
    R=R[R.rough>0]; R['lr']=np.log(R.rough)
    R['lr']-=R.groupby(['cause','Year']).lr.transform('mean')          # cause-year FE
    R['lr']-=R.groupby(['CountryCode','cause']).lr.transform('mean')   # country-cause FE (before collapsing)
    P=R.groupby(['CountryCode','Year']).lr.mean().rename('y').reset_index()
    on=P.CountryCode.map(treated)
    post=((on.notna())&(P.Year>=on)).values
    b,yw,pw=twfe(P,post); se=cluster_se(yw,pw,b,P.CountryCode.values)
    # permutation: assign treated onset years to random controls, drop real treated
    ctrl=P[~P.CountryCode.isin(treated.index)].reset_index(drop=True); cands=sorted(set(ctrl.CountryCode))
    perm=[]
    for _ in range(2000):
        pick=rng.choice(cands,len(treated),replace=False); fake=dict(zip(pick,rng.permutation(treated.values)))
        o2=ctrl.CountryCode.map(fake); pp=((o2.notna())&(ctrl.Year>=o2)).values
        perm.append(twfe(ctrl,pp)[0])
    perm=np.array(perm); pd.Series(perm).to_csv(f'{RES}/h3_perm_{kind}.csv',index=False); pperm=np.mean(np.abs(perm)>=abs(b))
    loo=[]
    for c in treated.index:
        Q=P[P.CountryCode!=c]; oq=Q.CountryCode.map(treated); loo.append(twfe(Q,((oq.notna())&(Q.Year>=oq)).values)[0])
    out.append(dict(test='H3',outcome=kind,est=np.exp(b),lo=np.exp(b-1.96*se),hi=np.exp(b+1.96*se),p_cluster=2*(1-__import__('scipy').stats.norm.cdf(abs(b/se))),
                    p_perm=pperm,loo_min=np.exp(min(loo)),loo_max=np.exp(max(loo)),n_treated=len(treated),n_countries=P.CountryCode.nunique()))
    # event study (relative to treated-country mean; controls define year effects already removed)
    T=P[P.CountryCode.isin(treated.index)].copy(); T['et']=T.Year-T.CountryCode.map(treated)
    T['bin']=pd.cut(T.et,[-40,-9,-6,-3,0,3,6,9,40],right=False)
    T['y']-=T.groupby('CountryCode').y.transform('mean')
    ev=T.groupby('bin').y.agg(['mean','sem','count']).reset_index(); ev['outcome']=kind; ev_all.append(ev)
O=pd.DataFrame(out); O.to_csv(f'{RES}/h3.csv',index=False); E=pd.concat(ev_all); E.to_csv(f'{RES}/h3_event.csv',index=False)
pd.set_option('display.width',220); print(O.round(4).to_string(index=False)); print(E.round(3).to_string(index=False))
