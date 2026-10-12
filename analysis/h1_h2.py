"""H1 smoothness dose-response; H2 covariate tracking with country bootstrap."""
import numpy as np, pandas as pd, statsmodels.formula.api as smf, warnings; warnings.filterwarnings('ignore')
from common import *
D=load_gbd(); CR=richness(D); CR.to_csv(f'{RES}/country_richness.csv',index=False)
print(CR.grp.value_counts().to_dict())
out=[]
# ---- H1
for kind in ['death','prev']:
    R=roughness_long(D,outcome_cols(D,kind))
    m=R.groupby(['CountryCode','cause']).rough.mean().reset_index().rename(columns={'CountryCode':'iso3'})
    m=m[m.rough>0].merge(CR,on='iso3'); m['lr']=np.log(m.rough); m['rich01']=m.nvr/32; m['lpop']=np.log(m['pop'])
    m=m.dropna(subset=['lr','rich01','lpop','sdi'])
    for lab,s in [('all',m),('pop5M',m[m['pop']>=5e6])]:
        f=smf.ols('lr ~ rich01 + lpop + sdi + C(cause)',s).fit(cov_type='cluster',cov_kwds={'groups':s.iso3.astype('category').cat.codes})
        b,se=f.params['rich01'],f.bse['rich01']
        out.append(dict(test='H1',outcome=kind,sample=lab,est=np.exp(b),lo=np.exp(b-1.96*se),hi=np.exp(b+1.96*se),p=f.pvalues['rich01'],n_countries=s.iso3.nunique(),n=len(s),
                        note='ratio of roughness, full VR (32y) vs none'))
    # binary contrast too
    s=m[m.grp!='mid'].copy(); s['poor']=(s.grp=='poor').astype(int)
    f=smf.ols('lr ~ poor + lpop + sdi + C(cause)',s).fit(cov_type='cluster',cov_kwds={'groups':s.iso3.astype('category').cat.codes})
    b,se=f.params['poor'],f.bse['poor']
    out.append(dict(test='H1',outcome=kind,sample='poor_vs_rich',est=np.exp(b),lo=np.exp(b-1.96*se),hi=np.exp(b+1.96*se),p=f.pvalues['poor'],n_countries=s.iso3.nunique(),n=len(s),note='ratio poor/rich'))
# ---- H2
def r2(y,X):
    X=np.column_stack([np.ones(len(y)),X]); b=np.linalg.lstsq(X,y,rcond=None)[0]; e=y-X@b; return 1-e.var()/y.var()
big=set(CR[CR["pop"]>=5e6].iso3); g=dict(zip(CR.iso3,CR.grp))
Dd=D[D.Year.isin(range(1991,2022,5))&D.CountryCode.isin(big)].sort_values(['CountryCode','Year']).copy()
rng=np.random.default_rng(2026)
for kind in ['death','prev']:
    rows={}
    for o in outcome_cols(D,kind):
        s=Dd[['CountryCode','Year',o]+X9].dropna(); s=s[s[o]>0].copy(); s['y']=np.log(s[o])
        for v in ['y']+X9: s['d'+v]=s.groupby('CountryCode')[v].diff()
        s=s.dropna(); s['grp']=s.CountryCode.map(g); rows[o]=s
    def stat(sample_countries):
        vals={'rich':[],'poor':[]}
        for o,s in rows.items():
            for G in vals:
                t=s[s.CountryCode.isin(sample_countries[G])]
                if len(t)<20: continue
                dX=t[['d'+x for x in X9]].values; sd=dX.std(0); sd[sd==0]=1
                vals[G].append(r2(t.dy.values,(dX-dX.mean(0))/sd))
        return np.mean(vals['poor'])-np.mean(vals['rich']), np.mean(vals['poor']), np.mean(vals['rich'])
    base={G:[c for c in big if g.get(c)==G] for G in ['rich','poor']}
    d0,p0,r0=stat({G:base[G] for G in base})
    boots=[]
    for _ in range(200):
        bs={G:list(rng.choice(base[G],len(base[G]),replace=True)) for G in base}
        # duplicates handled by isin (approximate cluster bootstrap)
        boots.append(stat(bs)[0])
    lo,hi=np.percentile(boots,[2.5,97.5])
    out.append(dict(test='H2',outcome=kind,sample='pop5M_5yr',est=d0,lo=lo,hi=hi,p=np.mean(np.array(boots)<=0)*2,n_countries=len(base['rich'])+len(base['poor']),n=len(rows),
                    note=f'mean R2 poor {p0:.3f} - rich {r0:.3f}'))
O=pd.DataFrame(out); O.to_csv(f'{RES}/h1_h2.csv',index=False)
pd.set_option('display.width',220); print(O.round(4).to_string(index=False))
