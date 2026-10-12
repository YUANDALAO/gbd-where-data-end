"""H4: GBD vs raw VR associations in data-rich countries. H5: EAPC properties by data richness and GBD-vs-raw EAPC agreement."""
import numpy as np, pandas as pd, statsmodels.formula.api as smf, warnings; warnings.filterwarnings('ignore')
from common import *
D=load_gbd(); CR=pd.read_csv(f'{RES}/country_richness.csv'); g=dict(zip(CR.iso3,CR.grp))
R=pd.read_csv(os.path.join(ROOT,'raw_vr_asr.csv'))
raw=R.pivot_table(index=['iso3','Year'],columns='cause',values='asr')
dth=R.pivot_table(index=['iso3','Year'],columns='cause',values='deaths')
raw['gshare']=(dth.GARBAGE/dth.ALL).values; raw=raw.reset_index()
raw=raw[raw.gshare<0.15]; n=raw.groupby('iso3').Year.nunique(); raw=raw[raw.iso3.isin(n[n>=15].index)]
rng=np.random.default_rng(2026)
# ---------- H4
pairs={'SocioDemographicIdx':['IHD','Stroke','COPD','TBLCancer','Diabetes','CKD','LiverCancer','ColonRectumCancer','StomachCancer','LRI','Dementias','CLD','BreastCancer','PancreaticCancer'],
'Smoking':['TBLCancer','COPD','IHD','Stroke','EsophagealCancer','BladderCancer','StomachCancer','PancreaticCancer'],
'HighBMI':['Diabetes','IHD','CKD','LiverCancer','ColonRectumCancer','KidneyCancer','Stroke'],'HighSBP':['Stroke','IHD','HHD','CKD','AF'],
'HighFBG':['Diabetes','CKD','IHD','Stroke'],'Alcohol':['CLD','LiverCancer','EsophagealCancer','BreastCancer'],
'PM25':['COPD','LRI','IHD','Stroke','TBLCancer','Diabetes'],'HouseholdSolidFuel':['COPD','LRI']}
rows=[]
for x,outs in pairs.items():
    for o in outs:
        s=D[['CountryCode','Year',o+'_Death',x]].rename(columns={'CountryCode':'iso3',o+'_Death':'gbd'}).merge(raw[['iso3','Year',o,'gshare']].rename(columns={o:'raw'}),on=['iso3','Year']).dropna()
        s=s[(s.gbd>0)&(s.raw>0)].copy(); s['lg']=np.log(s.gbd); s['lr']=np.log(s.raw); s['z']=(s[x]-s[x].mean())/s[x].std()
        for spec,ss,f in [('main',s,'{y}~z+C(iso3)+C(Year)'),('gctrl',s,'{y}~z+gshare+C(iso3)+C(Year)'),('strict',s[s.gshare<0.05],'{y}~z+C(iso3)+C(Year)')]:
            bG=smf.ols(f.format(y='lg'),ss).fit().params['z']; bR=smf.ols(f.format(y='lr'),ss).fit().params['z']
            rows.append(dict(spec=spec,x=x,o=o,bG=bG,bR=bR,nc=ss.iso3.nunique()))
H4=pd.DataFrame(rows); H4.to_csv(f'{RES}/h4_pairs.csv',index=False)
out=[]
for spec,t in H4.groupby('spec'):
    ratio=(t.bG.abs()/t.bR.abs()).values; bs=[np.median(rng.choice(ratio,len(ratio))) for _ in range(2000)]
    out.append(dict(test='H4',spec=spec,est=np.median(ratio),lo=np.percentile(bs,2.5),hi=np.percentile(bs,97.5),
        sign_disagree=int((np.sign(t.bG)!=np.sign(t.bR)).sum()),gbd_larger=float((t.bG.abs()>t.bR.abs()).mean()),n_pairs=len(t),n_countries=int(t.nc.median())))
# ---------- H5
def eapc(y,yr):
    X=np.column_stack([np.ones(len(yr)),yr-yr.mean()]); b=np.linalg.lstsq(X,y,rcond=None)[0]; e=y-X@b
    se=np.sqrt(e.var(ddof=2)/((yr-yr.mean())**2).sum()); return 100*(np.exp(b[1])-1),100*(np.exp(b[1]-1.96*se)-1),100*(np.exp(b[1]+1.96*se)-1)
E=[]
for o in outcome_cols(D,'death'):
    for c,t in D[['CountryCode','Year',o]].groupby('CountryCode'):
        t=t[t[o]>0]
        if len(t)<30: continue
        e,l,h=eapc(np.log(t[o].values),t.Year.values.astype(float)); E.append((c,o,e,l,h))
E=pd.DataFrame(E,columns=['iso3','cause','eapc','lo','hi']); E['grp']=E.iso3.map(g); E['width']=E.hi-E.lo; E['sig']=(E.lo>0)|(E.hi<0)
E.to_csv(f'{RES}/h5_eapc_gbd.csv',index=False)
s5=E.groupby('grp').agg(median_CI_width=('width','median'),share_sig=('sig','mean'),n=('sig','size'))
print('H5a GBD EAPC 1990-2021 by richness\n',s5.round(3))
# relative width controlling for cause: log width ratio poor vs rich
E2=E[E.grp!='mid'].copy(); E2['lw']=np.log(E2.width); E2['poor']=(E2.grp=='poor').astype(int)
f=smf.ols('lw~poor+C(cause)',E2).fit(cov_type='cluster',cov_kwds={'groups':E2.iso3.astype('category').cat.codes})
out.append(dict(test='H5a',spec='CI width poor/rich (cause FE)',est=np.exp(f.params['poor']),lo=np.exp(f.conf_int().loc['poor',0]),hi=np.exp(f.conf_int().loc['poor',1]),n_pairs=len(E2)))
out.append(dict(test='H5a',spec='share significant poor',est=E[E.grp=='poor'].sig.mean())); out.append(dict(test='H5a',spec='share significant rich',est=E[E.grp=='rich'].sig.mean()))
# H5b rich countries: GBD vs raw EAPC over identical years
causes=[c for c in raw.columns if c not in ('iso3','Year','gshare','GARBAGE','ALL') and c+'_Death' in D.columns]
B=[]
for o in causes:
    s=D[['CountryCode','Year',o+'_Death']].rename(columns={'CountryCode':'iso3',o+'_Death':'gbd'}).merge(raw[['iso3','Year',o]].rename(columns={o:'raw'}),on=['iso3','Year']).dropna()
    for c,t in s.groupby('iso3'):
        t=t[(t.gbd>0)&(t.raw>0)]
        if len(t)<15: continue
        yr=t.Year.values.astype(float); a=eapc(np.log(t.gbd.values),yr); b=eapc(np.log(t.raw.values),yr)
        B.append((c,o,len(t))+a+b)
B=pd.DataFrame(B,columns=['iso3','cause','nyrs','g','gl','gh','r','rl','rh']); B.to_csv(f'{RES}/h5_eapc_gbd_vs_raw.csv',index=False)
gs=(B.gl>0)|(B.gh<0); rs=(B.rl>0)|(B.rh<0)
res=dict(n=len(B),corr=np.corrcoef(B.g,B.r)[0,1],sign_agree=float((np.sign(B.g)==np.sign(B.r)).mean()),
         gbd_sig=float(gs.mean()),raw_sig=float(rs.mean()),gbd_sig_raw_not=float((gs&~rs).mean()),
         gbd_sig_opposite_raw_sig=float((gs&rs&(np.sign(B.g)!=np.sign(B.r))).mean()),
         raw_inside_gbdCI=float(((B.r>=B.gl)&(B.r<=B.gh)).mean()),median_abs_diff_pp=float((B.g-B.r).abs().median()))
print('H5b',{k:round(v,3) for k,v in res.items()})
for k,v in res.items(): out.append(dict(test='H5b',spec=k,est=v))
O=pd.DataFrame(out); O.to_csv(f'{RES}/h4_h5.csv',index=False); pd.set_option('display.width',220); print(O.round(3).to_string(index=False))
