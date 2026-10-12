"""Shared data loading for the GBD circularity audit. Raw WHO files live in WHO_DIR (not in repo)."""
import os, glob, numpy as np, pandas as pd
HERE=os.path.dirname(os.path.abspath(__file__))
ROOT=os.path.abspath(os.path.join(HERE,'..'))
# Public raw inputs (fetch_public_data.py). Override with WHO_MDB_DIR.
WHO_DIR=os.environ.get('WHO_MDB_DIR', os.path.join(HERE,'data_public'))
# GBD 2021 extract (GBD Results Tool). Override with GBD_CSV.
GBD_CSV=os.environ.get('GBD_CSV', os.path.join(ROOT,'data','gbd2021_extract.csv'))
RES=os.path.join(HERE,'results')
X9=['SocioDemographicIdx','MedianAge','Smoking','PM25','HighBMI','HouseholdSolidFuel','HighFBG','HighSBP','UnsafeWaterSource']

def load_gbd():
    return pd.read_csv(GBD_CSV)

def outcome_cols(D, kind):
    deaths=[c for c in D.columns if c.endswith('_Death') and not c.startswith(('LiverCancer_','COVID'))]
    if kind=='death': return deaths
    if kind=='prev':  return [c[:-6] for c in deaths if c[:-6] in D.columns]
    raise ValueError(kind)

def vr_years():
    """Country-years (1980+) with national VR data in WHO MDB (ICD-9 or ICD-10)."""
    cache=os.path.join(RES,'vr_years_all.csv')
    if os.path.exists(cache): return pd.read_csv(cache)
    old=os.getcwd(); os.chdir(WHO_DIR)
    ns={}; exec(open(os.path.join(HERE,'isomap.py'),encoding='utf-8').read(),ns); CMAP=ns['CMAP']
    files=[f for f in glob.glob('whomdb/Morticd10_part?')]+['icd9/Morticd9']
    Y=[]
    for f in files:
        x=pd.read_csv(f,usecols=['Country','Admin1','Year'],dtype={'Admin1':str},low_memory=False)
        Y.append(x[x.Admin1.isna()][['Country','Year']].drop_duplicates())
    os.chdir(old)
    Y=pd.concat(Y).drop_duplicates(); Y['iso3']=Y.Country.map(CMAP); Y=Y.dropna()
    Y=Y[Y.Year>=1980][['iso3','Year']].drop_duplicates(); Y.to_csv(cache,index=False); return Y

def richness(D):
    Y=vr_years()
    y9021=Y[(Y.Year>=1990)&(Y.Year<=2021)]
    c=pd.DataFrame({'iso3':sorted(D.CountryCode.unique())})
    c['nvr']=c.iso3.map(y9021.groupby('iso3').Year.nunique()).fillna(0)
    c['onset']=c.iso3.map(Y.groupby('iso3').Year.min())
    wb=pd.read_csv(os.path.join(ROOT,'wb_vr.csv')).groupby('iso3').vr.max()
    c['wb']=c.iso3.map(wb)
    c['grp']=np.where((c.nvr>=15)&(c.wb.fillna(0)>=90),'rich',np.where(c.nvr==0,'poor','mid'))
    cs=D[D.Year==2019].set_index('CountryCode')
    c['pop']=c.iso3.map(cs.Population); c['sdi']=c.iso3.map(cs.SocioDemographicIdx)
    return c

def roughness_long(D, cols):
    """Per country-cause-year |second difference| of log rate."""
    L=[]
    D=D.sort_values(['CountryCode','Year'])
    for o in cols:
        t=D[['CountryCode','Year',o]].copy(); t=t[t[o]>0]; t['l']=np.log(t[o])
        g=t.groupby('CountryCode').l
        t['rough']=(t.l-2*g.shift(1)+g.shift(2)).abs(); t['cause']=o
        L.append(t.dropna(subset=['rough'])[['CountryCode','Year','cause','rough']])
    return pd.concat(L,ignore_index=True)
