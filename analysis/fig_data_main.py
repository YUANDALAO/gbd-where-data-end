"""Inputs for the four main figures that are not already in results/ (all descriptive; the tests are in h1-h6 scripts).
Writes results/figdata/:
  rough_matrix.csv      country x cause mean log roughness of GBD death trajectories, 1992-2021, cause-centred (Fig 1c)
  rough_cause_ratio.csv per cause, unadjusted ratio of geometric-mean roughness, complete VR v no national VR (Fig 1c)
  h1_by_category.csv    H1 binary contrast (no VR v complete VR, adjusted for log population and SDI, cause FE,
                        country-clustered SE) estimated within each cause category (Fig 1f; added after the primary analysis)
  trajectories.csv      log GBD death rate, country-centred, for 4 causes in the 6 most populous countries of each group (Fig 1d)
  h2_partial_r2.csv     H2 drop-one-covariate partial R2, mean over causes, by group (Fig 2d; added after the primary analysis)
  h3_country.csv        per treated country: ratio of mean adjusted roughness after v before onset, and leave-one-out
                        pooled estimate (Fig 3a)
"""
import os
import numpy as np, pandas as pd, statsmodels.formula.api as smf, warnings; warnings.filterwarnings('ignore')
from common import *

OUT = os.path.join(RES, 'figdata'); os.makedirs(OUT, exist_ok=True)
CAT = {'Cardiovascular': 'CVD HHD IHD RHD AF CMY NRVHD PAH Stroke IschemicStroke ICH',
       'Neoplasms': 'TotalCancers Neoplasms EsophagealCancer StomachCancer LarynxCancer TBLCancer BreastCancer CervicalCancer UterineCancer '
                    'ProstateCancer ColonRectumCancer LOCancer NasopharynxCancer GBTCancer PancreaticCancer MSM NMSCancer OvarianCancer '
                    'TesticularCancer KidneyCancer BladderCancer BCNSCancer ThyroidCancer Mesothelioma EyeCancer STSarcomas MBCNeoplasm '
                    'MMHNeoplasm NeuroPNTumors HodgkinLymphoma NonHodgkinLymphoma BurkittLymphoma MultipleMyeloma Leukemia',
       'Chronic respiratory': 'CRD COPD Asthma ILD CoalWrkPneumoconiosis Silicosis Asbestosis Pneumoconiosis OtherPneumoconiosis OtherCRD',
       'Digestive': 'UGIDiseases PUD GandD Appendicitis IBD PIIO Hernia VID CLD GallBiliaryDiseases Pancreatitis NAFLD',
       'Neurological & mental': 'NeuroDisorders Dementias ParkinsonDisease IdiopathicEpilepsy MultipleSclerosis MotorNeuronDisease '
                                'MentalDisorders EatingDisorders AlcoholUseDisorder',
       'Diabetes & kidney': 'Diabetes T1DM T2DM CKD CKD1 CKD2 AGN',
       'Infectious, maternal & neonatal': 'URI LRI Tuberculosis RIandTB MDRTB XDRTB NutriDeficien PEM NeonatalDisorders PTBirth '
                                          'NeonatalInfections Measles VZVandHZ DiarrhealDiseases EntericInfection OtherIID HIVandAIDS HBV HCV EctopicPregnancy',
       'Other NCDs': 'EMBIDisorders SIDS RA MSD SkinSD BSkinDiseases UTIandIN Urolithiasis UterineFibroids'}
CAUSE_CAT = {c: k for k, v in CAT.items() for c in v.split()}

D = load_gbd(); CM = pd.read_csv(os.path.join(RES, 'fig_country_metrics.csv')); grp = dict(zip(CM.iso3, CM.grp))
deaths = outcome_cols(D, 'death'); assert set(c[:-6] for c in deaths) == set(CAUSE_CAT), 'cause list changed'
pd.DataFrame({'cause': list(CAUSE_CAT), 'category': list(CAUSE_CAT.values())}).to_csv(os.path.join(OUT, 'cause_category.csv'), index=False)

# ---- Fig 1c, 1f: roughness matrix and per-category H1
R = roughness_long(D, deaths); R = R[(R.Year >= 1992) & (R.rough > 0)].copy(); R['lr'] = np.log(R.rough); R['cause'] = R.cause.str[:-6]
M = R.groupby(['CountryCode', 'cause']).lr.mean().rename('lr').reset_index().rename(columns={'CountryCode': 'iso3'})
M['lr_c'] = M.lr - M.groupby('cause').lr.transform('mean'); M.to_csv(os.path.join(OUT, 'rough_matrix.csv'), index=False)
M['grp'] = M.iso3.map(grp)
G = M[M.grp.isin(['rich', 'poor'])].groupby(['cause', 'grp']).lr.mean().unstack()
pd.DataFrame({'cause': G.index, 'ratio': np.exp(G.rich - G.poor).values}).to_csv(os.path.join(OUT, 'rough_cause_ratio.csv'), index=False)
# same model as the H1 binary contrast in h1_h2.py (country-cause mean roughness over all years), within each category
R1 = roughness_long(D, deaths); m = R1.groupby(['CountryCode', 'cause']).rough.mean().reset_index().rename(columns={'CountryCode': 'iso3'})
m = m[m.rough > 0].merge(CM[['iso3', 'grp', 'pop', 'sdi']], on='iso3'); m['lr'] = np.log(m.rough); m['lpop'] = np.log(m['pop'])
m = m[m.grp != 'mid'].dropna(subset=['lr', 'lpop', 'sdi']); m['poor'] = (m.grp == 'poor').astype(int); m['cat'] = m.cause.str[:-6].map(CAUSE_CAT)
rows = []
for k, s in list(m.groupby('cat')) + [('All causes', m)]:
    f = smf.ols('lr ~ poor + lpop + sdi + C(cause)', s).fit(cov_type='cluster', cov_kwds={'groups': s.iso3.astype('category').cat.codes})
    b, se = -f.params['poor'], f.bse['poor']                          # complete VR v no VR
    rows.append(dict(category=k, ratio=np.exp(b), lo=np.exp(b - 1.96 * se), hi=np.exp(b + 1.96 * se), n_causes=s.cause.nunique(), n_countries=s.iso3.nunique()))
pd.DataFrame(rows).to_csv(os.path.join(OUT, 'h1_by_category.csv'), index=False); print(pd.DataFrame(rows).round(3).to_string(index=False))

# ---- Fig 1d: trajectories
pick = {g: CM[CM.grp == g].nlargest(6, 'pop').iso3.tolist() for g in ['rich', 'poor']}
T = []
for o in ['IHD', 'Stroke', 'COPD', 'Diabetes']:
    for g, isos in pick.items():
        t = D[D.CountryCode.isin(isos)][['CountryCode', 'Year', o + '_Death']].rename(columns={o + '_Death': 'rate'})
        t = t[t.rate > 0].copy(); t['l'] = np.log(t.rate); t['l'] -= t.groupby('CountryCode').l.transform('mean'); t['cause'] = o; t['grp'] = g; T.append(t)
pd.concat(T).to_csv(os.path.join(OUT, 'trajectories.csv'), index=False); print('trajectory countries', pick)

# ---- Fig 2d: H2 model (same sample and specification as h1_h2.py)
def fit(y, X):
    Xc = np.column_stack([np.ones(len(y)), X]); b = np.linalg.lstsq(Xc, y, rcond=None)[0]; e = y - Xc @ b; return 1 - e.var() / y.var(), b
big = set(CM[CM['pop'] >= 5e6].iso3)
Dd = D[D.Year.isin(range(1991, 2022, 5)) & D.CountryCode.isin(big)].sort_values(['CountryCode', 'Year']).copy()
PR = []
for o in deaths:
    s = Dd[['CountryCode', 'Year', o] + X9].dropna(); s = s[s[o] > 0].copy(); s['y'] = np.log(s[o])
    for v in ['y'] + X9: s['d' + v] = s.groupby('CountryCode')[v].diff()
    s = s.dropna(); s['grp'] = s.CountryCode.map(grp)
    for g in ['rich', 'poor']:
        t = s[s.grp == g]
        if len(t) < 20: continue
        dX = t[['d' + x for x in X9]].values; mu, sd = dX.mean(0), dX.std(0); sd[sd == 0] = 1; Z = (dX - mu) / sd
        full = fit(t.dy.values, Z)[0]
        for k, x in enumerate(X9): PR.append(dict(cause=o[:-6], grp=g, covariate=x, partial=full - fit(t.dy.values, np.delete(Z, k, 1))[0]))
PR = pd.DataFrame(PR).groupby(['covariate', 'grp']).partial.mean().unstack().reset_index(); PR.to_csv(os.path.join(OUT, 'h2_partial_r2.csv'), index=False); print(PR.round(4))

# ---- Fig 3a: per-country ratio and leave-one-out (same panel and TWFE as h3_onset.py)
src = open(os.path.join(HERE, 'h3_onset.py'), encoding='utf-8').read(); ns = {}; exec(src.split('for kind in')[0], ns)
treated, keep, twfe = ns['treated'], ns['keep'], ns['twfe']
P = roughness_long(D[D.CountryCode.isin(keep)], deaths); P = P[P.rough > 0]; P['lr'] = np.log(P.rough)
P['lr'] -= P.groupby(['cause', 'Year']).lr.transform('mean'); P['lr'] -= P.groupby(['CountryCode', 'cause']).lr.transform('mean')
P = P.groupby(['CountryCode', 'Year']).lr.mean().rename('y').reset_index()
O = pd.read_csv(os.path.join(RES, 'fig_onset_countries.csv')); rows = []
for c in treated.index:
    o = O[O.CountryCode == c]; Q = P[P.CountryCode != c]; oq = Q.CountryCode.map(treated)
    rows.append(dict(iso3=c, onset=int(treated[c]), ratio=np.exp(o[o.et >= 0].adj.mean() - o[o.et < 0].adj.mean()),
                     loo=np.exp(twfe(Q, ((oq.notna()) & (Q.Year >= oq)).values)[0])))
pd.DataFrame(rows).to_csv(os.path.join(OUT, 'h3_country.csv'), index=False); print(pd.DataFrame(rows).round(3).to_string(index=False))
