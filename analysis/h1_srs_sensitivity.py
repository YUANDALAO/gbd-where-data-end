"""Sensitivity of H1 to countries that have national sample registration but no WHO VR data (post hoc).
Rule: countries in the no-VR group with a nationally representative sample registration system operating throughout
1990-2021 - India (Sample Registration System) and Bangladesh (Sample Vital Registration System). WHO VR data cannot show
these data, so the main analysis counts them as having no primary data, which biases the contrast towards the null.
Variants: (1) as in the main analysis; (2) the two countries excluded; (3) the two countries moved to the partial-VR group
(binary contrast) and given 16 years of data on the dose-response scale (the midpoint).
Output: results/h1_srs_sensitivity.csv
"""
import numpy as np, pandas as pd, statsmodels.formula.api as smf, warnings
warnings.filterwarnings('ignore')
from common import load_gbd, richness, roughness_long, outcome_cols, RES

SRS = ['IND', 'BGD']
D = load_gbd(); CR = richness(D); out = []
for kind in ['death', 'prev']:
    R = roughness_long(D, outcome_cols(D, kind))
    m0 = R.groupby(['CountryCode', 'cause']).rough.mean().reset_index().rename(columns={'CountryCode': 'iso3'})
    m0 = m0[m0.rough > 0].merge(CR, on='iso3'); m0['lr'] = np.log(m0.rough); m0['lpop'] = np.log(m0['pop'])
    m0 = m0.dropna(subset=['lr', 'nvr', 'lpop', 'sdi'])
    variants = {'main': m0, 'SRS countries excluded': m0[~m0.iso3.isin(SRS)],
                'SRS countries as partial data': m0.assign(grp=np.where(m0.iso3.isin(SRS), 'mid', m0.grp),
                                                           nvr=np.where(m0.iso3.isin(SRS), 16, m0.nvr))}
    for lab, m in variants.items():
        m = m.assign(rich01=m.nvr / 32)
        f = smf.ols('lr ~ rich01 + lpop + sdi + C(cause)', m).fit(cov_type='cluster', cov_kwds={'groups': m.iso3.astype('category').cat.codes})
        b, se = f.params['rich01'], f.bse['rich01']
        out.append(dict(outcome=kind, variant=lab, contrast='32 v 0 years of VR', est=np.exp(b), lo=np.exp(b - 1.96 * se),
                        hi=np.exp(b + 1.96 * se), p=f.pvalues['rich01'], n_countries=m.iso3.nunique()))
        s = m[m.grp != 'mid'].assign(poor=lambda x: (x.grp == 'poor').astype(int))
        f = smf.ols('lr ~ poor + lpop + sdi + C(cause)', s).fit(cov_type='cluster', cov_kwds={'groups': s.iso3.astype('category').cat.codes})
        b, se = f.params['poor'], f.bse['poor']
        out.append(dict(outcome=kind, variant=lab, contrast='no VR v complete VR', est=np.exp(b), lo=np.exp(b - 1.96 * se),
                        hi=np.exp(b + 1.96 * se), p=f.pvalues['poor'], n_countries=s.iso3.nunique()))
O = pd.DataFrame(out); O.to_csv(f'{RES}/h1_srs_sensitivity.csv', index=False)
pd.set_option('display.width', 200); print(O.round(4).to_string(index=False))
