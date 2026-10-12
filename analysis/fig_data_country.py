"""Country-level figure data for the BMJ figures (descriptive; the tests are in h1_h2.py, h3_onset.py, h4_h5.py).
results/fig_country_metrics.csv : per country - years of VR, group, SDI 2019, population, GBD region,
    relative roughness of death trajectories (geometric mean over causes, cause-adjusted, 1992-2021; 1 = average country),
    relative width of the conventional 95% CI of GBD EAPC (median over causes of width / cause median),
    and, for data-rich countries with raw VR, share of series whose raw EAPC lies inside the GBD CI and share agreeing in sign.
results/fig_onset_countries.csv : per treated country and year, cause-adjusted log roughness (country-cause and
    cause-year means removed), for the 13 countries whose national VR began in 1998-2013.
"""
import os
import numpy as np, pandas as pd
from common import load_gbd, outcome_cols, roughness_long, richness, RES

D = load_gbd(); C = richness(D)
reg = D[D.Year == 2019].set_index('CountryCode').RegionName
deaths = outcome_cols(D, 'death')
R = roughness_long(D, deaths); R = R[(R.Year >= 1992) & (R.rough > 0)].copy(); R['lr'] = np.log(R.rough)
R['lr_c'] = R.lr - R.groupby('cause').lr.transform('mean')
C['rough_rel'] = C.iso3.map(np.exp(R.groupby('CountryCode').lr_c.mean()))
C['region'] = C.iso3.map(reg)

E = pd.read_csv(os.path.join(RES, 'h5_eapc_gbd.csv'))
E['w_rel'] = E.width / E.groupby('cause').width.transform('median')
C['ci_width_rel'] = C.iso3.map(E.groupby('iso3').w_rel.median())

V = pd.read_csv(os.path.join(RES, 'h5_eapc_gbd_vs_raw.csv'))
V['inside'] = (V.r >= V.gl) & (V.r <= V.gh); V['sign'] = np.sign(V.g) == np.sign(V.r)
A = V.groupby('iso3').agg(n_series=('cause', 'size'), inside=('inside', 'mean'), sign=('sign', 'mean'))
C = C.join(A, on='iso3')
C.to_csv(os.path.join(RES, 'fig_country_metrics.csv'), index=False)

# onset countries (same definition as h3_onset.py)
cnt = None
try:
    from common import vr_years
    Y = vr_years(); cnt = Y[(Y.Year >= 1990) & (Y.Year <= 2021)].groupby('iso3').Year.nunique()
except Exception:
    pass
treated = C[(C.onset >= 1998) & (C.onset <= 2013) & (C.iso3.map(cnt).fillna(0) >= 5)].set_index('iso3').onset
R['lr_cy'] = R.lr - R.groupby(['cause', 'Year']).lr.transform('mean')
T = R[R.CountryCode.isin(treated.index)].copy()
T['adj'] = T.lr_cy - T.groupby(['CountryCode', 'cause']).lr_cy.transform('mean')
O = T.groupby(['CountryCode', 'Year']).adj.mean().reset_index()
O['onset'] = O.CountryCode.map(treated); O['et'] = O.Year - O.onset
O.to_csv(os.path.join(RES, 'fig_onset_countries.csv'), index=False)
print('countries', C.rough_rel.notna().sum(), 'with agreement', C.inside.notna().sum(), 'treated', len(treated))
print(treated.astype(int).to_dict())
print(C.groupby('grp')[['rough_rel', 'ci_width_rel', 'inside']].median().round(3))
