"""Supplementary tables S1-S7 -> results/Supplementary_tables.xlsx (+ one CSV per table in results/supp/).
Run after h1_h2.py, h3_onset.py, h3_robust.py, h4_h5.py, figs.py (figs.py writes h2_percause_death.csv).
"""
import numpy as np, pandas as pd, pycountry, warnings; warnings.filterwarnings('ignore')
from common import *
from build_raw_vr import ICD

OUT = os.path.join(RES, 'supp'); os.makedirs(OUT, exist_ok=True)
D = load_gbd(); CR = pd.read_csv(f'{RES}/country_richness.csv')
reg = D[D.Year == 2019].set_index('CountryCode').RegionName

def cname(iso):
    try: return pycountry.countries.get(alpha_3=iso).name
    except Exception: return iso

T = {}
# S1 ICD-10 mapping
T['S1_ICD10_mapping'] = pd.DataFrame([(k, '; '.join(f'{a}-{b}' if a != b else a for a, b in v)) for k, v in ICD.items()],
                                     columns=['Cause (GBD label in this study)', 'ICD-10 three-character codes'])

# S2 H4 pairs
P = pd.read_csv(f'{RES}/h4_pairs.csv')
S2 = P.pivot_table(index=['x', 'o'], columns='spec', values=['bG', 'bR']).round(3)
S2.columns = [f'{b}_{s}' for b, s in S2.columns]; S2 = S2.reset_index()
S2['ratio_main'] = (S2.bG_main.abs() / S2.bR_main.abs()).round(2)
S2['sign_disagree_main'] = np.sign(S2.bG_main) != np.sign(S2.bR_main)
S2 = S2.rename(columns={'x': 'Exposure', 'o': 'Cause of death'})
T['S2_H4_exposure_outcome_pairs'] = S2

# S3 H1/H2 all models + H2 per cause
T['S3a_H1_H2_models'] = pd.read_csv(f'{RES}/h1_h2.csv').round(4)
h2 = pd.read_csv(f'{RES}/h2_percause_death.csv').round(3); h2['difference'] = (h2.poor - h2.rich).round(3)
T['S3b_H2_R2_by_cause_deaths'] = h2.rename(columns={'rich': 'R2 data-rich', 'poor': 'R2 no-VR'}).sort_values('difference', ascending=False)

# S4 H3: treated countries, leave-one-out, event study, robust estimators
Y = vr_years(); cnt = Y[Y.Year <= 2021].groupby('iso3').Year.nunique()
tr = CR[(CR.onset >= 1998) & (CR.onset <= 2013) & (CR.iso3.map(cnt).fillna(0) >= 5)].copy()
tr['country'] = tr.iso3.map(cname); tr['VR years 1990-2021'] = tr.iso3.map(cnt).astype(int)
T['S4a_H3_onset_countries'] = tr[['iso3', 'country', 'onset', 'VR years 1990-2021', 'wb', 'pop', 'sdi']].rename(
    columns={'onset': 'first VR year', 'wb': 'max completeness (%)', 'pop': 'population 2019', 'sdi': 'SDI 2019'}).sort_values('first VR year')

def twfe(panel, post):
    y = panel.y.values.copy(); p = post.astype(float).copy(); c = panel.CountryCode.values; t = panel.Year.values
    for _ in range(50):
        for grp in (c, t):
            y -= pd.Series(y).groupby(grp).transform('mean').values; p -= pd.Series(p).groupby(grp).transform('mean').values
    return (p @ y) / (p @ p)
onset = tr.set_index('iso3').onset
keep = set(onset.index) | set(CR[CR.onset <= 1990].iso3) | set(CR[CR.onset.isna()].iso3)
loo = []
for kind in ['death', 'prev']:
    R = roughness_long(D[D.CountryCode.isin(keep)], outcome_cols(D, kind)); R = R[R.rough > 0]; R['lr'] = np.log(R.rough)
    R['lr'] -= R.groupby(['cause', 'Year']).lr.transform('mean'); R['lr'] -= R.groupby(['CountryCode', 'cause']).lr.transform('mean')
    Pn = R.groupby(['CountryCode', 'Year']).lr.mean().rename('y').reset_index()
    for c in ['(none)'] + list(onset.index):
        Q = Pn[Pn.CountryCode != c]; o = Q.CountryCode.map(onset)
        loo.append((kind, c, round(float(np.exp(twfe(Q, ((o.notna()) & (Q.Year >= o)).values))), 3)))
T['S4b_H3_leave_one_out'] = pd.DataFrame(loo, columns=['outcome', 'country omitted', 'roughness ratio'])
ev = pd.read_csv(f'{RES}/h3_event.csv'); ev['lo'] = ev['mean'] - 1.96 * ev['sem']; ev['hi'] = ev['mean'] + 1.96 * ev['sem']
T['S4c_H3_event_study'] = ev.round(3)
T['S4d_H3_robust_estimators'] = pd.read_csv(f'{RES}/h3_robust.csv').round(3)
T['S4e_H3_TWFE_primary'] = pd.read_csv(f'{RES}/h3.csv').round(4)

# S5 H5 EAPC
E = pd.read_csv(f'{RES}/h5_eapc_gbd.csv')
T['S5a_H5_EAPC_by_group'] = E.groupby('grp').agg(series=('sig', 'size'), median_CI_width=('width', 'median'),
                                                  IQR_low=('width', lambda s: s.quantile(.25)), IQR_high=('width', lambda s: s.quantile(.75)),
                                                  share_significant=('sig', 'mean')).round(3).reset_index()
B = pd.read_csv(f'{RES}/h5_eapc_gbd_vs_raw.csv')
B['gs'] = (B.gl > 0) | (B.gh < 0); B['rs'] = (B.rl > 0) | (B.rh < 0)
B['inside'] = (B.r >= B.gl) & (B.r <= B.gh); B['opp'] = B.gs & B.rs & (np.sign(B.g) != np.sign(B.r))
T['S5b_H5_agreement_by_cause'] = B.groupby('cause').apply(lambda d: pd.Series({
    'series': len(d), 'correlation': np.corrcoef(d.g, d.r)[0, 1], 'same_sign': (np.sign(d.g) == np.sign(d.r)).mean(),
    'raw_inside_GBD_CI': d.inside.mean(), 'GBD_sig_raw_not': (d.gs & ~d.rs).mean(), 'opposite_significant': d.opp.mean()})).round(3).reset_index()

# S6 country list
S6 = CR.copy(); S6['country'] = S6.iso3.map(cname); S6['GBD region'] = S6.iso3.map(reg)
T['S6_countries_data_availability'] = S6[['iso3', 'country', 'GBD region', 'nvr', 'onset', 'wb', 'grp', 'pop', 'sdi']].rename(
    columns={'nvr': 'VR years 1990-2021', 'onset': 'first VR year (>=1980)', 'wb': 'max completeness (%)', 'grp': 'group',
             'pop': 'population 2019', 'sdi': 'SDI 2019'}).sort_values(['group', 'iso3'])

# S7 literature counts
T['S7_PubMed_counts'] = pd.read_csv(f'{RES}/h6_pubmed.csv')

with pd.ExcelWriter(f'{RES}/Supplementary_tables.xlsx') as xw:
    for k, v in T.items():
        v.to_excel(xw, sheet_name=k[:31], index=False); v.to_csv(f'{OUT}/{k}.csv', index=False)
print({k: v.shape for k, v in T.items()})
