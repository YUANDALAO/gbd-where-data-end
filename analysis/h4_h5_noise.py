"""Agreement analyses allowing for the sampling error of raw registration data (added after the primary analysis).
H4: the 50 exposure-outcome pairs of h4_h5.py re-estimated with country-clustered SEs; pairs with a significant raw
    association, sign disagreements among them, and z tests of the GBD-raw difference.
H5b: z test of the difference between GBD and raw EAPC combining both standard errors (from h5_eapc_gbd_vs_raw.csv).
Outputs: results/h4_pairs_se.csv, results/h4_h5_noise.csv
"""
import numpy as np, pandas as pd, statsmodels.formula.api as smf, warnings; warnings.filterwarnings('ignore')
from scipy import stats
import os
exec(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'h4_h5.py'), encoding='utf-8').read().split('rows=[]')[0])

rows = []
for x, outs in pairs.items():
    for o in outs:
        s = D[['CountryCode', 'Year', o + '_Death', x]].rename(columns={'CountryCode': 'iso3', o + '_Death': 'gbd'}).merge(
            raw[['iso3', 'Year', o, 'gshare']].rename(columns={o: 'raw'}), on=['iso3', 'Year']).dropna()
        s = s[(s.gbd > 0) & (s.raw > 0)].copy(); s['lg'] = np.log(s.gbd); s['lr'] = np.log(s.raw); s['z'] = (s[x] - s[x].mean()) / s[x].std()
        fit = lambda y: smf.ols(f'{y}~z+C(iso3)+C(Year)', s).fit(cov_type='cluster', cov_kwds={'groups': pd.factorize(s.iso3)[0]})
        a, b = fit('lg'), fit('lr')
        rows.append(dict(x=x, o=o, bG=a.params['z'], seG=a.bse['z'], bR=b.params['z'], seR=b.bse['z']))
H = pd.DataFrame(rows); H['tR'] = H.bR / H.seR; H['z_diff'] = (H.bG - H.bR) / np.sqrt(H.seG ** 2 + H.seR ** 2)
H.to_csv(f'{RES}/h4_pairs_se.csv', index=False)
sig = H[H.tR.abs() > 1.96]
E = pd.read_csv(f'{RES}/h5_eapc_gbd_vs_raw.csv'); seg, ser = (E.gh - E.gl) / 3.92, (E.rh - E.rl) / 3.92
zE = (E.g - E.r) / np.sqrt(seg ** 2 + ser ** 2)
out = pd.DataFrame([
    dict(test='H4', stat='pairs with significant raw association', value=len(sig)),
    dict(test='H4', stat='median abs(bG)/abs(bR), raw-significant pairs', value=float((sig.bG.abs() / sig.bR.abs()).median())),
    dict(test='H4', stat='sign disagreements, raw-significant pairs', value=int((np.sign(sig.bG) != np.sign(sig.bR)).sum())),
    dict(test='H4', stat='pairs with significant GBD-raw difference', value=int((H.z_diff.abs() > 1.96).sum())),
    dict(test='H5b', stat='series with significant GBD-raw EAPC difference (share)', value=float((zE.abs() > 1.96).mean())),
    dict(test='H5b', stat='median SE ratio raw/GBD', value=float((ser / seg).median())),
    dict(test='H5b', stat='GBD EAPC inside raw 95% CI (share)', value=float(((E.g >= E.rl) & (E.g <= E.rh)).mean()))])
out.to_csv(f'{RES}/h4_h5_noise.csv', index=False); print(out.to_string(index=False))
