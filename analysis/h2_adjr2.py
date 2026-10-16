"""H2 with adjusted R-squared (added after the primary analysis): same data, groups and 200 country bootstraps
as h1_h2.py, with R2 replaced by 1 - (1 - R2)(n - 1)/(n - k - 1), k = 9 covariates. Output: results/h2_adjr2.csv
"""
import numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
from common import *
D = load_gbd(); CR = pd.read_csv(f'{RES}/country_richness.csv')
def r2a(y, X):
    Xc = np.column_stack([np.ones(len(y)), X]); b = np.linalg.lstsq(Xc, y, rcond=None)[0]; e = y - Xc @ b
    r2 = 1 - e.var() / y.var(); n, k = len(y), X.shape[1]; return 1 - (1 - r2) * (n - 1) / (n - k - 1)
big = set(CR[CR['pop'] >= 5e6].iso3); g = dict(zip(CR.iso3, CR.grp))
Dd = D[D.Year.isin(range(1991, 2022, 5)) & D.CountryCode.isin(big)].sort_values(['CountryCode', 'Year']).copy()
rng = np.random.default_rng(2026); out = []
for kind in ['death', 'prev']:
    rows = {}
    for o in outcome_cols(D, kind):
        s = Dd[['CountryCode', 'Year', o] + X9].dropna(); s = s[s[o] > 0].copy(); s['y'] = np.log(s[o])
        for v in ['y'] + X9: s['d' + v] = s.groupby('CountryCode')[v].diff()
        rows[o] = s.dropna()
    def stat(sc):
        vals = {'rich': [], 'poor': []}
        for o, s in rows.items():
            for G in vals:
                t = s[s.CountryCode.isin(sc[G])]
                if len(t) < 20: continue
                dX = t[['d' + x for x in X9]].values; sd = dX.std(0); sd[sd == 0] = 1
                vals[G].append(r2a(t.dy.values, (dX - dX.mean(0)) / sd))
        return np.mean(vals['poor']) - np.mean(vals['rich']), np.mean(vals['poor']), np.mean(vals['rich'])
    base = {G: [c for c in big if g.get(c) == G] for G in ['rich', 'poor']}
    d0, p0, r0 = stat(base); boots = []
    for _ in range(200):
        boots.append(stat({G: list(rng.choice(base[G], len(base[G]))) for G in base})[0])
    out.append(dict(outcome=kind, adj_r2_poor=p0, adj_r2_rich=r0, diff=d0, lo=np.percentile(boots, 2.5), hi=np.percentile(boots, 97.5)))
    print(out[-1], flush=True)
pd.DataFrame(out).to_csv(f'{RES}/h2_adjr2.csv', index=False)
