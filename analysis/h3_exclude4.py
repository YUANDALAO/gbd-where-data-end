"""Sensitivity analysis for H3 (added after the primary analysis): drop the four treated countries whose WHO
reporting onset may not mark the start of data available to GBD - Serbia and Montenegro (previously part of Yugoslavia),
Iran (national death registration system used by GBD before WHO reporting) and Turkiye (earlier urban death statistics).
Same two-way fixed-effects model, 2000 permutations and leave-one-out as h3_onset.py. Output: results/h3_exclude4.csv
"""
import numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
from scipy import stats
from common import *
import importlib.util
spec = importlib.util.spec_from_file_location('h3', os.path.join(os.path.dirname(os.path.abspath(__file__)), 'h3_onset.py'))
src = open(spec.origin, encoding='utf-8').read()
exec(src.split('for kind in')[0])                                   # data, treated/always/never sets, twfe(), cluster_se()
DROP = ['SRB', 'MNE', 'IRN', 'TUR']
treated = treated.drop([c for c in DROP if c in treated.index]); keep = set(treated.index) | always | never
rng = np.random.default_rng(2026); out = []
for kind in ['death', 'prev']:
    R = roughness_long(D[D.CountryCode.isin(keep)], outcome_cols(D, kind)); R = R[R.rough > 0]; R['lr'] = np.log(R.rough)
    R['lr'] -= R.groupby(['cause', 'Year']).lr.transform('mean'); R['lr'] -= R.groupby(['CountryCode', 'cause']).lr.transform('mean')
    P = R.groupby(['CountryCode', 'Year']).lr.mean().rename('y').reset_index()
    on = P.CountryCode.map(treated); post = ((on.notna()) & (P.Year >= on)).values
    b, yw, pw = twfe(P, post); se = cluster_se(yw, pw, b, P.CountryCode.values)
    ctrl = P[~P.CountryCode.isin(treated.index)].reset_index(drop=True); cands = sorted(set(ctrl.CountryCode)); perm = []
    for _ in range(2000):
        pick = rng.choice(cands, len(treated), replace=False); fake = dict(zip(pick, rng.permutation(treated.values)))
        o2 = ctrl.CountryCode.map(fake); perm.append(twfe(ctrl, ((o2.notna()) & (ctrl.Year >= o2)).values)[0])
    pperm = np.mean(np.abs(np.array(perm)) >= abs(b)); loo = []
    for c in treated.index:
        Q = P[P.CountryCode != c]; oq = Q.CountryCode.map(treated); loo.append(twfe(Q, ((oq.notna()) & (Q.Year >= oq)).values)[0])
    out.append(dict(test='H3 excluding SRB MNE IRN TUR', outcome=kind, est=np.exp(b), lo=np.exp(b - 1.96 * se), hi=np.exp(b + 1.96 * se),
                    p_cluster=2 * stats.norm.sf(abs(b / se)), p_perm=pperm, loo_min=np.exp(min(loo)), loo_max=np.exp(max(loo)),
                    n_treated=len(treated), n_countries=P.CountryCode.nunique()))
    print(out[-1], flush=True)
pd.DataFrame(out).to_csv(f'{RES}/h3_exclude4.csv', index=False)
