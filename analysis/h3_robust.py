"""H3 robustness: heterogeneity-robust staggered DiD.
- Callaway & Sant'Anna (2021) group-time ATTs, universal base period g-1-delta, simple and event-time aggregation
- Borusyak, Jaravel & Spiess (2024) imputation estimator
Controls: never-VR only (primary), or never-VR + always-VR (constant-status comparison).
delta = 0 (no anticipation) or 6 (six-year anticipation window, motivated by the event study).
Inference: country cluster bootstrap, 999 draws.
"""
import numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
from common import *

B_DRAWS = 999
rng = np.random.default_rng(2026)
D = load_gbd(); CR = pd.read_csv(f'{RES}/country_richness.csv')
Y = vr_years(); cnt = Y[Y.Year <= 2021].groupby('iso3').Year.nunique()
treated = CR[(CR.onset >= 1998) & (CR.onset <= 2013) & (CR.iso3.map(cnt).fillna(0) >= 5)].set_index('iso3').onset.astype(int)
always = set(CR[CR.onset <= 1990].iso3); never = set(CR[CR.onset.isna()].iso3)


def panel(kind):
    """Country-year mean of log roughness after removing cause-year and country-cause means (as in h3_onset.py)."""
    keep = set(treated.index) | always | never
    R = roughness_long(D[D.CountryCode.isin(keep)], outcome_cols(D, kind))
    R = R[R.rough > 0]; R['lr'] = np.log(R.rough)
    R['lr'] -= R.groupby(['cause', 'Year']).lr.transform('mean')
    R['lr'] -= R.groupby(['CountryCode', 'cause']).lr.transform('mean')
    P = R.groupby(['CountryCode', 'Year']).lr.mean().unstack('Year')   # countries x years (wide)
    return P


def cs_att(W, tr, ctrl, delta):
    """Callaway-Sant'Anna with universal base g-1-delta. W: wide panel. Returns overall ATT and event-time dict."""
    years = W.columns.values; C = W.loc[ctrl]
    cells = []  # (g, e, att, n_g)
    for g, units in tr.groupby(tr).groups.items():
        base = g - 1 - delta
        if base not in years: continue
        T = W.loc[list(units)]
        dT = T.sub(T[base], axis=0).mean(); dC = C.sub(C[base], axis=0).mean()
        for t in years:
            if t == base: continue
            cells.append((g, t - g, dT[t] - dC[t], len(units)))
    Ct = pd.DataFrame(cells, columns=['g', 'e', 'att', 'n'])
    post = Ct[Ct.e >= -delta]
    overall = np.average(post.att, weights=post.n)
    ev = Ct.groupby('e').apply(lambda d: np.average(d.att, weights=d.n))
    return overall, ev


def bjs_att(W, tr, ctrl, delta):
    """Imputation estimator: fit unit + year FE on untreated cells, impute Y(0) for treated post cells."""
    units = list(tr.index) + list(ctrl)
    L = W.loc[units].stack().rename('y').reset_index(); L.columns = ['u', 't', 'y']
    g = L.u.map(tr)
    untreated = g.isna() | (L.t < g - delta)
    Lu = L[untreated]
    # alternating projections for unit and year effects on untreated cells
    a = pd.Series(0.0, index=units); b = pd.Series(0.0, index=sorted(L.t.unique()))
    for _ in range(100):
        a = (Lu.y - Lu.t.map(b)).groupby(Lu.u).mean().reindex(units)
        b = (Lu.y - Lu.u.map(a)).groupby(Lu.t).mean().reindex(b.index)
    Lt = L[~untreated].copy()
    Lt['res'] = Lt.y - Lt.u.map(a) - Lt.t.map(b)
    return Lt.res.mean()


def boot(fn, W, tr, ctrl, delta):
    trl, cl = list(tr.index), list(ctrl); out = []
    for _ in range(B_DRAWS):
        bt = rng.choice(trl, len(trl), replace=True); bc = rng.choice(cl, len(cl), replace=True)
        # relabel duplicates so resampled units are distinct rows
        Wt = W.loc[bt].copy(); Wt.index = [f'{u}#{i}' for i, u in enumerate(bt)]
        Wc = W.loc[bc].copy(); Wc.index = [f'{u}#c{i}' for i, u in enumerate(bc)]
        trb = pd.Series(tr.loc[bt].values, index=Wt.index)
        out.append(fn(pd.concat([Wt, Wc]), trb, list(Wc.index), delta))
    return np.array(out)


rows, evs = [], []
for kind in ['death', 'prev']:
    W = panel(kind)
    tr = treated[treated.index.isin(W.index)]
    for ctrl_lab, ctrl in [('never', sorted(never & set(W.index))), ('never+always', sorted((never | always) & set(W.index)))]:
        for delta in [0, 6]:
            o, ev = cs_att(W, tr, ctrl, delta)
            bs = boot(lambda *a: cs_att(*a)[0], W, tr, ctrl, delta)
            rows.append(dict(outcome=kind, estimator='Callaway-Sant\'Anna', controls=ctrl_lab, anticipation=delta,
                             ratio=np.exp(o), lo=np.exp(np.percentile(bs, 2.5)), hi=np.exp(np.percentile(bs, 97.5)),
                             p_boot=min(1, 2 * min((bs <= 0).mean(), (bs >= 0).mean())), n_treated=len(tr), n_controls=len(ctrl)))
            if ctrl_lab == 'never':
                e = ev.rename('att').reset_index(); e['outcome'] = kind; e['anticipation'] = delta; evs.append(e)
            b0 = bjs_att(W, tr, ctrl, delta); bs = boot(bjs_att, W, tr, ctrl, delta)
            rows.append(dict(outcome=kind, estimator='Borusyak-Jaravel-Spiess', controls=ctrl_lab, anticipation=delta,
                             ratio=np.exp(b0), lo=np.exp(np.percentile(bs, 2.5)), hi=np.exp(np.percentile(bs, 97.5)),
                             p_boot=min(1, 2 * min((bs <= 0).mean(), (bs >= 0).mean())), n_treated=len(tr), n_controls=len(ctrl)))
            print(rows[-2]); print(rows[-1])
O = pd.DataFrame(rows); O.to_csv(f'{RES}/h3_robust.csv', index=False)
pd.concat(evs).to_csv(f'{RES}/h3_robust_event.csv', index=False)
pd.set_option('display.width', 220); print(O.round(3).to_string(index=False))
