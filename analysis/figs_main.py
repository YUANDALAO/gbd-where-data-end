"""Main figures 1-4 for the BMJ manuscript. Every value is read from results/ (h1-h6 scripts, fig_data_country.py,
fig_data_main.py); nothing is simulated. Output: figures/Figure1-4 (.png 300 dpi, .pdf)."""
import os
import numpy as np, pandas as pd, matplotlib as mpl, matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm, Normalize, LinearSegmentedColormap
import statsmodels.api as sm
import figs_country as fc
from common import RES

FD = os.path.join(RES, 'figdata'); FIG = fc.FIG; MM = fc.MM
BLUE, YEL, RED, INK, INK2, GRID, GRAY = fc.BLUE, fc.YEL, fc.RED, fc.INK, fc.INK2, fc.GRID, fc.GRAY
GK = ['rich', 'mid', 'poor']; GC = {'rich': BLUE, 'mid': YEL, 'poor': RED}; GN = {'rich': 'Complete VR', 'mid': 'Partial VR', 'poor': 'No national VR'}
CATS = ['Cardiovascular', 'Neoplasms', 'Chronic respiratory', 'Digestive', 'Neurological & mental', 'Diabetes & kidney',
        'Infectious, maternal & neonatal', 'Other NCDs']
CATC = dict(zip(CATS, ['#123f78', BLUE, '#9cc0ea', '#5b8a72', '#8c6bb1', YEL, '#b07a00', '#7a1d17']))
SHORT = {'Infectious, maternal & neonatal': 'Infectious & maternal', 'Neurological & mental': 'Neuro & mental'}
SEQR = LinearSegmentedColormap.from_list('r', ['#f7f6f3', '#f4c7c2', RED, '#7a1d17'])
mpl.rcParams.update({'font.size': 6.2, 'axes.labelsize': 6.2, 'xtick.labelsize': 5.6, 'ytick.labelsize': 5.6,
                     'axes.titlesize': 6.6, 'legend.fontsize': 5.6})
C = pd.read_csv(os.path.join(RES, 'fig_country_metrics.csv'))
CC = dict(pd.read_csv(os.path.join(FD, 'cause_category.csv')).values) | {'LiverCancer': 'Neoplasms'}
NICE = {'SocioDemographicIdx': 'SDI', 'MedianAge': 'Median age', 'Smoking': 'Smoking', 'PM25': 'PM2.5', 'HighBMI': 'High BMI',
        'HouseholdSolidFuel': 'Solid fuel', 'HighFBG': 'High FPG', 'HighSBP': 'High SBP', 'UnsafeWaterSource': 'Unsafe water', 'Alcohol': 'Alcohol'}
LET = []


def head(fig, ax, s, title, pad=4):
    if title: ax.set_title(title, loc='left', fontsize=6.6, fontweight='bold', pad=pad, y=1.0)
    LET.append((ax, s))


def legend_top(ax, handles, labels, ncol):
    ax.legend(handles, labels, ncol=ncol, loc='lower left', bbox_to_anchor=(0, 1.0), borderaxespad=0.6, handlelength=1.2,
              columnspacing=0.9, handletextpad=0.4, fontsize=5.4)


def dot(c, s=4, **k): return plt.Line2D([], [], marker='o', ls='', color=c, ms=s, **k)


def logx(ax, ticks):
    ax.set_xscale('log'); ax.set_xticks(ticks); ax.set_xticklabels([f'{t:g}' for t in ticks]); ax.minorticks_off()


def save(fig, n):
    fig.canvas.draw(); R = fig.canvas.get_renderer(); inv = fig.transFigure.inverted()
    for ax, s in LET:
        bb = inv.transform(ax.get_tightbbox(R).get_points()); x0 = min(bb[0, 0], ax.get_position().x0 - 0.028)
        fig.text(max(x0, 0.004), min(bb[1, 1], 0.995), s, fontsize=9, fontweight='bold', va='top', ha='left')
    LET.clear()
    fig.savefig(f'{FIG}/{n}.pdf'); fig.savefig(f'{FIG}/{n}.png', dpi=300); plt.close(fig); print('saved', n)


def cat_strip(ax, cats, n_total):
    """Vertical colour strip for cause categories (cats: category per row, rows already grouped)."""
    y0 = 0; mids, labs = [], []
    for k in pd.unique(cats):
        n = int((np.array(cats) == k).sum()); ax.add_patch(plt.Rectangle((0, y0), 1, n, color=CATC[k], lw=0)); mids.append(y0 + n / 2); labs.append(SHORT.get(k, k)); y0 += n
    ax.set_xlim(0, 1); ax.set_ylim(n_total, 0); ax.set_xticks([]); ax.grid(False)
    for sp in ax.spines.values(): sp.set_visible(False)
    ax.set_yticks(mids); ax.set_yticklabels(labs, fontsize=5.6); ax.tick_params(axis='y', length=0)
    return np.cumsum([int((np.array(cats) == k).sum()) for k in pd.unique(cats)])


# ======================================================================== Figure 1
def fig1():
    M = pd.read_csv(os.path.join(FD, 'rough_matrix.csv')); CR = pd.read_csv(os.path.join(FD, 'rough_cause_ratio.csv')).set_index('cause').ratio
    W = M.pivot(index='cause', columns='iso3', values='lr_c')
    cols = C[C.iso3.isin(W.columns)].assign(g=lambda d: d.grp.map({'poor': 0, 'mid': 1, 'rich': 2})).sort_values(['g', 'nvr', 'rough_rel'])
    rows = pd.DataFrame({'cause': W.index, 'cat': W.index.map(CC)}).assign(k=lambda d: d.cat.map(CATS.index), r=lambda d: d.cause.map(CR)).sort_values(['k', 'r'])
    Z = W.loc[rows.cause, cols.iso3].values; nK, nC = Z.shape
    fig = plt.figure(figsize=(180 * MM, 240 * MM))
    G = fig.add_gridspec(4, 1, height_ratios=[0.78, 1.75, 0.62, 0.85], hspace=0.42, left=0.135, right=0.985, top=0.965, bottom=0.045)
    r1 = G[0].subgridspec(1, 2, width_ratios=[1.55, 1], wspace=0.18)
    ax = fig.add_subplot(r1[0]); fc.draw_map(ax, dict(zip(C.iso3, C.nvr)), fc.SEQ, Normalize(0, 32))
    fc.cbar(fig, ax, fc.SEQ, Normalize(0, 32), [0, 15, 32], ['0', '15', '32'], 'VR years, 1990–2021', pos=(0.0, 0.05, 0.16, 0.035))
    head(fig, ax, 'a', 'National vital registration (VR) data, WHO Mortality Database')
    P = pd.read_csv(os.path.join(RES, 'h6_pubmed.csv')); ax = fig.add_subplot(r1[1])
    ax.bar(P.year, P.gbd_papers, color='#d8d7d2', width=.78, label='All GBD-based papers'); ax.bar(P.year, P.gbd_trend_papers, color=BLUE, width=.78, label='Trend analyses (EAPC, joinpoint, APC)')
    ax.set_ylabel('PubMed records'); ax.set_xticks([2012, 2016, 2020, 2024]); ax.legend(loc='upper left', fontsize=5.4)
    i2 = ax.inset_axes([0.1, 0.36, 0.4, 0.34]); sh = P[P.year >= 2016]; i2.plot(sh.year, sh.gbd_trend_papers / sh.gbd_papers * 100, color=BLUE, lw=1, marker='o', ms=1.8)
    i2.set_title('% trend analyses', fontsize=5.2, loc='left', pad=1); i2.tick_params(labelsize=4.8); i2.set_xticks([2016, 2025]); i2.set_ylim(0, 70)
    head(fig, ax, 'b', 'GBD-based papers indexed per year')
    r2 = G[1].subgridspec(2, 3, height_ratios=[0.05, 1], width_ratios=[0.025, 1, 0.16], hspace=0.03, wspace=0.015)
    strip = fig.add_subplot(r2[0, 1]); strip.imshow(cols.nvr.values[None, :], aspect='auto', cmap=fc.SEQ, vmin=0, vmax=32, extent=[0, nC, 0, 1]); strip.axis('off')
    gpos = cols.g.values
    for g, k in [(0, 'poor'), (1, 'mid'), (2, 'rich')]:
        idx = np.where(gpos == g)[0]; strip.text(idx.mean() + .5, 1.25, f'{GN[k]} (n={len(idx)})', ha='center', va='bottom', fontsize=5.6, color=GC[k], fontweight='bold')
    bounds = cat_strip(fig.add_subplot(r2[1, 0]), rows.cat.values, nK)
    ax = fig.add_subplot(r2[1, 1]); lim = np.nanpercentile(np.abs(Z), 97)
    im = ax.imshow(Z, aspect='auto', cmap=fc.DIV, norm=TwoSlopeNorm(0, -lim, lim), extent=[0, nC, nK, 0], interpolation='nearest'); ax.set_facecolor('#f7f6f3')
    for b in bounds[:-1]: ax.axhline(b, color='white', lw=0.9)
    for b in np.where(np.diff(gpos) != 0)[0]: ax.axvline(b + 1, color=INK, lw=0.6)
    ax.set_yticks([]); ax.set_xticks([]); ax.grid(False); ax.set_xlabel(f'{nC} countries, grouped by VR status and ordered by years of VR data')
    head(fig, strip, 'c', f'Roughness of every GBD death trajectory, 1992–2021: {nK} causes × {nC} countries', pad=12)
    side = fig.add_subplot(r2[1, 2]); side.barh(np.arange(nK) + .5, rows.r.values, color=[CATC[k] for k in rows.cat], height=.82, lw=0)
    side.axvline(1, color=INK, lw=.6); side.set_ylim(nK, 0); side.set_yticks([]); logx(side, [1, 3, 10, 30]); side.grid(True, axis='x'); side.grid(False, axis='y')
    side.spines['left'].set_visible(False); side.set_xlabel('Roughness ratio,\ncomplete v no VR'); side.set_title('per cause', fontsize=5.6, loc='left', pad=3)
    cax = ax.inset_axes([0.0, -0.085, 0.2, 0.018]); cb = fig.colorbar(im, cax=cax, orientation='horizontal', ticks=[-lim, 0, lim])
    cb.ax.set_xticklabels(['smoother', '0', 'rougher'], fontsize=5.2); cb.outline.set_visible(False); cb.ax.tick_params(length=1.5)
    T = pd.read_csv(os.path.join(FD, 'trajectories.csv')); r3 = G[2].subgridspec(1, 4, wspace=0.28)
    for k, (cause, lab) in enumerate([('IHD', 'Ischaemic heart disease'), ('Stroke', 'Stroke'), ('COPD', 'COPD'), ('Diabetes', 'Diabetes')]):
        ax = fig.add_subplot(r3[k])
        for (g, iso), t in T[T.cause == cause].groupby(['grp', 'CountryCode']): ax.plot(t.Year, t.l + (0.6 if g == 'rich' else -0.6), color=GC[g], lw=.6, alpha=.9)
        ax.text(0.03, 0.97, lab, transform=ax.transAxes, fontsize=5.8, va='top', fontweight='bold'); ax.set_xticks([1990, 2005, 2020]); ax.set_yticks([])
        if k == 0: ax.set_ylabel('Log death rate\n(country-centred, offset)'); head(fig, ax, 'd', 'Six most populous countries per group', pad=12)
        if k == 2: legend_top(ax, [plt.Line2D([], [], color=BLUE, lw=1), plt.Line2D([], [], color=RED, lw=1)], ['Complete VR', 'No national VR'], 2)
    r4 = G[3].subgridspec(1, 3, width_ratios=[1.1, 1, 1], wspace=0.95)
    ax = fig.add_subplot(r4[0]); d = C.dropna(subset=['rough_rel'])
    for g in GK:
        s = d[d.grp == g]; ax.scatter(s.nvr, s.rough_rel, s=np.clip(np.sqrt(s['pop']) / 900, 2, 40), color=GC[g], alpha=.65, lw=.3, ec='white', label=GN[g])
    lw = sm.nonparametric.lowess(np.log(d.rough_rel), d.nvr, frac=.5); ax.plot(lw[:, 0], np.exp(lw[:, 1]), color=INK, lw=1)
    ax.set_yscale('log'); ax.set_yticks([0.3, 1, 3]); ax.set_yticklabels(['0.3', '1', '3']); ax.minorticks_off()
    ax.set_xlabel('Years with national VR data, 1990–2021'); ax.set_ylabel('Relative roughness'); head(fig, ax, 'e', f'Dose–response, {len(d)} countries')
    ax.legend(loc='lower right', fontsize=5, markerscale=.6, handletextpad=.2, borderaxespad=.2)
    H = pd.read_csv(os.path.join(FD, 'h1_by_category.csv')).set_index('category').loc[CATS + ['All causes']].reset_index(); ax = fig.add_subplot(r4[1])
    for i, r in H.iterrows():
        c = CATC.get(r.category, INK); ax.plot([r.lo, r.hi], [i, i], color=c, lw=1.2); ax.plot(r.ratio, i, 's' if r.category == 'All causes' else 'o', color=c, ms=3.5)
    ax.axvline(1, color=INK, lw=.6); ax.set_yticks(range(len(H))); ax.set_yticklabels([SHORT.get(c, c) for c in H.category]); ax.set_ylim(len(H) - .4, -.6); logx(ax, [1, 2, 4, 8]); ax.set_xlim(.9, 8)
    ax.grid(True, axis='x'); ax.grid(False, axis='y'); ax.tick_params(axis='y', length=0); ax.set_xlabel('Roughness ratio, complete v no VR\n(adjusted, 95% CI)')
    head(fig, ax, 'f', 'By cause category')
    ax = fig.add_subplot(r4[2]); regs = d.groupby('region').rough_rel.median().sort_values().index.tolist(); rr = np.random.default_rng(1)
    for i, rg in enumerate(regs):
        for j, g in enumerate(GK):
            v = d[(d.region == rg) & (d.grp == g)].rough_rel; ax.scatter(v, i + (j - 1) * .22 + rr.uniform(-.05, .05, len(v)), s=5, color=GC[g], lw=0, alpha=.85)
    ax.axvline(1, color=INK, lw=.6); ax.set_yticks(range(len(regs))); ax.set_yticklabels(regs); ax.set_ylim(len(regs) - .4, -.6); logx(ax, [0.3, 1, 3])
    ax.grid(True, axis='x'); ax.grid(False, axis='y'); ax.tick_params(axis='y', length=0); ax.set_xlabel('Relative roughness'); head(fig, ax, 'g', 'Within region')
    save(fig, 'Figure1')


# ======================================================================== Figure 2
def fig2():
    fig = plt.figure(figsize=(180 * MM, 200 * MM))
    G = fig.add_gridspec(2, 2, width_ratios=[1.9, 1], height_ratios=[0.85, 1], hspace=0.35, wspace=0.55, left=0.085, right=0.985, top=0.95, bottom=0.07)
    H2 = pd.read_csv(os.path.join(RES, 'h2_percause_death.csv')).dropna(); H2['cat'] = H2.cause.map(CC); H2['k'] = H2.cat.map(CATS.index)
    L = G[0, 0].subgridspec(1, 2, width_ratios=[0.8, 1], wspace=0.45)
    ax = fig.add_subplot(L[0]); H1 = pd.read_csv(os.path.join(RES, 'h1_h2.csv')); h2 = H1[H1.test == 'H2'].set_index('outcome')
    for k, (o, lab) in enumerate([('death', 'Deaths'), ('prev', 'Prevalence')]):
        nt = h2.loc[o, 'note'].split(); pr, ri = float(nt[3]), float(nt[6])
        ax.bar(k - .19, ri, width=.36, color=BLUE, lw=0); ax.bar(k + .19, pr, width=.36, color=RED, lw=0)
        ax.text(k - .19, ri + .005, f'{ri:.2f}', ha='center', va='bottom', fontsize=5.2); ax.text(k + .19, pr + .005, f'{pr:.2f}', ha='center', va='bottom', fontsize=5.2)
        ax.text(k, max(pr, ri) + .05, f'+{h2.loc[o, "est"]:.3f}\n({h2.loc[o, "lo"]:.3f} to {h2.loc[o, "hi"]:.3f})', ha='center', va='bottom', fontsize=5)
    ax.set_xticks([0, 1]); ax.set_xticklabels(['Deaths\n(112 causes)', 'Prevalence\n(110 causes)']); ax.set_ylim(0, .42); ax.set_xlim(-.6, 1.6)
    ax.set_ylabel('Mean R² across causes'); legend_top(ax, [plt.Rectangle((0, 0), 1, 1, color=BLUE), plt.Rectangle((0, 0), 1, 1, color=RED)], ['Complete VR', 'No national VR'], 2)
    head(fig, ax, 'a', 'Covariate tracking (hypothesis 2)', pad=12)
    ax = fig.add_subplot(L[1]); hi = max(H2.rich.max(), H2.poor.max()) * 1.05
    ax.plot([0, hi], [0, hi], color=INK, lw=.6, ls='--'); ax.scatter(H2.rich, H2.poor, s=10, c=[CATC[k] for k in H2.cat], lw=.3, ec='white', alpha=.9)
    ax.set_xlim(0, hi); ax.set_ylim(0, hi); ax.set_xlabel('R², countries with complete VR'); ax.set_ylabel('R², countries without national VR'); ax.grid(True, axis='both')
    ax.text(.03 * hi, .97 * hi, f'{int((H2.poor > H2.rich).sum())} of {len(H2)} causes\nabove the diagonal', fontsize=5.6, va='top'); head(fig, ax, 'b', 'Covariate R², cause by cause')
    ax = fig.add_subplot(G[:, 1]); o = H2.assign(d=H2.poor - H2.rich).sort_values(['k', 'd']); yy = np.arange(len(o))
    ax.hlines(yy, o.rich, o.poor, color=[CATC[k] for k in o.cat], lw=.9, alpha=.55)
    ax.scatter(o.rich, yy, s=5, color=BLUE, zorder=3, lw=0); ax.scatter(o.poor, yy, s=5, color=RED, zorder=3, lw=0)
    pos = np.r_[0, np.cumsum(o.groupby('k', sort=True).size().values)]; ax.set_yticks(pos[:-1] + np.diff(pos) / 2 - .5)
    ax.set_yticklabels([SHORT.get(CATS[k], CATS[k]) for k in sorted(o.k.unique())], fontsize=5.6)
    for p_ in pos[1:-1]: ax.axhline(p_ - .5, color=GRID, lw=.6)
    ax.set_ylim(len(o) - .5, -.5); ax.tick_params(axis='y', length=0); ax.grid(True, axis='x'); ax.grid(False, axis='y')
    ax.set_xlabel('R² of 5-year change in log death rate\non changes in nine covariates'); legend_top(ax, [dot(BLUE), dot(RED)], ['Complete VR', 'No national VR'], 2)
    head(fig, ax, 'c', f'All {len(o)} causes', pad=12)
    B = G[1, 0].subgridspec(1, 3, width_ratios=[0.7, 0.85, 0.85], wspace=1.25)
    PR = pd.read_csv(os.path.join(FD, 'h2_partial_r2.csv')).sort_values('poor', ascending=False); ax = fig.add_subplot(B[0]); Mx = PR[['rich', 'poor']].values
    ax.imshow(Mx, aspect='auto', cmap=SEQR, vmin=0, vmax=Mx.max() * 1.1); ax.set_xticks([0, 1]); ax.set_xticklabels(['Full', 'None'])
    ax.set_yticks(range(len(PR))); ax.set_yticklabels([NICE[c] for c in PR.covariate]); ax.grid(False); ax.tick_params(length=0); ax.set_xlabel('National VR')
    for i in range(len(PR)):
        for j in range(2): ax.text(j, i, f'{Mx[i, j]:.3f}', ha='center', va='center', fontsize=4.8, color='white' if Mx[i, j] > Mx.max() * .6 else INK)
    head(fig, ax, 'd', 'Partial R²')
    E = pd.read_csv(os.path.join(RES, 'h5_eapc_gbd.csv')); E['w_rel'] = E.width / E.groupby('cause').width.transform('median'); E['lw'] = np.log(E.w_rel)
    E['cat'] = E.cause.str[:-6].map(CC); ax = fig.add_subplot(B[1])
    for k, g in enumerate(GK):
        v = E[E.grp == g].lw.dropna().values; v = v[np.isfinite(v)]; parts = ax.violinplot(v, positions=[k], widths=.85, showextrema=False, points=200)
        for b in parts['bodies']: b.set_facecolor(GC[g]); b.set_alpha(.4); b.set_edgecolor('none')
        q = np.percentile(v, [25, 50, 75]); ax.vlines(k, q[0], q[2], color=GC[g], lw=2.5); ax.scatter(k, q[1], color='white', s=8, zorder=3, lw=.5, ec=GC[g])
    ax.set_xticks(range(3)); ax.set_xticklabels(['Full', 'Part', 'None']); ax.set_xlabel('National VR'); ax.set_ylim(-3, 3)
    ax.set_yticks(np.log([.1, .3, 1, 3, 10])); ax.set_yticklabels(['0.1', '0.3', '1', '3', '10']); ax.set_ylabel('EAPC CI width, relative to cause median')
    head(fig, ax, 'e', 'CI width')
    ax = fig.add_subplot(B[2]); Wm = E.groupby(['cat', 'grp']).w_rel.median().unstack().loc[CATS, GK]
    ax.imshow(np.log(Wm.values), aspect='auto', cmap=fc.DIV, norm=TwoSlopeNorm(0, -.8, .8)); ax.set_xticks(range(3)); ax.set_xticklabels(['Full', 'Part', 'None'], fontsize=5)
    ax.set_yticks(range(len(CATS))); ax.set_yticklabels([SHORT.get(c, c) for c in CATS]); ax.grid(False); ax.tick_params(length=0)
    for i in range(len(CATS)):
        for j in range(3): ax.text(j, i, f'{Wm.values[i, j]:.2f}', ha='center', va='center', fontsize=4.8)
    ax.set_xlabel('National VR'); head(fig, ax, 'f', 'Median, by category')
    save(fig, 'Figure2')


# ======================================================================== Figure 3
def fig3():
    O = pd.read_csv(os.path.join(RES, 'fig_onset_countries.csv')); HC = pd.read_csv(os.path.join(FD, 'h3_country.csv')).sort_values(['onset', 'iso3'])
    flag = {'SRB', 'MNE', 'IRN', 'TUR'}; isos = HC.iso3.tolist(); names = [fc.cname(i) for i in isos]; on = HC.onset.values
    et = np.arange(O.et.min(), O.et.max() + 1).astype(int); E = np.full((len(isos), len(et)), np.nan)
    for i, c in enumerate(isos):
        o = O[O.CountryCode == c]; E[i, (o.et - et[0]).astype(int)] = o.adj
    fig = plt.figure(figsize=(180 * MM, 222 * MM))
    G = fig.add_gridspec(3, 1, height_ratios=[1.25, 0.72, 0.95], hspace=0.3, left=0.105, right=0.985, top=0.965, bottom=0.05)
    A = G[0].subgridspec(2, 4, height_ratios=[0.2, 1], width_ratios=[0.03, 1, 0.2, 0.2], hspace=0.04, wspace=0.04)
    top = fig.add_subplot(A[0, 1]); n_et = np.sum(np.isfinite(E), 0); mu = np.where(n_et >= 5, np.nanmean(np.where(np.isfinite(E), E, np.nan), 0), np.nan)
    lim = np.nanpercentile(np.abs(E), 97)
    top.bar(et, np.nan_to_num(mu), width=.82, color=[fc.DIV(TwoSlopeNorm(0, -lim / 2, lim / 2)(v)) if np.isfinite(v) else 'none' for v in mu], lw=0)
    top.axhline(0, color=INK2, lw=.5); top.set_xticks([]); top.set_yticks([]); top.grid(False); top.spines['left'].set_visible(False)
    top.text(0.0, 1.0, 'mean (years with ≥5 countries)', transform=top.transAxes, fontsize=5.2, color=INK2, va='top')
    head(fig, top, 'a', 'Adjusted log roughness of GBD death trajectories, aligned at the first year of national VR data', pad=4)
    strip = fig.add_subplot(A[1, 0]); strip.imshow(on[:, None], aspect='auto', cmap=fc.SEQ, vmin=1994, vmax=2014); strip.set_xticks([]); strip.grid(False)
    strip.set_yticks(range(len(isos))); strip.set_yticklabels([f'{n}{"*" if i in flag else ""}  {o}' for n, i, o in zip(names, isos, on)]); strip.tick_params(axis='y', length=0)
    for sp in strip.spines.values(): sp.set_visible(False)
    ax = fig.add_subplot(A[1, 1]); ext = [et[0] - .5, et[-1] + .5, len(isos) - .5, -.5]
    im = ax.imshow(E, aspect='auto', cmap=fc.DIV, norm=TwoSlopeNorm(0, -lim, lim), extent=ext, interpolation='nearest'); ax.set_facecolor('#f7f6f3')
    ax.axvline(-.5, color=INK, lw=.9); ax.axvline(-6.5, color=INK, lw=.6, ls=(0, (2, 2)))
    for i in range(1, len(isos)): ax.axhline(i - .5, color='white', lw=.6)
    ax.set_yticks([]); ax.grid(False); ax.set_xlabel('Years relative to first year of national VR data'); top.set_xlim(ext[0], ext[1])
    hold = fig.add_subplot(A[0, 2:]); hold.axis('off'); cax = hold.inset_axes([0.08, 0.5, 0.84, 0.15])
    cb = fig.colorbar(im, cax=cax, orientation='horizontal', ticks=[-lim, 0, lim]); cb.ax.set_xticklabels(['smoother', '0', 'rougher'], fontsize=5)
    cb.outline.set_visible(False); cb.ax.tick_params(length=1.5); cax.set_title('Log roughness\n(dashed: 6 years before onset)', fontsize=5, loc='left', pad=2)
    r1 = fig.add_subplot(A[1, 2]); lr = np.log(HC.ratio.values)
    bars = r1.barh(range(len(isos)), lr, color=[BLUE if v > 0 else RED for v in lr], height=.72, lw=0)
    for b, i in zip(bars, isos):
        if i in flag: b.set_hatch('////'); b.set_edgecolor('white')
    r1.axvline(0, color=INK, lw=.6); r1.set_ylim(len(isos) - .5, -.5); r1.set_yticks([]); r1.set_xticks(np.log([.5, 1, 2, 4])); r1.set_xticklabels(['0.5', '1', '2', '4'])
    r1.grid(True, axis='x'); r1.grid(False, axis='y'); r1.spines['left'].set_visible(False); r1.set_xlabel('After / before'); r1.set_title('Country ratio', fontsize=5.6, loc='left', pad=3)
    H3 = pd.read_csv(os.path.join(RES, 'h3.csv')).set_index('outcome'); b0 = H3.loc['death', 'est']
    r2 = fig.add_subplot(A[1, 3]); r2.axvline(b0, color=INK, lw=.6, ls='--'); r2.hlines(range(len(isos)), b0, HC.loo, color=INK2, lw=.8)
    r2.scatter(HC.loo, range(len(isos)), s=9, color=[RED if i in flag else INK for i in isos], zorder=3, lw=0)
    r2.set_ylim(len(isos) - .5, -.5); r2.set_yticks([]); r2.set_xlim(1.3, 1.72); r2.set_xticks([1.4, 1.5, 1.6]); r2.grid(True, axis='x'); r2.grid(False, axis='y')
    r2.spines['left'].set_visible(False); r2.set_xlabel('Pooled ratio'); r2.set_title('Country omitted', fontsize=5.6, loc='left', pad=3)
    B = G[1].subgridspec(2, 7, hspace=0.25, wspace=0.12); ylim = np.nanpercentile(np.abs(O.adj), 99) * 1.1
    for i, (c, n, o_) in enumerate(zip(isos, names, on)):
        ax = fig.add_subplot(B[i // 7, i % 7]); s = O[O.CountryCode == c].sort_values('Year'); post = s.Year >= o_
        ax.axvspan(o_ - .5, 2021.5, color='#eef3fa', lw=0); ax.plot(s.Year, s.adj, color=GRAY, lw=.6)
        ax.scatter(s.Year, s.adj, s=2.5, c=[BLUE if p else RED for p in post], lw=0, zorder=3)
        ax.plot([s.Year.min(), o_ - .5], [s.adj[~post].mean()] * 2, color=RED, lw=1.3); ax.plot([o_ - .5, 2021], [s.adj[post].mean()] * 2, color=BLUE, lw=1.3)
        ax.set_ylim(-ylim, ylim); ax.set_xlim(1991.5, 2021.5); ax.set_yticks([-1, 0, 1] if i % 7 == 0 else []); ax.set_xticks([1995, 2015] if i // 7 == 1 or i >= 6 else [])
        ax.text(.04, .95, n + ('*' if c in flag else ''), transform=ax.transAxes, fontsize=5.4, fontweight='bold', va='top')
        if i == 0: head(fig, ax, 'b', 'Each country: yearly adjusted log roughness, mean before (red) and from (blue) the first VR year')
    ax = fig.add_subplot(B[1, 6]); ax.axis('off')
    ax.text(0.05, 0.85, '* Onset date uncertain:\nformer Yugoslavia (Serbia,\nMontenegro); earlier\nnational or urban death\ndata (Iran, Türkiye)', fontsize=5, color=INK2, va='top', transform=ax.transAxes)
    Cg = G[2].subgridspec(1, 3, width_ratios=[1, 0.8, 1.75], wspace=0.42)
    EV = pd.read_csv(os.path.join(RES, 'h3_event.csv')); EV = EV[EV.outcome == 'death'].reset_index(drop=True)
    labs = ['≤−10', '−9 to −7', '−6 to −4', '−3 to −1', '0 to 2', '3 to 5', '6 to 8', '≥9']; x = np.arange(len(EV)); ax = fig.add_subplot(Cg[0])
    ax.bar(x, EV['mean'], width=.72, color=[fc.DIV(TwoSlopeNorm(0, -.6, .6)(v)) for v in EV['mean']], lw=0)
    ax.errorbar(x, EV['mean'], yerr=1.96 * EV['sem'], fmt='none', ecolor=INK, lw=.7, capsize=1.5); ax.axhline(0, color=INK2, lw=.5); ax.axvline(3.5, color=INK, lw=.6)
    ax.set_xticks(x); ax.set_xticklabels(labs, fontsize=5, rotation=45, ha='right', rotation_mode='anchor'); ax.set_xlabel('Years relative to onset'); ax.set_ylabel('Log roughness, country-centred (95% CI)')
    for xi, n in zip(x, EV['count']): ax.text(xi, ax.get_ylim()[0], f'{n}', ha='center', va='bottom', fontsize=4.4, color=INK2)
    head(fig, ax, 'c', 'Pooled event study')
    perm = pd.read_csv(os.path.join(RES, 'h3_perm_death.csv')).iloc[:, 0].values; obs = np.log(b0); ax = fig.add_subplot(Cg[1])
    h, e = np.histogram(perm, bins=np.linspace(-max(abs(perm).max(), obs) * 1.05, max(abs(perm).max(), obs) * 1.05, 45)); c_ = (e[:-1] + e[1:]) / 2
    ax.bar(c_, h, width=e[1] - e[0], color=[RED if abs(v) >= obs else '#d8d7d2' for v in c_], lw=0); ax.axvline(obs, color=RED, lw=.9)
    ax.text(e[0], h.max() * .98, f'observed {b0:.2f}\nP={H3.loc["death", "p_perm"]:.3f}', fontsize=5.2, color=RED, va='top')
    ax.set_xlabel(f'Log ratio, {len(perm)} placebo onsets'); ax.set_ylabel('Count'); head(fig, ax, 'd', 'Permutation test')
    X4 = pd.read_csv(os.path.join(RES, 'h3_exclude4.csv')).set_index('outcome'); RB = pd.read_csv(os.path.join(RES, 'h3_robust.csv'))
    rb = lambda est, ant: RB[(RB.outcome == 'death') & (RB.estimator == est) & (RB.controls == 'never') & (RB.anticipation == ant)].iloc[0]
    def row(lab, est, lo, hi, p, c): return (lab, (est, lo, hi, p, c))
    rows = [('Primary', None), row('Two-way fixed effects', b0, H3.loc['death', 'lo'], H3.loc['death', 'hi'], H3.loc['death', 'p_perm'], INK), ('Sensitivity', None),
            row('Excluding 4 uncertain onsets*', X4.loc['death', 'est'], X4.loc['death', 'lo'], X4.loc['death', 'hi'], X4.loc['death', 'p_perm'], BLUE),
            row('Prevalence outcome', H3.loc['prev', 'est'], H3.loc['prev', 'lo'], H3.loc['prev', 'hi'], H3.loc['prev', 'p_perm'], BLUE), ('Heterogeneity-robust', None)]
    for est, nm in [("Callaway-Sant'Anna", 'Callaway–Sant\'Anna'), ('Borusyak-Jaravel-Spiess', 'Imputation (BJS)')]:
        for ant in [0, 6]:
            r = rb(est, ant); rows.append(row(nm if ant == 0 else '  6-year anticipation', r.ratio, r.lo, r.hi, r.p_boot, GRAY))
    ax = fig.add_subplot(Cg[2]); tr = ax.get_yaxis_transform(); ylab = []
    for i, (lab, v) in enumerate(rows):
        if v is None: ax.text(-0.02, i, lab, transform=tr, ha='right', va='center', fontsize=5.6, fontweight='bold'); ylab.append(''); continue
        e_, l, h_, p, c = v; ylab.append(lab)
        ax.plot([l, h_], [i, i], color=c, lw=1.2); ax.plot(e_, i, 's' if c == INK else 'o', color=c, ms=3.6)
        ax.text(1.03, i, f'{e_:.2f} ({l:.2f} to {h_:.2f})', transform=tr, va='center', fontsize=5.3)
        ax.text(1.68, i, '<0.001' if p < 0.001 else f'{p + 1e-9:.3f}', transform=tr, va='center', fontsize=5.3)
    ax.text(1.03, -1, 'Ratio (95% CI)', transform=tr, fontsize=5.3, fontweight='bold', va='center'); ax.text(1.68, -1, 'P†', transform=tr, fontsize=5.3, fontweight='bold', va='center')
    ax.axvline(1, color=INK2, lw=.6); ax.set_yticks(range(len(rows))); ax.set_yticklabels(ylab, fontsize=5.6); ax.set_ylim(len(rows) - .5, -1.5); ax.tick_params(axis='y', length=0)
    logx(ax, [.75, 1, 1.5, 2, 3]); ax.set_xlim(.7, 3); ax.grid(True, axis='x'); ax.grid(False, axis='y'); ax.set_xlabel('Roughness ratio, after v before onset')
    p_ = ax.get_position(); ax.set_position([p_.x0 + 0.075, p_.y0, p_.width * 0.44, p_.height]); head(fig, ax, 'e', 'All estimators')
    save(fig, 'Figure3')


# ======================================================================== Figure 4
def fig4():
    V = pd.read_csv(os.path.join(RES, 'h5_eapc_gbd_vs_raw.csv')); V['se_g'], V['se_r'] = (V.gh - V.gl) / 3.92, (V.rh - V.rl) / 3.92
    V['z'] = (V.g - V.r) / np.sqrt(V.se_g ** 2 + V.se_r ** 2); V['differ'] = V.z.abs() > 1.96; V['cat'] = V.cause.map(CC); share = V.differ.mean() * 100
    cty = C[C.iso3.isin(V.iso3)].sort_values('pop'); cs = V.groupby('cause').differ.mean()
    rows = pd.DataFrame({'cause': cs.index, 'cat': cs.index.map(CC), 'sh': cs.values}).assign(k=lambda d: d.cat.map(CATS.index)).sort_values(['k', 'sh'], ascending=[True, False])
    Z = V.pivot(index='cause', columns='iso3', values='z').reindex(index=rows.cause, columns=cty.iso3).values; nK, nC = Z.shape
    fig = plt.figure(figsize=(180 * MM, 228 * MM))
    G = fig.add_gridspec(3, 1, height_ratios=[1.3, 0.85, 0.8], hspace=0.42, left=0.135, right=0.955, top=0.965, bottom=0.065)
    A = G[0].subgridspec(2, 3, height_ratios=[0.24, 1], width_ratios=[0.025, 1, 0.18], hspace=0.04, wspace=0.02)
    top = fig.add_subplot(A[0, 1]); shc = cty.differ.values * 100
    top.bar(np.arange(nC) + .5, shc, width=.8, color=RED, alpha=.8, lw=0); top.axhline(share, color=INK, lw=.5, ls='--')
    top.set_xlim(0, nC); top.set_xticks([]); top.set_ylim(0, 100); top.set_yticks([0, 50, 100]); top.grid(True, axis='y'); top.set_ylabel('% differ', fontsize=5.4)
    head(fig, top, 'a', f'GBD minus raw VR trend (z statistic), {nK} causes × {nC} countries with complete VR', pad=4)
    bounds = cat_strip(fig.add_subplot(A[1, 0]), rows.cat.values, nK)
    ax = fig.add_subplot(A[1, 1]); ax.set_facecolor('#f7f6f3')
    im = ax.imshow(np.clip(Z, -6, 6), aspect='auto', cmap=fc.DIV, norm=TwoSlopeNorm(0, -6, 6), extent=[0, nC, nK, 0], interpolation='nearest')
    for b in bounds[:-1]: ax.axhline(b, color='white', lw=.9)
    ax.set_yticks(np.arange(nK) + .5); ax.set_yticklabels([]); ax.tick_params(axis='y', length=0); ax.set_xticks([]); ax.grid(False)
    ax.set_xlabel('Countries ordered by population (small → large)')
    cax = ax.inset_axes([0.0, -0.11, 0.22, 0.022]); cb = fig.colorbar(im, cax=cax, orientation='horizontal', ticks=[-6, -1.96, 0, 1.96, 6])
    cb.ax.set_xticklabels(['GBD lower', '', '0', '', 'GBD higher'], fontsize=5); cb.outline.set_visible(False); cb.ax.tick_params(length=1.5)
    side = fig.add_subplot(A[1, 2]); side.barh(np.arange(nK) + .5, rows.sh.values * 100, color=[CATC[k] for k in rows.cat], height=.78, lw=0)
    side.axvline(share, color=INK, lw=.5, ls='--'); side.set_ylim(nK, 0); side.set_yticks([]); side.set_xlim(0, 100); side.set_xticks([0, 50, 100])
    side.grid(True, axis='x'); side.grid(False, axis='y'); side.spines['left'].set_visible(False); side.set_xlabel('% of countries\ndiffering')
    side.set_title('per cause', fontsize=5.6, loc='left', pad=3)
    B = G[1].subgridspec(1, 3, wspace=0.5); ax = fig.add_subplot(B[0]); lo_, hi_ = np.percentile(np.r_[V.r, V.g], [0.5, 99.5])
    ax.hexbin(V.r.clip(lo_, hi_), V.g.clip(lo_, hi_), gridsize=30, extent=(lo_, hi_, lo_, hi_), cmap=LinearSegmentedColormap.from_list('h', ['#f3f2ef', '#9cc0ea', BLUE, '#123f78']),
              mincnt=1, bins='log', lw=0)
    ax.plot([lo_, hi_], [lo_, hi_], color=RED, lw=.8, ls='--'); ax.set_xlim(lo_, hi_); ax.set_ylim(lo_, hi_); ax.set_aspect('equal'); ax.grid(False)
    ax.set_xlabel('EAPC, raw VR (%/year)'); ax.set_ylabel('EAPC, GBD (%/year)')
    ax.text(.04, .96, f'n={len(V)}; r={np.corrcoef(V.r, V.g)[0, 1]:.2f}\ndashed: line of identity', transform=ax.transAxes, fontsize=5.2, va='top', color=INK2); head(fig, ax, 'b', 'GBD v raw trends')
    V['dec'] = pd.qcut(V.se_r, 10, labels=False) + 1; sh = V.groupby('dec').differ.mean().values * 100; dec = np.arange(1, 11); ax = fig.add_subplot(B[1])
    ax.bar(dec, sh, color=RED, width=.78, lw=0, label='Differ'); ax.bar(dec, 100 - sh, bottom=sh, color='#d8d7d2', width=.78, lw=0, label='Agree')
    for x_, v in zip(dec, sh): ax.text(x_, v - 3, f'{v:.0f}', ha='center', va='top', fontsize=4.8, color='white')
    ax.set_ylim(0, 100); ax.set_xticks(dec); ax.set_xlabel('Decile of raw-trend SE (1 = most precise)'); ax.set_ylabel('% of series'); ax.grid(False)
    legend_top(ax, *ax.get_legend_handles_labels(), 2); head(fig, ax, 'c', 'Disagreement by precision', pad=12)
    ax = fig.add_subplot(B[2]); d = cty.dropna(subset=['differ'])
    ax.scatter(d['pop'], d.differ * 100, s=np.clip(np.sqrt(d['pop']) / 700, 3, 40), color=RED, alpha=.6, lw=.3, ec='white')
    lw = sm.nonparametric.lowess(d.differ * 100, np.log10(d['pop']), frac=.6); ax.plot(10 ** lw[:, 0], lw[:, 1], color=INK, lw=.9)
    ax.set_xscale('log'); ax.set_xticks([1e5, 1e6, 1e7, 1e8]); ax.set_xticklabels(['0.1', '1', '10', '100']); ax.minorticks_off(); ax.set_ylim(0, 100); ax.grid(True, axis='both')
    ax.set_xlabel('Population, 2019 (millions)'); ax.set_ylabel('% of series differing'); head(fig, ax, 'd', 'By country size')
    Cg = G[2].subgridspec(1, 2, width_ratios=[0.75, 2], wspace=0.25)
    P = pd.read_csv(os.path.join(RES, 'h4_pairs_se.csv')); exps = list(dict.fromkeys(P.x)); EC = dict(zip(exps, list(CATC.values())))
    H4 = pd.read_csv(os.path.join(RES, 'h4_h5.csv')); med = H4[(H4.test == 'H4') & (H4.spec == 'main')].est.iloc[0]
    ax = fig.add_subplot(Cg[0]); sig = P.tR.abs() >= 1.96; lim = np.abs(np.r_[P.bG, P.bR]).max() * 1.08
    ax.plot([-lim, lim], [-lim, lim], color=INK2, lw=.6, ls='--'); ax.plot([-lim, lim], [-lim * med, lim * med], color=RED, lw=.8)
    ax.scatter(P.bR[sig], P.bG[sig], s=13, c=[EC[x] for x in P.x[sig]], lw=.3, ec='white', zorder=3)
    ax.scatter(P.bR[~sig], P.bG[~sig], s=13, facecolor='white', edgecolors=[EC[x] for x in P.x[~sig]], lw=.8, zorder=3)
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_aspect('equal'); ax.set_xlabel('Association, raw VR'); ax.set_ylabel('Association, GBD'); ax.grid(True, axis='both')
    ax.text(.97, .04, f'red: median ratio {med:.2f}', transform=ax.transAxes, color=RED, fontsize=5, ha='right')
    legend_top(ax, [dot(INK2, 3.5), dot('white', 3.5, mec=INK2)], [f'raw P<0.05 ({sig.sum()})', 'raw P≥0.05'], 2)
    head(fig, ax, 'e', f'{len(P)} exposure–outcome pairs', pad=12)
    outs = list(dict.fromkeys(P.o)); R_ = P.assign(ratio=P.bG.abs() / P.bR.abs()).pivot(index='x', columns='o', values='ratio').reindex(index=exps, columns=outs)
    S_ = P.assign(s=sig).pivot(index='x', columns='o', values='s').reindex(index=exps, columns=outs)
    ax = fig.add_subplot(Cg[1]); ax.set_facecolor('#f7f6f3')
    im = ax.imshow(np.clip(R_.values, 0, 2), aspect='auto', cmap=LinearSegmentedColormap.from_list('a', [RED, '#f4c7c2', '#f0efec', '#9cc0ea', BLUE]), norm=TwoSlopeNorm(1, 0, 2))
    ON = {'TBLCancer': 'Lung ca', 'ColonRectumCancer': 'Colorectal ca', 'StomachCancer': 'Stomach ca', 'LiverCancer': 'Liver ca', 'BreastCancer': 'Breast ca',
          'PancreaticCancer': 'Pancreatic ca', 'EsophagealCancer': 'Oesophageal ca', 'BladderCancer': 'Bladder ca', 'KidneyCancer': 'Kidney ca', 'Dementias': 'Dementia'}
    ax.set_xticks(range(len(outs))); ax.set_xticklabels([ON.get(o, o) for o in outs], rotation=45, ha='right', fontsize=5.2)
    ax.set_yticks(range(len(exps))); ax.set_yticklabels([NICE.get(x, x) for x in exps], fontsize=5.4)
    for t, x in zip(ax.get_yticklabels(), exps): t.set_color(EC[x]); t.set_fontweight('bold')
    ax.tick_params(length=0); ax.grid(False)
    for i in range(len(exps)):
        for j in range(len(outs)):
            v = R_.values[i, j]
            if np.isfinite(v): ax.text(j, i, f'{v:.2f}' if v < 10 else '>10', ha='center', va='center', fontsize=4.5, fontweight='bold' if S_.values[i, j] else 'normal', color=INK if S_.values[i, j] else INK2)
    cb = fig.colorbar(im, ax=ax, fraction=.025, pad=.01, ticks=[0, .5, 1, 1.5, 2]); cb.ax.set_yticklabels(['0', '0.5', '1', '1.5', '≥2']); cb.outline.set_visible(False); cb.ax.tick_params(labelsize=5, length=1.5)
    head(fig, ax, 'f', '|GBD| / |raw| association for each pair (bold: raw P<0.05; grey cell: pair not analysed)')
    save(fig, 'Figure4')


if __name__ == '__main__':
    import sys
    for f in (sys.argv[1:] or ['fig1', 'fig2', 'fig3', 'fig4']): globals()[f]()
