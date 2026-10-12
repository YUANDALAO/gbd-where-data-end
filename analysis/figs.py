"""Publication figures for the GBD circularity audit.
Output: figures/Fig1-4 (.pdf vector + .png 300 dpi). Run after h1_h2.py, h3_onset.py, h4_h5.py, h6_biblio.py.
Usage: python figs.py [1 2 3 4]   (default: all)
"""
import sys, numpy as np, pandas as pd, matplotlib as mpl, matplotlib.pyplot as plt, warnings
warnings.filterwarnings('ignore')
from common import *

# Colour-blind-safe palette (blue/orange, blue/red separable under common colour-vision deficiencies)
BLUE, ORANGE, RED, GRAY = '#2a78d6', '#eb6834', '#e34948', '#9a9994'
INK, INK2, GRID, SHADE = '#0b0b0b', '#52514e', '#e6e5e1', '#f3f2ef'
mpl.rcParams.update({
    'font.family': 'Arial', 'font.size': 7, 'axes.labelsize': 7, 'axes.titlesize': 7.5,
    'xtick.labelsize': 6.5, 'ytick.labelsize': 6.5, 'axes.edgecolor': INK2, 'axes.labelcolor': INK,
    'xtick.color': INK2, 'ytick.color': INK2, 'text.color': INK, 'axes.linewidth': 0.6,
    'xtick.major.width': 0.6, 'ytick.major.width': 0.6, 'xtick.major.size': 2.5, 'ytick.major.size': 2.5,
    'axes.spines.top': False, 'axes.spines.right': False, 'axes.grid': True, 'axes.grid.axis': 'y',
    'grid.color': GRID, 'grid.linewidth': 0.5, 'axes.axisbelow': True, 'legend.frameon': False,
    'legend.fontsize': 6.5, 'pdf.fonttype': 42, 'svg.fonttype': 'none', 'lines.linewidth': 1.5})
MM = 1 / 25.4
FIG = os.path.join(HERE, 'figures'); os.makedirs(FIG, exist_ok=True)

def tag(ax, s):
    ax.text(-0.14, 1.04, s, transform=ax.transAxes, fontsize=9, fontweight='bold', va='bottom')

def save(fig, name):
    fig.savefig(f'{FIG}/{name}.pdf', bbox_inches='tight')
    fig.savefig(f'{FIG}/{name}.png', dpi=300, bbox_inches='tight')
    plt.close(fig); print('saved', name)

def r2(y, X):
    X = np.column_stack([np.ones(len(y)), X]); b = np.linalg.lstsq(X, y, rcond=None)[0]; e = y - X @ b
    return 1 - e.var() / y.var()

D = load_gbd(); CR = pd.read_csv(f'{RES}/country_richness.csv'); g = dict(zip(CR.iso3, CR.grp))
RAW = pd.read_csv(os.path.join(ROOT, 'raw_vr_asr.csv'))
which = set(sys.argv[1:]) or {'1', '2', '3', '4'}


# ---------------- Figure 1: motivation ----------------
def fig1():
    fig, axs = plt.subplots(1, 2, figsize=(180 * MM, 64 * MM), gridspec_kw={'width_ratios': [1, 1.25], 'wspace': 0.3})
    ax = axs[0]; B = pd.read_csv(f'{RES}/h6_pubmed.csv'); B = B[B.year >= 2014]
    ax.bar(B.year, B.gbd_trend_papers, width=0.72, color=BLUE, edgecolor='white', linewidth=0.8)
    for y, v in zip(B.year, B.gbd_trend_papers):
        if y in (2016, 2020, 2023, 2025):
            ax.text(y, v + 40, f'{v:,}', ha='center', va='bottom', fontsize=6.3, color=INK)
    ax.set_xticks(B.year[::2]); ax.set_ylim(0, 2600); ax.set_ylabel('PubMed records per year')
    ax.set_title('GBD-based trend analyses\n(EAPC, joinpoint, age-period-cohort, decomposition)', loc='left')
    tag(ax, 'a')

    ax = axs[1]; on = 2009
    t = D[D.CountryCode == 'TUR'].sort_values('Year').set_index('Year')
    gb = t.IHD_Death * 1e5
    rr = RAW[(RAW.iso3 == 'TUR') & (RAW.cause == 'IHD')].set_index('Year').asr.sort_index() * 1e5
    rr = rr * gb.reindex(rr.index).mean() / rr.mean()   # different age standards -> rescale raw to GBD level
    ax.axvspan(1989.5, on - 0.5, color=SHADE, zorder=0, lw=0)
    ax.plot(gb.index, gb.values, color=BLUE, lw=1.6, zorder=3)
    ax.plot(rr.index, rr.values, 'o', ms=4.5, mfc=ORANGE, mec='white', mew=0.8, zorder=4)
    top = max(gb.max(), rr.max()) * 1.04
    ax.set_ylim(min(gb.min(), rr.min()) * 0.9, top * 1.06); ax.set_xlim(1989.5, 2021.5)
    ax.text(1990.4, top, 'No national VR data', color=INK2, fontsize=6.5, va='bottom')
    ax.text(on + 0.3, top, 'VR data from 2009', color=INK2, fontsize=6.5, va='bottom')
    ax.annotate('GBD 2021 estimate', (1993, gb.loc[1993]), xytext=(1990.6, gb.loc[1993] * 0.66),
                fontsize=6.5, color=INK, arrowprops=dict(arrowstyle='-', color=INK2, lw=0.5))
    ax.annotate('WHO raw VR\n(rescaled to GBD level)', (rr.index[2], rr.values[2]), xytext=(2012.5, top * 0.93),
                fontsize=6.5, color=INK, va='top', arrowprops=dict(arrowstyle='-', color=INK2, lw=0.5))
    d2 = np.abs(np.diff(np.log(gb.values), 2)); yrs = gb.index.values[2:]
    ax.text(0.99, 0.03, f'Roughness of GBD trajectory, after vs before data: ×{d2[yrs >= on].mean() / d2[yrs < on].mean():.1f}',
            transform=ax.transAxes, ha='right', fontsize=6.3, color=INK2)
    ax.set_ylabel('Age-standardised IHD deaths per 100,000')
    ax.set_title('Illustrative example: ischaemic heart disease, Türkiye', loc='left')
    tag(ax, 'b'); save(fig, 'Fig1_motivation')


# ---------------- Figure 2: H1 dose-response, H2 per-cause ----------------
def fig2():
    fig, axs = plt.subplots(1, 2, figsize=(180 * MM, 72 * MM), gridspec_kw={'wspace': 0.32})
    ax = axs[0]; bins = [-0.5, 0.5, 10.5, 20.5, 29.5, 32.5]; labs = ['0', '1–10', '11–20', '21–29', '30–32']
    for kind, col, lab, dx in [('death', BLUE, 'Deaths', -0.07), ('prev', ORANGE, 'Prevalence', 0.07)]:
        Rl = roughness_long(D, outcome_cols(D, kind)); Rl = Rl[Rl.rough > 0]
        m = Rl.groupby(['CountryCode', 'cause']).rough.mean().reset_index()
        m['l'] = np.log(m.rough); m['l'] -= m.groupby('cause').l.transform('mean')
        c = m.groupby('CountryCode').l.mean().reset_index().merge(CR, left_on='CountryCode', right_on='iso3')
        c['b'] = pd.cut(c.nvr, bins, labels=labs)
        s = c.groupby('b').l.agg(['mean', 'sem', 'count']); x = np.arange(len(labs)) + dx
        mu, lo, hi = np.exp(s['mean']), np.exp(s['mean'] - 1.96 * s['sem']), np.exp(s['mean'] + 1.96 * s['sem'])
        ax.errorbar(x, mu, yerr=[mu - lo, hi - mu], color=col, marker='o', ms=4.5, mec='white', mew=0.8,
                    lw=1.5, elinewidth=1, capsize=0, label=lab)
        ax.text(x[-1] + 0.16, mu.iloc[-1], lab, color=INK, va='center', fontsize=6.5)
        if kind == 'death': ncount = s['count'].values
    ax.set_yscale('log'); ax.set_yticks([0.4, 0.6, 1, 1.5, 2]); ax.set_yticklabels(['0.4', '0.6', '1.0', '1.5', '2.0'])
    ax.minorticks_off()
    ax.set_xticks(range(len(labs))); ax.set_xticklabels([f'{l}\n(n={n})' for l, n in zip(labs, ncount)])
    ax.set_xlim(-0.4, 4.95); ax.legend(loc='upper left', handlelength=1.4)
    ax.set_xlabel('Years of national VR data, 1990–2021 (WHO Mortality Database)')
    ax.set_ylabel('Relative roughness of GBD trajectory\n(geometric mean, cause-adjusted; 95% CI)')
    ax.set_title('More primary data, rougher GBD trajectories', loc='left'); tag(ax, 'a')

    big = set(CR[CR['pop'] >= 5e6].iso3)
    Dd = D[D.Year.isin(range(1991, 2022, 5)) & D.CountryCode.isin(big)].sort_values(['CountryCode', 'Year'])
    P = []
    for o in outcome_cols(D, 'death'):
        s = Dd[['CountryCode', 'Year', o] + X9].dropna(); s = s[s[o] > 0].copy(); s['y'] = np.log(s[o])
        for v in ['y'] + X9: s['d' + v] = s.groupby('CountryCode')[v].diff()
        s = s.dropna(); s['grp'] = s.CountryCode.map(g); vals = {}
        for G in ['rich', 'poor']:
            t = s[s.grp == G]; dX = t[['d' + x for x in X9]].values; sd = dX.std(0); sd[sd == 0] = 1
            vals[G] = r2(t.dy.values, (dX - dX.mean(0)) / sd)
        P.append((o.replace('_Death', ''), vals['rich'], vals['poor']))
    P = pd.DataFrame(P, columns=['cause', 'rich', 'poor']); P.to_csv(f'{RES}/h2_percause_death.csv', index=False)
    ax = axs[1]; ax.grid(True, axis='both'); lim = max(P.rich.max(), P.poor.max()) * 1.08
    ax.plot([0, lim], [0, lim], color=GRAY, lw=0.8, ls=(0, (3, 2)), zorder=1)
    ax.scatter(P.rich, P.poor, s=16, color=BLUE, edgecolor='white', linewidth=0.6, zorder=3)
    offs = [(18, -14), (22, 4), (-30, -22)]
    for (_, rw), off in zip(P.assign(d=P.poor - P.rich).nlargest(3, 'd').iterrows(), offs):
        ax.annotate(rw.cause, (rw.rich, rw.poor), xytext=off, textcoords='offset points', fontsize=6, color=INK2,
                    arrowprops=dict(arrowstyle='-', color=INK2, lw=0.4, shrinkA=0, shrinkB=2))
    ax.text(0.03, 0.97, f'{(P.poor > P.rich).mean() * 100:.0f}% of {len(P)} causes above the line of equality',
            transform=ax.transAxes, fontsize=6.5, va='top')
    ax.set_xlim(0, lim); ax.set_ylim(0, lim); ax.set_aspect('equal')
    ax.set_xlabel('R², data-rich countries'); ax.set_ylabel('R², countries with no VR data')
    ax.set_title('Covariates explain more of GBD death-rate change\nwhere data are absent (5-year changes, 9 covariates)', loc='left')
    tag(ax, 'b'); save(fig, 'Fig2_richness')


# ---------------- Figure 3: H3 onset ----------------
def fig3():
    E = pd.read_csv(f'{RES}/h3_event.csv'); H = pd.read_csv(f'{RES}/h3.csv').set_index('outcome')
    fig, axs = plt.subplots(1, 2, figsize=(180 * MM, 68 * MM), gridspec_kw={'width_ratios': [1.45, 1], 'wspace': 0.3})
    ax = axs[0]
    lab_map = {'[-40, -9)': '≤−10', '[-9, -6)': '−9 to\n−7', '[-6, -3)': '−6 to\n−4', '[-3, 0)': '−3 to\n−1',
               '[0, 3)': '0 to\n2', '[3, 6)': '3 to\n5', '[6, 9)': '6 to\n8', '[9, 40)': '≥9'}
    order = list(lab_map)
    ax.axvspan(-0.5, 3.5, color=SHADE, lw=0, zorder=0)
    ax.axhline(0, color=GRAY, lw=0.8, zorder=1)
    for kind, col, lab, dx in [('death', BLUE, 'Deaths', -0.08), ('prev', ORANGE, 'Prevalence', 0.08)]:
        e = E[E.outcome == kind].set_index('bin').reindex(order); x = np.arange(len(order)) + dx
        ax.errorbar(x, e['mean'], yerr=1.96 * e['sem'], color=col, marker='o', ms=4.5, mec='white', mew=0.8,
                    lw=1.5, elinewidth=1, capsize=0, label=lab)
    ax.axvline(3.5, color=INK2, lw=0.7, ls=(0, (3, 2)))
    ax.text(1.5, 0.6, 'Before VR data', ha='center', va='top', fontsize=6.5, color=INK2)
    ax.text(5.5, 0.6, 'After VR data', ha='center', va='top', fontsize=6.5, color=INK2)
    ax.set_xticks(range(len(order))); ax.set_xticklabels([lab_map[o] for o in order])
    ax.set_xlabel('Years relative to first national VR year'); ax.set_ylim(-0.72, 0.66)
    ax.set_ylabel('Log roughness of GBD trajectory\n(within-country, year- and cause-adjusted; 95% CI)')
    ax.legend(loc='lower right', handlelength=1.4)
    ax.set_title('Event study: 13 countries whose VR data begin 1998–2013', loc='left'); tag(ax, 'a')

    ax = axs[1]; k = 'death'
    perm = pd.read_csv(f'{RES}/h3_perm_{k}.csv').iloc[:, 0].values; obs = np.log(H.loc[k, 'est'])
    ax.hist(np.exp(perm), bins=40, color=GRAY, edgecolor='white', linewidth=0.5)
    ax.axvline(np.exp(obs), color=RED, lw=1.5)
    ax.text(np.exp(obs), ax.get_ylim()[1] * 0.95, f'  observed ×{H.loc[k, "est"]:.2f}\n  permutation p = {H.loc[k, "p_perm"]:.3f}',
            color=INK, fontsize=6.5, va='top')
    ax.set_xlabel('Post-onset roughness ratio, deaths'); ax.set_ylabel('Placebo assignments (of 2,000)')
    ax.set_title('Placebo onsets in never- or always-VR countries', loc='left'); tag(ax, 'b')
    save(fig, 'Fig3_onset')


# ---------------- Figure 4: H5 EAPC ----------------
def fig4():
    E = pd.read_csv(f'{RES}/h5_eapc_gbd.csv'); B = pd.read_csv(f'{RES}/h5_eapc_gbd_vs_raw.csv')
    fig, axs = plt.subplots(1, 3, figsize=(180 * MM, 70 * MM), gridspec_kw={'width_ratios': [1, 1.15, 0.8], 'wspace': 0.38})
    ax = axs[0]
    for grp, col, lab in [('poor', ORANGE, 'No VR data'), ('rich', BLUE, 'Data-rich')]:
        w = np.sort(E[E.grp == grp].width.values); y = np.arange(1, len(w) + 1) / len(w)
        ax.plot(w, y, color=col, lw=1.6, label=lab)
        med = np.median(w); ax.plot([med], [0.5], 'o', ms=4.5, color=col, mec='white', mew=0.8)
    ax.set_xscale('log'); ax.set_xlim(0.03, 20); ax.set_xticks([0.1, 1, 10]); ax.set_xticklabels(['0.1', '1', '10'])
    ax.minorticks_off(); ax.grid(True, axis='x')
    ax.set_xlabel('Width of 95% CI of EAPC (percentage points)'); ax.set_ylabel('Cumulative share of country-cause series')
    ax.legend(loc='lower right', handlelength=1.4)
    ax.set_title('Narrower "confidence" where\nthere are no data', loc='left'); tag(ax, 'a')

    ax = axs[1]; ax.grid(True, axis='both')
    gs = (B.gl > 0) | (B.gh < 0); rs = (B.rl > 0) | (B.rh < 0); opp = gs & rs & (np.sign(B.g) != np.sign(B.r))
    L = 10; x = B.r.clip(-L, L); y = B.g.clip(-L, L)
    ax.plot([-L, L], [-L, L], color=GRAY, lw=0.8, ls=(0, (3, 2)), zorder=1)
    ax.axhline(0, color=GRAY, lw=0.5); ax.axvline(0, color=GRAY, lw=0.5)
    ax.scatter(x[~opp], y[~opp], s=5, color=BLUE, alpha=0.35, linewidth=0, zorder=2, label='Other series')
    ax.scatter(x[opp], y[opp], s=9, color=RED, edgecolor='white', linewidth=0.3, zorder=3,
               label=f'Both significant, opposite sign ({opp.mean() * 100:.0f}%)')
    ax.set_xlim(-L, L); ax.set_ylim(-L, L); ax.set_aspect('equal')
    ax.set_xlabel('EAPC from WHO raw VR (%/year)'); ax.set_ylabel('EAPC from GBD 2021 (%/year), same years')
    ax.legend(loc='upper left', handlelength=1, fontsize=6, markerscale=1.5, borderaxespad=0.2)
    ax.set_title(f'GBD vs raw trend, same years\n({len(B):,} series, 78 data-rich countries)', loc='left'); tag(ax, 'b')

    ax = axs[2]; ax.grid(True, axis='x'); ax.grid(False, axis='y')
    cats = [('Raw EAPC inside GBD 95% CI', ((B.r >= B.gl) & (B.r <= B.gh)).mean()),
            ('Same sign', (np.sign(B.g) == np.sign(B.r)).mean()),
            ('GBD significant, raw not', (gs & ~rs).mean()),
            ('Both significant, opposite sign', opp.mean())]
    yy = np.arange(len(cats))[::-1]
    ax.barh(yy, [v * 100 for _, v in cats], height=0.42, color=BLUE, edgecolor='white', linewidth=0.8)
    for yi, (lab, v) in zip(yy, cats):
        ax.text(v * 100 + 2, yi, f'{v * 100:.0f}%', va='center', fontsize=6.5)
    for yi, (lab, v) in zip(yy, cats):
        ax.text(0, yi + 0.36, lab, va='bottom', ha='left', fontsize=6.3, color=INK)
    ax.set_yticks([]); ax.spines['left'].set_visible(False); ax.set_xlim(0, 100); ax.set_ylim(-0.5, len(cats) - 0.1)
    ax.set_xlabel('Share of series (%)'); ax.set_title('Agreement of GBD with raw data', loc='left'); tag(ax, 'c')
    save(fig, 'Fig4_eapc')


for k, f in [('1', fig1), ('2', fig2), ('3', fig3), ('4', fig4)]:
    if k in which: f()
