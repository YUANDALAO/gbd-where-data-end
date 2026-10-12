"""Country-level main figures for the BMJ manuscript (every country a dot or a map cell).
Fig1_maps:      (a) years of national VR 1990-2021; (b) relative roughness of GBD death trajectories;
                (c) growth of GBD trend papers; (d) illustrative trajectory, Turkiye.
Fig2_countries: 201 countries by SDI - (a) relative roughness, (b) relative width of the EAPC confidence interval.
Fig3_onset:     13 countries whose VR began in 1998-2013, one panel each, plus the pooled event study.
Fig4_agreement: 78 data-rich countries - (a) map of the share of series whose raw EAPC lies inside the GBD CI,
                (b) the same share and sign agreement, country by country.
Inputs: results/fig_country_metrics.csv, fig_onset_countries.csv (fig_data_country.py), h3*.csv, h6_pubmed.csv.
Palette: groups blue #2a78d6 / yellow #eda100 / red #e34948 (validated; yellow is always directly labelled).
"""
import json, os
import numpy as np, pandas as pd, matplotlib as mpl, matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.colors import LinearSegmentedColormap, LogNorm, Normalize
import pycountry
from common import HERE, ROOT, RES, load_gbd

BLUE, YEL, RED, GRAY = '#2a78d6', '#eda100', '#e34948', '#9a9994'
INK, INK2, GRID, SHADE = '#0b0b0b', '#52514e', '#e6e5e1', '#f3f2ef'
GCOL = {'rich': BLUE, 'mid': YEL, 'poor': RED}
GLAB = {'rich': 'Complete VR (≥15 years, ≥90% complete)', 'mid': 'Partial VR', 'poor': 'No national VR, 1990–2021'}
mpl.rcParams.update({
    'font.family': 'Arial', 'font.size': 7, 'axes.labelsize': 7, 'axes.titlesize': 7.5, 'xtick.labelsize': 6.5,
    'ytick.labelsize': 6.5, 'axes.edgecolor': INK2, 'axes.labelcolor': INK, 'xtick.color': INK2, 'ytick.color': INK2,
    'text.color': INK, 'axes.linewidth': 0.6, 'xtick.major.width': 0.6, 'ytick.major.width': 0.6, 'xtick.major.size': 2.5,
    'ytick.major.size': 2.5, 'axes.spines.top': False, 'axes.spines.right': False, 'axes.grid': True, 'axes.grid.axis': 'y',
    'grid.color': GRID, 'grid.linewidth': 0.5, 'axes.axisbelow': True, 'legend.frameon': False, 'legend.fontsize': 6.5,
    'pdf.fonttype': 42, 'lines.linewidth': 1.5})
MM = 1 / 25.4
FIG = os.path.join(HERE, 'figures')
GEO = json.load(open(os.path.join(HERE, 'ne_110m_countries.geojson'), encoding='utf-8'))   # Natural Earth 1:110m, public domain
C = pd.read_csv(os.path.join(RES, 'fig_country_metrics.csv'))
DIV = LinearSegmentedColormap.from_list('div', [RED, '#f0efec', BLUE])
SEQ = LinearSegmentedColormap.from_list('seq', ['#eef3fa', '#9cc0ea', BLUE, '#123f78'])
NAME = {'IRN': 'Iran', 'TUR': 'Türkiye', 'PSE': 'Palestine', 'RUS': 'Russia', 'USA': 'USA', 'GBR': 'UK',
        'KOR': 'South Korea', 'VNM': 'Viet Nam', 'COD': 'DR Congo', 'TZA': 'Tanzania', 'SYR': 'Syria', 'BOL': 'Bolivia',
        'VEN': 'Venezuela', 'MDA': 'Moldova'}


def cname(iso):
    if iso in NAME: return NAME[iso]
    c = pycountry.countries.get(alpha_3=iso)
    return getattr(c, 'common_name', None) or (c.name if c else iso)


def tag(ax, s, x=-0.1): ax.text(x, 1.03, s, transform=ax.transAxes, fontsize=9, fontweight='bold', va='bottom')


def save(fig, name):
    fig.savefig(f'{FIG}/{name}.pdf', bbox_inches='tight'); fig.savefig(f'{FIG}/{name}.png', dpi=300, bbox_inches='tight')
    plt.close(fig); print('saved', name)


def polys(feat):
    g = feat['geometry']; parts = g['coordinates'] if g['type'] == 'MultiPolygon' else [g['coordinates']]
    return [np.array(p[0]) for p in parts]


def iso_of(f): return f['properties'].get('ADM0_A3') or f['properties'].get('ISO_A3')


def draw_map(ax, values, cmap, norm, missing='#f7f6f3'):
    for f in GEO['features']:
        if f['properties'].get('CONTINENT') == 'Antarctica': continue
        v = values.get(iso_of(f), np.nan)
        col = cmap(norm(v)) if np.isfinite(v) else missing
        for p in polys(f): ax.add_patch(Polygon(p, closed=True, facecolor=col, edgecolor='white', linewidth=0.25))
    ax.set_xlim(-170, 180); ax.set_ylim(-57, 84); ax.set_aspect('equal'); ax.axis('off')


def cbar(fig, ax, cmap, norm, ticks, labels, title):
    cax = ax.inset_axes([0.0, 0.12, 0.17, 0.032])
    sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
    cb = fig.colorbar(sm, cax=cax, orientation='horizontal', ticks=ticks); cb.ax.set_xticklabels(labels, fontsize=5.8)
    cb.outline.set_visible(False); cb.ax.tick_params(length=2); cb.ax.minorticks_off(); cax.set_title(title, fontsize=6, loc='left', pad=2)


def fig1():
    fig = plt.figure(figsize=(180 * MM, 170 * MM))
    gs = fig.add_gridspec(3, 2, height_ratios=[1, 1, 0.62], hspace=0.2, wspace=0.28)
    ax = fig.add_subplot(gs[0, :])
    draw_map(ax, dict(zip(C.iso3, C.nvr)), SEQ, Normalize(0, 32))
    cbar(fig, ax, SEQ, Normalize(0, 32), [0, 16, 32], ['0', '16', '32'], 'Years with VR data')
    n0 = int((C.nvr == 0).sum())
    ax.set_title(f'Years with national vital registration (VR) data, 1990–2021: none in {n0} of {len(C)} countries', loc='left'); tag(ax, 'a', -0.02)
    ax = fig.add_subplot(gs[1, :])
    norm = LogNorm(0.25, 4)
    draw_map(ax, dict(zip(C.iso3, C.rough_rel.clip(0.25, 4))), DIV, norm)
    cbar(fig, ax, DIV, norm, [0.25, 1, 4], ['¼', '1', '4'], 'Roughness (1 = average)')
    ax.set_title('Roughness of GBD 2021 death-rate trajectories: smooth where VR data are absent', loc='left'); tag(ax, 'b', -0.02)
    for iso in ['IND', 'RUS', 'IDN', 'NGA', 'BRA', 'ETH']:
        for f in GEO['features']:
            if iso_of(f) == iso:
                p = max(polys(f), key=len); x, y = p[:, 0].mean(), p[:, 1].mean()
                v = C.set_index('iso3').rough_rel.get(iso)
                ax.text(x, y, f'{cname(iso)}\n×{v:.1f}', fontsize=5.3, ha='center', va='center', color=INK)

    ax = fig.add_subplot(gs[2, 0]); B = pd.read_csv(f'{RES}/h6_pubmed.csv'); B = B[B.year >= 2014]
    ax.bar(B.year, B.gbd_trend_papers, width=0.72, color=GRAY, edgecolor='white', linewidth=0.6)
    for y, v in zip(B.year, B.gbd_trend_papers):
        if y in (2016, 2020, 2023, 2025): ax.text(y, v + 40, f'{v:,}', ha='center', va='bottom', fontsize=6.2)
    ax.set_xticks(B.year[::2]); ax.set_ylim(0, 2600); ax.set_ylabel('PubMed records per year')
    ax.set_title('Secondary analyses of GBD trends per year', loc='left'); tag(ax, 'c')

    ax = fig.add_subplot(gs[2, 1]); D = load_gbd(); RAW = pd.read_csv(os.path.join(ROOT, 'raw_vr_asr.csv')); on = 2009
    t = D[D.CountryCode == 'TUR'].sort_values('Year').set_index('Year'); gb = t.IHD_Death * 1e5
    rr = RAW[(RAW.iso3 == 'TUR') & (RAW.cause == 'IHD')].set_index('Year').asr.sort_index() * 1e5
    rr = rr * gb.reindex(rr.index).mean() / rr.mean()
    ax.axvspan(1989.5, on - 0.5, color=SHADE, zorder=0, lw=0)
    ax.plot(gb.index, gb.values, color=RED, lw=1.6, zorder=3)
    ax.plot(rr.index, rr.values, 'o', ms=3.8, mfc=BLUE, mec='white', mew=0.7, zorder=4)
    top = max(gb.max(), rr.max()) * 1.04; ax.set_ylim(min(gb.min(), rr.min()) * 0.82, top * 1.08); ax.set_xlim(1989.5, 2021.5)
    ax.text(1990.4, top, 'No VR data', color=INK2, fontsize=6.2, va='bottom'); ax.text(on + 0.3, top, 'VR from 2009', color=INK2, fontsize=6.2, va='bottom')
    ax.text(1996, gb.loc[1996] * 1.06, 'GBD 2021 estimate', color=RED, fontsize=6.3)
    ax.text(2009.2, rr.min() * 0.97, 'WHO raw VR (rescaled)', color=BLUE, fontsize=6.3, va='top')
    ax.set_ylabel('IHD deaths per 100 000\n(age standardised)')
    ax.set_title('Example: ischaemic heart disease, Türkiye', loc='left'); tag(ax, 'd')
    save(fig, 'Fig1_maps')


def scatter_panel(ax, col, ylab, title, letter, labels):
    labels = labels[col]
    d = C.dropna(subset=[col, 'sdi'])
    for g in ['rich', 'mid', 'poor']:
        s = d[d.grp == g]
        ax.scatter(s.sdi, s[col], s=4 + 120 * np.sqrt(s['pop'] / d['pop'].max()), color=GCOL[g], alpha=0.8,
                   edgecolor='white', linewidth=0.4, zorder=3, label=f'{GLAB[g]} (n={len(s)})')
        # median within SDI quintiles to show separation at the same development level
        q = pd.qcut(d.sdi, 5, labels=False); s = s.assign(q=q[s.index])
        m = s.groupby('q').agg(x=('sdi', 'median'), y=(col, 'median'), n=(col, 'size')); m = m[m.n >= 4]
        ax.plot(m.x, m.y, color=GCOL[g], lw=1.6, zorder=4, path_effects=[mpl.patheffects.withStroke(linewidth=2.8, foreground='white')])
    ax.set_yscale('log'); ax.axhline(1, color=GRAY, lw=0.7, zorder=1)
    ax.set_yticks([0.25, 0.5, 1, 2, 4]); ax.set_yticklabels(['¼', '½', '1', '2', '4']); ax.minorticks_off()
    ax.set_xlabel('Socio-demographic Index, 2019'); ax.set_ylabel(ylab); ax.grid(True, axis='both')
    for iso, dx, dy in labels:
        r = d[d.iso3 == iso]
        if r.empty: continue
        x, y = r.sdi.iloc[0], r[col].iloc[0]
        ax.annotate(cname(iso), (x, y), xytext=(dx, dy), textcoords='offset points', fontsize=5.6, color=INK,
                    ha='left' if dx >= 0 else 'right', va='center', arrowprops=dict(arrowstyle='-', color=INK2, lw=0.4))
    ax.set_title(title, loc='left'); tag(ax, letter, -0.13)


def fig2():
    fig, axs = plt.subplots(1, 2, figsize=(180 * MM, 92 * MM), gridspec_kw={'wspace': 0.3})
    L = {'rough_rel': [('IND', -14, 12), ('CHN', 14, -10), ('USA', 14, -14), ('IDN', 14, -8), ('NGA', -14, -10), ('PAK', -16, 10),
                       ('BRA', 12, 10), ('RUS', -14, 8), ('ETH', 14, 12), ('JPN', -14, 10), ('BGD', -14, 8), ('TUR', 14, 6)],
         'ci_width_rel': [('IND', 12, 12), ('CHN', 14, -12), ('USA', 14, -16), ('IDN', 12, -10), ('NGA', -12, -12), ('PAK', -14, 8),
                          ('BRA', 16, -16), ('RUS', -14, 8), ('ETH', -14, 10), ('JPN', 14, 8), ('BGD', -14, -8), ('TUR', 14, 6)]}
    scatter_panel(axs[0], 'rough_rel', 'Roughness of GBD death trajectories\n(geometric mean over 112 causes; 1 = average country)',
                  'Roughness of GBD death trends', 'a', L)
    scatter_panel(axs[1], 'ci_width_rel', 'Width of conventional 95% CI of GBD EAPC\n(median over causes; 1 = typical width)',
                  'Apparent precision of GBD trends', 'b', L)
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, loc='lower center', ncol=3, bbox_to_anchor=(0.5, -0.06), markerscale=0.7, handletextpad=0.3)
    fig.text(0.5, -0.1, 'Each dot is one country, sized by 2019 population. Lines join group medians within quintiles of SDI.',
             ha='center', fontsize=6.2, color=INK2)
    save(fig, 'Fig2_countries')


def fig3():
    O = pd.read_csv(f'{RES}/fig_onset_countries.csv'); E = pd.read_csv(f'{RES}/h3_event.csv'); H = pd.read_csv(f'{RES}/h3.csv').set_index('outcome')
    order = O.groupby('CountryCode').onset.first().sort_values().index.tolist()
    fig = plt.figure(figsize=(180 * MM, 112 * MM)); gs = fig.add_gridspec(3, 8, hspace=0.75, wspace=0.3, width_ratios=[1, 1, 1, 1, 1, 0.55, 1.4, 1.4])
    lim = np.nanpercentile(np.abs(O.adj), 99) * 1.05
    for k, iso in enumerate(order):
        ax = fig.add_subplot(gs[k // 5, k % 5]); d = O[O.CountryCode == iso].sort_values('Year'); on = int(d.onset.iloc[0])
        ax.axvspan(1991.5, on - 0.5, color=SHADE, lw=0, zorder=0); ax.axhline(0, color=GRAY, lw=0.6, zorder=1)
        pre, post = d[d.Year < on], d[d.Year >= on]
        ax.plot(d.Year, d.adj, color=GRAY, lw=0.6, zorder=2)
        ax.scatter(pre.Year, pre.adj, s=5, color=RED, zorder=3, lw=0); ax.scatter(post.Year, post.adj, s=5, color=BLUE, zorder=3, lw=0)
        for s, col in [(pre, RED), (post, BLUE)]:
            if len(s): ax.plot([s.Year.min(), s.Year.max()], [s.adj.mean()] * 2, color=col, lw=1.8, zorder=4)
        ax.set_ylim(-lim, lim); ax.set_xlim(1991, 2022); ax.set_xticks([1995, 2005, 2015]); ax.set_xticklabels(["'95", "'05", "'15"])
        ax.tick_params(labelsize=5.6); ax.set_title(cname(iso) + chr(10) + f'VR from {on}', loc='left', fontsize=6.2, pad=2, linespacing=1.05)
        if k % 5: ax.set_yticklabels([])
        else: ax.set_ylabel('Log roughness', fontsize=6)
    ax = fig.add_subplot(gs[:, 6:])
    lab_map = {'[-40, -9)': '≤−10', '[-9, -6)': '−9', '[-6, -3)': '−6', '[-3, 0)': '−3', '[0, 3)': '0', '[3, 6)': '3', '[6, 9)': '6', '[9, 40)': '≥9'}
    e = E[E.outcome == 'death'].set_index('bin').reindex(list(lab_map)); x = np.arange(len(e))
    ax.axvspan(3.5, 7.5, color='#eef3fa', lw=0, zorder=0); ax.axhline(0, color=GRAY, lw=0.7)
    ax.errorbar(x, e['mean'], yerr=1.96 * e['sem'], color=INK, marker='o', ms=3.5, mec='white', mew=0.6, lw=1.2, elinewidth=0.9, capsize=0)
    ax.set_xticks(x); ax.set_xticklabels(list(lab_map.values()), fontsize=5.6); ax.set_xlabel('Years relative to first VR year (3-year bins)', fontsize=6.3)
    ax.set_title(f'All 13 countries pooled: ×{H.loc["death", "est"]:.2f} rougher\nafter onset (permutation p={H.loc["death", "p_perm"]:.3f})',
                 loc='left', fontsize=6.5)
    ax.set_ylabel('Log roughness (95% CI)', fontsize=6)
    note = fig.add_subplot(gs[2, 3:5]); note.axis('off')
    note.text(0.02, 0.95, 'Red: years before national VR data\nBlue: years with VR data\nBars: period means\n\n'
              'Log roughness of GBD death\ntrajectories, cause-year and\ncountry-cause means removed',
              transform=note.transAxes, fontsize=6, color=INK2, va='top')
    save(fig, 'Fig3_onset')


def fig4():
    d = C.dropna(subset=['inside']).copy(); d['inside_pct'] = d.inside * 100; d['sign_pct'] = d.sign * 100
    fig = plt.figure(figsize=(180 * MM, 160 * MM)); gs = fig.add_gridspec(2, 1, height_ratios=[1, 1.05], hspace=0.18)
    ax = fig.add_subplot(gs[0]); seq = LinearSegmentedColormap.from_list('agr', ['#7a1d17', RED, '#f4c7c2', '#f0efec'])
    draw_map(ax, dict(zip(d.iso3, d.inside_pct)), seq, Normalize(0, 50))
    cbar(fig, ax, seq, Normalize(0, 50), [0, 10, 20, 30, 40, 50], ['0', '10', '20', '30', '40', '50%'],
         'Raw VR trend inside GBD 95% CI (% of causes)')
    ax.set_title(f'In {len(d)} data-rich countries, the trend in the raw registration data fell inside GBD\'s conventional confidence interval for '
                 f'{(d.inside * d.n_series).sum() / d.n_series.sum() * 100:.0f}% of causes', loc='left'); tag(ax, 'a', -0.02)
    ax = fig.add_subplot(gs[1]); d = d.sort_values('inside_pct'); x = np.arange(len(d))
    ax.vlines(x, d.inside_pct, d.sign_pct, color=GRID, lw=1.2, zorder=1)
    ax.scatter(x, d.sign_pct, s=10, color=BLUE, zorder=3, lw=0, label='Same direction of trend (GBD vs raw VR)')
    ax.scatter(x, d.inside_pct, s=10, color=RED, zorder=3, lw=0, label='Raw trend inside GBD 95% CI')
    ax.set_xticks(x); ax.set_xticklabels(d.iso3, rotation=90, fontsize=4.6); ax.set_xlim(-1, len(d)); ax.set_ylim(0, 100)
    ax.set_ylabel('% of causes (series per country: median ' + f'{int(d.n_series.median())})'); ax.grid(True, axis='y')
    ax.legend(loc='lower left', bbox_to_anchor=(0, 1.0), ncol=2, handletextpad=0.2, markerscale=1.2)
    ax.set_title('Country by country: trends agree in direction, but GBD intervals rarely contain the observed trend', loc='left', pad=16)
    tag(ax, 'b', -0.06)
    save(fig, 'Fig4_agreement')


if __name__ == '__main__':
    import matplotlib.patheffects  # noqa
    fig1(); fig2(); fig3(); fig4()
