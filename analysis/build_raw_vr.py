"""Build age-standardised raw VR death rates from the WHO Mortality Database (ICD-10), 1990-2021.
Inputs (public): WHO MDB Morticd10_part1..6 + country_codes (in WHO_MDB_DIR/whomdb), UN WPP2024 population by
5-year age group (WHO_MDB_DIR/wpp.csv.gz). Output: ../raw_vr_asr.csv (iso3, Year, cause, deaths, asr).
Standard population: WHO World Standard 2000-2025, 18 groups (0-4 ... 85+).
"""
import glob, os, numpy as np, pandas as pd
from common import WHO_DIR, ROOT, HERE

W = np.array([8.86, 8.69, 8.60, 8.47, 8.22, 7.93, 7.61, 7.15, 6.59, 6.04, 5.37, 4.55, 3.72, 2.96, 2.21, 1.52, 0.91, 0.63])
W = W / W.sum()
# ICD-10 three-character ranges mapped to GBD causes (supplementary table S1)
ICD = {'IHD': [('I20', 'I25')], 'Stroke': [('I60', 'I69')], 'COPD': [('J40', 'J44')], 'TBLCancer': [('C33', 'C34')],
       'Diabetes': [('E10', 'E14')], 'LiverCancer': [('C22', 'C22')], 'CKD': [('N18', 'N18')],
       'ColonRectumCancer': [('C18', 'C21')], 'StomachCancer': [('C16', 'C16')], 'BreastCancer': [('C50', 'C50')],
       'EsophagealCancer': [('C15', 'C15')], 'PancreaticCancer': [('C25', 'C25')], 'ProstateCancer': [('C61', 'C61')],
       'LRI': [('J09', 'J18'), ('J20', 'J22')], 'Dementias': [('F01', 'F01'), ('F03', 'F03'), ('G30', 'G31')],
       'CLD': [('K70', 'K70'), ('K73', 'K74')], 'Asthma': [('J45', 'J46')], 'Tuberculosis': [('A15', 'A19')],
       'HIVandAIDS': [('B20', 'B24')], 'ParkinsonDisease': [('G20', 'G20')], 'Leukemia': [('C91', 'C95')],
       'BladderCancer': [('C67', 'C67')], 'KidneyCancer': [('C64', 'C65')], 'OvarianCancer': [('C56', 'C56')],
       'CervicalCancer': [('C53', 'C53')], 'HHD': [('I11', 'I11')], 'AF': [('I48', 'I48')], 'CMY': [('I42', 'I42')],
       'IdiopathicEpilepsy': [('G40', 'G41')], 'GARBAGE': [('R00', 'R99')]}


def agegroups(df, prefix):
    """WHO MDB age formats 00-02 -> 18 groups: 0-4 (cols 2-6), 5-9..80-84 (cols 7-22), 85+ (cols 23-25)."""
    a = np.zeros((len(df), 18))
    a[:, 0] = df[[f'{prefix}{i}' for i in range(2, 7)]].fillna(0).sum(1)
    for k, i in enumerate(range(7, 23)): a[:, k + 1] = df[f'{prefix}{i}'].fillna(0)
    a[:, 17] = df[[f'{prefix}{i}' for i in range(23, 26)]].fillna(0).sum(1)
    return a


def main():
    ns = {}; exec(open(os.path.join(HERE, 'isomap.py'), encoding='utf-8').read().replace("'whomdb/country_codes'", repr(os.path.join(WHO_DIR, 'whomdb', 'country_codes'))), ns)
    CMAP = ns['CMAP']
    parts = []
    for f in sorted(glob.glob(os.path.join(WHO_DIR, 'whomdb', 'Morticd10_part?'))):   # '?' avoids matching .zip files
        d = pd.read_csv(f, dtype={'Admin1': str, 'SubDiv': str, 'List': str, 'Cause': str, 'Frmat': str}, low_memory=False)
        d = d[d.Admin1.isna() & d.SubDiv.isna() & d.List.isin(['103', '104']) & d.Frmat.isin(['00', '01', '02'])
              & d.Sex.isin([1, 2]) & (d.Year >= 1990) & (d.Year <= 2021)]
        d['c3'] = d.Cause.str[:3]; parts.append(d)
    d = pd.concat(parts); d = d[d.c3.str.len() == 3]
    A = agegroups(d, 'Deaths'); out = []
    masks = {k: np.logical_or.reduce([(d.c3 >= lo) & (d.c3 <= hi) for lo, hi in r]) for k, r in ICD.items()}
    masks['ALL'] = (d.c3 != 'AAA').values   # detailed codes only; 'AAA' is the all-cause total row
    for name, m in masks.items():
        m = np.asarray(m)
        g = pd.DataFrame(A[m]); g['Country'] = d.Country.values[m]; g['Year'] = d.Year.values[m]
        s = g.groupby(['Country', 'Year']).sum(); s['cause'] = name; out.append(s.reset_index())
    Dth = pd.concat(out); Dth['iso3'] = Dth.Country.map(CMAP); Dth = Dth.dropna(subset=['iso3'])
    w = pd.read_csv(os.path.join(WHO_DIR, 'wpp.csv.gz'), usecols=['ISO3_code', 'Time', 'AgeGrpStart', 'PopTotal'], low_memory=False)
    w = w[w.ISO3_code.notna() & (w.Time >= 1990) & (w.Time <= 2021)]
    w['g'] = np.minimum(w.AgeGrpStart // 5, 17)
    wp = (w.groupby(['ISO3_code', 'Time', 'g']).PopTotal.sum().unstack('g') * 1000).reset_index()
    wp.columns = ['iso3', 'Year'] + [f'p{i}' for i in range(18)]
    M = Dth.merge(wp, on=['iso3', 'Year'])
    dd = M[list(range(18))].values.astype(float); pp = M[[f'p{i}' for i in range(18)]].values
    M['asr'] = (dd / pp * W).sum(1); M['deaths'] = dd.sum(1)
    M = M[['iso3', 'Year', 'cause', 'deaths', 'asr']].sort_values(['iso3', 'Year', 'cause'])
    M.to_csv(os.path.join(ROOT, 'raw_vr_asr.csv'), index=False)
    print('countries', M.iso3.nunique(), 'rows', len(M))


if __name__ == '__main__':
    main()
