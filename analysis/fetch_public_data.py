"""Download every public input except the GBD extract into WHO_MDB_DIR (default: ./data_public).
- WHO Mortality Database raw files (ICD-10 parts 1-6, ICD-9, country codes)
- UN World Population Prospects 2024, population by 5-year age group
- World Bank SP.REG.DTHS.ZS (saved to ../wb_vr.csv)
GBD 2021 estimates must be downloaded from the GBD Results Tool (https://vizhub.healthdata.org/gbd-results/)
under IHME's terms; see README for the query used.
"""
import io, json, os, urllib.request, zipfile
import pandas as pd
from common import WHO_DIR, ROOT

WHO = 'https://cdn.who.int/media/docs/default-source/world-health-data-platform/mortality-raw-data/'
FILES = {'whomdb': [f'morticd10_part{i}.zip' for i in range(1, 7)] + ['mort_country_codes.zip'], 'icd9': ['morticd09.zip']}
WPP = ('https://population.un.org/wpp/assets/Excel%20Files/1_Indicator%20(Standard)/CSV_FILES/'
       'WPP2024_PopulationByAge5GroupSex_Medium.csv.gz')
WB = 'https://api.worldbank.org/v2/country/all/indicator/SP.REG.DTHS.ZS?format=json&per_page=20000&date=1990:2023'


def get(url):
    with urllib.request.urlopen(url, timeout=600) as r: return r.read()


def main():
    for sub, names in FILES.items():
        os.makedirs(os.path.join(WHO_DIR, sub), exist_ok=True)
        for n in names:
            print('WHO', n); zipfile.ZipFile(io.BytesIO(get(WHO + n))).extractall(os.path.join(WHO_DIR, sub))
    print('WPP 2024'); open(os.path.join(WHO_DIR, 'wpp.csv.gz'), 'wb').write(get(WPP))
    print('World Bank'); j = json.loads(get(WB))
    rows = [(x['countryiso3code'], int(x['date']), x['value']) for x in j[1] if x['value'] is not None]
    pd.DataFrame(rows, columns=['iso3', 'year', 'vr']).to_csv(os.path.join(ROOT, 'wb_vr.csv'), index=False)


if __name__ == '__main__':
    main()
