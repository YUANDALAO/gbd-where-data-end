import pandas as pd, pycountry
fix={'Bolivia':'BOL','Iran (Islamic Republic of)':'IRN','Republic of Korea':'KOR','Republic of Moldova':'MDA','Venezuela':'VEN','United Kingdom':'GBR','Syrian Arab Republic':'SYR','TFYR Macedonia':'MKD','Turkey':'TUR','Russian Federation':'RUS','Czech Republic':'CZE','United States of America':'USA','Saint Vincent and Grenadines':'VCT','Reunion':'REU','Hong Kong SAR':'HKG','Cape Verde':'CPV','Occupied Palestinian Territory':'PSE','Libyan Arab Jamahiriya':'LBY','China: Province of Taiwan only':'TWN','Micronesia (Federated States of)':'FSM','Virgin Islands (USA)':'VIR'}
def iso(n):
    if n in fix: return fix[n]
    if any(k in n for k in ['Former','United Kingdom,','Rodrigues','Antilles','Ryu Kyu','West Berlin']): return None
    try: return pycountry.countries.lookup(n).alpha_3
    except Exception:
        try: return pycountry.countries.search_fuzzy(n)[0].alpha_3
        except Exception: return None
codes=pd.read_csv('whomdb/country_codes',encoding='latin1'); codes['iso3']=codes.name.map(iso)
CMAP=dict(zip(codes.country,codes.iso3))
