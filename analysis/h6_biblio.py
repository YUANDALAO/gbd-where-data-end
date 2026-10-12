"""H6: PubMed counts of GBD secondary trend analyses per year."""
import urllib.request, urllib.parse, json, time, pandas as pd
from common import RES
def n(q):
    u='https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?db=pubmed&retmode=json&term='+urllib.parse.quote(q)
    for _ in range(5):
        try: time.sleep(0.8); return int(json.load(urllib.request.urlopen(u,timeout=30))['esearchresult']['count'])
        except Exception: time.sleep(3)
GBD='("Global Burden of Disease"[tiab] OR "GBD 2019"[tiab] OR "GBD 2021"[tiab] OR "GBD 2017"[tiab])'
TR='("estimated annual percentage change"[tiab] OR EAPC[tiab] OR joinpoint[tiab] OR "average annual percent change"[tiab] OR "age-period-cohort"[tiab] OR "decomposition analysis"[tiab])'
rows=[(y,n(f'{GBD} AND {y}[dp]'),n(f'{GBD} AND {TR} AND {y}[dp]')) for y in range(2012,2026)]
T=pd.DataFrame(rows,columns=['year','gbd_papers','gbd_trend_papers']); T.to_csv(f'{RES}/h6_pubmed.csv',index=False); print(T.to_string(index=False))
