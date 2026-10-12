# Where the data end, the model begins — analysis code

Code for the study "Where the data end, the model begins: data availability and model-generated
trends in Global Burden of Disease estimates for 201 countries".

## Inputs

| Input | Source | How to get it |
|---|---|---|
| WHO Mortality Database, raw ICD-9 and ICD-10 files | WHO | `python fetch_public_data.py` |
| UN World Population Prospects 2024, population by 5-year age group | UN DESA | `python fetch_public_data.py` |
| World Bank SP.REG.DTHS.ZS, completeness of death registration | World Bank API | `python fetch_public_data.py` |
| GBD 2021 age-standardised death and prevalence rates, both sexes, 1990–2021, plus the nine covariates | IHME GBD Results Tool (free login) | manual export; set `GBD_CSV` to the file |

The GBD extract cannot be redistributed under IHME's terms. The column naming expected is
`<Cause>` (prevalence) and `<Cause>_Death` (deaths), one row per `CountryCode` × `Year`.

## Run

```
set WHO_MDB_DIR=<folder filled by fetch_public_data.py>   # defaults to analysis/data_public
set GBD_CSV=<path to GBD extract>          # defaults to data/gbd2021_extract.csv
python run_all.py
```

Python 3.12, with pandas, numpy, statsmodels, scipy, matplotlib, openpyxl and pycountry.

## Scripts (in run order)

| Script | Produces |
|---|---|
| `build_raw_vr.py` | `../raw_vr_asr.csv`: raw VR death rates, age-standardised to the WHO 2000–2025 standard |
| `h1_h2.py` | H1 smoothness and H2 covariate tracking → `results/h1_h2.csv`, `results/country_richness.csv` |
| `h1_srs_sensitivity.py` | Post hoc sensitivity of H1 to India and Bangladesh (national sample registration, no WHO VR) → `results/h1_srs_sensitivity.csv` (table S8) |
| `h3_onset.py` | H3 VR-onset difference-in-differences (TWFE), permutation test, event study |
| `h3_robust.py` | H3 with Callaway–Sant'Anna and Borusyak–Jaravel–Spiess estimators, 0- and 6-year anticipation |
| `h4_h5.py` | H4 GBD vs raw VR associations; H5 EAPC precision and agreement |
| `h6_biblio.py` | PubMed counts per year (needs internet access) |
| `figs.py` | Statistical figures (earlier versions of figures 1–4) in `figures/` |
| `fig_data_country.py` | Country-level metrics for the main figures → `results/fig_country_metrics.csv`, `results/fig_onset_countries.csv` |
| `figs_country.py` | Main figures 1–4: maps, country scatter by SDI, 13 onset countries, country agreement (`Fig1_maps`, `Fig2_countries`, `Fig3_onset`, `Fig4_agreement`) |
| `supp_tables.py` | Supplementary tables S1–S7 → `results/Supplementary_tables.xlsx` |

The analysis plan, written before the formal runs, is `../ANALYSIS_PLAN.md`.
Random seeds are fixed (2026), so every number is reproducible.
