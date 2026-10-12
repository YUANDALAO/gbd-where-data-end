# Where the data end, the model begins

Analysis code and aggregate results for:

> Chen Y, et al. Where the data end, the model begins: data availability and model-generated trends in Global Burden of Disease estimates for 201 countries. (Manuscript under review.)

The study asks whether GBD 2021 cause-specific trends behave differently in countries with and without vital registration (VR) data: how smooth they are, how closely they track model covariates, how they change when VR begins, and how well their trends agree with raw VR.

## Layout

| Path | Contents |
|---|---|
| `ANALYSIS_PLAN.md` | Analysis plan with hypotheses H1–H6, written before the formal runs |
| `analysis/` | All scripts, in run order (see `analysis/README.md`) |
| `analysis/results/` | Aggregate results and the supplementary tables (`Supplementary_tables.xlsx`, `supp/`) |
| `analysis/figures/` | Figures 1–4 (PDF and PNG) and legends |
| `raw_vr_asr.csv` | Age-standardised death rates computed from the raw WHO Mortality Database (built by `analysis/build_raw_vr.py`) |
| `wb_vr.csv` | World Bank completeness of death registration (SP.REG.DTHS.ZS) |

## Reproducing

```
cd analysis
python fetch_public_data.py          # WHO Mortality Database, UN WPP 2024, World Bank (about 600 MB)
set GBD_CSV=<path to the GBD 2021 extract>
python run_all.py
```

The GBD 2021 estimates have to be exported from the IHME GBD Results Tool (free registration) and cannot be redistributed here; `analysis/README.md` lists the measures, years and column names needed. All other inputs are public and are downloaded by `fetch_public_data.py`. Random seeds are fixed, so the permutation and bootstrap results reproduce exactly.

Python 3.12; package versions in `requirements.txt`.

## Licence

Code: MIT (see `LICENSE`). Derived results are provided for verification of the published analysis; the underlying WHO, UN, World Bank and IHME data remain under their providers' terms.
