# GBD circularity audit — formal analysis plan

Locked 2026-09-30, before the formal runs. The feasibility results (README.md) motivated this plan.
Every hypothesis below is reported whichever way it comes out.

## Question
How far are GBD 2021 time trends, and the exposure–outcome associations that secondary analyses
derive from them, determined by the estimation model (covariates plus space-time smoothing) rather
than by observed data? And does this depend on how much primary data a country contributes?

## Data (all public)
| Item | Source |
|---|---|
| GBD 2021 outcomes and covariates | `featureDataTable_for_R.csv` (201 countries × 1990–2021; age-standardised rates) |
| Primary data richness (exposure of interest) | WHO Mortality Database raw ICD-9/ICD-10 files: number of national VR country-years, 1990–2021 |
| Secondary richness | World Bank SP.REG.DTHS.ZS (completeness of death registration with cause) |
| Raw comparator | WHO MDB ICD-10 deaths, age-standardised to the WHO 2000–2025 standard using UN WPP2024 populations |
| Optional robustness (manual) | IHME GBD 2021 Sources Tool CSV (ghdx.healthdata.org/gbd-2021/sources). The server refuses scripted queries, so this has to be exported by hand |

Richness definitions: `nvr` = VR country-years 1990–2021 (0–32, continuous, primary);
`rich` = nvr ≥ 15 and completeness ≥ 90%; `poor` = nvr = 0.

## Hypotheses and primary tests
**H1 — Smoothness.** GBD trajectories are smoother where there are fewer data.
Metric: roughness = mean |second difference of log rate| per country-cause.
Model: log(roughness) ~ nvr/32 + log(pop) + SDI + cause FE; SEs clustered by country.
Run separately for deaths, prevalence and incidence.
Prediction: negative coefficient.

**H2 — Covariate tracking.** Five-year changes in outcomes are explained by changes in GBD covariates
more strongly where there are fewer data.
Per cause: R² of Δ5 log rate on Δ5 of 9 covariates, compared between poor and rich, with a
country-bootstrap 95% CI; restricted to countries with population ≥ 5M.
Prediction: R²(poor) > R²(rich).

**H3 — VR onset (primary causal test).** Within a country, the arrival of VR data makes GBD death
trajectories rougher.
Sample: countries whose first VR year (ICD-9 or ICD-10) is 1998–2013 and that have ≥ 5 VR years.
Model: country-cause FE + cause-year FE, post-onset indicator, SEs clustered by country.
Inference: permutation p-value (2,000 draws), assigning pseudo-onset years to the same number of
never- or always-VR countries.
Negative-control outcome: prevalence. VR feeds the death models directly, so the effect on deaths
is predicted to be larger than the effect on prevalence.
Event-study bins are reported as well.

**H4 — Agreement with raw data (data-rich countries).** Within-country (TWFE) associations for
pre-specified GBD risk–outcome pairs, estimated on GBD versus raw VR.
Summaries: median |bGBD|/|bRaw|, random-effects pooled log ratio, and the sign-disagreement rate.
Sensitivity analyses: control for garbage-code share; restrict to years with garbage share < 5%.
Two-sided: either inflation or attenuation counts as disagreement.

**H5 — Consequences for typical secondary analyses (EAPC).**
EAPC per country-cause, 1990–2021 (log-linear OLS, 95% CI).
(a) CI width and the share of "significant" EAPCs, by richness.
(b) In rich countries, agreement between GBD EAPC and raw-VR EAPC over the same years:
correlation, sign agreement, and the share where GBD is significant but raw is not.
Prediction: narrower CIs and more nominally significant trends in data-poor countries.

**H6 — Scale of the literature (descriptive).** PubMed counts per year of GBD secondary analyses
that use EAPC or joinpoint.

## Decision rules
- The central claim ("GBD trends are partly model-generated") requires H3 (permutation p < 0.05)
  plus H1 or H2.
- If H3 fails, reposition the paper as descriptive: "data richness shapes GBD trend properties".
- H4 and H5 are reported whatever the direction.

## Known threats and what addresses them
- Real volatility differs by setting → H3's within-country design; population and SDI controls in H1.
- Raw VR is not ground truth (garbage redistribution, ICD transitions) → garbage-share control,
  ICD-10-only years in H4, and the fact that H3 does not depend on raw data being correct.
- Small-number noise in small countries → population ≥ 5M restrictions.
- Only 13 onset countries → permutation inference and leave-one-country-out.
