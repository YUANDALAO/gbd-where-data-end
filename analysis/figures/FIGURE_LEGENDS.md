# Figure legends (draft)

**Figure 1. Growth of GBD-based trend analyses, and an illustrative trajectory.**
(a) PubMed records per year that mention the Global Burden of Disease together with EAPC, joinpoint,
age-period-cohort or decomposition analysis (search run 2026-09-30; query in `h6_biblio.py`).
(b) Age-standardised ischaemic heart disease mortality in Türkiye, whose national vital registration (VR)
data in the WHO Mortality Database begin in 2009 (shaded area = no VR data).
- Blue line: GBD 2021 estimate.
- Orange points: WHO raw VR, age-standardised to the WHO 2000–2025 standard and rescaled to the GBD
  mean over 2009–2021, because the two use different standard populations.
- Roughness is the mean absolute second difference of the log rate.
- This panel is illustrative only; the systematic evidence is in Figures 2–3.

**Figure 2. Properties of GBD trajectories by availability of primary data.**
(a) Relative roughness of GBD 2021 trajectories, 1990–2021, by years of national VR data. Points are
geometric means across countries of cause-adjusted roughness (country × cause log roughness minus
the cause mean); bars are 95% CIs; n = countries.
Adjusted models (log population, SDI, cause fixed effects, SEs clustered by country): roughness ratio
for full versus no VR data was 3.55 (95% CI 2.75–4.57) for deaths and 1.54 (1.41–1.69) for prevalence.
(b) For each of 112 causes of death: R² of 5-year changes in the log age-standardised death rate
regressed on 5-year changes in nine GBD covariates, in countries with no VR data (y) versus data-rich
countries (x). Countries with population ≥ 5 million only. The dashed line is equality.

**Figure 3. Within-country effect of the arrival of VR data on GBD trajectories.**
(a) Event study for the 13 countries whose first national VR year falls in 1998–2013. The outcome is
log roughness after removing cause-year and country-cause means, expressed relative to each country's
mean. Bars are 95% CIs (country-level SEM × 1.96).
Pooled difference-in-differences estimate (country and year fixed effects; controls are 97 always-VR
and 66 never-VR countries): deaths ×1.51 (95% CI 1.06–2.15); prevalence ×1.17 (1.05–1.31).
(b) Distribution of the same estimator for deaths across 2,000 placebo assignments of the observed
onset years to randomly drawn control countries. The red line is the observed estimate.

**Figure 4. Consequences for estimated annual percentage change (EAPC).**
(a) Cumulative distribution of the width of the conventional 95% CI of EAPC (log-linear OLS, 1990–2021)
across country–cause series of GBD 2021 death rates, for countries with no VR data and data-rich
countries. Points mark the medians. With cause fixed effects, CIs in countries without data are 0.50×
as wide (0.45–0.55).
(b) EAPC estimated from GBD 2021 versus from WHO raw VR over identical years (≥ 15 years), for 2,232
country–cause series in 78 data-rich countries. Series where both estimates are significant but of
opposite sign are shown in red. Values are clipped at ±10%/year for display.
(c) Summary of agreement between the two.

*Data-rich* = ≥ 15 VR years 1990–2021 (WHO Mortality Database, ICD-9/10) and World Bank completeness of
death registration ≥ 90%. *No VR data* = no national VR country-years in the WHO Mortality Database.
