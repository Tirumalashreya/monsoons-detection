# Project Summary — Indian Monsoon Prediction

## What the original PDF brief said

**Title:** Predicting Annual Indian Monsoon Patterns — AI/ML for Climate Science, using ENSO (El Niño & La Niña)

**Stated objective:**
- Forecast Indian monsoon rainfall using ENSO conditions
- Analyze historical climate datasets
- Build predictive ML models "2026–2030"
- Create visualization dashboards

**Inputs listed:**
- ENSO indices (Niño 3.4, SOI)
- Historical rainfall from IMD
- Seasonal forecast vs. actual rainfall
- Geospatial datasets: river basins & agricultural zones

**Suggested (but unvalidated) AI/ML approaches:**
- EDA / correlation analysis
- Time-series: ARIMA, Prophet, LSTM
- Classification: Random Forest, XGBoost
- Regression: rainfall intensity
- Hybrid: stats + deep learning

**Deliverables:** data pipeline scripts, predictive ML model, dashboard (Streamlit/Power BI), research report.

**Workflow (as drawn):** Phase 1–2 Data Collection & Preprocessing → Phase 3 Modeling → Phase 4–5 Evaluation & Deployment → Phase 6 Reporting.

The brief never specified: target granularity (national vs. regional), what "predict 2026–2030" actually means, which predictor lags to use, or how to evaluate skill.

---

## What we're actually building (and why it differs)

| Brief said | We're actually doing | Why |
|---|---|---|
| LSTM / Prophet for time-series | Simple regression + tree ensembles (RF/XGBoost) | Only ~70–150 years of data; deep learning overfits badly at this sample size, Prophet doesn't fit annual-granularity data |
| "Predict 2026–2030" as if one output | 2026 = real point forecast (pre-season ENSO/IOD data already exists); 2027–2030 = conditional scenario tables (El Niño / La Niña / Neutral composites) | ENSO/IOD are only predictable ~6–12 months ahead — nobody can know 2028's ENSO state today |
| ENSO indices as generic input | Only **pre-season (spring, MAM average)** ENSO/IOD/SOI values used | Using concurrent-season values would leak future information into the forecast |
| No mention of national vs. regional | National model built first (from IITM's official area-weighted AISMR series), subdivision-level regional model next | National = best-documented, cleanest ENSO signal, validates the pipeline; regional = more useful for agriculture/water-management, noisier |
| Geospatial river-basin/agri-zone data as a "model input" | Reclassified as a downstream **dashboard/visualization** concern, not a predictor | It doesn't predict monsoon — it's for presenting the forecast, not producing it |
| No evaluation method specified | Leave-one-year-out cross-validation, benchmarked against a climatology baseline | Random train/test splits leak future info in time-series data; climatology is the real bar to beat |
| IOD not mentioned at all | Added IOD (DMI) as a third predictor alongside ENSO | IOD can offset/override ENSO's effect on the monsoon (e.g. 1997) — a well-documented gap in ENSO-only approaches |

---

## Data sources actually in use

| File | Source | Role | Coverage |
|---|---|---|---|
| `nino34_oni.txt` | NOAA CPC | ENSO predictor (ANOM column) | 1950–2026 |
| `dmi_iod.txt` | NOAA PSL | IOD predictor | 1870–2026 |
| `soi.txt` | NOAA (standardized table only — see note) | SOI predictor | 1951–2026 |
| `iitm_aismr.txt` | IITM Pune (via web archive) | **National target** (% departure from climatology) | 1871–2019 |
| `Sub_Division_IMD_2017.csv` | IMD / public dataset | **Regional target** (36 subdivisions) — not yet merged in | 1901–2017 |
| `rainfall_area-wt_India_1901-2015.csv` | IMD/OGD | National backup/cross-check | 1901–2015 |
| `india_crop_production_1997_2020.csv` | GitHub mirror of public crop stats | Agriculture-impact check (Kharif rice yield) | 1997–2020 |

**Data-quality issues found and fixed along the way:**
- `soi.txt` silently contains **two different tables** (raw pressure anomaly + standardized SOI) concatenated in one file — was accidentally reading both as one, causing duplicate years with conflicting values. Fixed to read only the standardized table.
- `read_fwf` in this environment was inferring columns as string dtype, not numeric — silently broke both `-999.9`/`-9999` sentinel filtering and any `.mean()` call. Fixed with explicit `pd.to_numeric` coercion everywhere.
- A stray sentinel-declaration line (`-9999`) in `dmi_iod.txt` was passing as a fake "year -9999" row — fixed with a real year-range filter.
- `Sub_Division_IMD_2017.csv` has isolated missing years for 3 subdivisions (Andaman & Nicobar, Arunachal Pradesh, Lakshadweep) — to be dropped, not interpolated.
- Naively averaging the 36 subdivisions to get a "national" number would be wrong (unweighted — biases toward small subdivisions); solved by using IITM's official pre-weighted AISMR series instead.

---

## Results

**National:** neither linear regression (RMSE 10.25) nor Random Forest (RMSE 10.98) beat the climatology baseline (RMSE 9.99). Correlations between spring ENSO/IOD/SOI and national rainfall departure are all weak (-0.11 to 0.17). Honest finding: national-level, spring-only ENSO/IOD/SOI does not show real predictive skill.

**Regional:** signal is real but heterogeneous. Looping the same approach across all 36 subdivisions: **10 of 36 beat their own baseline** (Arunachal Pradesh, Uttarakhand, Marathwada, Punjab, West/East Uttar Pradesh, Telangana, East Rajasthan, Bihar, Konkan & Goa). 5 subdivisions got clearly worse than baseline (Saurashtra & Kutch, Gujarat Region, Rayalseema, West Rajasthan, North Interior Karnataka). Final models (trained on all available years, not held out) saved to `models/` for the 10 skilled subdivisions only.

**Disaster preparedness:** added a `risk_flag` column to regional data (Deficient/drought risk, Excess/flood risk, Normal — ±10% departure bands) — turns the raw forecast into an actionable early-warning label.

**Agriculture:** raw correlation between Kharif rice yield and national rainfall departure is 0.33, but yield has a strong 1997–2020 upward trend (technology/irrigation, corr. 0.84 with year) that masks the weather signal. After detrending yield, correlation with rainfall departure rises to **0.57** — rainfall genuinely matters for yield, once technology-driven growth is separated out.

**Recommendation (answering the brief's closing slide, with real evidence):**
- *Agriculture:* the 10 skilled subdivisions can get a pre-season forecast before Kharif sowing, and rainfall is confirmed to meaningfully affect yield (0.57 correlation after detrending) — a real basis for crop-choice guidance, not a guess.
- *Water management:* same pre-season lead time lets reservoir/irrigation planners plan release schedules months ahead, for the subdivisions with demonstrated skill.
- *Disaster preparedness:* the `risk_flag` label is usable as-is for pre-positioning drought/flood response, per subdivision, per year.
- *Caveat, stated honestly:* this only holds for the 10 subdivisions with proven skill — recommending action off the national number or the other 26 subdivisions would be irresponsible, since no real skill was demonstrated there.

## Current status

- [x] All predictor files parsed, cleaned, reduced to pre-season (spring) features
- [x] Predictors merged into one table (`df`)
- [x] National target loaded and merged → `national_df`, saved to `data/processed/national_df.csv`
- [x] Baseline (climatology) + linear regression + Random Forest on national data — neither beat baseline
- [x] Leave-one-year-out cross-validation (national + all 36 subdivisions)
- [x] Regional (subdivision-level) model — 10/36 show real skill, final models trained and saved
- [x] Disaster-preparedness risk flag (regional)
- [x] Agriculture impact check (Kharif rice yield vs. rainfall, detrended)
- [x] 2026 real forecast (run final models on 2026 pre-season data); 2027–2030 via interactive scenario sliders
- [x] Dashboard — live interactive predictor (National/Regional), clickable choropleth map of all 36 subdivisions
- [x] National model swapped to DJF-ONI (weak but real improvement over baseline: RMSE 9.94 vs 9.99)
- [x] Agriculture vulnerability assessment (state-level, rainfall vs. detrended yield)
- [x] Infrastructure risk assessment (district-level, flood hazard x housing fragility x population)
- [x] Human vulnerability index (district-level, IPCC hazard x exposure x vulnerability framework)
- [x] Monthly breakdown forecast (June/July/August/September separately, not just seasonal total) — 27 of 144 subdivision-month combinations show real skill; closest honest alternative to a specific-date forecast, which isn't physically possible with ENSO/IOD/SOI
- [x] Pushed to GitHub: github.com/Tirumalashreya/monsoons-detection
- [ ] Close 2020–2025 rainfall data gap

