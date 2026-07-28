# Indian Monsoon Prediction

AI/ML project forecasting Indian monsoon rainfall using ENSO, IOD, and SOI climate indices — built from an initial project brief, refined and validated through iterative EDA, feature engineering, and honest model evaluation.

## What this project actually does

- Predicts monsoon-season (June-September) rainfall departure from climatology, using **pre-season** (lagged, not concurrent) ENSO/IOD/SOI values — a genuine forecast, not a hindcast.
- Evaluates **national** and **regional** (36 IMD subdivisions) approaches separately.
- Reports results honestly: where the model beats a climatology baseline, and where it doesn't.
- Extends into downstream impact: a drought/flood early-warning flag, and a rainfall-vs-crop-yield agriculture check.
- Ships an interactive Streamlit dashboard — a live predictor (not a static report) with a clickable map of all 36 subdivisions.

## Key results

**National:** no approach with 3 spring predictors (ENSO/IOD/SOI, March-May) beat the climatology baseline. A later lag window (ENSO only, Dec-Jan-Feb) shows weak but real improvement (RMSE 9.94 vs. baseline 9.99) — included in the dashboard as the current best-available national approach, clearly flagged as marginal, not proven.

**Regional:** genuinely heterogeneous. Of 36 subdivisions, **10 show real predictive skill** (beat their own climatology baseline in leave-one-year-out cross-validation): Arunachal Pradesh, Uttarakhand, Marathwada, Punjab, West Uttar Pradesh, East Uttar Pradesh, Telangana, East Rajasthan, Bihar, Konkan & Goa. 5 subdivisions get measurably worse than baseline. Final models trained on all available years are saved per skilled subdivision.

**Agriculture:** Kharif rice yield correlates 0.33 with rainfall departure raw, but yield has a strong 1997-2020 upward trend from irrigation/seed/fertilizer improvements (corr. 0.84 with year) that masks the weather signal. After detrending, correlation rises to **0.57** — rainfall genuinely affects yield, once technology-driven growth is separated out.

**Disaster preparedness:** a `risk_flag` classification (Deficient/drought risk, Excess/flood risk, Normal — ±10% departure bands) turns raw rainfall numbers into an actionable early-warning label.

**2026 forecast:** a real, live forecast (not backtested) — 2026's pre-season ENSO/IOD/SOI values already exist, so the 10 skilled models produce genuine 2026 predictions. 2027-2030 have no real forecast possible yet (that data doesn't exist until those years happen) — the dashboard instead offers interactive scenario testing for those years.

## Weather-impact vulnerability assessments

Three additional analyses extend the core forecast into downstream impact, using the IPCC risk framework (Risk = Hazard x Exposure x Vulnerability) where relevant:

**1. Agriculture vulnerability** (`agriculture_vulnerability.csv`) — state-level Kharif rice yield, detrended, correlated against each state's matched IMD subdivision(s) rainfall departure. Most weather-vulnerable states: **Chhattisgarh** (0.79 correlation) and **Karnataka** (0.68) — least buffered by irrigation. States like Haryana and Punjab show weak/negative correlation, consistent with heavy irrigation infrastructure decoupling yield from rainfall.

**2. Infrastructure risk** (`infrastructure_risk.csv`) — district-level, combining flood-hazard frequency (from this project's own `risk_flag` analysis) with housing fragility (% dilapidated housing, Census 2011) and population exposure. Most at-risk: West Bengal districts (Murshidabad, South/North 24 Parganas), reflecting high flood frequency combined with fragile housing stock.

**3. Human vulnerability index** (`human_vulnerability_index.csv`) — district-level, combining flood/drought hazard frequency with population (exposure) and % of workforce in agriculture (vulnerability - livelihoods directly exposed to rainfall variability). Most vulnerable: Nashik, Paschim Medinipur, East Godavari — large populations with high agricultural dependency in hazard-prone subdivisions.

All three use a state/district-to-IMD-subdivision crosswalk (built for this project, since state and IMD subdivision boundaries don't align) to connect Census 2011 and crop production data to the rainfall risk analysis above. This crosswalk is an approximation for states split across multiple subdivisions (e.g. Uttar Pradesh, Maharashtra) - a real limitation, noted rather than hidden.

## Why this differs from the original brief

The original project brief suggested LSTM/Prophet for time-series, generic ENSO indices as input, and didn't specify target granularity or evaluation method. This implementation deliberately diverges:

| Brief said | This project does | Why |
|---|---|---|
| LSTM / Prophet | Linear regression + shallow Random Forest | Only ~70-150 years of data; deep learning overfits badly at this sample size |
| ENSO as generic input | Only pre-season (lagged) values used | Concurrent-season values would leak future information into the forecast |
| No evaluation method specified | Leave-one-year-out CV vs. climatology baseline | Random splits leak future info in time-series data; climatology is the real bar to beat |
| No national/regional distinction | Both built and evaluated separately | National signal is weak/diluted; regional signal is real but heterogeneous |
| IOD not mentioned | Added as a third predictor | IOD can offset ENSO's effect on the monsoon (e.g. 1997) |

Full detail in [`PROJECT_SUMMARY.md`](PROJECT_SUMMARY.md).

## Repository structure

```
DATA/
  raw/          Original downloaded datasets (ENSO, IOD, SOI, IMD rainfall, crop production, subdivision boundaries)
  processed/    Cleaned, merged, model-ready tables and results
notebooks/
  eda.ipynb     Full pipeline: parsing -> feature engineering -> modeling -> evaluation -> forecasting
models/         Trained final models (.pkl) - only for approaches that beat baseline
dashboard/
  app.py        Interactive Streamlit predictor
```

## Data sources

| File | Source | Role |
|---|---|---|
| `nino34_oni.txt` | NOAA CPC | ENSO predictor |
| `dmi_iod.txt` | NOAA PSL | IOD predictor |
| `soi.txt` | NOAA (standardized table) | SOI predictor |
| `meiv2.data` | NOAA PSL | MEI (tested, didn't improve results) |
| `iitm_aismr.txt` | IITM Pune | National rainfall target (% departure from climatology) |
| `Sub_Division_IMD_2017.csv` | IMD / public dataset | Regional rainfall target (36 subdivisions) |
| `rainfall_area-wt_India_1901-2015.csv` | IMD/OGD | National rainfall cross-check |
| `india_crop_production_1997_2020.csv` | Public crop statistics | Agriculture-impact analysis |
| `imd_subdivision_boundaries.json` | IMD Mausam portal | Subdivision boundaries for the dashboard map |
| `india_census_2011_districts.csv` | Census of India 2011 | Population, agricultural workforce, housing condition - vulnerability assessments |

## Running it

```bash
# Notebook (full pipeline, reproducible end-to-end)
jupyter nbconvert --to notebook --execute --inplace notebooks/eda.ipynb

# Dashboard
cd dashboard
streamlit run app.py
```

## Data quality issues found and fixed

- `soi.txt` silently contains two different tables (raw pressure anomaly + standardized SOI) concatenated in one file — fixed to read only the standardized table.
- `read_fwf` in this environment inferred columns as string dtype, silently breaking sentinel-value filtering and `.mean()` calls — fixed with explicit numeric coercion.
- A stray sentinel-declaration line in `dmi_iod.txt` was passing as a fake "year -9999" row — fixed with a real year-range filter.
- Naively averaging the 36 subdivisions to get a "national" number would be statistically wrong (unweighted, biases toward small subdivisions) — solved by using IITM's official pre-weighted national series instead.

## Honest limitations

- No proven national forecast — the DJF-ONI model shows weak, not strong, evidence.
- Only 10 of 36 subdivisions have real predictive skill; using this for the other 26 or for the national number would not be evidence-based.
- 2020-2025 rainfall data gap not yet closed for the regional target (`Sub_Division_IMD_2017.csv` stops at 2017).
- Multi-year-ahead forecasts (2027-2030) are not physically possible with this approach — ENSO/IOD/SOI are only predictable 6-12 months ahead. The dashboard offers scenario exploration instead of false point forecasts.
