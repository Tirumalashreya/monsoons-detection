# Indian Monsoon Prediction

AI/ML system forecasting Indian monsoon rainfall using ENSO, IOD, and SOI climate indices, extended into downstream climate-impact analysis (agriculture, infrastructure, human vulnerability). Built iteratively from an initial project brief, with every modeling decision validated by honest evaluation against a climatology baseline — not just built to run, but checked for whether it actually works.

Live dashboard code: `dashboard/app.py`. Full pipeline: `notebooks/monsoon_prediction_pipeline.ipynb`.

---

## Quick start

### 1. Clone

```bash
git clone https://github.com/Tirumalashreya/monsoons-detection.git
cd monsoons-detection
```

### 2. Set up the environment

```bash
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Run the dashboard (fastest way to see it working)

```bash
cd dashboard
streamlit run app.py
```

Opens at `http://localhost:8501` (or whatever port Streamlit picks). No setup beyond the above — all models and data are pre-built and committed to the repo, so the dashboard runs immediately without needing to re-run the notebook first.

### 4. (Optional) Re-run the full pipeline from raw data

Only needed if you want to reproduce or modify the analysis itself:

```bash
jupyter nbconvert --to notebook --execute --inplace notebooks/monsoon_prediction_pipeline.ipynb
```

This re-parses all raw data, re-trains every model, regenerates every CSV in `data/processed/` and every `.pkl` in `models/`, in place. Takes under a minute.

---

## What's been built

**Core forecasting:**
- A national-level monsoon rainfall model (ENSO/IOD/SOI → seasonal rainfall departure)
- A regional model for each of India's 36 IMD meteorological subdivisions
- A monthly-breakdown model (June/July/August/September separately, not just one seasonal number)
- All three evaluated honestly with leave-one-year-out cross-validation against a climatology baseline — not just fit and reported

**Downstream impact analysis:**
- Disaster-preparedness risk flagging (drought/flood early warning per subdivision)
- Agriculture vulnerability assessment (state-level, rainfall vs. crop yield)
- Infrastructure risk assessment (district-level, flood hazard × housing fragility × population)
- Human vulnerability index (district-level, IPCC hazard × exposure × vulnerability framework)

**Interactive dashboard** (Streamlit) with four views:
1. **Regional Predictor** — clickable map of all 36 subdivisions, live sliders for ENSO/IOD/SOI, real-time forecast
2. **National Predictor** — same idea at national level, explicitly flagged as weaker evidence
3. **Monthly Breakdown** — June–September forecasts individually, for subdivision-months with proven skill
4. **Vulnerability Assessments** — interactive state/district exploration of the three downstream analyses

---

## How to use the frontend (dashboard)

After running `streamlit run app.py`, the app opens in your browser with a sidebar on the left (**View** selector) and content on the right. There are four views.

### 1. Regional Predictor (default view)

1. In the sidebar, make sure **View** is set to "Regional Predictor".
2. Look at the map at the top of the page. Every dot is one of India's 36 IMD meteorological subdivisions.
   - **Red** = Deficient (drought risk)
   - **Blue** = Excess (flood risk)
   - **Green** = Normal
   - **Grey** = No reliable model — this subdivision never showed real predictive skill, so it's intentionally left unpredicted rather than guessed
3. Click any **colored** dot to select that subdivision (clicking a grey dot shows a warning instead, since there's no model for it). You can also just use the **Subdivision** dropdown in the sidebar directly.
4. In the sidebar, pick a **Forecast year** (2026–2030).
   - 2026 is a real forecast — the sliders below pre-fill with 2026's actual observed pre-season ENSO/IOD/SOI readings.
   - 2027–2030 have no real data yet (those years haven't happened) — sliders start at neutral (0,0,0) so you can manually explore "what if" scenarios (e.g. drag ENSO positive to simulate an El Niño year).
5. Drag the three sliders (**Spring Niño 3.4**, **Spring IOD**, **Spring SOI**) — the "Predicted rainfall departure" and risk label on the right update live as you move them.
6. Expand **"[Subdivision] — historical accuracy and trend"** to see that subdivision's actual cross-validated performance (baseline vs. model RMSE) and its rainfall history.
7. Expand **"All 36 subdivisions — full skill table"** to see every subdivision's evaluation result, including the 26 without a reliable model.

### 2. National Predictor

1. Set **View** to "National Predictor".
2. Same interaction as Regional, but only one slider (Dec–Jan–Feb Niño 3.4), since that's the only predictor that showed any real improvement nationally.
3. Read the caption above the slider — it explicitly says this model only weakly beats the baseline, so treat the number as a lead worth watching, not a confident forecast the way the regional predictions are.
4. Expand **"National historical trend"** and **"Agriculture link"** for supporting context.

### 3. Monthly Breakdown

1. Set **View** to "Monthly Breakdown".
2. Pick a subdivision from the dropdown — only subdivisions with at least one skilled month are listed.
3. You'll see a metric card for each month (June/July/August/September) that has a real model for that subdivision, with its predicted % departure and risk label. Months without a reliable model are simply not shown (and a caption tells you which ones are missing) — nothing is guessed.
4. Expand **"All 27 subdivision-month combinations with real skill"** to see the full list across all subdivisions.

### 4. Vulnerability Assessments

1. Set **View** to "Vulnerability Assessments".
2. In the sidebar, pick which of the three assessments to view: **Agriculture**, **Infrastructure**, or **Human Vulnerability**.
3. **Agriculture:** pick a state from the dropdown — see its rainfall-yield correlation, its rank among all states, and a bar chart comparing every state.
4. **Infrastructure / Human Vulnerability:** pick a state and a number of districts to show (slider) — see that state's top districts ranked by risk/vulnerability score, plus which district is highest nationally.

---

## Methodology

**Target:** seasonal (June–September, "JJAS") rainfall, expressed as % departure from each region's own long-term climatological mean — not raw mm, so results are comparable across regions of very different rainfall totals.

**Predictors:** Niño 3.4 (ENSO), Dipole Mode Index (IOD), Southern Oscillation Index (SOI) — all used at **pre-season lag only** (spring average, or Dec–Jan–Feb for the national model). Concurrent-season values are never used, because that would leak information a real forecast wouldn't have — you can't know June–September ENSO before June starts.

**Evaluation:** leave-one-year-out cross-validation, always compared against a climatology baseline (predicting the historical average). A model that doesn't beat this baseline is reported as such — "didn't beat baseline" is a real, useful finding in this project, not a failure to hide.

**Models:** linear regression and shallow Random Forest only. Given ~65–70 years of usable data (and often less for finer breakdowns), anything more complex (LSTM, deep learning) would overfit — this was tested and confirmed, not just assumed.

**Final models:** trained on 100% of available data (no held-out year) — but only for the subdivisions/months that already demonstrated real skill during cross-validation. No final model exists for combinations that didn't prove themselves.

---

## Key results

**National:** the original 3-predictor spring model (ENSO+IOD+SOI, March–May) never beat the climatology baseline. A later lag window (ENSO alone, Dec–Jan–Feb) shows weak but real improvement (RMSE 9.94 vs. baseline 9.99) — this is what's in the dashboard now, clearly labeled as marginal evidence, not a settled result. Four other improvement attempts (a new index called MEI, Random Forest/Ridge regression, rolling up regional models, adding more predictors) were tried and did not help — reported honestly rather than omitted.

**Regional:** genuinely heterogeneous skill. Of 36 subdivisions, **10 show real predictive skill**: Arunachal Pradesh, Uttarakhand, Marathwada, Punjab, West Uttar Pradesh, East Uttar Pradesh, Telangana, East Rajasthan, Bihar, Konkan & Goa. 5 subdivisions get measurably *worse* than baseline (Saurashtra & Kutch, Gujarat Region, Rayalseema, West Rajasthan, North Interior Karnataka) — using this model there would be actively misleading.

**Monthly breakdown:** of 144 subdivision-month combinations tested (36 subdivisions × 4 months), **27 show real skill** — the closest honest alternative to date-specific prediction, which isn't physically possible (ENSO/IOD/SOI carry no day-level signal, and no daily rainfall data exists in this pipeline).

**Agriculture:** national Kharif rice yield correlates 0.33 with rainfall departure raw — but yield has a strong 1997–2020 upward trend from irrigation/technology (correlation 0.84 with year alone) that masks the weather signal. After removing that trend, the correlation with rainfall rises to **0.57**. State-level breakdown: **Chhattisgarh (0.79) and Karnataka (0.68)** are the most weather-exposed states (least irrigation-buffered).

**Infrastructure & human vulnerability:** built on the IPCC risk framework (Risk = Hazard × Exposure × Vulnerability), combining this project's own hazard flags with Census 2011 population and housing data. Most infrastructure-at-risk: West Bengal districts (Murshidabad, 24 Parganas). Most human-vulnerable: Nashik, Paschim Medinipur, East Godavari.

**2026 forecast:** genuinely live, not backtested — 2026's pre-season data already exists (partway through the year), so the 10 skilled regional models and 27 skilled monthly models produce real predictions, not simulated ones.

---

## Repository structure

```
data/
  raw/                    Original downloaded datasets — never modified in place
  processed/               Cleaned, merged, model-ready tables and analysis outputs
notebooks/
  monsoon_prediction_pipeline.ipynb   Full pipeline: preprocessing -> feature engineering ->
                                       correlation analysis -> model training/evaluation ->
                                       final models -> forecasting -> downstream impact
models/
  *_final_model.pkl        Regional models — one per subdivision with proven skill
  monthly/                 Monthly models — one per subdivision-month with proven skill
  National_DJF_ONI_model.pkl
dashboard/
  app.py                   Interactive Streamlit application
PROJECT_SUMMARY.md          Detailed build log: what changed vs. the original brief, and why
requirements.txt
```

---

## Data sources

| File | Source | Role | Coverage |
|---|---|---|---|
| `nino34_oni.txt` | NOAA CPC | ENSO predictor (ANOM column) | 1950–2026 |
| `dmi_iod.txt` | NOAA PSL | IOD predictor | 1870–2026 |
| `soi.txt` | NOAA (standardized table) | SOI predictor | 1951–2026 |
| `meiv2.data` | NOAA PSL | MEI — tested as a predictor, didn't improve results | 1979–2026 |
| `iitm_aismr.txt` | IITM Pune | National rainfall target (% departure from climatology) | 1871–2019 |
| `Sub_Division_IMD_2017.csv` | IMD / public dataset | Regional + monthly rainfall target (36 subdivisions) | 1901–2017 |
| `rainfall_area-wt_India_1901-2015.csv` | IMD/OGD | National rainfall cross-check | 1901–2015 |
| `india_crop_production_1997_2020.csv` | Public crop statistics | Agriculture vulnerability analysis | 1997–2020 |
| `imd_subdivision_boundaries.json` | IMD Mausam portal | Subdivision boundaries (map reference) | — |
| `india_census_2011_districts.csv` | Census of India 2011 | Population, agricultural workforce, housing condition | 2011 |

All predictor and target files are real, publicly sourced data — verified by direct inspection (not assumed authentic), with several data-quality bugs found and fixed along the way (below).

---

## Data quality issues found and fixed

- `soi.txt` silently contains **two different tables** (raw pressure anomaly + standardized SOI) concatenated in one file — was accidentally reading both as one, causing duplicate years with conflicting values. Fixed to read only the standardized table.
- `read_fwf` in this environment inferred columns as string dtype, not numeric — silently broke both missing-value filtering and any `.mean()` calculation. Fixed with explicit numeric coercion everywhere.
- A stray sentinel-declaration line in `dmi_iod.txt` was passing as a fake "year -9999" row — fixed with a real year-range filter.
- `Sub_Division_IMD_2017.csv` has isolated missing years for 3 subdivisions — dropped, not interpolated (interpolating rainfall risks inventing signal that isn't there).
- Naively averaging the 36 subdivisions to get a "national" number would be statistically wrong (unweighted — biases toward small subdivisions) — solved by using IITM's officially pre-weighted national series instead.
- A state-name mismatch ("ODISHA" vs. the census file's "ORISSA", "PUDUCHERRY" vs. "PONDICHERRY") was silently dropping 34 districts from the vulnerability outputs — found and fixed.

---

## Why this differs from the original brief

The original brief suggested LSTM/Prophet for time-series, generic ENSO indices as input, and didn't specify target granularity or evaluation method. This implementation deliberately diverges:

| Brief said | This project does | Why |
|---|---|---|
| LSTM / Prophet | Linear regression + shallow Random Forest | Only ~65–70 years of usable data; deep learning overfits badly at this sample size (tested, confirmed) |
| ENSO as generic input | Only pre-season (lagged) values used | Concurrent-season values would leak future information into the forecast |
| No evaluation method specified | Leave-one-year-out CV vs. climatology baseline | Random splits leak future info in time-series data; climatology is the real bar to beat |
| No national/regional distinction | Both built and evaluated separately | National signal is weak/diluted; regional signal is real but heterogeneous |
| IOD not mentioned | Added as a third predictor | IOD can offset ENSO's effect on the monsoon (e.g. 1997) |
| "Predict 2026–2030" as one output | 2026 = real forecast; 2027–2030 = interactive scenario exploration | ENSO/IOD/SOI are only predictable ~6–12 months ahead — nobody can know 2028's ENSO state today |

Full build log and reasoning: [`PROJECT_SUMMARY.md`](PROJECT_SUMMARY.md).

---

## Honest limitations

- **No proven national forecast** — the best national model shows weak, not strong, evidence (marginal RMSE improvement over baseline).
- **Only 10 of 36 subdivisions** and **27 of 144 subdivision-months** have real predictive skill — using this for the rest would not be evidence-based, and the dashboard deliberately withholds predictions there rather than guessing.
- **2020–2025 rainfall data gap** not yet closed for the regional target (source file stops at 2017).
- **Multi-year-ahead forecasts (2027–2030) are not physically possible** with this approach, or any ENSO-based approach — the dashboard offers scenario exploration instead of manufacturing false point forecasts.
- **Specific-date prediction is not possible** — this is seasonal/monthly climate forecasting, not short-range weather forecasting; those are different fields with different data and methods.
- The state/district-to-IMD-subdivision crosswalk used in the vulnerability assessments is an approximation for states split across multiple subdivisions (e.g. Uttar Pradesh, Maharashtra) — noted rather than hidden.
