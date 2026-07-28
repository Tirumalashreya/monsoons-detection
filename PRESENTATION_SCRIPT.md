# Presentation Script

Two parts: explaining the pipeline/code (notebook), and explaining the dashboard (frontend). Written as talking points you can read from or paraphrase — not just bullet labels.

---

# Part 1 — Explaining the code (notebook pipeline)

### Opening (say this first)

"This project forecasts the Indian monsoon using three ocean-atmosphere climate indices — ENSO, IOD, and SOI — evaluated separately at the national level, the regional level (36 IMD subdivisions), and a monthly level. Every result here was tested against a baseline before being trusted — I'll show you where it worked and where it honestly didn't."

### Setup section

"Standard imports — pandas and numpy for data handling, matplotlib/seaborn for plotting. Nothing modeling-specific here yet."

### Data Preprocessing section

"This section loads five raw files, each from a different real source:
- **Niño 3.4 (ENSO)** from NOAA CPC — sea-surface temperature anomaly in the Pacific.
- **DMI (IOD)** from NOAA PSL — the Indian Ocean's own temperature dipole, which can offset ENSO's effect on the monsoon.
- **SOI** from NOAA — an atmospheric pressure-based version of the same ENSO signal.
- **IMD subdivision rainfall** — the actual rainfall records, 1901 to 2017, our prediction target.
- **IITM's national rainfall series** — same idea, but nationally aggregated, and already properly area-weighted.

Worth mentioning: none of these files were clean. The SOI file, for example, secretly contains *two* different tables concatenated together — I had to find that by inspecting the raw text, not by trusting the column headers."

### Feature Engineering section

"Here's the most important design decision in the whole project: I only use **pre-season** values of these indices — the spring average, before the monsoon season starts — not the same-season values. If I used June–September ENSO to predict June–September rainfall, that's not a forecast, that's cheating — you can't know that data before the season happens. Everything downstream respects this rule."

### Exploratory Correlation Analysis section

"Before building any model, I checked whether these predictors actually correlate with national rainfall at all. They're weak — around 0.11 to 0.17. That's a warning sign I paid attention to, not something I ignored to keep going."

### Model Training — National section

"I tried linear regression and Random Forest, both evaluated with leave-one-year-out cross-validation against a climatology baseline — meaning, 'how much better is this than just predicting the historical average every year?' Neither model beat that baseline. I'm reporting that honestly instead of hiding it — it's a real finding about the limits of national-level ENSO prediction, not a bug."

### Model Training — Regional section

"Same exact test, but run separately for each of the 36 subdivisions, since national aggregation can hide regional signal. Result: **10 of 36 subdivisions show real, evaluated skill** — Arunachal Pradesh, Uttarakhand, Marathwada, Punjab, and six others. 5 subdivisions get *worse* than baseline. This heterogeneity is the actual finding — the monsoon-ENSO relationship isn't uniform across India, and pretending it is would be dishonest."

### Disaster Risk Flagging section

"I turned the raw rainfall departure numbers into a simple three-category flag — Deficient, Excess, Normal — using ±10% bands. This is what makes the forecast actionable for something like early warning, rather than just a number."

### Regional Zone Summary section

"I grouped the 36 subdivisions into North/South/East/West and looked at aggregate patterns — East India is driest on average with the highest drought-year frequency, South is wettest. This is descriptive, not a new model."

### Final Models and 2026 Forecast section

"For only the 10 subdivisions that proved real skill, I trained one final model each, using *all* available years — no held-out year this time, since the evaluation phase is done. Then I fed in 2026's actual, already-observed pre-season data — since we're partway through 2026, that data genuinely exists — to produce a real 2026 forecast, not a hypothetical one."

### Monthly Breakdown Forecast section

"Someone asked whether we could predict for a specific date. That's not possible — ENSO-type indices carry no day-level signal. But I tested the next-best thing: predicting June, July, August, and September *separately* instead of one seasonal number. 27 of 144 subdivision-month combinations showed real skill — a genuine, if partial, improvement in granularity."

### Downstream Impact sections (Agriculture / Infrastructure / Human Vulnerability)

"These extend the forecast into actual impact. For agriculture, I detrended crop yield to separate the technology-driven improvement from the weather-driven signal — once separated, rainfall's real effect on yield nearly doubles, from a 0.33 to a 0.57 correlation. For infrastructure and human vulnerability, I used the IPCC's standard risk framework — hazard times exposure times vulnerability — combining our own rainfall hazard data with real Census 2011 population and housing data."

### Closing line

"Every number in this project either beat a baseline or is reported as not beating one — nothing here is dressed up to look better than it is."

---

# Part 2 — Explaining the Streamlit frontend

### Opening (say this first, before touching the app)

"This isn't a static report — it's a live predictor. You can move the actual climate index values and watch the forecast respond in real time. Let me walk through it."

### Regional Predictor (open with this view)

"This map shows all 36 of India's meteorological subdivisions. [Point at colors] Red means deficient rainfall risk, blue means excess/flood risk, green is normal — and grey means *no reliable model*. That grey is deliberate: those 26 subdivisions never proved real predictive skill in testing, so rather than guess, the app just doesn't color them.

[Click a red or blue dot] I'll click on this one — see how it selects the subdivision below. [Point at sliders] These three sliders are the actual pre-season index values — Niño 3.4, IOD, SOI. They're pre-filled with 2026's *real* observed readings right now. [Drag a slider] Watch what happens to the forecast number and the risk label as I move this — that's a live model prediction, not a lookup table.

[Point at Forecast year dropdown] If I switch to 2027 or later, the sliders reset to zero — because that year's real data doesn't exist yet, since it hasn't happened. I'm not going to pretend otherwise; this is where you'd manually explore a scenario, like 'what if this turns into an El Niño year.'"

### National Predictor

"Same interaction, but nationally. Only one slider here — Dec-to-Feb ENSO — because that's the only approach out of everything I tried that even weakly beat the baseline nationally. I kept the honest caption right on the page: this is a lead worth watching, not something to act on with confidence."

### Monthly Breakdown

"This answers 'can you predict for a specific date' as honestly as the data allows. Not a specific day — but June, July, August, and September separately, for the subdivisions where that finer breakdown actually showed skill. [Select a subdivision] Notice this one might only show three months, not four — the missing month isn't hidden by accident, it's because that specific combination never proved real skill, so it's excluded on purpose."

### Vulnerability Assessments

"This is the downstream-impact side. [Switch sub-view] Agriculture shows which states' crop yields are most sensitive to rainfall — Chhattisgarh and Karnataka top the list. Infrastructure and Human Vulnerability let you pick a state and see its highest-risk districts, using real Census data combined with our own hazard analysis."

### Anticipated questions, and honest answers

**"Why isn't the whole map colored in?"**
"Because only 10 of 36 subdivisions actually demonstrated real predictive skill when tested. Coloring the rest would mean showing a forecast with no evidence behind it."

**"Can this predict a specific date, like July 31st?"**
"No — that's a fundamentally different problem, short-range weather forecasting, which needs daily data and different models entirely. This project is seasonal/monthly climate forecasting. The Monthly Breakdown view is the closest honest answer."

**"Why does the national model seem weaker than the regional ones?"**
"Because it is, and I'm not going to claim otherwise. Averaging across all of India dilutes a signal that's real but regional — some areas respond to ENSO, some don't, and nationally they partly cancel out."

**"How do I know 2026's forecast is real and not made up?"**
"Because 2026's pre-season months have already happened — we're partway through the year — so that input data genuinely exists, unlike 2027 onward."
