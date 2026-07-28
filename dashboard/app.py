import os

import joblib
import pandas as pd
import streamlit as st

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROCESSED_DIR = os.path.join(BASE_DIR, "..", "DATA", "processed")
MODELS_DIR = os.path.join(BASE_DIR, "..", "models")

st.set_page_config(page_title="Indian Monsoon Prediction", layout="wide")


@st.cache_data
def load_data():
    national_df = pd.read_csv(os.path.join(PROCESSED_DIR, "national_df.csv"))
    national_djf_df = pd.read_csv(os.path.join(PROCESSED_DIR, "national_djf_df.csv"))
    regional_df = pd.read_csv(os.path.join(PROCESSED_DIR, "regional_df.csv"))
    subdivision_results = pd.read_csv(os.path.join(PROCESSED_DIR, "subdivision_model_results.csv"))
    forecast_2026 = pd.read_csv(os.path.join(PROCESSED_DIR, "forecast_2026.csv"))
    agriculture_vulnerability = pd.read_csv(os.path.join(PROCESSED_DIR, "agriculture_vulnerability.csv"))
    infrastructure_risk = pd.read_csv(os.path.join(PROCESSED_DIR, "infrastructure_risk.csv"))
    human_vulnerability_index = pd.read_csv(os.path.join(PROCESSED_DIR, "human_vulnerability_index.csv"))
    return (national_df, national_djf_df, regional_df, subdivision_results, forecast_2026,
            agriculture_vulnerability, infrastructure_risk, human_vulnerability_index)


@st.cache_resource
def load_regional_model(subdivision_name):
    fname = subdivision_name.replace(" ", "_").replace("&", "and")
    return joblib.load(os.path.join(MODELS_DIR, f"{fname}_final_model.pkl"))


@st.cache_resource
def load_national_model():
    return joblib.load(os.path.join(MODELS_DIR, "National_DJF_ONI_model.pkl"))


def risk_band(pct_dep):
    if pct_dep <= -10:
        return "Deficient (drought risk)"
    elif pct_dep >= 10:
        return "Excess (flood risk)"
    return "Normal"


(national_df, national_djf_df, regional_df, subdivision_results, forecast_2026,
 agriculture_vulnerability, infrastructure_risk, human_vulnerability_index) = load_data()

skilled = subdivision_results[subdivision_results["improvement"] > 0]["SUBDIVISION"].tolist()
skilled_2026 = forecast_2026.set_index("SUBDIVISION")
climatology_mean = national_df["pct_departure_national"].mean()

NATIONAL_YEAR_DEFAULTS = {
    2026: -0.37,
    2027: 0.0,
    2028: 0.0,
    2029: 0.0,
    2030: 0.0,
}

st.title("Indian Monsoon Prediction")
st.caption("Live predictor — adjust ENSO/IOD/SOI, see the forecast update")

YEAR_DEFAULTS = {
    2026: (0.51, 0.24, -0.1),
    2027: (0.0, 0.0, 0.0),
    2028: (0.0, 0.0, 0.0),
    2029: (0.0, 0.0, 0.0),
    2030: (0.0, 0.0, 0.0),
}

mode = st.sidebar.radio("View", ["Regional Predictor", "National Predictor", "Vulnerability Assessments"])

if mode in ("Regional Predictor", "National Predictor"):
    forecast_year = st.sidebar.selectbox("Forecast year", [2026, 2027, 2028, 2029, 2030])
    if forecast_year > 2026:
        st.sidebar.caption(
            f"{forecast_year}'s real pre-season ENSO/IOD/SOI data doesn't exist yet "
            "(that year hasn't happened) — sliders start at neutral. Adjust them yourself "
            "to test a scenario, e.g. El Niño (positive ONI) or La Niña (negative ONI)."
        )
    else:
        st.sidebar.caption("2026's real observed pre-season values are pre-filled below.")
    default_oni, default_iod, default_soi = YEAR_DEFAULTS[forecast_year]

if mode == "Regional Predictor":
    sub_choice = st.sidebar.selectbox("Subdivision (only ones with a proven working model)", sorted(skilled))
    st.sidebar.caption(f"{len(skilled)} of 36 subdivisions have demonstrated real forecast skill — only those are selectable here.")

    model = load_regional_model(sub_choice)
    st.header(f"{sub_choice} — {forecast_year}")

    col_inputs, col_result = st.columns([1, 1])

    with col_inputs:
        st.subheader("Pre-season conditions")
        oni_val = st.slider("Spring Niño 3.4 (ENSO) anomaly", -2.5, 2.5, default_oni, 0.01, key=f"reg_oni_{forecast_year}")
        iod_val = st.slider("Spring IOD (Dipole Mode Index)", -1.5, 1.5, default_iod, 0.01, key=f"reg_iod_{forecast_year}")
        soi_val = st.slider("Spring SOI (standardized)", -3.0, 3.0, default_soi, 0.01, key=f"reg_soi_{forecast_year}")

    with col_result:
        st.subheader("Predicted monsoon outcome")
        pred = model.predict([[oni_val, iod_val, soi_val]])[0]
        risk = risk_band(pred)

        st.metric("Predicted rainfall departure", f"{pred:.1f}%")
        if "Deficient" in risk:
            st.error(risk)
        elif "Excess" in risk:
            st.info(risk)
        else:
            st.success(risk)

        is_default = abs(oni_val - default_oni) < 0.001 and abs(iod_val - default_iod) < 0.001 and abs(soi_val - default_soi) < 0.001
        if is_default and forecast_year == 2026:
            st.caption("This matches the real 2026 forecast (2026's actual pre-season values).")
        elif is_default:
            st.caption(f"{forecast_year}'s real pre-season data doesn't exist yet — this is a neutral placeholder, not a real forecast.")

    st.divider()

    with st.expander(f"{sub_choice} — historical accuracy and trend"):
        baseline_rmse = subdivision_results.loc[subdivision_results["SUBDIVISION"] == sub_choice, "baseline_rmse"].iloc[0]
        model_rmse = subdivision_results.loc[subdivision_results["SUBDIVISION"] == sub_choice, "model_rmse"].iloc[0]
        c1, c2 = st.columns(2)
        c1.metric("Climatology baseline RMSE", f"{baseline_rmse:.2f}")
        c2.metric("Model RMSE", f"{model_rmse:.2f}", delta=f"{model_rmse - baseline_rmse:.2f}", delta_color="inverse")
        sub_hist = regional_df[regional_df["SUBDIVISION"] == sub_choice].sort_values("YEAR")
        st.line_chart(sub_hist.set_index("YEAR")["jjas_pct_departure"])

    with st.expander("All 36 subdivisions — full skill table"):
        display = subdivision_results.copy()
        display["Skilled"] = display["SUBDIVISION"].isin(skilled)
        st.dataframe(display.sort_values("improvement", ascending=False), use_container_width=True)

    st.caption(
        "Recommendation: this predictor is only reliable for the 10 subdivisions listed above — "
        "using it for agriculture planning, water management, or drought/flood early warning is "
        "supported by evidence there. It should not be extended to the national number or the "
        "other 26 subdivisions, where no real skill was demonstrated."
    )

elif mode == "National Predictor":
    st.header(f"National — {forecast_year}")
    st.caption(
        "Uses Dec-Jan-Feb ONI (an earlier lag than the regional models) — the one "
        "approach, of five tried, that showed real improvement over baseline. Still "
        "weak/marginal evidence (RMSE 9.94 vs. baseline 9.99), not strong proven skill "
        "like the regional models — treat this as the best available, not a settled result."
    )

    national_model = load_national_model()
    nat_default = NATIONAL_YEAR_DEFAULTS[forecast_year]

    col_inputs, col_result = st.columns([1, 1])

    with col_inputs:
        st.subheader("Pre-season conditions")
        djf_oni_val = st.slider("Dec-Jan-Feb Niño 3.4 (ENSO) anomaly", -2.5, 2.5, nat_default, 0.01, key=f"nat_djf_{forecast_year}")
        if forecast_year > 2026:
            st.caption(f"{forecast_year} has no real data yet — slider defaults to neutral (0.0).")
        else:
            st.caption("2026's real observed DJF ONI value is pre-filled.")

    with col_result:
        st.subheader("Predicted vs. baseline")
        pred = national_model.predict([[djf_oni_val]])[0]
        st.metric("Model prediction", f"{pred:.1f}%")
        st.metric("Climatology baseline", f"{climatology_mean:.1f}%")
        st.caption(
            "Unlike the old model, this one does edge out the baseline in evaluation — "
            "but only by a small margin, so treat it as a lead worth watching, not a "
            "confident forecast the way the regional models are."
        )

    st.divider()
    with st.expander("National historical trend"):
        st.line_chart(national_df.set_index("YEAR")["pct_departure_national"])

    with st.expander("Agriculture link (national-level evidence, not a forecast)"):
        col1, col2 = st.columns(2)
        col1.metric("Raw correlation (Kharif rice yield vs. rainfall)", "0.33")
        col2.metric("After removing tech/irrigation trend", "0.57")
        st.caption(
            "This shows rainfall genuinely matters for yield nationally, once the "
            "1997-2020 technology-driven yield trend is removed - useful context, but "
            "not something this unreliable model should be used to act on."
        )

else:
    st.header("Weather-Impact Vulnerability Assessments")
    st.caption(
        "Three downstream analyses connecting the monsoon forecast to real impact — "
        "agriculture, infrastructure, and population. Uses a state/district-to-IMD-"
        "subdivision crosswalk (an approximation for states split across subdivisions)."
    )

    st.subheader("1. Agriculture vulnerability")
    st.caption(
        "State-level Kharif rice yield, detrended, correlated against matched-subdivision "
        "rainfall departure. Higher correlation = more weather-exposed, less irrigation-buffered."
    )
    agri_display = agriculture_vulnerability.rename(columns={
        "state": "State", "n": "Years", "rain_yield_corr": "Rainfall-Yield Correlation",
        "drought_yr_residual": "Avg Yield Impact (Drought Years)",
        "flood_yr_residual": "Avg Yield Impact (Flood Years)",
    })
    st.dataframe(agri_display, use_container_width=True)
    top_agri = agriculture_vulnerability.iloc[0]
    st.info(f"Most weather-vulnerable state: **{top_agri['state']}** (correlation {top_agri['rain_yield_corr']:.2f})")

    st.divider()

    st.subheader("2. Infrastructure risk")
    st.caption(
        "District-level: flood-hazard frequency (from this project's own risk flag) "
        "combined with % dilapidated housing (Census 2011) and population exposure. "
        "Top 20 shown, ranked by risk score."
    )
    infra_cols = ["District name", "State name", "Population", "dilapidated_pct", "rural_pct", "pct_excess_years", "infra_flood_risk"]
    infra_top = infrastructure_risk.nlargest(20, "infra_flood_risk")[infra_cols].rename(columns={
        "District name": "District", "State name": "State",
        "dilapidated_pct": "Dilapidated Housing %", "rural_pct": "Rural %",
        "pct_excess_years": "Flood-Risk Years %", "infra_flood_risk": "Infrastructure Risk Score",
    })
    st.dataframe(infra_top, use_container_width=True)
    top_infra = infrastructure_risk.nlargest(1, "infra_flood_risk").iloc[0]
    st.info(f"Highest infrastructure risk: **{top_infra['District name']}, {top_infra['State name']}**")

    st.divider()

    st.subheader("3. Human vulnerability index")
    st.caption(
        "District-level, IPCC framework (Risk = Hazard x Exposure x Vulnerability): "
        "flood/drought hazard frequency x population x % workforce in agriculture. "
        "Top 20 shown, ranked by vulnerability score."
    )
    human_cols = ["District name", "State name", "Population", "agri_worker_pct", "pct_deficient_years", "pct_excess_years", "human_vulnerability_index"]
    human_top = human_vulnerability_index.nlargest(20, "human_vulnerability_index")[human_cols].rename(columns={
        "District name": "District", "State name": "State",
        "agri_worker_pct": "Agricultural Workforce %", "pct_deficient_years": "Drought Years %",
        "pct_excess_years": "Flood Years %", "human_vulnerability_index": "Vulnerability Score",
    })
    st.dataframe(human_top, use_container_width=True)
    top_human = human_vulnerability_index.nlargest(1, "human_vulnerability_index").iloc[0]
    st.info(f"Most vulnerable district: **{top_human['District name']}, {top_human['State name']}**")

    st.divider()
    st.caption(
        "All three use a crosswalk mapping states/districts to IMD subdivisions, since "
        "their boundaries don't align — an approximation, most accurate for states that "
        "map to a single subdivision, noted rather than hidden."
    )
