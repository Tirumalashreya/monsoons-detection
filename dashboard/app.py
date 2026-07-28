import os

import joblib
import pandas as pd
import plotly.express as px
import streamlit as st

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROCESSED_DIR = os.path.join(BASE_DIR, "..", "DATA", "processed")
MODELS_DIR = os.path.join(BASE_DIR, "..", "models")

SUBDIVISION_COORDS = {
    "Andaman & Nicobar Islands": (11.7, 92.7), "Arunachal Pradesh": (28.2, 94.7),
    "Assam & Meghalaya": (26.2, 92.9), "Bihar": (25.6, 85.9), "Chhattisgarh": (21.3, 81.9),
    "Coastal Andhra Pradesh": (16.5, 80.6), "Coastal Karnataka": (13.5, 74.8),
    "East Madhya Pradesh": (23.5, 80.5), "East Rajasthan": (26.5, 75.5),
    "East Uttar Pradesh": (26.5, 82.5), "Gangetic West Bengal": (23.0, 87.8),
    "Gujarat Region": (22.5, 72.5), "Haryana Delhi & Chandigarh": (29.5, 76.5),
    "Himachal Pradesh": (31.9, 77.2), "Jammu & Kashmir": (33.5, 75.0),
    "Jharkhand": (23.6, 85.3), "Kerala": (10.5, 76.3), "Konkan & Goa": (16.5, 73.5),
    "Lakshadweep": (10.5, 72.6), "Madhya Maharashtra": (18.5, 74.5),
    "Matathwada": (18.9, 76.5), "Naga Mani Mizo Tripura": (24.8, 93.5),
    "North Interior Karnataka": (16.0, 76.0), "Orissa": (20.5, 84.5),
    "Punjab": (30.9, 75.5), "Rayalseema": (14.5, 78.5),
    "South Interior Karnataka": (13.0, 76.5), "Saurashtra & Kutch": (22.5, 70.0),
    "Sub Himalayan West Bengal & Sikkim": (27.0, 88.5), "Tamil Nadu": (11.0, 78.5),
    "Telangana": (18.0, 79.0), "Uttarakhand": (30.1, 79.2), "Vidarbha": (20.9, 78.5),
    "West Madhya Pradesh": (23.0, 76.5), "West Rajasthan": (26.5, 71.5),
    "West Uttar Pradesh": (27.5, 79.5),
}

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


def build_subdivision_map(skilled, forecast_2026):
    forecast_lookup = forecast_2026.set_index("SUBDIVISION")
    rows = []
    for name, (lat, lon) in SUBDIVISION_COORDS.items():
        if name in skilled:
            row = forecast_lookup.loc[name]
            status = row["risk_flag"]
            pred = row["predicted_2026_pct_departure"]
        else:
            status = "No reliable model"
            pred = None
        rows.append({"SUBDIVISION": name, "lat": lat, "lon": lon, "status": status, "predicted_2026_pct_departure": pred})
    map_df = pd.DataFrame(rows)

    color_map = {
        "Deficient (drought risk)": "#d62728",
        "Excess (flood risk)": "#1f77b4",
        "Normal": "#2ca02c",
        "No reliable model": "#c7c7c7",
    }
    fig = px.scatter_geo(
        map_df,
        lat="lat",
        lon="lon",
        color="status",
        color_discrete_map=color_map,
        hover_name="SUBDIVISION",
        hover_data={"lat": False, "lon": False, "predicted_2026_pct_departure": True},
        custom_data=["SUBDIVISION"],
        scope="asia",
    )
    fig.update_traces(marker=dict(size=16, line=dict(width=1, color="white")))
    fig.update_geos(
        lataxis_range=[6, 38], lonaxis_range=[66, 98],
        showcountries=True, showcoastlines=True, showland=True,
        landcolor="rgb(235,235,235)",
    )
    fig.update_layout(margin={"r": 0, "t": 0, "l": 0, "b": 0}, height=480, legend_title_text="2026 status")
    return map_df, fig


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
    st.header(f"All 36 subdivisions — {forecast_year}")
    st.caption(
        "Click a dot to select that subdivision below. Grey = no reliable model "
        "(26 of 36). Red/blue/green = has a real, evaluated forecast (10 of 36)."
    )
    map_df, map_fig = build_subdivision_map(skilled, forecast_2026)
    map_event = st.plotly_chart(map_fig, on_select="rerun", key="subdiv_scatter_map", selection_mode="points")

    if map_event and map_event.selection and map_event.selection.points:
        custom_data = map_event.selection.points[0].get("customdata")
        clicked_name = custom_data[0] if custom_data else None
        if clicked_name:
            if clicked_name in skilled:
                st.session_state["sub_choice_widget"] = clicked_name
            else:
                st.warning(f"{clicked_name} has no reliable model — pick one of the colored dots instead.")

    st.divider()

    sub_choice = st.sidebar.selectbox(
        "Subdivision (only ones with a proven working model)",
        sorted(skilled),
        key="sub_choice_widget",
    )
    st.sidebar.caption(f"{len(skilled)} of 36 subdivisions have demonstrated real forecast skill — only those are selectable here.")

    model = load_regional_model(sub_choice)
    st.subheader(f"{sub_choice} — {forecast_year}")

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
        "pick a state to explore, rather than scrolling raw tables."
    )

    assessment = st.sidebar.radio("Assessment", ["Agriculture", "Infrastructure", "Human Vulnerability"])

    if assessment == "Agriculture":
        st.subheader("Agriculture vulnerability")
        st.caption(
            "State-level Kharif rice yield, detrended, correlated against matched-subdivision "
            "rainfall departure. Higher correlation = more weather-exposed, less irrigation-buffered."
        )
        agri_state = st.selectbox("State", sorted(agriculture_vulnerability["state"].unique()))
        row = agriculture_vulnerability[agriculture_vulnerability["state"] == agri_state].iloc[0]

        c1, c2, c3 = st.columns(3)
        c1.metric("Rainfall-Yield Correlation", f"{row['rain_yield_corr']:.2f}")
        c2.metric("Yield Impact, Drought Years", f"{row['drought_yr_residual']:.2f}" if pd.notna(row["drought_yr_residual"]) else "n/a")
        c3.metric("Yield Impact, Flood Years", f"{row['flood_yr_residual']:.2f}" if pd.notna(row["flood_yr_residual"]) else "n/a")

        rank = int((agriculture_vulnerability["rain_yield_corr"] > row["rain_yield_corr"]).sum()) + 1
        st.caption(f"{agri_state} ranks #{rank} of {len(agriculture_vulnerability)} states by weather sensitivity.")

        chart_data = agriculture_vulnerability.set_index("state")["rain_yield_corr"].sort_values()
        st.bar_chart(chart_data)

    elif assessment == "Infrastructure":
        st.subheader("Infrastructure risk")
        st.caption(
            "District-level: flood-hazard frequency (this project's own risk flag) combined "
            "with % dilapidated housing (Census 2011) and population exposure."
        )
        infra_states = sorted(infrastructure_risk["State name"].unique())
        infra_state = st.selectbox("State", infra_states, key="infra_state")
        top_n = st.slider("Number of districts to show", 3, 20, 5)

        state_infra = infrastructure_risk[infrastructure_risk["State name"] == infra_state]
        state_top = state_infra.nlargest(top_n, "infra_flood_risk")[
            ["District name", "Population", "dilapidated_pct", "rural_pct", "pct_excess_years", "infra_flood_risk"]
        ].rename(columns={
            "District name": "District", "dilapidated_pct": "Dilapidated Housing %",
            "rural_pct": "Rural %", "pct_excess_years": "Flood-Risk Years %",
            "infra_flood_risk": "Infrastructure Risk Score",
        })
        st.dataframe(state_top, use_container_width=True)

        national_rank_district = infrastructure_risk.nlargest(1, "infra_flood_risk").iloc[0]
        st.caption(f"Highest infrastructure risk nationally: {national_rank_district['District name']}, {national_rank_district['State name']}")

    else:
        st.subheader("Human vulnerability index")
        st.caption(
            "District-level, IPCC framework (Risk = Hazard x Exposure x Vulnerability): "
            "flood/drought hazard frequency x population x % workforce in agriculture."
        )
        human_states = sorted(human_vulnerability_index["State name"].unique())
        human_state = st.selectbox("State", human_states, key="human_state")
        top_n_h = st.slider("Number of districts to show", 3, 20, 5, key="human_topn")

        state_human = human_vulnerability_index[human_vulnerability_index["State name"] == human_state]
        state_top_h = state_human.nlargest(top_n_h, "human_vulnerability_index")[
            ["District name", "Population", "agri_worker_pct", "pct_deficient_years", "pct_excess_years", "human_vulnerability_index"]
        ].rename(columns={
            "District name": "District", "agri_worker_pct": "Agricultural Workforce %",
            "pct_deficient_years": "Drought Years %", "pct_excess_years": "Flood Years %",
            "human_vulnerability_index": "Vulnerability Score",
        })
        st.dataframe(state_top_h, use_container_width=True)

        national_rank_human = human_vulnerability_index.nlargest(1, "human_vulnerability_index").iloc[0]
        st.caption(f"Most vulnerable district nationally: {national_rank_human['District name']}, {national_rank_human['State name']}")

    st.divider()
    st.caption(
        "All three use a crosswalk mapping states/districts to IMD subdivisions, since "
        "their boundaries don't align — an approximation, most accurate for states that "
        "map to a single subdivision, noted rather than hidden."
    )
