import json
import os

import joblib
import pandas as pd
import plotly.express as px
import streamlit as st

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROCESSED_DIR = os.path.join(BASE_DIR, "..", "DATA", "processed")
MODELS_DIR = os.path.join(BASE_DIR, "..", "models")
RAW_DIR = os.path.join(BASE_DIR, "..", "DATA", "raw")

GEO_NAME_MAP = {
    "Andaman & Nicobar Islands": "A & N ISLAND",
    "Arunachal Pradesh": "ARUNACHAL PRADESH",
    "Assam & Meghalaya": "ASSAM & MEGHALAYA",
    "Bihar": "BIHAR",
    "Chhattisgarh": "CHHATTISGARH",
    "Coastal Andhra Pradesh": "COASTAL ANDHRA PRADESH",
    "Coastal Karnataka": "COASTAL KARNATAKA",
    "East Madhya Pradesh": "EAST MADHYA PRADESH",
    "East Rajasthan": "EAST RAJASTHAN",
    "East Uttar Pradesh": "EAST UTTAR PRADESH",
    "Gangetic West Bengal": "GANGETIC WEST BENGAL",
    "Gujarat Region": "GUJARAT REGION",
    "Haryana Delhi & Chandigarh": "HAR. CHD & DELHI",
    "Himachal Pradesh": "HIMACHAL PRADESH",
    "Jammu & Kashmir": "JAMMU & KASHMIR",
    "Jharkhand": "JHARKHAND",
    "Kerala": "KERALA",
    "Konkan & Goa": "KONKAN & GOA",
    "Lakshadweep": "LAKSHADWEEP",
    "Madhya Maharashtra": "MADHYA MAHARASHTRA",
    "Matathwada": "MARATHWADA",
    "Naga Mani Mizo Tripura": "N M M T",
    "North Interior Karnataka": "N. I. KARNATAKA",
    "Orissa": "ORISSA",
    "Punjab": "PUNJAB",
    "Rayalseema": "RAYALASEEMA",
    "South Interior Karnataka": "S. I. KARNATAKA",
    "Saurashtra & Kutch": "SAURASHTRA & KUTCH",
    "Sub Himalayan West Bengal & Sikkim": "SHWB & SIKKIM",
    "Tamil Nadu": "TAMILNADU & PONDICHERY",
    "Telangana": "TELANGANA",
    "Uttarakhand": "UTTARAKHAND",
    "Vidarbha": "VIDARBHA",
    "West Madhya Pradesh": "WEST MADHYA PRADESH",
    "West Rajasthan": "WEST RAJASTHAN",
    "West Uttar Pradesh": "WEST UTTAR PRADESH",
}

st.set_page_config(page_title="Indian Monsoon Prediction", layout="wide")


@st.cache_data
def load_data():
    national_df = pd.read_csv(os.path.join(PROCESSED_DIR, "national_df.csv"))
    national_djf_df = pd.read_csv(os.path.join(PROCESSED_DIR, "national_djf_df.csv"))
    regional_df = pd.read_csv(os.path.join(PROCESSED_DIR, "regional_df.csv"))
    subdivision_results = pd.read_csv(os.path.join(PROCESSED_DIR, "subdivision_model_results.csv"))
    forecast_2026 = pd.read_csv(os.path.join(PROCESSED_DIR, "forecast_2026.csv"))
    return national_df, national_djf_df, regional_df, subdivision_results, forecast_2026


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


@st.cache_resource
def load_geojson():
    with open(os.path.join(RAW_DIR, "imd_subdivision_boundaries.json")) as f:
        return json.load(f)


def build_map_df(subdivision_results, forecast_2026, skilled):
    forecast_lookup = forecast_2026.set_index("SUBDIVISION")
    rows = []
    for name, geo_name in GEO_NAME_MAP.items():
        if name in skilled:
            row = forecast_lookup.loc[name]
            status = row["risk_flag"]
            pred = row["predicted_2026_pct_departure"]
        else:
            status = "No reliable model"
            pred = None
        rows.append({"SUBDIVISION": name, "geo_name": geo_name, "status": status, "predicted_2026_pct_departure": pred})
    return pd.DataFrame(rows)


def make_map_figure(map_df, geojson):
    color_map = {
        "Deficient (drought risk)": "#d62728",
        "Excess (flood risk)": "#1f77b4",
        "Normal": "#2ca02c",
        "No reliable model": "#c7c7c7",
    }
    fig = px.choropleth(
        map_df,
        geojson=geojson,
        locations="geo_name",
        featureidkey="properties.subdivisio",
        color="status",
        color_discrete_map=color_map,
        hover_name="SUBDIVISION",
        hover_data={"geo_name": False, "predicted_2026_pct_departure": True},
    )
    fig.update_geos(fitbounds="locations", visible=False)
    fig.update_layout(margin={"r": 0, "t": 0, "l": 0, "b": 0}, height=520, legend_title_text="2026 status")
    return fig


national_df, national_djf_df, regional_df, subdivision_results, forecast_2026 = load_data()
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

mode = st.sidebar.radio("Predictor", ["Regional", "National"])
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

if mode == "Regional":
    st.header(f"All 36 subdivisions — {forecast_year}")
    st.caption(
        "Click any subdivision on the map to select it below. Grey = no reliable model "
        "(26 of 36) — colored ones (10 of 36) have a real, evaluated forecast."
    )

    geojson = load_geojson()
    map_df = build_map_df(subdivision_results, forecast_2026, skilled)
    map_fig = make_map_figure(map_df, geojson)
    map_event = st.plotly_chart(map_fig, on_select="rerun", key="subdiv_map", selection_mode="points")

    if map_event and map_event.selection and map_event.selection.points:
        clicked_geo_name = map_event.selection.points[0].get("location")
        clicked_name = next((n for n, g in GEO_NAME_MAP.items() if g == clicked_geo_name), None)
        if clicked_name and clicked_name in skilled:
            st.session_state["sub_choice_widget"] = clicked_name
        elif clicked_name:
            st.warning(f"{clicked_name} has no reliable model — pick one of the 10 colored subdivisions instead.")

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

else:
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
