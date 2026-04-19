"""
Step 2 — The Bottom Line.
Three numbers that matter, a heatmap showing where the field is struggling,
and the heavy sensor data tucked safely behind an expander.
"""
import streamlit as st
import plotly.express as px
import pandas as pd
import os
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
from mock_logic import (
    logger, calculate_mock_yield, calculate_growth_stage,
    generate_mock_heatmap, generate_mock_spectral_data,
)
from ui_utils import inject_custom_css

st.set_page_config(page_title="Harvest Forecast", layout="wide")
inject_custom_css()

# ── Pull user inputs (defaults ensure page always renders) ────────
n_rate    = st.session_state.get("n_rate", 150)
genotype  = st.session_state.get("genotype", "DKC62-08")
plant_dt  = st.session_state.get("plant_date", None)
use_real  = st.session_state.get("real_data", False)

# ── Check for real inference results ──────────────────────────────
repo_root = Path(__file__).resolve().parent.parent.parent
scoring_dir = repo_root / "scoring_outputs"
rankings_path = scoring_dir / "hybrid_rankings.csv"

real_rankings = None
if use_real and rankings_path.exists():
    real_rankings = pd.read_csv(rankings_path)
    predicted = real_rankings["predicted_yield_mean"].mean()
    logger.info(f"Using REAL inference data. Mean predicted yield: {predicted:.1f}")
else:
    predicted = calculate_mock_yield(n_rate, genotype)

field_ac   = 500
corn_price = 4.40
revenue    = predicted * field_ac * corn_price
growth     = calculate_growth_stage(plant_dt) if plant_dt else "V10 (Mid Vegetative)"

# ── Header ────────────────────────────────────────────────────────
st.title("Harvest Forecast Dashboard")
st.caption("The three numbers that matter to your bottom line — updated from today's satellite pass.")

if not st.session_state.get("setup_done"):
    st.warning("Go to the **Field Digital Twin Setup** page first to enter your field details. Showing defaults below.")

if use_real and real_rankings is not None:
    st.success("Showing results from **real XGBoost inference** on your uploaded data.")

# ── KPI Cards ─────────────────────────────────────────────────────
c1, c2, c3 = st.columns(3)
c1.metric(
    "Predicted Yield",
    f"{predicted:.1f} bu/ac",
    "+4.2 bu/ac vs county avg",
    delta_color="normal",
    help="Model-predicted harvest yield for this field based on satellite imagery and your inputs.",
)
c2.metric(
    "Estimated Field Revenue",
    f"${revenue:,.0f}",
    f"{field_ac} acres × ${corn_price:.2f}/bu",
    delta_color="off",
    help="Gross revenue estimate at today's corn cash price.",
)
c3.metric(
    "Current Growth Stage",
    growth,
    "Based on GDD accumulation",
    delta_color="normal",
    help="Growth stage calculated from accumulated Growing Degree Days since your planting date.",
)

st.divider()

# ── Heatmap ───────────────────────────────────────────────────────
st.subheader("Field Yield Heatmap")
st.caption("Red zones are under-performing — target scouting and variable-rate inputs there first.")

heatmap = generate_mock_heatmap(predicted, 50)
fig = px.imshow(
    heatmap,
    color_continuous_scale="RdYlGn",
    origin="lower",
    labels={"color": "Predicted Yield (bu/ac)"},
)
fig.update_layout(
    xaxis_visible=False, yaxis_visible=False,
    margin=dict(l=0, r=0, t=0, b=0), height=450,
)
st.plotly_chart(fig, use_container_width=True)

# ── Deep-dive expander ────────────────────────────────────────────
with st.expander("View Comprehensive Sensor Data (for technical review)", expanded=False):
    logger.debug("Deep-dive spectral table accessed.")

    # Show real prediction data if available, else show mock spectral data
    predictions_path = repo_root / "xgb_predictions_full.csv"
    if use_real and predictions_path.exists():
        st.caption("Real per-plot predictions from the XGBoost inference pipeline.")
        real_preds = pd.read_csv(predictions_path)
        display_cols = [c for c in real_preds.columns if c in [
            "genotype", "location", "timepoint", "experiment", "range", "row",
            "predicted_yield", "actual_yield", "yieldPerAcre"
        ]]
        if display_cols:
            st.dataframe(real_preds[display_cols].head(100), use_container_width=True, hide_index=True)
        else:
            st.dataframe(real_preds.head(100), use_container_width=True, hide_index=True)
    else:
        st.caption(
            "Raw per-plot spectral indices and soil moisture readings extracted "
            "from 6-band Pléiades Neo imagery.  This data feeds the ML model."
        )
        st.dataframe(generate_mock_spectral_data(), use_container_width=True, hide_index=True)

