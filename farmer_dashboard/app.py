"""
Step 1 — The One-Click Setup.
Draw your field → enter three things you already know → click Generate.

Uses a Plotly scattermapbox instead of folium to avoid external dependency issues.
"""
import streamlit as st
import time
import numpy as np
import plotly.graph_objects as go

st.set_page_config(page_title="HybridScout – Field Setup", layout="wide")

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent))
from ui_utils import inject_custom_css
from mock_logic import logger, get_genotype_names
inject_custom_css()

# ── Header ────────────────────────────────────────────────────────
st.title("Field Digital Twin Setup")
st.caption("Draw your field, enter three things you already know, and let our satellite AI do the rest.")

col_map, col_inputs = st.columns([2, 1])

# ── Left: Interactive Map ─────────────────────────────────────────
with col_map:
    st.subheader("1. Select Your Field")
    st.info("Click anywhere on the map to place a field marker, then hit **Generate** on the right.")

    # Plotly Mapbox — zero extra dependencies
    fig = go.Figure(go.Scattermapbox(
        lat=[41.2586],
        lon=[-95.9378],
        mode="markers",
        marker=dict(size=14, color="#10b981"),
        text=["Your Field"],
    ))
    fig.update_layout(
        mapbox=dict(
            style="carto-positron",
            center=dict(lat=41.2586, lon=-95.9378),
            zoom=12,
        ),
        margin=dict(l=0, r=0, t=0, b=0),
        height=420,
    )
    st.plotly_chart(fig, use_container_width=True)

    # Always show the preview to keep the demo flowing
    if st.session_state.get("setup_done"):
        st.success("Field boundary captured — 30 cm satellite imagery loaded.")
        rng = np.random.default_rng(7)
        patch = rng.uniform(0.15, 0.70, (12, 12, 3))
        patch[:, :, 1] += 0.25
        patch = np.clip(patch, 0, 1)
        import plotly.express as px
        fig_patch = px.imshow(patch)
        fig_patch.update_layout(
            xaxis_visible=False, yaxis_visible=False,
            margin=dict(l=0, r=0, t=0, b=0), height=140,
        )
        st.plotly_chart(fig_patch, use_container_width=True)
        st.caption("Preview: false-color composite (NIR / Red Edge / Green)")

# ── Right: The 3 Inputs ──────────────────────────────────────────
with col_inputs:
    st.subheader("2. Upload Your Data")
    sat_file = st.file_uploader(
        "Upload Satellite / Drone ZIP",
        type=["zip"],
        help="Upload a ZIP of 6-band TIF files from your drone or satellite provider.",
    )
    csv_file = st.file_uploader(
        "Upload Planting CSV",
        type=["csv"],
        help="Your planting map CSV with genotype, range, row, and nitrogen columns.",
    )
    if sat_file:
        st.success(f"`{sat_file.name}` uploaded ({sat_file.size / 1e6:.1f} MB)")
    if csv_file:
        st.success(f"`{csv_file.name}` uploaded ({csv_file.size / 1e3:.1f} KB)")

    st.markdown("---")
    st.subheader("3. Your Field Details")
    st.caption("We only need three things you already know by heart.")

    genotype = st.selectbox(
        "Seed Hybrid Planted",
        get_genotype_names(),
        help="The commercial hybrid you purchased this season.",
    )
    planting_date = st.date_input(
        "Planting Date",
        help="When did the planter go in the ground?",
    )
    nitrogen = st.slider(
        "Nitrogen Applied (lbs / acre)",
        min_value=0, max_value=250, value=150, step=10,
        help="Total actual N applied (starter + side-dress).",
    )

    st.markdown("---")

    if st.button("Generate Digital Twin & Forecast Yield", type="primary"):
        logger.info(f"Twin generation requested — G={genotype}, E={planting_date}, M={nitrogen}")
        with st.spinner("Calculating Growing Degree Days & analyzing 6 spectral bands…"):
            time.sleep(2)

        st.session_state["setup_done"] = True
        st.session_state["genotype"]   = genotype
        st.session_state["plant_date"] = planting_date
        st.session_state["n_rate"]     = nitrogen

        st.success("Digital Twin generated. Navigate to **Harvest Forecast** to view results.")
        st.rerun()
