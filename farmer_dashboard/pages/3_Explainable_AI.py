"""
Step 4 — Earning Trust: The 'Why'.
A SHAP waterfall chart that speaks plain English so a farmer
trusts the number they just saw on the Forecast page.
"""
import streamlit as st
import plotly.graph_objects as go
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
from mock_logic import logger, calculate_mock_yield
from ui_utils import inject_custom_css

st.set_page_config(page_title="Why This Yield?", layout="wide")
inject_custom_css()

# ── Pull context ──────────────────────────────────────────────────
genotype  = st.session_state.get("genotype", "DKC62-08")
n_rate    = st.session_state.get("n_rate", 150)
predicted = calculate_mock_yield(n_rate, genotype)

# ── Header ────────────────────────────────────────────────────────
st.title("Why This Yield?")

if not st.session_state.get("setup_done"):
    st.warning("Go to **Field Digital Twin Setup** first to set your inputs. Using defaults below.")

col_chart, col_explain = st.columns([1.4, 1])

# ── Mock SHAP drivers ────────────────────────────────────────────
drivers = [
    ("County Avg Baseline",             155.0,  "absolute"),
    ("Strong NDRE Vegetation",           12.0,  "relative"),
    ("Optimal Soil Moisture",             8.5,  "relative"),
    ("N Below Hybrid Optimum",           -3.2,  "relative"),
    ("Late Planting Penalty",            -5.0,  "relative"),
]
running = sum(v for _, v, _ in drivers)
drivers.append(("Your Predicted Yield", round(running, 1), "total"))

labels   = [d[0] for d in drivers]
values   = [d[1] for d in drivers]
measures = [d[2] for d in drivers]

with col_chart:
    st.subheader("Yield Driver Breakdown")
    st.caption("Each bar shows how a single factor pushed your yield up (green) or down (red).")

    fig = go.Figure(go.Waterfall(
        orientation="v",
        measure=measures,
        x=labels,
        y=values,
        text=[
            f"+{v:.1f}" if m == "relative" and v > 0
            else f"{v:.1f}"
            for v, m in zip(values, measures)
        ],
        textposition="outside",
        connector={"line": {"color": "rgba(100,100,100,0.4)"}},
        increasing={"marker": {"color": "#10b981"}},
        decreasing={"marker": {"color": "#ef4444"}},
        totals={"marker":     {"color": "#3b82f6"}},
    ))
    fig.update_layout(
        yaxis_title="Predicted Yield (bu/ac)",
        showlegend=False,
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=20, r=20, t=20, b=20),
        height=420,
    )
    st.plotly_chart(fig, use_container_width=True)

# ── Plain-English explanation ─────────────────────────────────────
with col_explain:
    st.subheader("What This Means")

    with st.container(border=True):
        st.markdown(
            "**1. Your plants look great.** "
            "Satellite imagery shows a very high NDRE vegetation index, "
            "which added **+12.0 bu/ac**. Healthy leaf tissue reflects "
            "strongly in the Red Edge band."
        )
        st.markdown(
            "**2. Soil moisture is ideal.** "
            "The model detected favorable moisture levels in the root zone, "
            "contributing **+8.5 bu/ac** to the forecast."
        )
        st.markdown(
            "**3. You could use a bit more nitrogen.** "
            "Your application rate is slightly below what this specific "
            "hybrid responds best to, costing **−3.2 bu/ac**. "
            "Consider a late side-dress to close this gap."
        )
        st.markdown(
            "**4. Planting was a little late.** "
            "The crop missed early-season Growing Degree Days (heat units), "
            "resulting in a **−5.0 bu/ac** penalty that can't be recovered."
        )

    st.success(
        f"**Bottom line:** Your field is predicted at **{running:.1f} bu/ac** — "
        f"above the 155 bu/ac county baseline.  The only actionable lever left "
        f"is nitrogen: a targeted side-dress could recover up to 3 bu/ac."
    )

logger.info("Explainable AI page rendered successfully.")
