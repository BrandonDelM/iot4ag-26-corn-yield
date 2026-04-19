"""
Step 3 — The Checkbook Simulator.
Drag a slider, swap a seed, and instantly see whether the change
makes or loses money — before you spend a dime.
"""
import streamlit as st
import pandas as pd
import plotly.express as px
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
from mock_logic import logger, calculate_mock_yield, get_genotype_names
from ui_utils import inject_custom_css

st.set_page_config(page_title="What-If Simulator", layout="wide")
inject_custom_css()

# ── Pull baseline from session state ──────────────────────────────
base_n     = st.session_state.get("n_rate", 150)
base_geno  = st.session_state.get("genotype", "DKC62-08")
baseline   = calculate_mock_yield(base_n, base_geno)
field_ac   = 500
corn_price = 4.40
n_cost_lb  = 0.70

# ── Header ────────────────────────────────────────────────────────
st.title("What-If Scenario Simulator")
st.caption("Ask 'what if I change my seed or nitrogen rate?' and see the financial impact in real-time.")

if not st.session_state.get("setup_done"):
    st.warning("Go to **Field Digital Twin Setup** first to set your baseline. Using defaults below.")

col_ctrl, col_result = st.columns([1, 2])

# ── Left: Controls ────────────────────────────────────────────────
with col_ctrl:
    st.subheader("Change a Variable")
    st.caption("Adjust one thing at a time to isolate the financial impact.")

    geno_list = get_genotype_names()
    default_idx = geno_list.index(base_geno) if base_geno in geno_list else 0

    new_geno = st.selectbox(
        "Swap Seed Hybrid",
        geno_list,
        index=default_idx,
        help="What if you had planted a different hybrid?",
    )
    new_n = st.slider(
        "Adjust Nitrogen (lbs/ac)",
        0, 250, value=min(base_n + 50, 250), step=10,
        help="Slide up to add a side-dress, or down to see savings from cutting back.",
    )

    st.info(
        "**Key insight:** Higher nitrogen does NOT always mean higher profit. "
        "Our model captures diminishing returns unique to each hybrid's genetics."
    )

# ── Compute scenario ─────────────────────────────────────────────
new_yield = calculate_mock_yield(new_n, new_geno)
gap       = new_yield - baseline
extra_n   = new_n - base_n
n_cost_delta = extra_n * n_cost_lb

revenue_change = gap * corn_price * field_ac
cost_change    = n_cost_delta * field_ac
net_profit     = revenue_change - cost_change

logger.info(f"Scenario: {base_geno}→{new_geno}, N {base_n}→{new_n}, net ${net_profit:+,.0f}")

# ── Right: Results ────────────────────────────────────────────────
with col_result:
    st.subheader("Financial Impact")
    st.caption("All numbers reflect your 500-acre field at $4.40/bu corn price.")

    m1, m2, m3 = st.columns(3)
    m1.metric(
        "New Predicted Yield",
        f"{new_yield:.1f} bu/ac",
        f"{gap:+.1f} bu/ac vs baseline",
        delta_color="normal" if gap >= 0 else "inverse",
    )
    m2.metric(
        "Gap Bushels (whole field)",
        f"{gap * field_ac:+,.0f} bu",
        "Bushels gained or lost",
        delta_color="normal" if gap >= 0 else "inverse",
    )
    m3.metric(
        "Net Profit / Loss",
        f"${net_profit:+,.0f}",
        "After nitrogen cost",
        delta_color="normal" if net_profit >= 0 else "inverse",
    )

    st.divider()

    # ── Side-by-side bar chart ────────────────────────────────────
    df = pd.DataFrame({
        "Scenario": ["Your Current Plan", "Simulated Change"],
        "Gross Revenue ($)": [
            baseline * field_ac * corn_price,
            new_yield * field_ac * corn_price,
        ],
    })

    bar_color = "#10b981" if net_profit >= 0 else "#ef4444"
    fig = px.bar(
        df, x="Scenario", y="Gross Revenue ($)",
        text_auto="$.4s",
        color="Scenario",
        color_discrete_map={
            "Your Current Plan": "#64748b",
            "Simulated Change": bar_color,
        },
    )
    fig.update_layout(height=360, margin=dict(t=20, b=0), showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

    # ── Breakdown expander ────────────────────────────────────────
    with st.expander("View Detailed Cost Breakdown", expanded=False):
        breakdown = pd.DataFrame({
            "Item": ["Extra Yield Revenue", "Nitrogen Cost Change", "Net Profit / Loss"],
            "Per Acre ($/ac)": [
                f"${gap * corn_price:+,.2f}",
                f"${-n_cost_delta:+,.2f}",
                f"${gap * corn_price - n_cost_delta:+,.2f}",
            ],
            "Whole Field ($/500 ac)": [
                f"${revenue_change:+,.0f}",
                f"${-cost_change:+,.0f}",
                f"${net_profit:+,.0f}",
            ],
        })
        st.dataframe(breakdown, use_container_width=True, hide_index=True)
