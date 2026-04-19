"""
Gap Bushels / ROI — the business closer.

'If you switched from your current hybrid to the top stability-tier hybrid,
how many bushels do you pick up on 500 acres, and what's that in dollars?'
"""
from __future__ import annotations

import plotly.express as px
import streamlit as st

from hybridscout.app.app_utils import load_rankings, require_artifacts
from hybridscout.ml import io_paths as P


st.set_page_config(page_title="Gap Bushels / ROI · HybridScout", layout="wide")
st.title("Gap Bushels / ROI")
st.caption(
    "Switching the field's chosen hybrid to a top stability-tier hybrid has a "
    "dollar value. This page makes that number concrete."
)

if not require_artifacts(P.HYBRID_RANKINGS_CSV):
    st.stop()

rankings = load_rankings()

# ── Inputs ──
st.sidebar.header("Scenario")
acreage = st.sidebar.number_input("Acreage", 10, 50_000, 500, step=50)
corn_price = st.sidebar.slider("Corn cash price ($/bu)", 3.50, 6.00, 4.40, 0.05,
                               help="Pull the current USDA NASS / AMS figure before pitching.")
current = st.sidebar.selectbox(
    "Current hybrid",
    rankings.sort_values("genotype")["genotype"].tolist(),
    index=0,
)
k = st.sidebar.slider("Compare against top K stability-tier hybrids", 1, 20, 5)

current_row = rankings[rankings["genotype"] == current].iloc[0]

# ── Target selection: top-K breeder score with stability_tier in {High} ──
pool = rankings[rankings["stability_tier"] == "High"]
if len(pool) < k:
    pool = rankings
top_k = pool.sort_values("breeder_score", ascending=False).head(k)
target_mean = top_k["predicted_yield_mean"].mean()
target_std = top_k["predicted_yield_std"].mean()

delta_bu_per_ac = target_mean - current_row["predicted_yield_mean"]
total_bu = delta_bu_per_ac * acreage
total_usd = total_bu * corn_price

# ── KPI cards ──
c1, c2, c3 = st.columns(3)
c1.metric("Δ yield (target − current)",
          f"{delta_bu_per_ac:+.1f} bu/ac",
          f"vs {current_row['predicted_yield_mean']:.1f} bu/ac current")
c2.metric(f"Total bushels on {acreage:,} ac",
          f"{total_bu:+,.0f} bu",
          f"±{target_std:.1f} bu/ac stability (high-tier)")
c3.metric("Dollar impact @ market price",
          f"${total_usd:+,.0f}",
          f"${corn_price:.2f}/bu")

if delta_bu_per_ac <= 0:
    st.info(
        "The current hybrid is already at or above the top stability-tier "
        "mean. Gain comes from swapping the *weakest* plots, not the strongest."
    )
else:
    st.success(
        f"Switching from **{current}** to the average of the top {k} "
        f"stability-tier hybrids picks up **{delta_bu_per_ac:.1f} bu/ac** — "
        f"**${total_usd:,.0f}** on a {acreage:,}-acre operation at "
        f"${corn_price:.2f}/bu corn."
    )

st.divider()

# ── Target list ──
st.subheader(f"Top {k} stability-tier candidates")
show_cols = ["genotype", "predicted_yield_mean", "predicted_yield_std",
             "breeder_score", "yield_tier", "stability_tier", "plot_count"]
st.dataframe(top_k[show_cols].reset_index(drop=True), width="stretch")

# ── Sensitivity ──
st.divider()
st.subheader("Sensitivity to assumptions")
price_range = [corn_price * f for f in [0.85, 0.925, 1.0, 1.075, 1.15]]
yield_range = [delta_bu_per_ac * f for f in [0.7, 0.85, 1.0, 1.15, 1.3]]

import pandas as pd
grid = pd.DataFrame([
    {"corn_price": p, "delta_yield": d,
     "total_usd": p * d * acreage}
    for p in price_range for d in yield_range
])
heat = grid.pivot(index="delta_yield", columns="corn_price", values="total_usd")
fig = px.imshow(
    heat.values,
    x=[f"${p:.2f}" for p in heat.columns],
    y=[f"{d:.1f} bu/ac" for d in heat.index],
    color_continuous_scale="RdYlGn",
    aspect="auto",
    labels=dict(x="Corn price", y="Δ yield", color="$"),
    height=400,
)
st.plotly_chart(fig, width="stretch")

st.caption(
    "Green = ROI holds even under ±15% price / ±30% yield haircut. Red = the "
    "hybrid switch is marginal under pessimistic assumptions. **Cite** USDA "
    "NASS Crop Production, USDA ERS Fertilizer Use and Price, and ISU "
    "Extension Ag Decision Maker A1-20 at pitch time — don't hardcode numbers."
)
