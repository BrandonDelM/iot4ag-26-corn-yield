"""Hybrid Leaderboard — yield × stability ranking."""
from __future__ import annotations

import plotly.express as px
import streamlit as st

from hybridscout.app.app_utils import load_rankings, require_artifacts
from hybridscout.ml import io_paths as P


st.set_page_config(page_title="Leaderboard · HybridScout", layout="wide")
st.title("Hybrid Leaderboard")
st.caption(
    "Breeder score = 0.70 · yield_norm + 0.30 · stability_norm. Stability tier "
    "= lower predicted-yield std across plots (more consistent across environments)."
)

if not require_artifacts(P.HYBRID_RANKINGS_CSV):
    st.stop()

rankings = load_rankings()

# ── Sidebar filters ──
st.sidebar.header("Filters")
top_n = st.sidebar.slider("Top N", 5, min(100, len(rankings)), 20)
yield_tiers = st.sidebar.multiselect(
    "Yield tier", options=sorted(rankings["yield_tier"].unique()),
    default=sorted(rankings["yield_tier"].unique()),
)
stab_tiers = st.sidebar.multiselect(
    "Stability tier", options=["High", "Medium", "Low"],
    default=["High", "Medium", "Low"],
)
sort_by = st.sidebar.selectbox(
    "Sort by", options=["breeder_score", "predicted_yield_mean",
                         "predicted_yield_std", "rank_error"],
    index=0,
)
ascending = sort_by in {"predicted_yield_std", "rank_error"}

view = rankings[
    rankings["yield_tier"].isin(yield_tiers)
    & rankings["stability_tier"].isin(stab_tiers)
].sort_values(sort_by, ascending=ascending).head(top_n)

# ── KPI row ──
c1, c2, c3 = st.columns(3)
c1.metric("Hybrids shown", len(view))
c2.metric("Mean predicted yield", f"{view['predicted_yield_mean'].mean():.1f} bu/ac")
c3.metric("Mean stability (std)", f"±{view['predicted_yield_std'].mean():.1f} bu/ac")

st.divider()

# ── Yield vs stability quadrant chart ──
st.subheader("Yield vs stability — quadrant view")
fig = px.scatter(
    rankings,
    x="predicted_yield_mean",
    y="predicted_yield_std",
    color="yield_tier",
    symbol="stability_tier",
    hover_data=["genotype", "plot_count", "breeder_score",
                "rank_predicted", "rank_actual"],
    labels={"predicted_yield_mean": "Mean predicted yield (bu/ac)",
            "predicted_yield_std": "Std of predicted yield  (↓ = more stable)"},
    color_discrete_map={"Top 25%": "#2ecc71", "Top 50%": "#f39c12",
                         "Bottom 50%": "#e74c3c"},
    height=500,
)
med_y = rankings["predicted_yield_mean"].median()
med_s = rankings["predicted_yield_std"].median()
fig.add_vline(x=med_y, line_dash="dash", line_color="gray", opacity=0.5)
fig.add_hline(y=med_s, line_dash="dash", line_color="gray", opacity=0.5)
fig.update_yaxes(autorange="reversed")  # up = more stable
st.plotly_chart(fig, width="stretch")

st.caption(
    "Top-left quadrant = **high yield, high stability** — the commercial "
    "candidate archetype. Track the filled markers; those are the hybrids "
    "to advance."
)

# ── Ranking bar chart ──
st.subheader(f"Top {len(view)} by {sort_by}")
bar = px.bar(
    view.sort_values("predicted_yield_mean"),
    x="predicted_yield_mean",
    y="genotype",
    orientation="h",
    error_x="predicted_yield_std",
    color="stability_tier",
    color_discrete_map={"High": "#2ecc71", "Medium": "#f39c12", "Low": "#e74c3c"},
    labels={"predicted_yield_mean": "Mean predicted yield (bu/ac)",
            "genotype": ""},
    height=max(400, 20 * len(view)),
)
st.plotly_chart(bar, width="stretch")

# ── Full table ──
st.subheader("Full ranking table")
display_cols = [
    "rank_breeder", "genotype", "predicted_yield_mean", "predicted_yield_std",
    "actual_yield_mean", "plot_count", "breeder_score",
    "yield_tier", "stability_tier", "rank_predicted", "rank_actual", "rank_error",
]
st.dataframe(
    view[display_cols].reset_index(drop=True),
    width="stretch",
    height=500,
)

st.download_button(
    "Download filtered CSV",
    data=view.to_csv(index=False).encode(),
    file_name="hybrid_rankings_filtered.csv",
    mime="text/csv",
)
