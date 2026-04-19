"""
Nitrogen What-If (hero page #2).

Lincoln plots only. Pick a hybrid; see its predicted yield at the three
real trial N rates (75 / 150 / 225 lb/ac), and mark the response tier.
"""
from __future__ import annotations

import pandas as pd  # noqa: F401
import plotly.express as px
import streamlit as st

from hybridscout.app.app_utils import (
    load_oof, load_nitrogen_response, require_artifacts,
)
from hybridscout.ml import io_paths as P


st.set_page_config(page_title="Nitrogen What-If · HybridScout", layout="wide")
st.title("Nitrogen What-If")
st.caption(
    "Lincoln only — the trial had three real N rates (Low / Medium / High ≈ "
    "75 / 150 / 225 lb actual N / ac). MO Valley ran a single 160 lb rate so "
    "the N response view doesn't apply there."
)

if not require_artifacts(P.OOF_CSV, P.NITROGEN_RESPONSE_CSV):
    st.stop()

oof = load_oof()
n_resp = load_nitrogen_response()

lincoln = oof[oof["location"] == "Lincoln"].copy()
if lincoln.empty:
    st.warning("No Lincoln OOF rows found.")
    st.stop()

# ── Genotype picker ──
st.sidebar.header("Pick hybrid(s)")
all_genos = sorted(lincoln["genotype"].unique())
default_top = (
    n_resp.dropna(subset=["n_response"])
          .sort_values("n_response", ascending=False)
          .head(3)["genotype"].tolist()
    if "n_response" in n_resp else all_genos[:3]
)
picked = st.sidebar.multiselect(
    "Genotypes (max 6)", all_genos,
    default=[g for g in default_top if g in all_genos][:3],
    max_selections=6,
)

if not picked:
    st.info("Pick at least one hybrid in the sidebar.")
    st.stop()

# ── Per-genotype N curves ──
n_order = ["Low", "Medium", "High"]
curves = (lincoln[lincoln["genotype"].isin(picked)]
          .groupby(["genotype", "nitrogenTreatment"])["predicted_yield"]
          .mean().reset_index())
fig = px.line(
    curves.sort_values(["genotype", "nitrogenTreatment"]),
    x="nitrogenTreatment",
    y="predicted_yield",
    color="genotype",
    markers=True,
    category_orders={"nitrogenTreatment": n_order},
    labels={"nitrogenTreatment": "Nitrogen rate (Low ≈ 75 · Medium ≈ 150 · High ≈ 225 lb/ac)",
            "predicted_yield": "Mean predicted yield (bu/ac)"},
    height=450,
)
st.plotly_chart(fig, width="stretch")

# ── N response table (High − Low bu/ac) ──
st.subheader("N response tier (High-rate − Low-rate, bu/ac)")
view = n_resp[n_resp["genotype"].isin(picked)].copy()
cols = [c for c in ["genotype", "Low", "Medium", "High",
                    "n_response", "n_response_tier"] if c in view.columns]
st.dataframe(view[cols].reset_index(drop=True), width="stretch")

st.caption(
    "High responder = yield climbs steeply with N → risky in low-input "
    "systems, great in intensive ones. Low responder = flat curve → "
    "robust under N stress. Peer-reviewed methodology: Ahmad et al. 2025 "
    "(XAI for fertilization guidance, Frontiers Plant Science).\n\n"
    "**GxExM Warning:** This simulator does not apply a flat mathematical boost. "
    "Our XGBoost model leverages the SHAP TreeExplainer to account for localized "
    "environmental variables (e.g., cumulative rainfall, soil type) before predicting yield. "
    "The modeled response is uniquely constrained by the specific hybrid's genetics (G) interacting "
    "with local trial environment (E) and nitrogen rate (M)."
)

# ── Economic sensitivity ──
st.divider()
st.subheader("Is the extra N worth it?")
c1, c2 = st.columns(2)
with c1:
    corn_price = st.slider("Corn cash price ($/bu)", 3.50, 6.00, 4.40, 0.05,
                           help="Pull current USDA NASS / AMS figure before pitching.")
    n_price = st.slider("Nitrogen cost ($/lb actual N)", 0.30, 1.20, 0.70, 0.05,
                        help="USDA ERS Fertilizer Use and Price; ISU A1-20.")
with c2:
    delta_n = st.slider("Extra N above Low-rate (lb/ac)", 0, 150, 150, step=25)

if "n_response" in view and view["n_response"].notna().any():
    st.markdown("**Break-even table (per acre, per hybrid)**")
    view2 = view[["genotype", "n_response"]].dropna().copy()
    view2["Δ revenue ($/ac)"] = view2["n_response"] * corn_price
    view2["Δ N cost ($/ac)"] = delta_n * n_price
    view2["Net Δ ($/ac)"] = view2["Δ revenue ($/ac)"] - view2["Δ N cost ($/ac)"]
    
    avg_roi = view2["Net Δ ($/ac)"].mean()
    color = "normal" if avg_roi >= 0 else "inverse"
    st.metric(
        label="Average Estimated Net ROI ($/ac)",
        value=f"${avg_roi:,.2f}",
        delta=f"{avg_roi:+.2f} per acre",
        delta_color=color,
        help="Average Return on Investment across the selected hybrids based on your N-rate slider."
    )
    
    view2 = view2.rename(columns={"n_response": "Δ yield (bu/ac)"})
    st.dataframe(view2.reset_index(drop=True), width="stretch")
    st.caption(
        "Δ yield = predicted yield at High N − predicted yield at Low N. "
        "Positive Net Δ → the extra N pays at the entered prices."
    )
