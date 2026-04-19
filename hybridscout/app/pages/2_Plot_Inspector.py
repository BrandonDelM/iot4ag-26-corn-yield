"""
Plot Inspector (hero page #1).

For a chosen plot: render the 6-band TIF as RGB, show the OOF prediction with
actual, and explain the top drivers via SHAP on the final (all-rows) model.
"""
from __future__ import annotations

import glob

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import rasterio
import shap
import streamlit as st

from hybridscout.app.app_utils import (
    load_features, load_oof, load_model_bundle, load_explainer,
    require_artifacts,
)
from hybridscout.ml import io_paths as P


st.set_page_config(page_title="Plot Inspector · HybridScout", layout="wide")
st.title("Plot Inspector")
st.caption(
    "Pick a plot → see the RGB composite of the 6-band TIF, the honest OOF "
    "prediction vs ground truth, and the top SHAP drivers from the final model."
)

if not require_artifacts(P.FEATURES_PARQUET, P.OOF_CSV, P.MODEL_PKL, P.EXPLAINER_PKL):
    st.stop()

feats = load_features()
oof = load_oof()
bundle = load_model_bundle()
explainer = load_explainer()

# ─────────────────────────────────────────────
# Plot selector
# ─────────────────────────────────────────────

oof = oof.copy()
oof["plot_id"] = (oof["location"].astype(str) + " · exp " + oof["experiment"].astype(str)
                  + " · R" + oof["range"].astype(str) + " · row " + oof["row"].astype(str)
                  + " · " + oof["genotype"].astype(str))

st.sidebar.header("Pick a plot")
location = st.sidebar.selectbox("Location", sorted(oof["location"].unique()))
oof_loc = oof[oof["location"] == location]
genotype = st.sidebar.selectbox(
    "Genotype (optional filter)",
    options=["(all)"] + sorted(oof_loc["genotype"].unique()),
)
if genotype != "(all)":
    oof_loc = oof_loc[oof_loc["genotype"] == genotype]

plot_id = st.sidebar.selectbox(
    "Plot",
    sorted(oof_loc["plot_id"].unique())[:400],
    index=0,
)

sel = oof_loc[oof_loc["plot_id"] == plot_id]
tp_options = sorted(sel["timepoint"].unique())
tp = st.sidebar.selectbox("Timepoint", tp_options, index=len(tp_options) - 1)

row = sel[sel["timepoint"] == tp].iloc[0]

# ─────────────────────────────────────────────
# Find the matching TIF on disk
# ─────────────────────────────────────────────

def find_tif(location, timepoint, experiment, range_, row_) -> str | None:
    pattern = f"{location}-{timepoint}-{experiment}_{range_}_{row_}.*"
    hits = glob.glob(str(P.SATELLITE_ROOT / "**" / pattern), recursive=True)
    return hits[0] if hits else None


tif_path = find_tif(row["location"], row["timepoint"], row["experiment"],
                    row["range"], row["row"])

# ─────────────────────────────────────────────
# Top row: RGB preview + prediction card
# ─────────────────────────────────────────────

left, right = st.columns([1.2, 1])

with left:
    st.subheader(f"{plot_id}  ·  {tp}")
    if tif_path:
        with rasterio.open(tif_path) as src:
            bands = src.read().astype(np.float32)
        r, g, b = bands[0], bands[1], bands[2]
        mask = r > 0

        def norm(arr, mask):
            vals = arr[mask]
            if len(vals) == 0:
                return arr
            lo, hi = np.percentile(vals, 2), np.percentile(vals, 98)
            return np.clip((arr - lo) / max(hi - lo, 1e-6), 0, 1)

        rgb = np.dstack([norm(r, mask), norm(g, mask), norm(b, mask)])
        rgb[~mask] = 0
        st.image(rgb, caption=tif_path.split("/")[-1], width="stretch")
    else:
        st.info("TIF not found for this plot/timepoint.")

with right:
    st.subheader("Prediction vs actual")
    actual = float(row["actual_yield"])
    pred = float(row["predicted_yield"])
    err = pred - actual
    m1, m2, m3 = st.columns(3)
    m1.metric("Actual yield", f"{actual:.1f} bu/ac")
    m2.metric("Predicted (OOF)", f"{pred:.1f} bu/ac", f"{err:+.1f}")
    m3.metric("|error|", f"{abs(err):.1f} bu/ac")

    st.caption(
        f"OOF = hybrid's genotype was held out of this fold. "
        f"Nitrogen rate: **{row['nitrogenTreatment']}**."
    )

    # All-TP trajectory
    st.markdown("**Yield trajectory across timepoints**")
    traj = sel.sort_values("timepoint")
    traj_fig = px.line(
        traj, x="timepoint", y=["actual_yield", "predicted_yield"],
        markers=True,
        labels={"value": "bu/ac", "variable": ""},
    )
    traj_fig.update_layout(height=280, legend=dict(orientation="h", y=-0.2))
    st.plotly_chart(traj_fig, width="stretch")

# ─────────────────────────────────────────────
# SHAP waterfall
# ─────────────────────────────────────────────

st.divider()
st.subheader("Top drivers (SHAP on final model)")
st.caption(
    "Bars are bu/ac push away from the average predicted yield. Positive (→) = "
    "model raised its estimate; negative (←) = model lowered it. "
    "Peer-reviewed XAI methodology (Ahmad et al. 2025 · Shammi et al. 2025)."
)

# Reconstruct the final-model feature vector for this plot-TP row from feats
feat_row = feats[
    (feats["location"] == row["location"])
    & (feats["experiment"].astype(str) == str(row["experiment"]))
    & (feats["range"].astype(str) == str(row["range"]))
    & (feats["row"].astype(str) == str(row["row"]))
    & (feats["timepoint"] == row["timepoint"])
]

if feat_row.empty or bundle is None:
    st.info("Can't build SHAP vector for this row.")
else:
    feat_row = feat_row.iloc[0]
    raw_cols = bundle["raw_feature_cols"]
    X_raw = (feat_row.reindex(raw_cols)
             .apply(pd.to_numeric, errors="coerce")
             .values.reshape(1, -1).astype(np.float32))
    X_imp = bundle["imputer"].transform(X_raw)
    X_sel = bundle["selector"].transform(X_imp)
    geno_mean = bundle["genotype_means"].get(row["genotype"], bundle["global_mean"])
    X_final = np.column_stack([X_sel, [[geno_mean]]]).astype(np.float32)
    sv = explainer(X_final)

    col_fig, col_text = st.columns([1.5, 1])
    
    with col_fig:
        fig = plt.figure(figsize=(8, 6))
        shap.plots.waterfall(sv[0], show=False, max_display=8)
        st.pyplot(fig, use_container_width=True)

    with col_text:
        st.subheader("Explainable AI Insights")
        st.info(
            "This confirms our model is not a 'black box.' Below are the specific features "
            "that contributed most strongly to this unique plot's predicted yield."
        )
        vals = sv.values[0]
        names = bundle["feature_cols"]
        top_idx = np.argsort(-np.abs(vals))[:3]
        
        for i in top_idx:
            direction = "normal" if vals[i] > 0 else "inverse"
            st.metric(
                label=f"Driver: {names[i]}",
                value=f"{vals[i]:+.2f} bu/ac",
                delta="Elevated Yield" if vals[i] > 0 else "Reduced Yield",
                delta_color=direction
            )
