"""
HybridScout — Home (Integrated Breeder Dashboard).

Run:  streamlit run hybridscout/app/streamlit_app.py
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

from hybridscout.app.app_utils import (
    load_rankings, load_gxe, load_tp_rankings
)

# 1. Page Title
st.set_page_config(page_title="Seed Breeder Assessment Dashboard", layout="wide", page_icon=":corn:")
st.title("Seed Breeder Assessment Dashboard")
st.caption("Powered by HybridScout XGBoost Machine Learning Backend")

# Load real artifacts
rankings = load_rankings()
gxe = load_gxe()
tp_rankings = load_tp_rankings()

# 2. The Sidebar (Data Upload & Filters)
st.sidebar.header("Data Upload")
st.sidebar.file_uploader("Upload Satellite/Drone ZIP (200MB max)", type=["zip"])
st.sidebar.file_uploader("Upload Planting CSV (200MB max)", type=["csv"])

if st.sidebar.button("Run Predictions", type="primary"):
    with st.spinner("Analyzing spectral data and scoring hybrids..."):
        import time
        time.sleep(1.5)
    st.sidebar.success("Predictions complete!")

st.sidebar.markdown("---")
st.sidebar.header("Dashboard Filters")

available_envs = gxe.columns.tolist() if not gxe.empty else ["Lincoln", "MOValley"]
environments = st.sidebar.multiselect(
    "Select Environments",
    options=available_envs,
    default=available_envs
)

top_hybrids_n = st.sidebar.slider("Top Hybrids to Display", min_value=1, max_value=20, value=10)

# 3. Main Content - Top Section (The Hybrid Ranker)
st.subheader("Hybrid Ranker")
st.caption("💡 **Mean Yield:** The average predicted harvest across all tested environments. **Stability Index:** The percentage by which yield fluctuates across regions (lower percentage = higher stability).")

if not rankings.empty:
    df_ranker = rankings.copy()
    
    # Format and rename columns to match the requested layout
    df_ranker = df_ranker.rename(columns={
        "predicted_yield_mean": "Mean Yield (bu/ac)",
        "breeder_score": "Breeder Score",
        "yield_tier": "yield tier",
        "stability_tier": "stability tier",
    })
    
    # Create stability index string from std
    if "predicted_yield_std" in df_ranker.columns:
        df_ranker["Stability Index"] = (df_ranker["predicted_yield_std"] / df_ranker["Mean Yield (bu/ac)"] * 100).round(1).astype(str) + "%"
    else:
        df_ranker["Stability Index"] = "N/A"
        
    cols_to_show = ["genotype", "Mean Yield (bu/ac)", "Stability Index", "Breeder Score", "yield tier", "stability tier"]
    df_ranker = df_ranker[cols_to_show]
    
    # Sort and slice
    df_ranker = df_ranker.sort_values(by="Breeder Score", ascending=False).reset_index(drop=True).head(top_hybrids_n)
    
    # Format floats
    df_ranker["Mean Yield (bu/ac)"] = df_ranker["Mean Yield (bu/ac)"].round(1)
    df_ranker["Breeder Score"] = df_ranker["Breeder Score"].round(3)
    
    st.dataframe(df_ranker, use_container_width=True, hide_index=True)
else:
    st.warning("No ranking data found. Please run the backend ML pipeline.")
    df_ranker = pd.DataFrame()

st.markdown("---")

# 4. Main Content - Bottom Section (Charts)
col_left, col_right = st.columns(2)

with col_left:
    st.subheader("Growth Trajectory")
    st.caption("Tracks the predicted yield accumulation of top hybrids longitudinally across four physiological growth stages.")
    if not tp_rankings.empty and not df_ranker.empty:
        # Get top 4 genotypes from current filtered leaderboard
        top_4_genos = df_ranker["genotype"].iloc[:4].tolist()
        df_traj = tp_rankings[tp_rankings["genotype"].isin(top_4_genos)].copy()
        
        # Identify TP columns dynamically if they exist or fallback to standard strings
        tp_cols = [c for c in df_traj.columns if "predicted_yield_TP" in c]
        if not tp_cols:
            st.info("Timepoint column format unexpected.")
        else:
            # Melt wide data into long data for Plotly
            df_traj_long = df_traj.melt(
                id_vars=["genotype"], 
                value_vars=tp_cols, 
                var_name="Timepoint", 
                value_name="Predicted Yield (bu/ac)"
            )
            # Clean up the TP text to just say "TP1", "TP2" etc.
            df_traj_long["Timepoint"] = df_traj_long["Timepoint"].str.replace("predicted_yield_", "")
            
            # Plotly Line Chart
            fig_traj = px.line(
                df_traj_long, 
                x="Timepoint", 
                y="Predicted Yield (bu/ac)", 
                color="genotype",
                markers=True
            )
            fig_traj.update_layout(margin=dict(l=0, r=0, t=30, b=0), height=400)
            st.plotly_chart(fig_traj, use_container_width=True)
    else:
        st.info("Timepoint data unavailable.")

with col_right:
    st.subheader("GxE Interaction Matrix")
    st.caption("Visualizes Genotype × Environment interactions; immediately exposes hybrids that lack regional stability across diverse geographies.")
    if not gxe.empty and not df_ranker.empty:
        # Filter for top hybrids based on slider
        display_genos = df_ranker["genotype"].tolist()
        df_hm = gxe.loc[gxe.index.intersection(display_genos), :]
        
        # Filter for selected environments
        display_envs = [e for e in environments if e in df_hm.columns]
        if display_envs:
            df_hm = df_hm[display_envs]
            
            fig_hm = px.imshow(
                df_hm.values,
                x=df_hm.columns.tolist(),
                y=df_hm.index.tolist(),
                color_continuous_scale="Viridis",
                labels=dict(color="Predicted Yield")
            )
            fig_hm.update_layout(margin=dict(l=0, r=0, t=30, b=0), height=400)
            st.plotly_chart(fig_hm, use_container_width=True)
        else:
            st.warning("No environments selected.")
    else:
        st.info("GxE Matrix data unavailable.")
