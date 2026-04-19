import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# 1. Page Title
st.set_page_config(page_title="Seed Breeder Assessment Dashboard", layout="wide")
st.title("Seed Breeder Assessment Dashboard")

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
environments = st.sidebar.multiselect(
    "Select Environments",
    options=["Lincoln", "MOValley", "Ames", "Crawfordsville", "Scottsbluff"],
    default=["Lincoln", "MOValley"]
)

top_hybrids_n = st.sidebar.slider("Top Hybrids to Display", min_value=1, max_value=20, value=10)

# 3. Main Content - Top Section (The Hybrid Ranker)
st.subheader("Hybrid Ranker")
st.caption("💡 **Mean Yield:** The average predicted harvest across all tested environments. **Stability Index:** The percentage by which yield fluctuates across regions (lower percentage = higher stability).")

np.random.seed(42)
fake_genotypes = [
    "LH185 X LH82", "K201 X O5426", "PHP02 X LH185", "DKC62-08", "P1197AM", 
    "NK1082", "DKC59-82", "P0953AM", "B73 X Mo17", "PH207 X PHG47"
]

# Clamp to available genotypes so arrays always match
n_display = min(top_hybrids_n, len(fake_genotypes))

rank_data = {
    "genotype": fake_genotypes[:n_display],
    "Mean Yield (bu/ac)": np.round(np.random.uniform(47.0, 49.6, n_display), 1),
    "Stability Index": [f"{np.random.randint(1, 5)}%" for _ in range(n_display)],
    "Breeder Score": np.round(np.random.uniform(25.000, 31.980, n_display), 3),
    "yield tier": [f"Tier {np.random.randint(1, 4)}" for _ in range(n_display)],
    "stability tier": ["Elite" if i < 3 else f"Tier {np.random.randint(1, 4)}" for i in range(n_display)]
}

df_ranker = pd.DataFrame(rank_data)
# Sort to make it look like a leaderboard
df_ranker = df_ranker.sort_values(by="Breeder Score", ascending=False).reset_index(drop=True)

st.dataframe(df_ranker, use_container_width=True, hide_index=True)

st.markdown("---")

# 4. Main Content - Bottom Section (Charts)
col_left, col_right = st.columns(2)

with col_left:
    st.subheader("Growth Trajectory")
    st.caption("Tracks the predicted yield accumulation of top hybrids longitudinally across four physiological growth stages.")
    
    # Mock line chart data (4 timepoints, top 4 genotypes)
    timepoints = ["TP1", "TP2", "TP3", "TP4"]
    top_4_genos = df_ranker["genotype"].iloc[:4].tolist()
    
    traj_data = []
    for g in top_4_genos:
        base = np.random.uniform(20.0, 30.0)
        for i, tp in enumerate(timepoints):
            base += np.random.uniform(2.0, 8.0) # Yield increases over time
            traj_data.append({"Timepoint": tp, "Genotype": g, "Predicted Yield": base})
            
    df_traj = pd.DataFrame(traj_data)
    
    fig_traj = px.line(
        df_traj, 
        x="Timepoint", 
        y="Predicted Yield", 
        color="Genotype",
        markers=True
    )
    fig_traj.update_layout(margin=dict(l=0, r=0, t=30, b=0), height=400)
    st.plotly_chart(fig_traj, use_container_width=True)

with col_right:
    st.subheader("GxE Interaction Matrix")
    st.caption("Visualizes Genotype × Environment interactions; immediately exposes hybrids that lack regional stability across diverse geographies.")
    
    # Mock heatmap data (10 Genotypes x 5 Environments)
    all_envs = ["Lincoln", "MOValley", "Ames", "Crawfordsville", "Scottsbluff"]
    # Only show heat matrix for selected environments from the sidebar
    display_envs = environments if environments else all_envs
    display_genos = fake_genotypes[:10]
    
    matrix = np.random.uniform(40.0, 55.0, (len(display_genos), len(display_envs)))
    
    fig_hm = px.imshow(
        matrix,
        x=display_envs,
        y=display_genos,
        color_continuous_scale="Viridis",
        labels=dict(color="Predicted Yield")
    )
    fig_hm.update_layout(margin=dict(l=0, r=0, t=30, b=0), height=400)
    st.plotly_chart(fig_hm, use_container_width=True)
