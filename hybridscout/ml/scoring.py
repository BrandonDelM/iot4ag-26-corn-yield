"""
Seed-breeder scoring engine.

Consumes OOF predictions (xgb_predictions_full.csv) from train_model.py and
produces hybrid-level rankings, stability scores, GxE matrix, per-TP early
ranking reliability, and nitrogen-response tiers.

All outputs written to hybridscout/artifacts/ for the Streamlit app.

Run:  python -m hybridscout.ml.scoring
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from . import io_paths as P

YIELD_WEIGHT = 0.70
STABILITY_WEIGHT = 0.30
MIN_PLOTS_PER_GENOTYPE = 2


# ─────────────────────────────────────────────
# Load OOF predictions
# ─────────────────────────────────────────────

def load_predictions(oof_csv=P.OOF_CSV) -> pd.DataFrame:
    if not oof_csv.exists():
        raise FileNotFoundError(
            f"Could not find {oof_csv}. Run `python -m hybridscout.ml.train_model` first."
        )
    df = pd.read_csv(oof_csv)
    print(f"Loaded {len(df)} OOF rows from {oof_csv.name}")
    print(f"  Genotypes : {df['genotype'].nunique()}")
    print(f"  Locations : {sorted(df['location'].unique())}")
    print(f"  Timepoints: {sorted(df['timepoint'].unique())}")
    return df


# ─────────────────────────────────────────────
# Hybrid rankings with stability + breeder score
# ─────────────────────────────────────────────

def compute_hybrid_rankings(df: pd.DataFrame,
                            yield_weight: float = YIELD_WEIGHT,
                            stability_weight: float = STABILITY_WEIGHT,
                            min_plots: int = MIN_PLOTS_PER_GENOTYPE) -> pd.DataFrame:
    grp = df.groupby("genotype").agg(
        predicted_yield_mean=("predicted_yield", "mean"),
        predicted_yield_std=("predicted_yield", "std"),
        actual_yield_mean=("actual_yield", "mean"),
        actual_yield_std=("actual_yield", "std"),
        plot_count=("predicted_yield", "count"),
    ).reset_index()

    grp = grp[grp["plot_count"] >= min_plots].copy()
    grp["predicted_yield_std"] = grp["predicted_yield_std"].fillna(0)

    grp["rank_predicted"] = grp["predicted_yield_mean"].rank(ascending=False).astype(int)
    grp["rank_actual"] = grp["actual_yield_mean"].rank(ascending=False).astype(int)
    grp["rank_error"] = (grp["rank_predicted"] - grp["rank_actual"]).abs()

    top25 = grp["predicted_yield_mean"].quantile(0.75)
    top50 = grp["predicted_yield_mean"].quantile(0.50)
    grp["yield_tier"] = "Bottom 50%"
    grp.loc[grp["predicted_yield_mean"] >= top50, "yield_tier"] = "Top 50%"
    grp.loc[grp["predicted_yield_mean"] >= top25, "yield_tier"] = "Top 25%"

    low_thresh = grp["predicted_yield_std"].quantile(0.33)
    high_thresh = grp["predicted_yield_std"].quantile(0.67)
    grp["stability_tier"] = "Medium"
    grp.loc[grp["predicted_yield_std"] <= low_thresh, "stability_tier"] = "High"
    grp.loc[grp["predicted_yield_std"] >= high_thresh, "stability_tier"] = "Low"

    y_range = grp["predicted_yield_mean"].max() - grp["predicted_yield_mean"].min()
    s_range = grp["predicted_yield_std"].max() - grp["predicted_yield_std"].min()
    y_norm = (grp["predicted_yield_mean"] - grp["predicted_yield_mean"].min()) / (y_range + 1e-9)
    s_norm = 1 - (grp["predicted_yield_std"] - grp["predicted_yield_std"].min()) / (s_range + 1e-9)
    grp["breeder_score"] = (yield_weight * y_norm + stability_weight * s_norm).round(3)
    grp["rank_breeder"] = grp["breeder_score"].rank(ascending=False).astype(int)

    grp = grp.sort_values("rank_breeder").reset_index(drop=True)

    rho = spearmanr(grp["rank_predicted"], grp["rank_actual"]).statistic
    print(f"\nHybrid rankings: {len(grp)} genotypes (min_plots={min_plots})")
    print(f"  Spearman predicted vs actual rank: {rho:.3f}")
    return grp


# ─────────────────────────────────────────────
# GxE matrix (genotype × location)
# ─────────────────────────────────────────────

def compute_gxe_matrix(df: pd.DataFrame) -> pd.DataFrame:
    gxe = df.groupby(["genotype", "location"])["predicted_yield"].mean().reset_index()
    matrix = gxe.pivot(index="genotype", columns="location", values="predicted_yield").round(1)
    matrix["_mean"] = matrix.mean(axis=1)
    matrix = matrix.sort_values("_mean", ascending=False).drop(columns="_mean")
    print(f"\nGxE matrix: {matrix.shape[0]} genotypes × {matrix.shape[1]} locations")
    return matrix


# ─────────────────────────────────────────────
# Per-timepoint early-ranking reliability
# ─────────────────────────────────────────────

def compute_timepoint_rankings(df: pd.DataFrame) -> pd.DataFrame:
    """Per-TP mean predicted yield, plus Spearman vs full-season ranking."""
    timepoints = sorted(df["timepoint"].unique())
    full_rank = df.groupby("genotype")["predicted_yield"].mean().rank(ascending=False)

    tp_means = {}
    print("\nTimepoint ranking reliability (Spearman r vs full-season):")
    for tp in timepoints:
        tp_df = df[df["timepoint"] == tp]
        tp_mean = tp_df.groupby("genotype")["predicted_yield"].mean()
        tp_rank = tp_mean.rank(ascending=False)
        common = full_rank.index.intersection(tp_rank.index)
        if len(common) >= 3:
            r, p = spearmanr(full_rank[common], tp_rank[common])
            print(f"  {tp}: r={r:.3f}  (p={p:.4f})  n={len(common)} genotypes")
        tp_means[tp] = tp_mean

    wide = pd.DataFrame(tp_means)
    wide.columns = [f"predicted_yield_{tp}" for tp in wide.columns]
    return wide.reset_index()


# ─────────────────────────────────────────────
# Nitrogen response (Lincoln only: Low/Medium/High rates)
# ─────────────────────────────────────────────

def compute_nitrogen_response(df: pd.DataFrame) -> pd.DataFrame:
    lincoln = df[df["location"] == "Lincoln"].copy()
    if lincoln.empty:
        print("\nNitrogen response: no Lincoln rows, skipping.")
        return pd.DataFrame()

    n_resp = (lincoln.groupby(["genotype", "nitrogenTreatment"])["predicted_yield"]
              .mean().unstack(fill_value=np.nan).reset_index())

    if {"High", "Low"}.issubset(n_resp.columns):
        n_resp["n_response"] = n_resp["High"] - n_resp["Low"]
        valid = n_resp["n_response"].notna()
        if valid.sum() >= 3:
            n_resp.loc[valid, "n_response_tier"] = pd.qcut(
                n_resp.loc[valid, "n_response"].rank(method="first"),
                q=3, labels=["Low responder", "Medium responder", "High responder"],
            )

    print(f"\nNitrogen response: {len(n_resp)} Lincoln genotypes")
    return n_resp


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def main():
    df = load_predictions()

    rankings = compute_hybrid_rankings(df)
    rankings.to_csv(P.HYBRID_RANKINGS_CSV, index=False)
    print(f"saved → {P.HYBRID_RANKINGS_CSV}")

    gxe = compute_gxe_matrix(df)
    gxe.to_csv(P.GXE_MATRIX_CSV)
    print(f"saved → {P.GXE_MATRIX_CSV}")

    tp_rankings = compute_timepoint_rankings(df)
    tp_rankings.to_csv(P.TIMEPOINT_RANKINGS_CSV, index=False)
    print(f"saved → {P.TIMEPOINT_RANKINGS_CSV}")

    n_response = compute_nitrogen_response(df)
    n_response.to_csv(P.NITROGEN_RESPONSE_CSV, index=False)
    print(f"saved → {P.NITROGEN_RESPONSE_CSV}")

    print("\nTop 10 hybrids by breeder score:")
    cols = ["genotype", "predicted_yield_mean", "predicted_yield_std",
            "breeder_score", "yield_tier", "stability_tier"]
    print(rankings[cols].head(10).to_string(index=False))


if __name__ == "__main__":
    main()
