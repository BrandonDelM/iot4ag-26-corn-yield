import pandas as pd
import numpy as np
from scipy.stats import spearmanr
import os

def compute_hybrid_rankings(df):
    grp = df.groupby('genotype').agg(
        predicted_yield_mean=('predicted_yield', 'mean'),
        predicted_yield_std=('predicted_yield', 'std'),
        actual_yield_mean=('actual_yield', 'mean')
    ).reset_index()

    # Fill NaN for genotypes with only 1 plot
    grp['predicted_yield_std'] = grp['predicted_yield_std'].fillna(0)

    # --- NEW STABILITY PERCENTAGE LOGIC ---
    # We want 100% to be "Most Stable"
    # Logic: 1 - (Your SD / Max SD in the whole trial)
    max_sd_observed = grp['predicted_yield_std'].max()
    if max_sd_observed == 0:
        grp['stability_pct'] = 1.0 # Everything is perfectly stable
    else:
        # Higher is better: 1.0 is no variance, 0.0 is the most variant hybrid
        grp['stability_pct'] = 1.0 - (grp['predicted_yield_std'] / max_sd_observed)

    # Breeder Score (using the new percentage is cleaner)
    grp['breeder_score'] = grp['predicted_yield_mean'] * grp['stability_pct']

    # Tiers (Top 25% = Tier 1)
    grp['yield_tier'] = pd.qcut(grp['predicted_yield_mean'], 4, labels=['Tier 4', 'Tier 3', 'Tier 2', 'Tier 1'])
    grp['stability_tier'] = pd.qcut(grp['predicted_yield_std'].rank(method='first'), 4, 
                                    labels=['Elite', 'Stable', 'Variable', 'Risky'])

    # Ranking Correlation (Only if harvest data exists)
    grp['rank_predicted'] = grp['predicted_yield_mean'].rank(ascending=False).astype(int)
    
    if grp['actual_yield_mean'].notna().any():
        grp['rank_actual'] = grp['actual_yield_mean'].rank(ascending=False, na_option='keep')
        valid = grp['rank_actual'].notna()
        if valid.sum() > 1:
            r = spearmanr(grp['rank_predicted'][valid], grp['rank_actual'][valid]).statistic
            print(f"  Hybrid Rank Correlation: {r:.3f}")
        grp['rank_actual'] = grp['rank_actual'].astype('Int64')
    else:
        grp['rank_actual'] = pd.NA

    return grp.sort_values('breeder_score', ascending=False)

def compute_gxe_matrix(df):
    """
    Creates the Genotype x Location matrix for the environment filters.
    """
    matrix = df.pivot_table(
        index='genotype', 
        columns='location', 
        values='predicted_yield', 
        aggfunc='mean'
    )
    # Reset index ensures 'genotype' is a column and locations are headers
    return matrix.reset_index()

def compute_timepoint_rankings(df):
    """
    Creates the Genotype x Timepoint matrix for growth charts.
    """
    tp_pivot = df.pivot_table(
        index='genotype', 
        columns='timepoint', 
        values='predicted_yield', 
        aggfunc='mean'
    )
    return tp_pivot.reset_index()

if __name__ == "__main__":
    # Test block for standalone execution
    if os.path.exists('xgb_predictions_full.csv'):
        df_test = pd.read_csv('xgb_predictions_full.csv')
        rankings = compute_hybrid_rankings(df_test)
        print("\nTop 5 Hybrids:")
        print(rankings[['genotype', 'breeder_score', 'yield_tier']].head(5))