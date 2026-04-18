"""
Seed Breeder Scoring Engine
============================
Consumes xgb_predictions_full.csv (output of patch_features.py) and
produces hybrid-level rankings, stability scores, and early prediction
reliability metrics ready for the dashboard.

Output files:
  hybrid_rankings.csv     — one row per genotype, all scoring metrics
  gxe_matrix.csv          — genotype × location predicted yield matrix
  timepoint_rankings.csv  — rankings at each timepoint (TP1, TP2, TP3...)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
import os


# ─────────────────────────────────────────────
# 1. LOAD PREDICTIONS
# ─────────────────────────────────────────────

def load_predictions(filepath='xgb_predictions_full.csv',
                     groundtruth_csv=None):
    if not os.path.exists(filepath):
        raise FileNotFoundError(
            f"\nCould not find '{filepath}'.\n"
            f"Re-run patch_features.py first to generate this file."
        )

    df = pd.read_csv(filepath)
    print(f"Loaded {len(df)} rows from {filepath}")

    # If metadata columns are missing, join from ground truth CSV
    required = ['genotype', 'nitrogenTreatment']
    missing  = [c for c in required if c not in df.columns]

    if missing and groundtruth_csv and os.path.exists(groundtruth_csv):
        print(f"  Metadata columns {missing} not in CSV — joining from ground truth...")
        gt = pd.read_csv(groundtruth_csv)
        for col in ['range', 'row', 'experiment']:
            df[col] = df[col].astype(str)
            gt[col] = gt[col].astype(str)
        df = df.merge(
            gt[['location', 'range', 'row', 'experiment',
                'genotype', 'nitrogenTreatment', 'block']],
            on=['location', 'range', 'row', 'experiment'],
            how='left'
        )
        print(f"  Join complete — genotype nulls: {df['genotype'].isnull().sum()}")

    elif missing:
        raise KeyError(
            f"\nMissing columns: {missing}\n"
            f"Pass groundtruth_csv='path/to/train_HIPS_HYBRIDS_2023_V2.3.csv' "
            f"to load_predictions() to join them automatically, or re-run "
            f"patch_features.py to regenerate the predictions file with metadata."
        )

    print(f"\n  Genotypes   : {df['genotype'].nunique()}")
    print(f"  Locations   : {df['location'].unique().tolist()}")
    print(f"  Timepoints  : {sorted(df['timepoint'].unique().tolist())}")
    print(f"  N treatments: {df['nitrogenTreatment'].unique().tolist()}")
    return df


# ─────────────────────────────────────────────
# 2. HYBRID RANKINGS
# ─────────────────────────────────────────────

def compute_hybrid_rankings(df):
    """
    Aggregate plot-level predictions to genotype level.
    Returns a DataFrame with one row per genotype and these columns:

      predicted_yield_mean  — average predicted yield across all plots
      predicted_yield_std   — std dev (stability — lower is more consistent)
      actual_yield_mean     — ground truth average (for validation)
      plot_count            — how many plots this genotype appeared in
      rank_predicted        — rank by predicted yield (1 = best)
      rank_actual           — rank by actual yield (for comparison)
      stability_tier        — High / Medium / Low stability label
      yield_tier            — Top25 / Top50 / Bottom50 label
    """
    grp = df.groupby('genotype').agg(
        predicted_yield_mean=('predicted_yield', 'mean'),
        predicted_yield_std=('predicted_yield', 'std'),
        actual_yield_mean=('actual_yield', 'mean'),
        actual_yield_std=('actual_yield', 'std'),
        plot_count=('predicted_yield', 'count'),
    ).reset_index()

    # Fill NaN std for genotypes with only 1 plot
    grp['predicted_yield_std'] = grp['predicted_yield_std'].fillna(0)

    # Ranks (1 = highest yield)
    grp['rank_predicted'] = grp['predicted_yield_mean'].rank(ascending=False).astype(int)
    grp['rank_actual']    = grp['actual_yield_mean'].rank(ascending=False).astype(int)
    grp['rank_error']     = (grp['rank_predicted'] - grp['rank_actual']).abs()

    # Yield tiers
    top25 = grp['predicted_yield_mean'].quantile(0.75)
    top50 = grp['predicted_yield_mean'].quantile(0.50)
    grp['yield_tier'] = 'Bottom 50%'
    grp.loc[grp['predicted_yield_mean'] >= top50, 'yield_tier'] = 'Top 50%'
    grp.loc[grp['predicted_yield_mean'] >= top25, 'yield_tier'] = 'Top 25%'

    # Stability tiers (lower std = more stable)
    low_thresh  = grp['predicted_yield_std'].quantile(0.33)
    high_thresh = grp['predicted_yield_std'].quantile(0.67)
    grp['stability_tier'] = 'Medium'
    grp.loc[grp['predicted_yield_std'] <= low_thresh,  'stability_tier'] = 'High'
    grp.loc[grp['predicted_yield_std'] >= high_thresh, 'stability_tier'] = 'Low'

    # Composite breeder score: rewards high yield AND high stability
    # Normalise both to 0-1 then combine (70% yield, 30% stability)
    y_norm = (grp['predicted_yield_mean'] - grp['predicted_yield_mean'].min()) / \
             (grp['predicted_yield_mean'].max() - grp['predicted_yield_mean'].min())
    s_norm = 1 - (grp['predicted_yield_std'] - grp['predicted_yield_std'].min()) / \
             (grp['predicted_yield_std'].max() - grp['predicted_yield_std'].min() + 1e-9)
    grp['breeder_score'] = (0.70 * y_norm + 0.30 * s_norm).round(3)
    grp['rank_breeder']  = grp['breeder_score'].rank(ascending=False).astype(int)

    grp = grp.sort_values('rank_predicted').reset_index(drop=True)

    print(f"\nHybrid rankings computed for {len(grp)} genotypes")
    print(f"  Rank correlation (predicted vs actual): "
          f"{spearmanr(grp['rank_predicted'], grp['rank_actual']).statistic:.3f}")

    return grp


# ─────────────────────────────────────────────
# 3. GxE MATRIX
# ─────────────────────────────────────────────

def compute_gxe_matrix(df):
    """
    Pivot to a genotype × location matrix of mean predicted yield.
    This is the core GxE view — which hybrids thrive where.
    """
    gxe = df.groupby(['genotype', 'location'])['predicted_yield'].mean().reset_index()
    matrix = gxe.pivot(index='genotype', columns='location', values='predicted_yield')
    matrix = matrix.round(1)

    # Sort by row mean (best overall performers at top)
    matrix['_mean'] = matrix.mean(axis=1)
    matrix = matrix.sort_values('_mean', ascending=False).drop(columns='_mean')

    print(f"\nGxE matrix: {matrix.shape[0]} genotypes × {matrix.shape[1]} locations")
    return matrix


# ─────────────────────────────────────────────
# 4. TIMEPOINT RANKINGS
# ─────────────────────────────────────────────

def compute_timepoint_rankings(df):
    """
    Compute hybrid rankings at each individual timepoint.
    Key metric: how early can we reliably rank hybrids?

    Returns a DataFrame and prints Spearman correlation of each
    timepoint's ranking against the full-season ranking.
    """
    timepoints = sorted(df['timepoint'].unique())

    # Full-season ranking (all timepoints combined) as ground truth
    full_ranking = df.groupby('genotype')['predicted_yield'].mean().rank(ascending=False)

    tp_rankings = {}
    print("\nTimepoint ranking reliability (Spearman r vs full-season):")
    for tp in timepoints:
        tp_df = df[df['timepoint'] == tp]
        tp_rank = tp_df.groupby('genotype')['predicted_yield'].mean().rank(ascending=False)

        # Align on common genotypes
        common = full_ranking.index.intersection(tp_rank.index)
        r, p = spearmanr(full_ranking[common], tp_rank[common])
        print(f"  {tp}: r={r:.3f}  (p={p:.4f})  n={len(common)} genotypes")

        tp_rankings[tp] = tp_df.groupby('genotype')['predicted_yield'].mean()

    tp_df_wide = pd.DataFrame(tp_rankings)
    tp_df_wide.columns = [f'predicted_yield_{tp}' for tp in tp_df_wide.columns]
    tp_df_wide = tp_df_wide.reset_index()
    return tp_df_wide


# ─────────────────────────────────────────────
# 5. NITROGEN RESPONSE
# ─────────────────────────────────────────────

def compute_nitrogen_response(df):
    """
    For each genotype, compute yield response to nitrogen treatment.
    Hybrids that respond strongly to N are riskier in low-input systems.
    """
    n_map = {'Low': 0, 'Medium': 1, 'High': 2}
    df = df.copy()
    df['n_level'] = df['nitrogenTreatment'].map(n_map)

    n_response = df.groupby(['genotype', 'nitrogenTreatment'])['predicted_yield'] \
                   .mean().unstack(fill_value=np.nan).reset_index()

    # N response = yield at High N minus yield at Low N
    if 'High' in n_response.columns and 'Low' in n_response.columns:
        n_response['n_response'] = n_response['High'] - n_response['Low']
        n_response['n_response_tier'] = pd.qcut(
            n_response['n_response'].rank(method='first'),
            q=3, labels=['Low responder', 'Medium responder', 'High responder']
        )

    print(f"\nNitrogen response computed for {len(n_response)} genotypes")
    return n_response


# ─────────────────────────────────────────────
# 6. PLOTS
# ─────────────────────────────────────────────

def plot_yield_vs_stability(rankings_df, output_path='yield_vs_stability.png'):
    """
    Scatter plot: predicted yield (x) vs stability (y, inverted so up = more stable).
    Quadrants label the four breeder archetypes.
    """
    fig, ax = plt.subplots(figsize=(10, 7))

    colors = {'Top 25%': '#2ecc71', 'Top 50%': '#f39c12', 'Bottom 50%': '#e74c3c'}
    for tier, grp in rankings_df.groupby('yield_tier'):
        ax.scatter(grp['predicted_yield_mean'], grp['predicted_yield_std'],
                   label=tier, color=colors.get(tier, 'gray'),
                   alpha=0.75, edgecolors='k', linewidths=0.4, s=60)

    # Quadrant lines at medians
    med_yield = rankings_df['predicted_yield_mean'].median()
    med_std   = rankings_df['predicted_yield_std'].median()
    ax.axvline(med_yield, color='gray', linestyle='--', linewidth=0.8, alpha=0.6)
    ax.axhline(med_std,   color='gray', linestyle='--', linewidth=0.8, alpha=0.6)

    # Quadrant labels
    ax.text(med_yield + 1, med_std * 0.15, 'High yield\nHigh stability',
            fontsize=9, color='#27ae60', ha='left')
    ax.text(rankings_df['predicted_yield_mean'].min(), med_std * 0.15,
            'Low yield\nHigh stability', fontsize=9, color='#7f8c8d', ha='left')
    ax.text(med_yield + 1, rankings_df['predicted_yield_std'].max() * 0.9,
            'High yield\nLow stability', fontsize=9, color='#e67e22', ha='left')
    ax.text(rankings_df['predicted_yield_mean'].min(),
            rankings_df['predicted_yield_std'].max() * 0.9,
            'Low yield\nLow stability', fontsize=9, color='#c0392b', ha='left')

    ax.set_xlabel('Mean predicted yield (bu/acre)')
    ax.set_ylabel('Std dev of predicted yield (lower = more stable)')
    ax.set_title('Hybrid yield vs stability — breeder quadrant view')
    ax.legend(title='Yield tier', loc='upper left')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.show()
    print(f"Saved {output_path}")


def plot_top_hybrids(rankings_df, n=20, output_path='top_hybrids.png'):
    """
    Horizontal bar chart of top N hybrids by breeder score,
    with error bars showing stability (std dev).
    """
    top = rankings_df.nsmallest(n, 'rank_breeder')
    top = top.sort_values('breeder_score')

    fig, ax = plt.subplots(figsize=(10, 8))
    bars = ax.barh(top['genotype'], top['predicted_yield_mean'],
                   xerr=top['predicted_yield_std'],
                   color='steelblue', alpha=0.8, ecolor='gray',
                   capsize=3, height=0.6)
    ax.set_xlabel('Predicted yield (bu/acre)  ±1 std dev')
    ax.set_title(f'Top {n} hybrids by breeder score (yield + stability)')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.show()
    print(f"Saved {output_path}")


def plot_gxe_heatmap(gxe_matrix, output_path='gxe_heatmap.png', top_n=30):
    """
    Heatmap of predicted yield for top N genotypes across locations.
    """
    import matplotlib.colors as mcolors
    top = gxe_matrix.head(top_n)

    fig, ax = plt.subplots(figsize=(max(6, len(top.columns) * 2), max(8, top_n * 0.35)))
    im = ax.imshow(top.values, aspect='auto', cmap='RdYlGn')
    ax.set_xticks(range(len(top.columns)))
    ax.set_xticklabels(top.columns, rotation=30, ha='right', fontsize=9)
    ax.set_yticks(range(len(top.index)))
    ax.set_yticklabels(top.index, fontsize=7)
    plt.colorbar(im, ax=ax, label='Predicted yield (bu/acre)')
    ax.set_title(f'GxE predicted yield — top {top_n} genotypes')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.show()
    print(f"Saved {output_path}")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

if __name__ == '__main__':

    PREDICTIONS_FILE = 'xgb_predictions_full.csv'
    GROUNDTRUTH_CSV  = '../GroundTruth/train_HIPS_HYBRIDS_2023_V2.3.csv'
    OUTPUT_DIR       = './scoring_outputs'
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ── Load (joins ground truth automatically if metadata columns are missing) ──
    df = load_predictions(PREDICTIONS_FILE, groundtruth_csv=GROUNDTRUTH_CSV)

    # ── Hybrid rankings ──
    rankings = compute_hybrid_rankings(df)
    rankings.to_csv(f'{OUTPUT_DIR}/hybrid_rankings.csv', index=False)
    print(f"\nTop 10 hybrids by breeder score:")
    print(rankings[['genotype', 'predicted_yield_mean', 'predicted_yield_std',
                     'breeder_score', 'yield_tier', 'stability_tier']].head(10).to_string(index=False))

    # ── GxE matrix ──
    gxe = compute_gxe_matrix(df)
    gxe.to_csv(f'{OUTPUT_DIR}/gxe_matrix.csv')
    print(f"\nTop 5 genotypes across locations:")
    print(gxe.head(5).to_string())

    # ── Timepoint rankings ──
    tp_rankings = compute_timepoint_rankings(df)
    tp_rankings.to_csv(f'{OUTPUT_DIR}/timepoint_rankings.csv', index=False)

    # ── Nitrogen response ──
    n_response = compute_nitrogen_response(df)
    n_response.to_csv(f'{OUTPUT_DIR}/nitrogen_response.csv', index=False)

    # ── Plots ──
    plot_yield_vs_stability(rankings,
                            output_path=f'{OUTPUT_DIR}/yield_vs_stability.png')
    plot_top_hybrids(rankings, n=20,
                     output_path=f'{OUTPUT_DIR}/top_hybrids.png')
    plot_gxe_heatmap(gxe, output_path=f'{OUTPUT_DIR}/gxe_heatmap.png')

    print(f"\nAll outputs saved to {OUTPUT_DIR}/")
    print("Feed hybrid_rankings.csv and gxe_matrix.csv into the dashboard.")