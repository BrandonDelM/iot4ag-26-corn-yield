import pandas as pd
import matplotlib.pyplot as plt
import cv2 as cv
import rasterio
import os
import glob
import warnings
import numpy as np

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
import joblib

warnings.filterwarnings("ignore", message="invalid value encountered in scalar divide")


# ─────────────────────────────────────────────
# HELPER
# ─────────────────────────────────────────────

def norm(band):
    band_min, band_max = band.min(), band.max()
    if band_max == band_min:
        return np.zeros_like(band, dtype=np.float32)
    return (band - band_min) / (band_max - band_min)


def indices_from_arrays(r, g, b, nir=None, re=None):
    """
    Compute vegetation indices from 2D NumPy arrays.
    nir and re are optional (satellite only).
    Returns a dict of {index_name: 2D array}.
    """
    with np.errstate(divide='ignore', invalid='ignore'):
        GLI   = np.where((2*g + r + b) != 0, (2*g - r - b) / (2*g + r + b), np.nan)
        NGRDI = np.where((r + g) != 0, (r - g) / (r + g), np.nan)

    result = {'GLI': GLI, 'NGRDI': NGRDI, 'Red': r, 'Green': g, 'Blue': b}

    if nir is not None:
        with np.errstate(divide='ignore', invalid='ignore'):
            NDVI  = np.where((nir + r)       != 0, (nir - r)       / (nir + r),       np.nan)
            GNDVI = np.where((nir + g)       != 0, (nir - g)       / (nir + g),       np.nan)
            SAVI  = np.where((nir + r + 0.5) != 0, 1.5*(nir - r)   / (nir + r + 0.5), np.nan)
            NDRE  = np.where((nir + re)      != 0, (nir - re)       / (nir + re),      np.nan) if re is not None else None
        result.update({'NDVI': NDVI, 'GNDVI': GNDVI, 'SAVI': SAVI, 'NIR': nir})
        if NDRE is not None:
            result['NDRE'] = NDRE

    return result


def patch_stats(patch_2d, mask_2d):
    """
    Given a 2D array and a boolean mask of valid pixels,
    return mean, median, std for the valid pixels.
    """
    valid = patch_2d[mask_2d]
    valid = valid[~np.isnan(valid)]
    if len(valid) == 0:
        return np.nan, np.nan, np.nan
    return float(np.mean(valid)), float(np.median(valid)), float(np.std(valid))


# ─────────────────────────────────────────────
# PATCH-BASED SATELLITE FEATURE EXTRACTION
# ─────────────────────────────────────────────

def satelliteimage_patches(inputpath, grid_size=4):
    """
    Divide a 6-band satellite .TIF into a grid_size × grid_size patch grid.
    For each patch, compute mean/median/std of every vegetation index.

    Returns a single-row dict with columns like:
        patch_0_0_NDVI_mean, patch_0_1_NDVI_mean, ..., patch_3_3_SAVI_std
    """
    with rasterio.open(inputpath, 'r') as src:
        bands = src.read().astype(np.float32)   # (6, H, W)

    r, g, b = bands[0], bands[1], bands[2]
    nir, re, db = bands[3], bands[4], bands[5]

    # Valid pixel mask — removes zero-padded border pixels
    mask = r > 0

    # Compute all indices as full 2D arrays (fast — no pixel loop)
    idx_arrays = indices_from_arrays(r, g, b, nir=nir, re=re)

    H, W = r.shape
    ph = H // grid_size   # patch height
    pw = W // grid_size   # patch width

    row_dict = {'Imagename': os.path.basename(inputpath)}

    for pi in range(grid_size):
        for pj in range(grid_size):
            # Slice this patch out of every array
            rs = slice(pi * ph, (pi + 1) * ph)
            cs = slice(pj * pw, (pj + 1) * pw)

            patch_mask = mask[rs, cs]

            # Skip completely empty patches (all border zeros)
            if patch_mask.sum() == 0:
                continue

            prefix = f'patch_{pi}_{pj}'

            for idx_name, idx_arr in idx_arrays.items():
                patch_vals = idx_arr[rs, cs]
                mean_, med_, std_ = patch_stats(patch_vals, patch_mask)
                row_dict[f'{prefix}_{idx_name}_mean']   = mean_
                row_dict[f'{prefix}_{idx_name}_median'] = med_
                row_dict[f'{prefix}_{idx_name}_std']    = std_

    return row_dict


# ─────────────────────────────────────────────
# PATCH-BASED RGB FEATURE EXTRACTION
# ─────────────────────────────────────────────

def RGB_patches(inputpath, grid_size=4):
    """
    Divide an RGB image into a grid_size × grid_size patch grid.
    For each patch, compute mean/median/std of GLI and NGRDI.
    Returns a single-row dict.
    """
    img = cv.imread(inputpath)
    if img is None:
        return {}
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB).astype(np.float32)
    r, g, b = img[:,:,0], img[:,:,1], img[:,:,2]

    mask = r > 0
    idx_arrays = indices_from_arrays(r, g, b)

    H, W = r.shape
    ph = H // grid_size
    pw = W // grid_size

    row_dict = {'Imagename': os.path.basename(inputpath)}

    for pi in range(grid_size):
        for pj in range(grid_size):
            rs = slice(pi * ph, (pi + 1) * ph)
            cs = slice(pj * pw, (pj + 1) * pw)
            patch_mask = mask[rs, cs]

            if patch_mask.sum() == 0:
                continue

            prefix = f'patch_{pi}_{pj}'
            for idx_name, idx_arr in idx_arrays.items():
                patch_vals = idx_arr[rs, cs]
                mean_, med_, std_ = patch_stats(patch_vals, patch_mask)
                row_dict[f'{prefix}_{idx_name}_mean']   = mean_
                row_dict[f'{prefix}_{idx_name}_median'] = med_
                row_dict[f'{prefix}_{idx_name}_std']    = std_

    return row_dict


# ─────────────────────────────────────────────
# COLLECT ALL IMAGES
# ─────────────────────────────────────────────

def collect_all_satellite(image_root='../Satellite', grid_size=4):
    all_tif = glob.glob(os.path.join(image_root, '**', '*.TIF'), recursive=True)
    all_tif += glob.glob(os.path.join(image_root, '**', '*.tif'), recursive=True)

    if not all_tif:
        print(f"No TIF files found under {image_root}")
        return pd.DataFrame()

    results = []
    for i, filepath in enumerate(all_tif):
        print(f"  Satellite [{i+1}/{len(all_tif)}]: {os.path.basename(filepath)}")
        try:
            row = satelliteimage_patches(filepath, grid_size=grid_size)
            results.append(row)
        except Exception as e:
            print(f"    Warning: skipped {filepath} — {e}")

    return pd.DataFrame(results) if results else pd.DataFrame()


def collect_all_rgb(image_root='../UAV', grid_size=4):
    all_rgb = glob.glob(os.path.join(image_root, '**', '*.jpg'), recursive=True)
    all_rgb += glob.glob(os.path.join(image_root, '**', '*.png'), recursive=True)

    if not all_rgb:
        print(f"No RGB files found under {image_root}")
        return pd.DataFrame()

    results = []
    for i, filepath in enumerate(all_rgb):
        print(f"  RGB [{i+1}/{len(all_rgb)}]: {os.path.basename(filepath)}")
        try:
            row = RGB_patches(filepath, grid_size=grid_size)
            results.append(row)
        except Exception as e:
            print(f"    Warning: skipped {filepath} — {e}")

    return pd.DataFrame(results) if results else pd.DataFrame()


# ─────────────────────────────────────────────
# PARSE FILENAME INTO PLOT METADATA
# ─────────────────────────────────────────────

def parse_filename(imagename_col):
    def parse_one(name):
        try:
            base  = os.path.splitext(name)[0]
            parts = base.split('-')
            location   = parts[0]
            timepoint  = parts[1]
            rest       = parts[2].split('_')
            experiment = rest[0]
            rangeno    = rest[1]
            row        = rest[2]
            return location, timepoint, experiment, rangeno, row
        except Exception:
            return None, None, None, None, None

    parsed = imagename_col.apply(parse_one)
    return pd.DataFrame(parsed.tolist(),
                        columns=['location', 'timepoint', 'experiment', 'range', 'row'])


# ─────────────────────────────────────────────
# RANDOM FOREST MODEL
# ─────────────────────────────────────────────

def run_random_forest(features_df, groundtruth_df,
                      n_estimators=300, max_depth=15,
                      min_samples_leaf=10, max_features=0.3,
                      n_jobs=-1):

    # ── Merge features with ground truth ──
    sat_col = [c for c in features_df.columns if 'Imagename' in c]
    if not sat_col:
        print("Could not find Imagename column.")
        return

    meta = parse_filename(features_df[sat_col[0]])
    features_df = pd.concat([features_df.reset_index(drop=True),
                              meta.reset_index(drop=True)], axis=1)

    groundtruth_df['range']      = groundtruth_df['range'].astype(str)
    groundtruth_df['row']        = groundtruth_df['row'].astype(str)
    groundtruth_df['experiment'] = groundtruth_df['experiment'].astype(str)

    # ── Encode nitrogenTreatment as ordered number ──
    nitrogen_map = {'Low': 0, 'Medium': 1, 'High': 2}
    groundtruth_df['nitrogenTreatment_enc'] = (
        groundtruth_df['nitrogenTreatment'].map(nitrogen_map)
    )

    # ── Encode genotype as integer ──
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    groundtruth_df['genotype_enc'] = le.fit_transform(
        groundtruth_df['genotype'].astype(str)
    )

    # ── Encode location ──
    groundtruth_df['location_enc'] = (groundtruth_df['location'] == 'Lincoln').astype(int)

    # NOTE: genotype_mean_yield is intentionally NOT computed here.
    # Computing it on the full dataset before splitting is data leakage —
    # test plots would have their own yield included in their own feature.
    # It is computed from train data only after the plot-level split below.

    AGRONOMIC_FEATURES = [
        'nitrogenTreatment_enc',
        'GDDToAnthesis',
        'daysToAnthesis',
        'genotype_enc',
        'genotype_mean_yield',   # populated after split — see below
        'location_enc',
        'block',
    ]

    merged = features_df.merge(
        groundtruth_df[['location', 'range', 'row', 'experiment', 'yieldPerAcre',
                        'genotype', 'nitrogenTreatment']
                       + [f for f in AGRONOMIC_FEATURES if f != 'genotype_mean_yield']],
        on=['location', 'range', 'row', 'experiment'],
        how='inner'
    )

    print(f"\nMerged dataset: {len(merged)} rows")
    print(f"  Patch image features : {len([c for c in merged.columns if 'patch' in c])}")
    print(f"  Agronomic features   : {len(AGRONOMIC_FEATURES)}")
    if len(merged) == 0:
        print("No matching rows after merge.")
        return

    # ── IDEA 2: Temporal features ──
    # For each plot, compute the change in key indices between timepoints.
    # The rate of green-up or senescence is often more predictive than
    # the index value at any single point.
    print("\nBuilding temporal delta features...")

    # Get patch index mean columns only (use mean as the representative value per patch)
    index_names = ['NDVI', 'GNDVI', 'SAVI', 'GLI', 'NGRDI']
    patch_positions = [f'{pi}_{pj}' for pi in range(GRID_SIZE) for pj in range(GRID_SIZE)]

    # Add plot key to merged for grouping
    merged['_plot_key'] = (merged['location'].astype(str) + '_' +
                           merged['range'].astype(str) + '_' +
                           merged['row'].astype(str) + '_' +
                           merged['experiment'].astype(str))

    # Add timepoint back in for pivot (it was in meta but got dropped)
    if 'timepoint' in merged.columns:
        tp_col = 'timepoint'
    else:
        # Re-parse from Imagename if needed
        sat_col2 = [c for c in merged.columns if 'Imagename' in c]
        if sat_col2:
            tp_series = merged[sat_col2[0]].apply(
                lambda x: os.path.splitext(x)[0].split('-')[1] if isinstance(x, str) else None
            )
            merged['timepoint'] = tp_series
            tp_col = 'timepoint'
        else:
            tp_col = None

    temporal_rows = []
    if tp_col and merged[tp_col].nunique() > 1:
        timepoints = sorted(merged[tp_col].dropna().unique())
        print(f"  Timepoints found: {timepoints}")

        for plot_key, plot_group in merged.groupby('_plot_key'):
            plot_group = plot_group.sort_values(tp_col)
            tps = plot_group[tp_col].tolist()
            delta_dict = {'_plot_key': plot_key}

            # Compute delta between consecutive timepoints for each patch × index
            for idx_name in index_names:
                for patch_pos in patch_positions:
                    col = f'patch_{patch_pos}_{idx_name}_mean'
                    if col not in plot_group.columns:
                        continue
                    vals = plot_group[col].tolist()
                    for t in range(1, len(vals)):
                        delta_name = f'delta_{idx_name}_{patch_pos}_TP{t}to{t+1}'
                        delta_dict[delta_name] = vals[t] - vals[t-1]

            # Also compute whole-plot mean NDVI per timepoint (for trend summary)
            ndvi_patch_cols = [c for c in plot_group.columns if 'NDVI_mean' in c and 'patch' in c]
            for i, row_ in enumerate(plot_group.itertuples()):
                tp_label = getattr(row_, tp_col)
                plot_ndvi_vals = [getattr(row_, c.replace(' ', '_'), np.nan)
                                  for c in ndvi_patch_cols]
                delta_dict[f'mean_NDVI_{tp_label}'] = float(np.nanmean(plot_ndvi_vals))

            temporal_rows.append(delta_dict)

        if temporal_rows:
            temporal_df = pd.DataFrame(temporal_rows)
            n_delta_cols = len([c for c in temporal_df.columns if c.startswith('delta_')])
            print(f"  Created {n_delta_cols} temporal delta features across {len(temporal_rows)} plots")
            merged = merged.merge(temporal_df, on='_plot_key', how='left')
        else:
            print("  No temporal deltas computed — check timepoint column.")
    else:
        print("  Only one timepoint found — skipping temporal deltas.")

    # Clean up helper columns
    merged = merged.drop(columns=['_plot_key'], errors='ignore')

    # ── SPLIT BY PLOT (not by image row) ──────────────────────────────────
    # Each plot has multiple timepoint images all sharing the same yield target.
    # Splitting at the image-row level leaks the same plot across train/test.
    # Fix: group all timepoints of a plot together before splitting.
    plot_key_cols = ['location', 'range', 'row', 'experiment']
    unique_plots = (merged[plot_key_cols + ['yieldPerAcre', 'genotype']]
                    .drop_duplicates(subset=plot_key_cols)
                    .reset_index(drop=True))

    y_bins_plots = pd.qcut(unique_plots['yieldPerAcre'], q=4, labels=False, duplicates='drop')
    train_plots, temp_plots = train_test_split(
        unique_plots, test_size=0.30, random_state=42, stratify=y_bins_plots
    )
    temp_bins = pd.qcut(temp_plots['yieldPerAcre'], q=4, labels=False, duplicates='drop')
    val_plots, test_plots = train_test_split(
        temp_plots, test_size=0.50, random_state=42, stratify=temp_bins
    )

    def tag_split(df, train_p, val_p, key_cols):
        train_keys = set(map(tuple, train_p[key_cols].values))
        val_keys   = set(map(tuple, val_p[key_cols].values))
        return df[key_cols].apply(tuple, axis=1).map(
            lambda k: 'train' if k in train_keys else ('val' if k in val_keys else 'test')
        )

    merged['split'] = tag_split(merged, train_plots, val_plots, plot_key_cols)
    train_df = merged[merged['split'] == 'train'].copy()
    val_df   = merged[merged['split'] == 'val'].copy()
    test_df  = merged[merged['split'] == 'test'].copy()

    print(f"\nPlot-level split (no timepoint leakage):")
    print(f"  Unique plots — train: {train_df[plot_key_cols].drop_duplicates().shape[0]}"
          f" | val: {val_df[plot_key_cols].drop_duplicates().shape[0]}"
          f" | test: {test_df[plot_key_cols].drop_duplicates().shape[0]}")
    print(f"  Image rows  — train: {len(train_df)} | val: {len(val_df)} | test: {len(test_df)}")
    unseen = len(set(test_df['genotype']) - set(train_df['genotype']))
    print(f"  Genotypes in test not seen in train: {unseen}")

    # ── genotype_mean_yield from train only (leakage fix) ─────────────────
    train_geno_mean = (train_df.groupby('genotype')['yieldPerAcre']
                       .mean().rename('genotype_mean_yield'))
    global_mean = train_df['yieldPerAcre'].mean()
    for df_split in [train_df, val_df, test_df]:
        df_split['genotype_mean_yield'] = (
            df_split['genotype'].map(train_geno_mean).fillna(global_mean)
        )
    print(f"  genotype_mean_yield computed from train only (fallback: {global_mean:.1f})")

    # ── Prepare X and y ──
    drop_cols = [c for c in merged.columns if 'Imagename' in c] + \
                ['location', 'timepoint', 'experiment', 'range', 'row',
                 'genotype', 'nitrogenTreatment', 'split']
    feature_cols = [c for c in train_df.columns
                    if c not in drop_cols + ['yieldPerAcre']]

    def prep_xy(df):
        return (np.ascontiguousarray(df[feature_cols].values.astype(np.float32)),
                np.ascontiguousarray(df['yieldPerAcre'].values.astype(np.float32)))

    X_train, y_train = prep_xy(train_df)
    X_val,   y_val   = prep_xy(val_df)
    X_test,  y_test  = prep_xy(test_df)

    # ── Impute and select on train only (leakage fix) ──────────────────────
    imputer = SimpleImputer(strategy='median')
    X_train = imputer.fit_transform(X_train)   # fit on train only
    X_val   = imputer.transform(X_val)
    X_test  = imputer.transform(X_test)

    from sklearn.feature_selection import VarianceThreshold
    selector = VarianceThreshold(threshold=0.01)
    X_train = selector.fit_transform(X_train)  # fit on train only
    X_val   = selector.transform(X_val)
    X_test  = selector.transform(X_test)

    feature_cols_filtered = [f for f, keep in zip(feature_cols, selector.get_support()) if keep]
    n_patch = len([f for f in feature_cols_filtered if 'patch' in f])
    n_delta = len([f for f in feature_cols_filtered if f.startswith('delta_') or 'mean_NDVI' in f])
    n_agro  = len([f for f in feature_cols_filtered if 'patch' not in f
                   and not f.startswith('delta_') and 'mean_NDVI' not in f])
    print(f"  Features after variance filter: {len(feature_cols_filtered)} / {len(feature_cols)}")
    print(f"    → {n_patch} patch/image features")
    print(f"    → {n_delta} temporal delta features")
    print(f"    → {n_agro} agronomic features")

    # ── Train XGBoost ──
    print(f"\nTraining XGBoost ({n_estimators} trees)...")
    print(f"  max_depth={max_depth}, min_samples_leaf={min_samples_leaf}, max_features={max_features}")
    from xgboost import XGBRegressor
    model = XGBRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=max_features,
        min_child_weight=min_samples_leaf,
        random_state=42,
        n_jobs=n_jobs,
        verbosity=0,
        reg_alpha=0.1,
        reg_lambda=2.0
    )
    model.fit(X_train, y_train)
    print(f"  Done.")

    # ── 5-fold cross-validation on train set to check stability ──
    from sklearn.model_selection import cross_val_score
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='r2', n_jobs=-1)
    print(f"\n  5-fold CV on train: R² = {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
    print(f"  (if this is much lower than train R², the model is still overfitting)")

    # ── Evaluate on all three splits ──
    y_pred_train = model.predict(X_train)
    y_pred_val   = model.predict(X_val)
    y_pred_test  = model.predict(X_test)

    def print_metrics(label, y_true, y_pred):
        r2   = r2_score(y_true, y_pred)
        mae  = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        bias = np.mean(y_pred - y_true)
        print(f"\n  === {label} ===")
        print(f"  R²   : {r2:.3f}      (1.0 = perfect, 0 = no better than predicting the mean)")
        print(f"  MAE  : {mae:.2f} bu/acre  (average absolute error per plot)")
        print(f"  RMSE : {rmse:.2f} bu/acre  (penalises large errors more than MAE)")
        print(f"  MAPE : {mape:.1f}%         (average % error relative to actual yield)")
        print(f"  Bias : {bias:+.2f} bu/acre  (positive = over-predicts, negative = under-predicts)")
        return r2, mae, rmse

    print("\n========================================")
    print("       MODEL ACCURACY REPORT")
    print("========================================")
    r2_train, _, _ = print_metrics("TRAIN SET", y_train, y_pred_train)
    r2_val,   _, _ = print_metrics("VALIDATION", y_val, y_pred_val)
    r2_test,  _, _ = print_metrics("TEST SET  (true accuracy — never seen during training)", y_test, y_pred_test)

    gap = r2_train - r2_test
    print(f"\n  Overfit gap (train R² - test R²): {gap:.3f}")
    if gap > 0.2:
        print("  ⚠ Gap > 0.2 — model is overfitting. Try increasing max_depth limit,")
        print("    min_samples_leaf, or reducing GRID_SIZE to shrink the feature count.")
    elif gap > 0.1:
        print("  ⚠ Mild overfitting. Consider tuning further.")
    else:
        print("  ✓ Train/test gap looks reasonable.")

    # ── Feature importances — top 20 ──
    importance_df = pd.DataFrame({
        'feature':    feature_cols_filtered,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False).head(20)

    print("\n  === Top 20 Most Important Features ===")
    print(importance_df.to_string(index=False))

    # ── Save model ──
    joblib.dump({'model': model, 'imputer': imputer, 'selector': selector,
                 'feature_cols': feature_cols_filtered},
                'best_model.pkl')
    print("\nModel saved → best_model.pkl")

    # ── Plot 1: Predicted vs Actual (3 panels) ──
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, (label, y_true, y_pred, color) in zip(axes, [
        ('Train',      y_train, y_pred_train, 'steelblue'),
        ('Validation', y_val,   y_pred_val,   'darkorange'),
        ('Test',       y_test,  y_pred_test,  'green'),
    ]):
        r2  = r2_score(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        ax.scatter(y_true, y_pred, alpha=0.6, color=color, edgecolors='k', linewidths=0.4)
        mn, mx = np.concatenate([y_train, y_val, y_test]).min(), np.concatenate([y_train, y_val, y_test]).max()
        ax.plot([mn, mx], [mn, mx], 'r--', label='Perfect fit')
        ax.set_xlabel('Actual Yield (bu/acre)')
        ax.set_ylabel('Predicted Yield (bu/acre)')
        ax.set_title(f'{label}\nR²={r2:.3f}  MAE={mae:.1f} bu/acre')
        ax.legend(fontsize=8)
    plt.suptitle(f'XGBoost — Patch grid {GRID_SIZE}×{GRID_SIZE} — Predicted vs Actual', fontsize=13)
    plt.tight_layout()
    plt.savefig('predicted_vs_actual.png', dpi=150)
    plt.show()
    print("Saved predicted_vs_actual.png")

    # ── Plot 2: Feature importances ──
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.barh(importance_df['feature'][::-1], importance_df['importance'][::-1], color='steelblue')
    ax.set_xlabel('Importance')
    ax.set_title(f'Top 20 Feature Importances (XGBoost)')
    plt.tight_layout()
    plt.savefig('feature_importances.png', dpi=150)
    plt.show()
    print("Saved feature_importances.png")

    # ── Save model ──
    joblib.dump({'model': model, 'imputer': imputer, 'selector': selector,
                 'feature_cols': feature_cols_filtered},
                'best_model.pkl')
    print("\nModel saved → best_model.pkl")

    # Save genotype priors (train-only) for the inference pipeline
    train_geno_mean.to_csv('historical_genotype_yields.csv')
    print("Saved historical_genotype_yields.csv (train-only genotype priors)")

    # ── Save full predictions with metadata for scoring engine ──
    METADATA_COLS = ['location', 'range', 'row', 'experiment',
                     'genotype', 'nitrogenTreatment', 'timepoint']
    meta_cols_present = [c for c in METADATA_COLS if c in train_df.columns]

    def build_predictions_df(df, y_true, y_pred, split_name):
        df = df.reset_index(drop=True)
        out = df[meta_cols_present].copy()
        out['actual_yield']    = y_true
        out['predicted_yield'] = y_pred
        out['error']           = y_pred - y_true
        out['absolute_error']  = np.abs(y_pred - y_true)
        out['pct_error']       = np.where(
            y_true != 0, np.abs((y_true - y_pred) / y_true) * 100, np.nan
        )
        out['split'] = split_name
        return out

    all_preds = pd.concat([
        build_predictions_df(train_df, y_train, y_pred_train, 'train'),
        build_predictions_df(val_df,   y_val,   y_pred_val,   'val'),
        build_predictions_df(test_df,  y_test,  y_pred_test,  'test'),
    ], ignore_index=True)

    all_preds.to_csv('xgb_predictions_full.csv', index=False)
    print(f"Saved xgb_predictions_full.csv")
    print(f"  Columns : {all_preds.columns.tolist()}")
    print(f"  Rows    : {len(all_preds)} ({all_preds['split'].value_counts().to_dict()})")

    return model


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

GRID_SIZE = 4   # change to 3, 5, 6, 8 etc to try different patch granularities

if __name__ == '__main__':

    GROUNDTRUTH_CSV = '../GroundTruth/train_HIPS_HYBRIDS_2023_V2.3.csv'
    SATELLITE_ROOT  = '../Satellite'
    UAV_ROOT        = '../UAV'

    print("Loading ground truth...")
    groundtruth = pd.read_csv(GROUNDTRUTH_CSV)
    print(f"  {len(groundtruth)} plots loaded")

    print(f"\nProcessing satellite images (patch grid: {GRID_SIZE}×{GRID_SIZE})...")
    sat_df = collect_all_satellite(SATELLITE_ROOT, grid_size=GRID_SIZE)
    print(f"  {len(sat_df)} satellite images processed")
    if not sat_df.empty:
        print(f"  Features per image: {sat_df.shape[1] - 1} patch columns")

    if os.path.isdir(UAV_ROOT):
        print(f"\nProcessing UAV images (patch grid: {GRID_SIZE}×{GRID_SIZE})...")
        rgb_df = collect_all_rgb(UAV_ROOT, grid_size=GRID_SIZE)
        print(f"  {len(rgb_df)} UAV images processed")
    else:
        print(f"\nNo UAV folder found — skipping.")
        rgb_df = pd.DataFrame()

    # Combine
    if not sat_df.empty and not rgb_df.empty:
        sat_df = sat_df.rename(columns={'Imagename': 'Imagename_sat'})
        rgb_df = rgb_df.rename(columns={'Imagename': 'Imagename_rgb'})
        sat_df['join_key'] = sat_df['Imagename_sat'].apply(lambda x: os.path.splitext(x)[0])
        rgb_df['join_key'] = rgb_df['Imagename_rgb'].apply(lambda x: os.path.splitext(x)[0])
        features_df = sat_df.merge(rgb_df, on='join_key', how='outer')
        features_df['Imagename'] = features_df['Imagename_sat'].fillna(features_df['Imagename_rgb'])
    elif not sat_df.empty:
        features_df = sat_df
    elif not rgb_df.empty:
        features_df = rgb_df
    else:
        print("No image data found.")
        exit()

    print("\nRunning Random Forest with patch features...")
    model = run_random_forest(features_df, groundtruth,
                              n_estimators=300,
                              max_depth=6,
                              min_samples_leaf=20, 
                              max_features=0.2)