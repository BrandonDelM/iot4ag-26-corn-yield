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
                      n_jobs=1): # Set to 1 to avoid C++ threading segfaults

    from sklearn.preprocessing import LabelEncoder
    from sklearn.impute import SimpleImputer
    from sklearn.feature_selection import VarianceThreshold
    from sklearn.model_selection import train_test_split
    from xgboost import XGBRegressor
    import numpy as np

    # ── 1. Merge features with ground truth ──
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

    # Encode basic categoricals (No leakage risk here)
    nitrogen_map = {'Low': 0, 'Medium': 1, 'High': 2}
    groundtruth_df['nitrogenTreatment_enc'] = groundtruth_df['nitrogenTreatment'].map(nitrogen_map)
    
    le = LabelEncoder()
    groundtruth_df['genotype_enc'] = le.fit_transform(groundtruth_df['genotype'].astype(str))
    groundtruth_df['location_enc'] = (groundtruth_df['location'] == 'Lincoln').astype(int)

    AGRONOMIC_FEATURES = [
        'nitrogenTreatment_enc', 'GDDToAnthesis', 'daysToAnthesis',
        'genotype_enc', 'location_enc', 'block'
    ]

    merged = features_df.merge(
        groundtruth_df[['location', 'range', 'row', 'experiment', 'yieldPerAcre',
                        'genotype', 'nitrogenTreatment'] + AGRONOMIC_FEATURES],
        on=['location', 'range', 'row', 'experiment'],
        how='inner'
    )
    
    merged = merged.dropna(subset=['yieldPerAcre'])
    if len(merged) == 0:
        print("No matching rows after merge.")
        return

    # ── 2. Build temporal delta features ──
    print("\nBuilding temporal delta features...")
    # NOTE: You need to ensure GRID_SIZE is defined globally or passed in
    GRID_SIZE = 2 
    index_names = ['NDVI', 'GNDVI', 'SAVI', 'GLI', 'NGRDI']
    patch_positions = [f'{pi}_{pj}' for pi in range(GRID_SIZE) for pj in range(GRID_SIZE)]

    merged['_plot_key'] = (merged['location'].astype(str) + '_' +
                           merged['range'].astype(str) + '_' +
                           merged['row'].astype(str) + '_' +
                           merged['experiment'].astype(str))

    sat_col2 = [c for c in merged.columns if 'Imagename' in c]
    if sat_col2:
        merged['timepoint'] = merged[sat_col2[0]].apply(
            lambda x: os.path.splitext(x)[0].split('-')[1] if isinstance(x, str) else None
        )
        tp_col = 'timepoint'
    else:
        tp_col = None

    temporal_rows = []
    if tp_col and merged[tp_col].nunique() > 1:
        for plot_key, plot_group in merged.groupby('_plot_key'):
            plot_group = plot_group.sort_values(tp_col)
            vals_list = plot_group[tp_col].tolist()
            delta_dict = {'_plot_key': plot_key}

            for idx_name in index_names:
                for patch_pos in patch_positions:
                    col = f'patch_{patch_pos}_{idx_name}_mean'
                    if col not in plot_group.columns:
                        continue
                    vals = plot_group[col].tolist()
                    for t in range(1, len(vals)):
                        delta_name = f'delta_{idx_name}_{patch_pos}_TP{t}to{t+1}'
                        delta_dict[delta_name] = vals[t] - vals[t-1]

            ndvi_patch_cols = [c for c in plot_group.columns if 'NDVI_mean' in c and 'patch' in c]
            for row_ in plot_group.itertuples():
                tp_label = getattr(row_, tp_col)
                plot_ndvi_vals = [getattr(row_, c.replace(' ', '_'), np.nan) for c in ndvi_patch_cols]
                delta_dict[f'mean_NDVI_{tp_label}'] = float(np.nanmean(plot_ndvi_vals))

            temporal_rows.append(delta_dict)

        if temporal_rows:
            temporal_df = pd.DataFrame(temporal_rows)
            merged = merged.merge(temporal_df, on='_plot_key', how='left')
            print(f"  Created temporal features for {len(temporal_rows)} plots.")

    merged = merged.drop(columns=['_plot_key'], errors='ignore')

    # ── 3. SPLIT BY PLOT (not by image row) ──────────────────────────────────
    # The core leakage fix. Each unique plot (location+range+row+experiment)
    # appears at multiple timepoints. Splitting at the image-row level lets
    # the model train on "plot X at TP2 → yield 142" and test on
    # "plot X at TP3 → yield 142" — the same target, just a different photo.
    # Fix: assign every timepoint of a given plot to the same split.
    print("\nSplitting by plot (not image row) to prevent timepoint leakage...")

    # Build one row per unique plot with its yield for stratification
    plot_key_cols = ['location', 'range', 'row', 'experiment']
    unique_plots = (merged[plot_key_cols + ['yieldPerAcre', 'genotype']]
                    .drop_duplicates(subset=plot_key_cols)
                    .reset_index(drop=True))

    # Stratify by yield quartile so low/high yield plots are balanced
    y_bins = pd.qcut(unique_plots['yieldPerAcre'], q=4, labels=False, duplicates='drop')

    train_plots, temp_plots = train_test_split(
        unique_plots, test_size=0.30, random_state=42, stratify=y_bins
    )
    temp_bins = pd.qcut(temp_plots['yieldPerAcre'], q=4, labels=False, duplicates='drop')
    val_plots, test_plots = train_test_split(
        temp_plots, test_size=0.50, random_state=42, stratify=temp_bins
    )

    # Tag every image row with its split using the plot key
    def tag_split(df, train_p, val_p, test_p, key_cols):
        train_keys = set(map(tuple, train_p[key_cols].values))
        val_keys   = set(map(tuple, val_p[key_cols].values))
        keys       = df[key_cols].apply(tuple, axis=1)
        split_col  = keys.map(lambda k: 'train' if k in train_keys
                               else ('val' if k in val_keys else 'test'))
        return split_col

    merged['split'] = tag_split(merged, train_plots, val_plots, test_plots, plot_key_cols)

    train_df = merged[merged['split'] == 'train'].copy()
    val_df   = merged[merged['split'] == 'val'].copy()
    test_df  = merged[merged['split'] == 'test'].copy()

    print(f"  Unique plots  — train: {train_df[plot_key_cols].drop_duplicates().shape[0]}"
          f" | val: {val_df[plot_key_cols].drop_duplicates().shape[0]}"
          f" | test: {test_df[plot_key_cols].drop_duplicates().shape[0]}")
    print(f"  Image rows    — train: {len(train_df)} | val: {len(val_df)} | test: {len(test_df)}")
    print(f"  Genotypes in test not in train: "
          f"{len(set(test_df['genotype']) - set(train_df['genotype']))}")

    drop_cols = [c for c in merged.columns if 'Imagename' in c] + \
                ['location', 'timepoint', 'experiment', 'range', 'row',
                 'genotype', 'nitrogenTreatment', 'yieldPerAcre', 'split']

    # ── 4. LEAKAGE FIX #2: Calculate Priors strictly from Train Data ──
    # Genotype Means
    train_geno_mean = train_df.groupby('genotype')['yieldPerAcre'].mean().rename('genotype_mean_yield')
    global_mean = train_df['yieldPerAcre'].mean() 
    
    # Map the training means onto all splits
    for df in [train_df, val_df, test_df]:
        df = df.join(train_geno_mean, on='genotype')
        df['genotype_mean_yield'] = df['genotype_mean_yield'].fillna(global_mean)

    # GxE Means
    for df in [train_df, val_df, test_df]:
        df['env_str'] = df['location'] + "_" + df['nitrogenTreatment'].astype(str)
        
    train_gxe_mean = train_df.groupby(['genotype', 'env_str'])['yieldPerAcre'].mean().rename('gxe_mean_yield')
    
    # Avoid SettingWithCopy warnings by reassigning
# Avoid SettingWithCopy warnings by reassigning
    def apply_gxe(df):
        # 1. Join the GxE means
        df = df.join(train_gxe_mean, on=['genotype', 'env_str'])
        
        # 2. Safely fill NaNs ONLY if genotype_mean_yield actually exists
        if 'genotype_mean_yield' in df.columns:
            df['gxe_mean_yield'] = df['gxe_mean_yield'].fillna(df['genotype_mean_yield'])
        else:
            # Fallback to the global mean if the column somehow got dropped
            df['gxe_mean_yield'] = df['gxe_mean_yield'].fillna(global_mean)
            
        return df

    train_df = apply_gxe(train_df)
    val_df = apply_gxe(val_df)
    test_df = apply_gxe(test_df)

    # Re-assemble feature columns safely 
    feature_cols = [c for c in train_df.columns if c not in drop_cols and c != 'env_str']

    # Convert to Numpy Arrays
    def prep_xy(df):
        return (np.ascontiguousarray(df[feature_cols].values.astype(np.float32)), 
                np.ascontiguousarray(df['yieldPerAcre'].values.astype(np.float32)))

    X_train, y_train = prep_xy(train_df)
    X_val, y_val = prep_xy(val_df)
    X_test, y_test = prep_xy(test_df)

    # ── 5. LEAKAGE FIX #3: Fit Imputer & Selector strictly on Train Data ──
    imputer = SimpleImputer(strategy='median')
    X_train = imputer.fit_transform(X_train)
    X_val = imputer.transform(X_val)
    X_test = imputer.transform(X_test)

    selector = VarianceThreshold(threshold=0.01)
    X_train = selector.fit_transform(X_train)
    X_val = selector.transform(X_val)
    X_test = selector.transform(X_test)
    
    feature_cols_filtered = [f for f, keep in zip(feature_cols, selector.get_support()) if keep]
    
    print(f"\nData split & prep complete:")
    print(f"  Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")
    print(f"  Active features: {len(feature_cols_filtered)}")

    # ── 6. Train XGBoost ──
    print(f"\nTraining XGBoost ({n_estimators} trees)...")
    model = XGBRegressor(
        n_estimators=n_estimators, max_depth=max_depth, learning_rate=0.05,
        subsample=0.8, colsample_bytree=max_features, min_child_weight=min_samples_leaf,
        random_state=42, n_jobs=n_jobs, verbosity=0
    )
    model.fit(X_train, y_train)

    # ── 7. Evaluate and Save ──
    y_pred_test  = model.predict(X_test)
    
    print("\n  === TRUE TEST SET METRICS ===")
    print(f"  R²   : {r2_score(y_test, y_pred_test):.3f}")
    print(f"  MAE  : {mean_absolute_error(y_test, y_pred_test):.2f} bu/acre")
    print(f"  RMSE : {np.sqrt(mean_squared_error(y_test, y_pred_test)):.2f} bu/acre")

    # Export everything needed for inference
    joblib.dump({
        'model': model, 
        'imputer': imputer, 
        'selector': selector,
        'feature_cols': feature_cols_filtered
    }, 'best_model.pkl')
    
    # Save the priors for the automated pipeline!
    train_geno_mean.to_csv('historical_genotype_yields.csv')
    print("Saved 'historical_genotype_yields.csv' for inference priors.")

    # ── Save full predictions with metadata for the scoring engine ──
    METADATA_COLS = ['location', 'range', 'row', 'experiment',
                     'genotype', 'nitrogenTreatment', 'timepoint']

    def save_split_predictions(df, X, y_true, y_pred, split_name):
        meta = df[METADATA_COLS].reset_index(drop=True)
        return pd.DataFrame({
            **{c: meta[c].values for c in METADATA_COLS},
            'actual_yield':    y_true,
            'predicted_yield': y_pred,
            'error':           y_pred - y_true,
            'absolute_error':  np.abs(y_pred - y_true),
            'pct_error':       np.abs((y_true - y_pred) / y_true) * 100,
            'split':           split_name
        })

    y_pred_train = model.predict(X_train)
    y_pred_val   = model.predict(X_val)

    all_preds = pd.concat([
        save_split_predictions(train_df.reset_index(drop=True), X_train, y_train, y_pred_train, 'train'),
        save_split_predictions(val_df.reset_index(drop=True),   X_val,   y_val,   y_pred_val,   'val'),
        save_split_predictions(test_df.reset_index(drop=True),  X_test,  y_test,  y_pred_test,  'test'),
    ], ignore_index=True)

    all_preds.to_csv('xgb_predictions_full.csv', index=False)
    print("Saved xgb_predictions_full.csv  (all splits + metadata)")
    
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
                              max_depth=10,          # tightened from 15
                              min_samples_leaf=15,   # tightened from 10
                              max_features=0.2)      # tightened from 0.3