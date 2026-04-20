import pandas as pd
import matplotlib.pyplot as plt
import cv2 as cv
import rasterio
import os
import glob
import warnings
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error

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
# PYTORCH LINEAR REGRESSION MODEL
# ─────────────────────────────────────────────

class LinearRegressionModel(nn.Module):
    def __init__(self, input_dim):
        super(LinearRegressionModel, self).__init__()
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, x):
        return self.linear(x)


def run_pytorch_regression(features_df, groundtruth_df,
                            epochs=500, lr=0.01, batch_size=32):

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

    merged = features_df.merge(
        groundtruth_df[['location', 'range', 'row', 'experiment', 'yieldPerAcre']],
        on=['location', 'range', 'row', 'experiment'],
        how='inner'
    )

    print(f"\nMerged dataset: {len(merged)} rows")
    print(f"Total features: {len([c for c in merged.columns if 'patch' in c])}")
    if len(merged) == 0:
        print("No matching rows after merge.")
        return

    # ── Prepare X and y ──
    drop_cols = [c for c in merged.columns if 'Imagename' in c] + \
                ['location', 'timepoint', 'experiment', 'range', 'row']
    feature_cols = [c for c in merged.columns
                    if c not in drop_cols + ['yieldPerAcre']]

    X = merged[feature_cols].values.astype(np.float32)
    y = merged['yieldPerAcre'].values.astype(np.float32)

    # ── Impute and scale ──
    imputer = SimpleImputer(strategy='median')
    X = imputer.fit_transform(X)

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # ── 70 / 15 / 15 split ──
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.30, random_state=42)
    X_val,   X_test, y_val,   y_test = train_test_split(X_temp, y_temp, test_size=0.50, random_state=42)

    print(f"\nData split:")
    print(f"  Train      : {len(X_train)} samples ({len(X_train)/len(X)*100:.0f}%)")
    print(f"  Validation : {len(X_val)} samples ({len(X_val)/len(X)*100:.0f}%)")
    print(f"  Test       : {len(X_test)} samples ({len(X_test)/len(X)*100:.0f}%)")

    # ── Tensors and DataLoader ──
    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
    X_val_t   = torch.tensor(X_val,   dtype=torch.float32)
    y_val_t   = torch.tensor(y_val,   dtype=torch.float32).unsqueeze(1)
    X_test_t  = torch.tensor(X_test,  dtype=torch.float32)

    train_loader = DataLoader(TensorDataset(X_train_t, y_train_t),
                              batch_size=batch_size, shuffle=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")

    model     = LinearRegressionModel(input_dim=X_train.shape[1]).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # ── Training loop ──
    print(f"\nTraining for {epochs} epochs...")
    train_losses, val_losses = [], []
    best_val_loss, best_state = float('inf'), None

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            loss = criterion(model(X_batch), y_batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        avg_train = epoch_loss / len(train_loader)
        train_losses.append(avg_train)

        model.eval()
        with torch.no_grad():
            val_loss = criterion(model(X_val_t.to(device)), y_val_t.to(device)).item()
        val_losses.append(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

        if (epoch + 1) % 50 == 0:
            print(f"  Epoch [{epoch+1}/{epochs}]  Train: {avg_train:.4f}  Val: {val_loss:.4f}")

    model.load_state_dict(best_state)
    torch.save({'model_state_dict': best_state,
                'input_dim': X_train.shape[1],
                'feature_cols': feature_cols,
                'imputer': imputer,
                'scaler': scaler}, 'best_model.pt')
    print(f"\nBest model saved (val loss: {best_val_loss:.4f}) → best_model.pt")

    # ── Evaluate ──
    model.eval()
    with torch.no_grad():
        y_pred_train = model(X_train_t.to(device)).cpu().numpy().flatten()
        y_pred_val   = model(X_val_t.to(device)).cpu().numpy().flatten()
        y_pred_test  = model(X_test_t.to(device)).cpu().numpy().flatten()

    def print_metrics(label, y_true, y_pred):
        r2   = r2_score(y_true, y_pred)
        mae  = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        bias = np.mean(y_pred - y_true)
        print(f"\n  === {label} ===")
        print(f"  R²   : {r2:.3f}")
        print(f"  MAE  : {mae:.2f} bu/acre")
        print(f"  RMSE : {rmse:.2f} bu/acre")
        print(f"  MAPE : {mape:.1f}%")
        print(f"  Bias : {bias:+.2f} bu/acre")
        return r2, mae, rmse

    print("\n========================================")
    print("       MODEL ACCURACY REPORT")
    print("========================================")
    print_metrics("TRAIN SET", y_train, y_pred_train)
    print_metrics("VALIDATION", y_val, y_pred_val)
    print_metrics("TEST SET  (true accuracy)", y_test, y_pred_test)

    # ── Plots ──
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, (label, y_true, y_pred, color) in zip(axes, [
        ('Train',      y_train, y_pred_train, 'steelblue'),
        ('Validation', y_val,   y_pred_val,   'darkorange'),
        ('Test',       y_test,  y_pred_test,  'green'),
    ]):
        r2  = r2_score(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        ax.scatter(y_true, y_pred, alpha=0.6, color=color, edgecolors='k', linewidths=0.4)
        mn, mx = y.min(), y.max()
        ax.plot([mn, mx], [mn, mx], 'r--')
        ax.set_xlabel('Actual Yield (bu/acre)')
        ax.set_ylabel('Predicted Yield (bu/acre)')
        ax.set_title(f'{label}  R²={r2:.3f}  MAE={mae:.1f}')
    plt.suptitle(f'Patch-based features ({GRID_SIZE}×{GRID_SIZE} grid) — Predicted vs Actual', fontsize=13)
    plt.tight_layout()
    plt.savefig('predicted_vs_actual.png', dpi=150)
    plt.show()

    plt.figure(figsize=(9, 4))
    plt.plot(train_losses, label='Train loss')
    plt.plot(val_losses,   label='Val loss')
    plt.xlabel('Epoch')
    plt.ylabel('MSE loss')
    plt.title('Training vs Validation Loss')
    plt.legend()
    plt.tight_layout()
    plt.savefig('training_loss.png', dpi=150)
    plt.show()

    pd.DataFrame({'actual': y_test, 'predicted': y_pred_test,
                  'error': y_pred_test - y_test,
                  'absolute_error': np.abs(y_pred_test - y_test),
                  'pct_error': np.abs((y_test - y_pred_test) / y_test) * 100}
                 ).to_csv('pytorch_predictions.csv', index=False)
    print("Saved pytorch_predictions.csv")

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

    print("\nRunning PyTorch regression with patch features...")
    model = run_pytorch_regression(features_df, groundtruth,
                                   epochs=500, lr=0.01, batch_size=32)