import os
import glob
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import rasterio
import datetime # NEW
import torch.nn.functional as F

import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from xgboost import XGBRegressor
import joblib

warnings.filterwarnings("ignore", message="invalid value encountered in scalar divide")
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# ─────────────────────────────────────────────
# 1. PYTORCH DATASET (CAE PREP)
# ─────────────────────────────────────────────

class CAESatelliteDataset(Dataset):
    def __init__(self, groundtruth_df, satellite_dir, grid_size=(1, 2)):
        self.grid_rows, self.grid_cols = grid_size
        self.satellite_dir = satellite_dir
        
        # Gather TIFs
        all_tifs = glob.glob(os.path.join(satellite_dir, '**', '*.TIF'), recursive=True)
        all_tifs += glob.glob(os.path.join(satellite_dir, '**', '*.tif'), recursive=True)
        
        # Parse filenames
        parsed_data = []
        for filepath in all_tifs:
            basename = os.path.basename(filepath)
            try:
                name_no_ext = os.path.splitext(basename)[0]
                parts = name_no_ext.split('-')
                location, timepoint = parts[0], parts[1]
                rest = parts[2].split('_')
                experiment, rangeno, row = rest[0], str(rest[1]), str(rest[2])
                
                parsed_data.append({
                    'filepath': filepath, 'location': location, 'timepoint': timepoint,
                    'experiment': experiment, 'range': rangeno, 'row': row
                })
            except Exception:
                continue
                
        images_df = pd.DataFrame(parsed_data)
        
        # Standardize and Merge
        groundtruth_df['range'] = groundtruth_df['range'].astype(str)
        groundtruth_df['row'] = groundtruth_df['row'].astype(str)
        groundtruth_df['experiment'] = groundtruth_df['experiment'].astype(str)
        
        self.data = pd.merge(
            images_df, groundtruth_df, 
            on=['location', 'experiment', 'range', 'row'], how='inner'
        )
        
        # Create GxE index
        self.data['environment_str'] = self.data['location'] + "_" + self.data['nitrogenTreatment'].astype(str)
        
        # Encode for CAE Correlation Loss
        self.geno_encoder = LabelEncoder()
        self.env_encoder = LabelEncoder()
        self.data['genotype_idx'] = self.geno_encoder.fit_transform(self.data['genotype'].astype(str))
        self.data['environment_idx'] = self.env_encoder.fit_transform(self.data['environment_str'])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row_data = self.data.iloc[idx]
        filepath = row_data['filepath']
        
        geno_id = torch.tensor(row_data['genotype_idx'], dtype=torch.long)
        env_id = torch.tensor(row_data['environment_idx'], dtype=torch.long)
        yield_val = torch.tensor(row_data['yieldPerAcre'], dtype=torch.float32)

        with rasterio.open(filepath, 'r') as src:
            bands = src.read().astype(np.float32)
            
        for b in range(bands.shape[0]):
            b_min, b_max = np.nanmin(bands[b]), np.nanmax(bands[b])
            if b_max > b_min:
                bands[b] = (bands[b] - b_min) / (b_max - b_min)
            else:
                bands[b] = 0.0
                
        # ... [keep your existing rasterio loading and NaN-filling code] ...
        bands = np.nan_to_num(bands, nan=0.0)

        # 1. Convert numpy array to tensor
        bands_tensor = torch.tensor(bands, dtype=torch.float32)

        # 2. NEW: Standardize spatial size to fix the DataLoader stacking error
        # Add a fake batch dim (1, C, H, W) for interpolate, resize to 24x12, then remove it
        bands_tensor = bands_tensor.unsqueeze(0)
        bands_tensor = F.interpolate(bands_tensor, size=(24, 12), mode='bilinear', align_corners=False)
        bands_tensor = bands_tensor.squeeze(0)

        # 3. Slice the newly standardized tensor into the 1x2 grid
        C, H, W = bands_tensor.shape
        ph, pw = H // self.grid_rows, W // self.grid_cols
        
        patches = []
        for pi in range(self.grid_rows):
            for pj in range(self.grid_cols):
                patch = bands_tensor[:, pi*ph:(pi+1)*ph, pj*pw:(pj+1)*pw]
                patches.append(patch)
                
        # 4. Stack into a single tensor: Shape will now consistently be (2, 6, 24, 6)
        patches_tensor = torch.stack(patches)

        return patches_tensor, geno_id, env_id, yield_val, idx


def cae_collate_fn(batch):
    """
    Flattens the (B, 2, 6, H, W) patches into (B*2, 6, H, W) for the encoder.
    """
    patches, geno_ids, env_ids, yields, indices = zip(*batch)
    
    # patches is a tuple of B tensors, each shape (2, 6, H, W)
    # Stacking gives (B, 2, 6, H, W). Reshaping gives (B*2, 6, H, W)
    stacked_patches = torch.stack(patches)
    B, num_patches, C, H, W = stacked_patches.shape
    flattened_patches = stacked_patches.view(B * num_patches, C, H, W)
    
    return {
        'patches': flattened_patches,
        'geno_ids': torch.stack(geno_ids),
        'env_ids': torch.stack(env_ids),
        'yields': torch.stack(yields),
        'indices': torch.tensor(indices, dtype=torch.long),
        'batch_size': B,
        'num_patches': num_patches
    }


# ─────────────────────────────────────────────
# 2. FEATURE EXTRACTION PIPELINE
# ─────────────────────────────────────────────

def extract_cae_features(dataset, cae_model, device='cuda', batch_size=16):
    """
    Passes the dataset through the trained CAE encoder and returns 
    a pandas DataFrame containing the disentangled latent vectors.
    """
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=cae_collate_fn)
    cae_model.eval()
    cae_model.to(device)
    
    all_latents = []
    all_indices = []
    
    print("\nExtracting features using Compositional Autoencoder...")
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            inputs = batch['patches'].to(device)
            B = batch['batch_size']
            num_patches = batch['num_patches']
            
            # Forward pass through encoder (Assumes CAE returns a concatenated latent vector)
            # Adjust this based on the exact output of the Baskar Group CAE
            latent_vectors = cae_model.encoder(inputs) # Shape: (B*2, latent_dim)
            
            # Reshape back to (B, 2, latent_dim) and average across the 2 patches
            latent_dim = latent_vectors.shape[1]
            reshaped_latents = latent_vectors.view(B, num_patches, latent_dim)
            plot_latents = reshaped_latents.mean(dim=1) # Shape: (B, latent_dim)
            
            all_latents.append(plot_latents.cpu().numpy())
            all_indices.extend(batch['indices'].numpy())
            
            if (i+1) % 10 == 0:
                print(f"  Processed batch {i+1}/{len(dataloader)}")
                
    # Create DataFrame of latents
    latents_matrix = np.vstack(all_latents)
    latent_cols = [f'latent_{i}' for i in range(latents_matrix.shape[1])]
    latents_df = pd.DataFrame(latents_matrix, columns=latent_cols)
    latents_df['data_index'] = all_indices
    
    # Merge latents back into the original dataset dataframe
    dataset.data['data_index'] = np.arange(len(dataset.data))
    final_df = pd.merge(dataset.data, latents_df, on='data_index').drop(columns=['data_index'])
    
    print(f"Extraction complete. Discovered {len(latent_cols)} latent features.")
    return final_df, latent_cols


# ─────────────────────────────────────────────
# 3. XGBOOST MODEL
# ─────────────────────────────────────────────

def run_xgboost_on_latents(features_df, latent_cols,
                           n_estimators=300, max_depth=15,
                           min_samples_leaf=10, max_features=0.3, 
                           run_id=None):

    # Generate a unique ID if one wasn't provided
    if run_id is None:
        run_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    # ── Encode Categories ──
    features_df['nitrogenTreatment_enc'] = features_df['nitrogenTreatment'].map({'Low': 0, 'Medium': 1, 'High': 2})
    features_df['location_enc'] = (features_df['location'] == 'Lincoln').astype(int)
    
    # Genotype specific mean
    geno_mean = features_df.groupby('genotype')['yieldPerAcre'].mean().rename('genotype_mean_yield')
    features_df = features_df.join(geno_mean, on='genotype')
    
    # GxE specific mean
    gxe_mean = features_df.groupby(['genotype', 'environment_str'])['yieldPerAcre'].mean().rename('gxe_mean_yield')
    features_df = features_df.join(gxe_mean, on=['genotype', 'environment_str'])

    AGRONOMIC_FEATURES = [
        'nitrogenTreatment_enc', 'GDDToAnthesis', 'daysToAnthesis',
        'genotype_idx', 'environment_idx', 'genotype_mean_yield', 
        'gxe_mean_yield', 'location_enc', 'block'
    ]

    feature_cols = latent_cols + AGRONOMIC_FEATURES
    merged = features_df.dropna(subset=['yieldPerAcre'] + AGRONOMIC_FEATURES)

    # Force contiguous memory to prevent C++ backend segfaults
    X = np.ascontiguousarray(merged[feature_cols].values.astype(np.float32))
    y = np.ascontiguousarray(merged['yieldPerAcre'].values.astype(np.float32))

    imputer = SimpleImputer(strategy='median')
    X = imputer.fit_transform(X)

    # ── Stratified Split ──
    y_bins = pd.qcut(y, q=4, labels=False)
    X_train, X_temp, y_train, y_temp, bins_train, bins_temp = train_test_split(
        X, y, y_bins, test_size=0.30, random_state=42, stratify=y_bins
    )
    X_val, X_test, y_val, y_test, _, _ = train_test_split(
        X_temp, y_temp, bins_temp, test_size=0.50, random_state=42, stratify=bins_temp
    )

    print(f"\nTraining XGBoost on {X_train.shape[1]} features (Latents + Agronomic)...")
    model = XGBRegressor(
        n_estimators=n_estimators, max_depth=max_depth, learning_rate=0.05,
        subsample=0.8, colsample_bytree=max_features, min_child_weight=min_samples_leaf,
        random_state=42, n_jobs=1, verbosity=0, reg_alpha=0.1, reg_lambda=2.0
    )
    model.fit(X_train, y_train)

    # ── Evaluation ──
    y_pred_test  = model.predict(X_test)
    r2_test = r2_score(y_test, y_pred_test)
    mae_test = mean_absolute_error(y_test, y_pred_test)

    print("\n  === TEST SET ===")
    print(f"  R²   : {r2_test:.3f}")
    print(f"  MAE  : {mae_test:.2f} bu/acre")
    print(f"  RMSE : {np.sqrt(mean_squared_error(y_test, y_pred_test)):.2f} bu/acre")

    # ── Plot 1: Predicted vs Actual (Error Plot) ──
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(y_test, y_pred_test, alpha=0.6, color='steelblue', edgecolors='k')
    
    # Draw the perfect fit line
    mn, mx = min(y_test.min(), y_pred_test.min()), max(y_test.max(), y_pred_test.max())
    ax.plot([mn, mx], [mn, mx], 'r--', label='Perfect Prediction')
    
    ax.set_xlabel('Actual Yield (bu/acre)')
    ax.set_ylabel('Predicted Yield (bu/acre)')
    ax.set_title(f'CAE + XGBoost Error | Run: {run_id}\nR² = {r2_test:.3f} | MAE = {mae_test:.1f}')
    ax.legend()
    plt.tight_layout()
    
    plot_filename = f'./data/error_plot_{run_id}.png'
    plt.savefig(plot_filename, dpi=150)
    plt.close() # Close to free memory
    print(f"\nSaved error plot → {plot_filename}")

    # ── Feature Importances ──
    importance_df = pd.DataFrame({
        'feature': feature_cols,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False).head(15)

    print("\n  === Top 15 Features ===")
    print(importance_df.to_string(index=False))

    # ── Save Model ──
    model_filename = f'./data/best_cae_xgb_model_{run_id}.pkl'
    joblib.dump({'model': model, 'imputer': imputer, 'feature_cols': feature_cols, 'run_id': run_id}, model_filename)
    print(f"Saved model → {model_filename}")

    # ── Save Predictions to CSV ──
    predictions_df = pd.DataFrame({
        'Actual_Yield': y_test,
        'Predicted_Yield': y_pred_test,
        'Error': y_pred_test - y_test,
        'Absolute_Error': np.abs(y_pred_test - y_test),
        'Pct_Error': np.abs((y_test - y_pred_test) / y_test) * 100
    })
    
    csv_filename = f'./data/predictions_{run_id}.csv'
    predictions_df.to_csv(csv_filename, index=False)
    print(f"Saved test predictions → {csv_filename}")

    return model


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

if __name__ == '__main__':
    GROUNDTRUTH_CSV = '../GroundTruth/train_HIPS_HYBRIDS_2023_V2.3.csv'
    SATELLITE_ROOT  = '../Satellite'
    GRID_SIZE       = (1, 2)

    print("Loading Dataset...")
    groundtruth = pd.read_csv(GROUNDTRUTH_CSV)
    
    # 1. Initialize the Dataset
    dataset = CAESatelliteDataset(groundtruth, SATELLITE_ROOT, grid_size=GRID_SIZE)
    print(f"Initialized Dataset with {len(dataset)} valid plot images.")

    # 2. Load the pre-trained CAE Model 
    # TODO: Initialize the actual CAE from the Baskar Group repo here
    import torch.nn as nn
    class DummyCAE(nn.Module):
        def __init__(self):
            super().__init__()
            # Dummy encoder that flattens and downsamples to 32 latent features
            self.encoder = nn.Sequential(
                nn.Conv2d(6, 16, kernel_size=3, stride=2, padding=1),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten() # Outputs (B, 16)
            )
    
    cae_model = DummyCAE()
    # cae_model.load_state_dict(torch.load('path_to_baskar_weights.pth'))
    
    # 3. Extract Latents
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    features_df, latent_cols = extract_cae_features(dataset, cae_model, device=device)

    # 4. Train XGBoost
    model = run_xgboost_on_latents(features_df, latent_cols, max_depth=6, min_samples_leaf=20, max_features=0.5)