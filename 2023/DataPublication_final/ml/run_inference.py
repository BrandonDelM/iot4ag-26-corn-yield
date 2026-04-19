import os
import zipfile
import shutil
import joblib
import pandas as pd
import numpy as np
import warnings
import patch_features 
import scoring_engine

# Ignore performance warnings during heavy dataframe manipulation
warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)

def process_zip_upload(zip_filepath, metadata_csv, model_pkl_path):
    extract_dir = './temp_unzipped_images'
    
    # 1. Clean and Prepare extraction directory
    if os.path.exists(extract_dir):
        shutil.rmtree(extract_dir)
    os.makedirs(extract_dir)

    print(f"1. Extracting {zip_filepath}...")
    with zipfile.ZipFile(zip_filepath, 'r') as zip_ref:
        zip_ref.extractall(extract_dir)

    # 2. Feature Extraction (Prioritize RGB, fallback to Multispectral TIF)
    print("\n2. Extracting patch features from drone images...")
    drone_features_df = patch_features.collect_all_rgb(extract_dir, grid_size=2)
    
    if drone_features_df.empty:
        print("  No JPG/PNG found. Checking for multispectral TIFs...")
        drone_features_df = patch_features.collect_all_satellite(extract_dir, grid_size=2)
        
    if drone_features_df.empty:
        print("❌ No valid images found in the zip file.")
        shutil.rmtree(extract_dir)
        return

    # 3. Parse Metadata from the Folder Structure
    print("\n3. Mapping images to Location and Timepoint folders...")
    path_meta = []
    for root, dirs, files in os.walk(extract_dir):
        for file in files:
            if file.lower().endswith(('.jpg', '.png', '.tif', '.tiff')):
                # Logic assumes structure: Satellite/Location/Timepoint/image.tif
                parts = root.replace('\\', '/').split('/')
                timepoint = parts[-1] if len(parts) >= 1 else 'Unknown'
                location = parts[-2] if len(parts) >= 2 else 'Unknown'

                # Extract Experiment_Range_Row from filename (e.g., ...-75_14_2.tif)
                name_no_ext = os.path.splitext(file)[0]
                try:
                    exp_parts = name_no_ext.split('-')[-1].split('_')
                    exp, rng, row = exp_parts[0], exp_parts[1], exp_parts[2]
                except (ValueError, IndexError):
                    exp, rng, row = "Unknown", "0", "0"

                path_meta.append({
                    'Imagename': file, 
                    'location': location, 
                    'timepoint': timepoint,
                    'experiment': str(exp), 
                    'range': str(rng), 
                    'row': str(row)
                })
                
    path_meta_df = pd.DataFrame(path_meta)
    drone_features_df = pd.merge(drone_features_df, path_meta_df, on='Imagename', how='inner')

    # 4. Merge with Agronomic Metadata (e.g., train_HIPS_HYBRIDS_2023_V2.3.csv)
    print("\n4. Synchronizing with metadata CSV...")
    metadata_df = pd.read_csv(metadata_csv)
    
    # Standardize join keys to strings and remove any hidden whitespace/decimals
    for col in ['location', 'experiment', 'range', 'row']:
        metadata_df[col] = metadata_df[col].astype(str).str.replace('.0', '', regex=False).str.strip()
        drone_features_df[col] = drone_features_df[col].astype(str).str.replace('.0', '', regex=False).str.strip()

    merged = pd.merge(drone_features_df, metadata_df, on=['location', 'experiment', 'range', 'row'], how='inner')

    if merged.empty:
        print("❌ 0 plots matched. Check if ZIP folder names match CSV location names exactly.")
        return

    # 5. XGBoost Inference
    print(f"\n5. Prepping data for XGBoost (Matched: {len(merged)} plots)...")
    artifacts = joblib.load(model_pkl_path)
    xgb_model = artifacts['model']
    final_features = artifacts['feature_cols']

    # Target Leakage Protection: Remove the ground truth from the input features
    # (The model shouldn't see 'yieldPerAcre' while trying to predict it)
    forbidden = ['actual_yield', 'yieldPerAcre', 'predicted_yield']
    model_inputs = [c for c in final_features if c not in forbidden]

    # Handle missing columns that the model expects
    missing_cols = [c for c in model_inputs if c not in merged.columns]
    if missing_cols:
        zero_data = pd.DataFrame(0.0, index=merged.index, columns=missing_cols)
        merged = pd.concat([merged, zero_data], axis=1)

    X = np.ascontiguousarray(merged[model_inputs].fillna(0).values.astype(np.float32))
    merged['predicted_yield'] = xgb_model.predict(X)
    
    # Map actual yield if present in CSV, else set to NaN
    merged['actual_yield'] = merged['yieldPerAcre'] if 'yieldPerAcre' in merged.columns else np.nan

    # Save master prediction file WITHOUT row index
    merged.to_csv('xgb_predictions_full.csv', index=False)
    
    # 6. Generate Dashboard Artifacts
    print("\n6. Running Scoring Engine...")
    os.makedirs('./scoring_outputs', exist_ok=True)
    
    # We save these specifically for the dashboard, using index=False to keep filters clean
    scoring_engine.compute_hybrid_rankings(merged).to_csv('./scoring_outputs/hybrid_rankings.csv', index=False)
    scoring_engine.compute_gxe_matrix(merged).to_csv('./scoring_outputs/gxe_matrix.csv', index=False)
    scoring_engine.compute_timepoint_rankings(merged).to_csv('./scoring_outputs/timepoint_rankings.csv', index=False)
    
    shutil.rmtree(extract_dir)
    print("\n✅ Pipeline Complete. Dashboard data updated.")

if __name__ == '__main__':
    # Default test run
    process_zip_upload('temp_upload.zip', 'train_HIPS_HYBRIDS_2023_V2.3.csv', 'best_model_2022_UAV.pkl')