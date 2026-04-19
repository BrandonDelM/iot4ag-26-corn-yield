"""Single source of path config for HybridScout."""
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = REPO_ROOT / "2023" / "DataPublication_final"

GROUND_TRUTH_CSV = DATA_ROOT / "GroundTruth" / "train_HIPS_HYBRIDS_2023_V2.3.csv"
SATELLITE_ROOT = DATA_ROOT / "Satellite"
UAV_ROOT = DATA_ROOT / "UAV"

ARTIFACTS = REPO_ROOT / "hybridscout" / "artifacts"
ARTIFACTS.mkdir(parents=True, exist_ok=True)

FEATURES_PARQUET = ARTIFACTS / "plots_features.parquet"
MODEL_PKL = ARTIFACTS / "model.pkl"
EXPLAINER_PKL = ARTIFACTS / "explainer.pkl"
METRICS_JSON = ARTIFACTS / "metrics.json"
OOF_CSV = ARTIFACTS / "xgb_predictions_full.csv"
FEATURE_IMPORTANCE_PNG = ARTIFACTS / "feature_importance.png"

HYBRID_RANKINGS_CSV = ARTIFACTS / "hybrid_rankings.csv"
GXE_MATRIX_CSV = ARTIFACTS / "gxe_matrix.csv"
TIMEPOINT_RANKINGS_CSV = ARTIFACTS / "timepoint_rankings.csv"
NITROGEN_RESPONSE_CSV = ARTIFACTS / "nitrogen_response.csv"

GRID_SIZE = 4
