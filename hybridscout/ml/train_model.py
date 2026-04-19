"""
Train corn yield XGBoost model with honest CV methodology.

CV-A: GroupKFold (5-fold) grouped by genotype — produces out-of-fold
      predictions. Matches Powadi et al. 2025 (Frontiers Plant Science).
CV-B: Cross-location hold-out — train Lincoln → test MOValley, reversed.

Target encoding (genotype_mean_yield) is computed on TRAIN ROWS ONLY per
fold/split, then mapped onto val rows, then fillna with global train mean.
No target leakage.

Final model retrained on all rows for the scoring engine + Streamlit.
SHAP TreeExplainer fitted on final model for the Plot Inspector page.

Run:  python -m hybridscout.ml.train_model
"""
from __future__ import annotations

import json
import warnings

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from scipy.stats import spearmanr
from sklearn.feature_selection import VarianceThreshold
from sklearn.ensemble import RandomForestRegressor
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold
from xgboost import XGBRegressor

from . import io_paths as P

warnings.filterwarnings("ignore", category=FutureWarning)

META_COLS = {
    "location", "experiment", "range", "row", "genotype", "nitrogenTreatment",
    "plantingDate", "timepoint", "Imagename",
}
TARGET = "yieldPerAcre"

XGB_PARAMS = dict(
    n_estimators=600,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=10,
    reg_alpha=0.1,
    reg_lambda=2.0,
    random_state=42,
    n_jobs=1,
    verbosity=0,
    missing=np.nan,
)


def _feature_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in META_COLS and c != TARGET]


def _prep_matrix(df: pd.DataFrame, imputer: IterativeImputer | None = None,
                 selector: VarianceThreshold | None = None):
    feat_cols = _feature_cols(df)
    X_raw = df[feat_cols].apply(pd.to_numeric, errors="coerce").values.astype(np.float32)

    if imputer is None:
        imputer = IterativeImputer(
            estimator=RandomForestRegressor(n_estimators=10, random_state=42),
            max_iter=10,
            random_state=42
        )
        X = imputer.fit_transform(X_raw)
    else:
        X = imputer.transform(X_raw)

    if selector is None:
        selector = VarianceThreshold(threshold=0.01)
        X = selector.fit_transform(X)
    else:
        X = selector.transform(X)

    kept_cols = [f for f, keep in zip(feat_cols, selector.get_support()) if keep]
    return X.astype(np.float32), imputer, selector, kept_cols


def _target_encode(train_df: pd.DataFrame, all_df: pd.DataFrame) -> np.ndarray:
    """Compute genotype mean yield on train rows only, map to every row of all_df."""
    train_means = train_df.groupby("genotype")[TARGET].mean()
    global_mean = float(train_df[TARGET].mean())
    return (all_df["genotype"].map(train_means)
            .fillna(global_mean)
            .values.astype(np.float32))


def _metrics(y_true, y_pred):
    return {
        "R2": float(r2_score(y_true, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "Spearman": float(spearmanr(y_true, y_pred).statistic),
        "n": int(len(y_true)),
    }


# ─────────────────────────────────────────────
# CV-A — 5-fold GroupKFold by genotype (OOF)
# ─────────────────────────────────────────────

def run_group_kfold(df: pd.DataFrame) -> tuple[np.ndarray, list[dict]]:
    y = df[TARGET].values.astype(np.float32)
    groups = df["genotype"].values
    oof = np.full(len(y), np.nan, dtype=np.float32)
    fold_metrics: list[dict] = []

    gkf = GroupKFold(n_splits=5)
    print("\n=== CV-A: 5-fold GroupKFold by genotype ===")

    for fold_i, (tr_idx, va_idx) in enumerate(gkf.split(df, y, groups=groups)):
        tr = df.iloc[tr_idx].copy()
        va = df.iloc[va_idx].copy()
        assert set(tr["genotype"]).isdisjoint(set(va["genotype"])), "genotype leakage"

        # Imputer + variance filter fit on train rows only.
        X_tr, imputer, selector, _ = _prep_matrix(tr)
        X_va = selector.transform(imputer.transform(
            va[_feature_cols(va)].apply(pd.to_numeric, errors="coerce").values.astype(np.float32)
        )).astype(np.float32)

        # Target-encode on train rows only.
        geno_all = _target_encode(tr, df)
        X_tr = np.column_stack([X_tr, geno_all[tr_idx]])
        X_va = np.column_stack([X_va, geno_all[va_idx]])

        model = XGBRegressor(**XGB_PARAMS)
        model.fit(X_tr, y[tr_idx])
        oof[va_idx] = model.predict(X_va)

        m = _metrics(y[va_idx], oof[va_idx])
        fold_metrics.append({"fold": fold_i + 1, **m,
                             "n_val_genotypes": int(len(set(groups[va_idx])))})
        print(f"  fold {fold_i+1}: R²={m['R2']:.3f}  RMSE={m['RMSE']:.1f}  "
              f"Spearman={m['Spearman']:.3f}  n={m['n']}")

    return oof, fold_metrics


# ─────────────────────────────────────────────
# CV-B — cross-location hold-out
# ─────────────────────────────────────────────

def run_cross_location(df: pd.DataFrame) -> list[dict]:
    y = df[TARGET].values.astype(np.float32)
    results = []

    print("\n=== CV-B: cross-location hold-out ===")
    for train_loc, test_loc in [("Lincoln", "MOValley"), ("MOValley", "Lincoln")]:
        tr = df[df["location"] == train_loc].copy()
        te = df[df["location"] == test_loc].copy()
        tr_idx = df["location"].values == train_loc

        X_tr, imputer, selector, _ = _prep_matrix(tr)
        X_te = selector.transform(imputer.transform(
            te[_feature_cols(te)].apply(pd.to_numeric, errors="coerce").values.astype(np.float32)
        )).astype(np.float32)

        geno_all = _target_encode(tr, df)
        X_tr = np.column_stack([X_tr, geno_all[tr_idx]])
        X_te = np.column_stack([X_te, geno_all[~tr_idx]])

        model = XGBRegressor(**XGB_PARAMS)
        model.fit(X_tr, y[tr_idx])
        y_pred = model.predict(X_te)
        m = _metrics(y[~tr_idx], y_pred)
        results.append({"train_location": train_loc, "test_location": test_loc, **m})
        print(f"  train {train_loc} ({tr_idx.sum()}) → test {test_loc} ({(~tr_idx).sum()}): "
              f"R²={m['R2']:.3f}  RMSE={m['RMSE']:.1f}  Spearman={m['Spearman']:.3f}")
    return results


# ─────────────────────────────────────────────
# Final model on all rows
# ─────────────────────────────────────────────

def train_final_model(df: pd.DataFrame):
    print("\n=== Final model: train on all rows (for scoring engine & Streamlit) ===")
    y = df[TARGET].values.astype(np.float32)
    raw_feature_cols = _feature_cols(df)
    X, imputer, selector, kept = _prep_matrix(df)
    # Genotype mean on all data — OK here because this model is never
    # evaluated on itself; it's only used for the scoring engine and Streamlit.
    geno_all = _target_encode(df, df)
    X_final = np.column_stack([X, geno_all])
    feature_cols_final = kept + ["genotype_mean_yield"]

    model = XGBRegressor(**XGB_PARAMS)
    model.fit(X_final, y)

    explainer = shap.TreeExplainer(model)

    bundle = {
        "model": model,
        "imputer": imputer,
        "selector": selector,
        "raw_feature_cols": raw_feature_cols,
        "feature_cols": feature_cols_final,
        "genotype_means": df.groupby("genotype")[TARGET].mean().to_dict(),
        "global_mean": float(y.mean()),
    }
    joblib.dump(bundle, P.MODEL_PKL)
    joblib.dump(explainer, P.EXPLAINER_PKL)
    print(f"  saved → {P.MODEL_PKL}")
    print(f"  saved → {P.EXPLAINER_PKL}")

    # Feature-importance plot
    imp = pd.DataFrame({"feature": feature_cols_final,
                        "importance": model.feature_importances_}
                       ).sort_values("importance", ascending=False).head(25)
    fig, ax = plt.subplots(figsize=(9, 8))
    ax.barh(imp["feature"][::-1], imp["importance"][::-1], color="steelblue")
    ax.set_xlabel("Importance")
    ax.set_title("Top 25 feature importances")
    plt.tight_layout()
    plt.savefig(P.FEATURE_IMPORTANCE_PNG, dpi=150)
    plt.close()
    print(f"  saved → {P.FEATURE_IMPORTANCE_PNG}")
    return bundle


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def main():
    df = pd.read_parquet(P.FEATURES_PARQUET)
    print(f"Loaded {len(df)} plots, {df.shape[1]} columns.")

    oof, fold_metrics = run_group_kfold(df)
    cv_b = run_cross_location(df)
    train_final_model(df)

    oof_metrics = _metrics(df[TARGET].values, oof)
    print("\n=== OOF (across all 5 folds) ===")
    for k, v in oof_metrics.items():
        print(f"  {k}: {v:.3f}" if isinstance(v, float) else f"  {k}: {v}")

    metrics = {
        "cv_a_groupkfold_by_genotype": {
            "per_fold": fold_metrics,
            "oof": oof_metrics,
        },
        "cv_b_cross_location": cv_b,
        "target_benchmarks_from_plan": {
            "per_tp_r2_powadi_2025_VIs": {"TP1": 0.75, "TP2": 0.80, "TP3": 0.77, "TP4": 0.70},
            "cross_location_target": 0.30,
            "spearman_target": 0.50,
        },
    }
    with open(P.METRICS_JSON, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nsaved → {P.METRICS_JSON}")

    # ── Write OOF predictions for scoring engine ──
    preds = df[["location", "experiment", "range", "row", "genotype",
                "nitrogenTreatment", "timepoint"]].copy()
    preds["actual_yield"] = df[TARGET].values
    preds["predicted_yield"] = oof
    preds["error"] = preds["predicted_yield"] - preds["actual_yield"]
    preds["absolute_error"] = preds["error"].abs()
    preds["pct_error"] = np.where(preds["actual_yield"] != 0,
                                  (preds["absolute_error"] / preds["actual_yield"]) * 100,
                                  np.nan)
    preds["split"] = "oof"
    preds.to_csv(P.OOF_CSV, index=False)
    print(f"saved → {P.OOF_CSV}")


if __name__ == "__main__":
    main()
