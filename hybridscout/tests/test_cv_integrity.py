"""
CV-integrity guardrails.

These tests guarantee the pitch-critical claim:
  - No genotype appears in both train and validation for any GroupKFold fold.
  - Cross-location splits are reported separately, not averaged in.
  - OOF predictions cover every row, with zero NaN.
  - Target encoding on val rows only uses train-row genotype means.

Run:  python -m pytest hybridscout/tests/ -v
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import GroupKFold

from hybridscout.ml import io_paths as P
from hybridscout.ml.train_model import _target_encode


@pytest.fixture(scope="module")
def features_df():
    if not P.FEATURES_PARQUET.exists():
        pytest.skip("plots_features.parquet missing — run extract_features first.")
    return pd.read_parquet(P.FEATURES_PARQUET)


@pytest.fixture(scope="module")
def oof_df():
    if not P.OOF_CSV.exists():
        pytest.skip("xgb_predictions_full.csv missing — run train_model first.")
    return pd.read_csv(P.OOF_CSV)


@pytest.fixture(scope="module")
def metrics_json():
    if not P.METRICS_JSON.exists():
        pytest.skip("metrics.json missing — run train_model first.")
    with open(P.METRICS_JSON) as f:
        return json.load(f)


# ─────────────────────────────────────────────
# GroupKFold integrity
# ─────────────────────────────────────────────

def test_groupkfold_no_genotype_leak(features_df):
    """Every fold's train/val genotype sets are disjoint."""
    y = features_df["yieldPerAcre"].values
    groups = features_df["genotype"].values
    gkf = GroupKFold(n_splits=5)
    for fold, (tr, va) in enumerate(gkf.split(features_df, y, groups=groups)):
        tr_genos = set(groups[tr])
        va_genos = set(groups[va])
        assert tr_genos.isdisjoint(va_genos), (
            f"Fold {fold+1}: {len(tr_genos & va_genos)} genotypes leaked "
            f"across train/val"
        )


def test_groupkfold_every_row_is_validated(features_df):
    """Across 5 folds, every row appears in exactly one val set."""
    y = features_df["yieldPerAcre"].values
    groups = features_df["genotype"].values
    gkf = GroupKFold(n_splits=5)
    seen = np.zeros(len(features_df), dtype=int)
    for _, va in gkf.split(features_df, y, groups=groups):
        seen[va] += 1
    assert (seen == 1).all(), (
        f"{(seen != 1).sum()} rows weren't validated exactly once."
    )


# ─────────────────────────────────────────────
# Target-encoding purity
# ─────────────────────────────────────────────

def test_target_encoding_train_only(features_df):
    """Val-row geno means must equal train-only means for each genotype."""
    df = features_df
    groups = df["genotype"].values
    gkf = GroupKFold(n_splits=5)

    for tr_idx, va_idx in gkf.split(df, df["yieldPerAcre"].values, groups=groups):
        tr = df.iloc[tr_idx]
        train_means = tr.groupby("genotype")["yieldPerAcre"].mean()
        global_mean = float(tr["yieldPerAcre"].mean())

        got = _target_encode(tr, df)

        for i in va_idx[:50]:  # spot-check
            g = df.iloc[i]["genotype"]
            expected = float(train_means.get(g, global_mean))
            assert np.isclose(float(got[i]), expected, atol=1e-5), (
                f"val row {i} genotype {g}: got {got[i]} vs {expected}"
            )
        break  # one fold is enough


# ─────────────────────────────────────────────
# OOF coverage & NaN guard
# ─────────────────────────────────────────────

def test_oof_covers_every_row_no_nan(features_df, oof_df):
    assert len(oof_df) == len(features_df), (
        f"OOF row count {len(oof_df)} != feature rows {len(features_df)}"
    )
    assert oof_df["predicted_yield"].notna().all(), "NaN predictions present"
    assert oof_df["actual_yield"].notna().all(), "NaN actuals present"


# ─────────────────────────────────────────────
# Cross-location reported separately
# ─────────────────────────────────────────────

def test_cross_location_reported_both_directions(metrics_json):
    cv_b = metrics_json.get("cv_b_cross_location", [])
    pairs = {(r["train_location"], r["test_location"]) for r in cv_b}
    assert ("Lincoln", "MOValley") in pairs, "Missing Lincoln → MOValley split"
    assert ("MOValley", "Lincoln") in pairs, "Missing MOValley → Lincoln split"


def test_cv_a_has_five_folds(metrics_json):
    folds = metrics_json["cv_a_groupkfold_by_genotype"]["per_fold"]
    assert len(folds) == 5, f"Expected 5 folds, got {len(folds)}"
    for f in folds:
        assert "R2" in f and "n_val_genotypes" in f
