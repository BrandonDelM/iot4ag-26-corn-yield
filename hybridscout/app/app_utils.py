"""
Shared Streamlit helpers: cached loaders for artifacts, model, explainer.

Home + page files import these so there's a single place to change IO.
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

from hybridscout.ml import io_paths as P


# ─────────────────────────────────────────────
# Cached loaders
# ─────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def load_features() -> pd.DataFrame:
    if not P.FEATURES_PARQUET.exists():
        return pd.DataFrame()
    return pd.read_parquet(P.FEATURES_PARQUET)


@st.cache_data(show_spinner=False)
def load_oof() -> pd.DataFrame:
    if not P.OOF_CSV.exists():
        return pd.DataFrame()
    return pd.read_csv(P.OOF_CSV)


@st.cache_data(show_spinner=False)
def load_rankings() -> pd.DataFrame:
    if not P.HYBRID_RANKINGS_CSV.exists():
        return pd.DataFrame()
    return pd.read_csv(P.HYBRID_RANKINGS_CSV)


@st.cache_data(show_spinner=False)
def load_gxe() -> pd.DataFrame:
    if not P.GXE_MATRIX_CSV.exists():
        return pd.DataFrame()
    return pd.read_csv(P.GXE_MATRIX_CSV, index_col=0)


@st.cache_data(show_spinner=False)
def load_tp_rankings() -> pd.DataFrame:
    if not P.TIMEPOINT_RANKINGS_CSV.exists():
        return pd.DataFrame()
    return pd.read_csv(P.TIMEPOINT_RANKINGS_CSV)


@st.cache_data(show_spinner=False)
def load_nitrogen_response() -> pd.DataFrame:
    if not P.NITROGEN_RESPONSE_CSV.exists():
        return pd.DataFrame()
    return pd.read_csv(P.NITROGEN_RESPONSE_CSV)


@st.cache_data(show_spinner=False)
def load_metrics() -> dict:
    if not P.METRICS_JSON.exists():
        return {}
    with open(P.METRICS_JSON) as f:
        return json.load(f)


@st.cache_resource(show_spinner=False)
def load_model_bundle() -> dict | None:
    if not P.MODEL_PKL.exists():
        return None
    return joblib.load(P.MODEL_PKL)


@st.cache_resource(show_spinner=False)
def load_explainer():
    if not P.EXPLAINER_PKL.exists():
        return None
    return joblib.load(P.EXPLAINER_PKL)


# ─────────────────────────────────────────────
# Gating helper
# ─────────────────────────────────────────────

def require_artifacts(*paths: Path) -> bool:
    """Return True if all artifacts exist, else render a guided message."""
    missing = [p for p in paths if not p.exists()]
    if not missing:
        return True
    st.warning(
        "Missing artifacts — run the pipeline first:\n\n"
        "```\n"
        "python -m hybridscout.ml.extract_features\n"
        "python -m hybridscout.ml.train_model\n"
        "python -m hybridscout.ml.scoring\n"
        "```"
    )
    st.caption("Files not found: " + ", ".join(p.name for p in missing))
    return False
