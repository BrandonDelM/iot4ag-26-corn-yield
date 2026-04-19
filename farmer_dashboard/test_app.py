"""Unit tests for farmer_dashboard mock logic."""
import pytest
import numpy as np
from datetime import date, timedelta
from mock_logic import (
    calculate_mock_yield,
    calculate_growth_stage,
    generate_mock_heatmap,
    generate_mock_spectral_data,
    get_genotype_names,
)


def test_genotype_catalog_populated():
    names = get_genotype_names()
    assert len(names) == 5
    assert "DKC62-08" in names


def test_yield_increases_with_nitrogen():
    low  = calculate_mock_yield(50,  "DKC62-08")
    high = calculate_mock_yield(200, "DKC62-08")
    assert high > low


def test_yield_diminishing_returns():
    """Going from 0→100 should add more than 100→200 (diminishing curve)."""
    gain_low  = calculate_mock_yield(100, "DKC62-08") - calculate_mock_yield(0,   "DKC62-08")
    gain_high = calculate_mock_yield(200, "DKC62-08") - calculate_mock_yield(100, "DKC62-08")
    assert gain_low > gain_high


def test_genotype_matters():
    best  = calculate_mock_yield(150, "DKC62-08")
    worst = calculate_mock_yield(150, "NK1082")
    assert best > worst


def test_growth_stage_emergence():
    planted_recently = date.today() - timedelta(days=10)
    assert "Emergence" in calculate_growth_stage(planted_recently)


def test_growth_stage_maturity():
    planted_long_ago = date.today() - timedelta(days=200)
    assert "Maturity" in calculate_growth_stage(planted_long_ago)


def test_heatmap_shape():
    hm = generate_mock_heatmap(190.0, 30)
    assert hm.shape == (30, 30)


def test_heatmap_has_stress_zone():
    """The SW corner should be deliberately lower than the field mean."""
    hm = generate_mock_heatmap(190.0, 50)
    sw_mean = hm[5:18, 5:18].mean()
    field_mean = hm.mean()
    assert sw_mean < field_mean


def test_spectral_data_columns():
    df = generate_mock_spectral_data()
    assert len(df) == 100
    assert "NDVI" in df.columns
    assert "Soil Moisture (%)" in df.columns
