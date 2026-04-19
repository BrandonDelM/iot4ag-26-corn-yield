"""
Mock data layer for the Farmer Digital Twin demo.
All functions return deterministic, presentation-ready data.
"""
import logging
import numpy as np
import pandas as pd
from datetime import date, timedelta

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("farmer_portal")

# ── Genotype catalog ──────────────────────────────────────────────
GENOTYPE_CATALOG = {
    "P0953AM":  {"label": "Pioneer P0953AM",  "adj": 2.0,  "n_response": 0.12},
    "DKC62-08": {"label": "DeKalb DKC62-08",  "adj": 8.5,  "n_response": 0.18},
    "P1197AM":  {"label": "Pioneer P1197AM",  "adj": 4.5,  "n_response": 0.14},
    "NK1082":   {"label": "NK Seeds NK1082",  "adj": -4.0, "n_response": 0.08},
    "DKC59-82": {"label": "DeKalb DKC59-82",  "adj": 0.0,  "n_response": 0.10},
}

def get_genotype_names() -> list[str]:
    return list(GENOTYPE_CATALOG.keys())


# ── Yield calculations ────────────────────────────────────────────
def calculate_mock_yield(n_rate: float, genotype: str = "DKC62-08") -> float:
    """
    Compute a mock predicted yield using a diminishing-returns nitrogen curve
    plus a genotype offset.  Returns bu/ac.
    """
    info = GENOTYPE_CATALOG.get(genotype, {"adj": 0.0, "n_response": 0.10})
    # Diminishing returns curve — realistic agronomic shape
    base = 155.0 + info["adj"]
    n_effect = info["n_response"] * n_rate * (1 - n_rate / 600)
    return round(base + n_effect, 1)


def calculate_growth_stage(planting_date: date) -> str:
    """Return a mocked growth stage string based on calendar offset from planting."""
    days_since = (date.today() - planting_date).days
    if days_since < 0:
        return "Not yet planted"
    stages = [
        (14, "VE (Emergence)"),
        (30, "V6 (Early Vegetative)"),
        (50, "V10 (Mid Vegetative)"),
        (65, "VT (Tasseling)"),
        (80, "R1 (Silking)"),
        (100, "R3 (Milk)"),
        (120, "R5 (Dent)"),
    ]
    for threshold, label in stages:
        if days_since <= threshold:
            return label
    return "R6 (Physiological Maturity)"


# ── Heatmap generation ────────────────────────────────────────────
def generate_mock_heatmap(base_yield: float, grid_size: int = 50) -> np.ndarray:
    """
    Produce a smoothed 2-D yield surface that looks like a real field scan.
    Includes a deliberate low-yield zone in the SW corner to demo the
    'red spot = problem area' narrative.
    """
    rng = np.random.default_rng(42)
    z = rng.normal(base_yield, 12, (grid_size, grid_size))

    # Inject a visible stress pocket (lower-left quadrant)
    z[5:18, 5:18] -= rng.uniform(18, 28, (13, 13))

    # Smooth for a natural gradient
    for _ in range(3):
        z_new = z.copy()
        for i in range(1, grid_size - 1):
            for j in range(1, grid_size - 1):
                z_new[i, j] = (z[i,j] + z[i-1,j] + z[i+1,j] + z[i,j-1] + z[i,j+1]) / 5
        z = z_new
    return z


# ── Spectral deep-dive data ──────────────────────────────────────
def generate_mock_spectral_data() -> pd.DataFrame:
    """Return a 100-row mock sensor table hidden inside the expander."""
    rng = np.random.default_rng(42)
    rows = 100
    return pd.DataFrame({
        "Plot ID":           [f"Plot-{i:03d}" for i in range(rows)],
        "NDVI":              rng.uniform(0.55, 0.92, rows).round(3),
        "NDRE":              rng.uniform(0.20, 0.55, rows).round(3),
        "SAVI":              rng.uniform(0.40, 0.80, rows).round(3),
        "Red Edge Refl.":    rng.uniform(0.08, 0.30, rows).round(3),
        "Deep Blue Refl.":   rng.uniform(0.02, 0.10, rows).round(3),
        "Soil Moisture (%)": rng.uniform(22, 48, rows).round(1),
    })
