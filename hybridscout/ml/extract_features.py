"""
Feature extraction: 6-band satellite TIFs → one row per plot × timepoint,
with patch-grid vegetation-index stats plus multi-TP delta features
broadcast across rows of the same plot, merged with ground truth.

Primitives (indices_from_arrays, patch_stats, satelliteimage_patches,
parse_filename) are vectorized over arrays — no per-pixel loops.

Run:  python -m hybridscout.ml.extract_features
Out:  hybridscout/artifacts/plots_features.parquet
"""
from __future__ import annotations

import glob
import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio

from . import io_paths as P

warnings.filterwarnings("ignore", message="invalid value encountered in scalar divide")


# ─────────────────────────────────────────────
# Vegetation-index primitives (vectorized)
# ─────────────────────────────────────────────

def indices_from_arrays(r, g, b, nir=None, re=None):
    with np.errstate(divide="ignore", invalid="ignore"):
        GLI = np.where((2 * g + r + b) != 0, (2 * g - r - b) / (2 * g + r + b), np.nan)
        NGRDI = np.where((r + g) != 0, (r - g) / (r + g), np.nan)

    result = {"GLI": GLI, "NGRDI": NGRDI, "Red": r, "Green": g, "Blue": b}

    if nir is not None:
        with np.errstate(divide="ignore", invalid="ignore"):
            NDVI = np.where((nir + r) != 0, (nir - r) / (nir + r), np.nan)
            GNDVI = np.where((nir + g) != 0, (nir - g) / (nir + g), np.nan)
            SAVI = np.where((nir + r + 0.5) != 0, 1.5 * (nir - r) / (nir + r + 0.5), np.nan)
            NDRE = np.where((nir + re) != 0, (nir - re) / (nir + re), np.nan) if re is not None else None
        result.update({"NDVI": NDVI, "GNDVI": GNDVI, "SAVI": SAVI, "NIR": nir})
        if NDRE is not None:
            result["NDRE"] = NDRE
    return result


def patch_stats(patch_2d, mask_2d):
    valid = patch_2d[mask_2d]
    valid = valid[~np.isnan(valid)]
    if len(valid) == 0:
        return np.nan, np.nan, np.nan
    return float(np.mean(valid)), float(np.median(valid)), float(np.std(valid))


def satelliteimage_patches(inputpath, grid_size=P.GRID_SIZE):
    """6-band TIF → dict of patch × index × stat features."""
    with rasterio.open(inputpath, "r") as src:
        bands = src.read().astype(np.float32)  # (6, H, W)

    r, g, b = bands[0], bands[1], bands[2]
    nir, re = bands[3], bands[4]
    mask = r > 0  # zero pixels are plot-boundary padding

    idx_arrays = indices_from_arrays(r, g, b, nir=nir, re=re)
    H, W = r.shape
    ph, pw = H // grid_size, W // grid_size
    row = {"Imagename": os.path.basename(inputpath)}

    for pi in range(grid_size):
        for pj in range(grid_size):
            rs = slice(pi * ph, (pi + 1) * ph)
            cs = slice(pj * pw, (pj + 1) * pw)
            patch_mask = mask[rs, cs]
            if patch_mask.sum() == 0:
                continue
            prefix = f"patch_{pi}_{pj}"
            for idx_name, idx_arr in idx_arrays.items():
                mean_, med_, std_ = patch_stats(idx_arr[rs, cs], patch_mask)
                row[f"{prefix}_{idx_name}_mean"] = mean_
                row[f"{prefix}_{idx_name}_median"] = med_
                row[f"{prefix}_{idx_name}_std"] = std_
    return row


def parse_filename(imagename_col):
    def parse_one(name):
        try:
            base = os.path.splitext(name)[0]
            parts = base.split("-")
            location = parts[0]
            timepoint = parts[1]
            rest = parts[2].split("_")
            return location, timepoint, rest[0], rest[1], rest[2]
        except Exception:
            return None, None, None, None, None

    parsed = imagename_col.apply(parse_one)
    return pd.DataFrame(parsed.tolist(),
                        columns=["location", "timepoint", "experiment", "range", "row"])


# ─────────────────────────────────────────────
# Collect all TIFs → long DF, build temporal deltas, merge ground truth
# ─────────────────────────────────────────────

def collect_all_satellite(image_root: Path, grid_size=P.GRID_SIZE) -> pd.DataFrame:
    tifs = glob.glob(str(image_root / "**" / "*.TIF"), recursive=True)
    tifs += glob.glob(str(image_root / "**" / "*.tif"), recursive=True)
    if not tifs:
        raise FileNotFoundError(f"No TIF files found under {image_root}")

    rows = []
    print(f"Extracting features from {len(tifs)} TIFs…")
    for i, fp in enumerate(tifs):
        if (i + 1) % 200 == 0:
            print(f"  {i+1}/{len(tifs)}")
        try:
            rows.append(satelliteimage_patches(fp, grid_size=grid_size))
        except Exception as e:
            print(f"  skip {fp}: {e}")
    return pd.DataFrame(rows)


def _add_temporal_deltas(df: pd.DataFrame, grid_size: int) -> pd.DataFrame:
    """
    For each plot, compute deltas of patch-mean NDVI/NDRE/GNDVI/SAVI between
    consecutive TPs, then broadcast the deltas onto all rows for that plot.
    """
    key = ["location", "experiment", "range", "row"]
    df = df.copy()
    df["_plot_key"] = df[key].astype(str).agg("_".join, axis=1)

    index_names = ["NDVI", "NDRE", "GNDVI", "SAVI"]
    patch_positions = [f"{pi}_{pj}" for pi in range(grid_size) for pj in range(grid_size)]

    delta_rows = []
    for plot_key, grp in df.groupby("_plot_key"):
        g = grp.sort_values("timepoint")
        tps = g["timepoint"].tolist()
        delta = {"_plot_key": plot_key}
        for idx in index_names:
            for pp in patch_positions:
                col = f"patch_{pp}_{idx}_mean"
                if col not in g.columns:
                    continue
                vals = g[col].tolist()
                for t in range(1, len(vals)):
                    delta[f"delta_{idx}_{pp}_{tps[t-1]}to{tps[t]}"] = vals[t] - vals[t - 1]
        delta_rows.append(delta)

    if delta_rows:
        delta_df = pd.DataFrame(delta_rows)
        df = df.merge(delta_df, on="_plot_key", how="left")
    return df.drop(columns=["_plot_key"])


def build_plots_features(image_root: Path = P.SATELLITE_ROOT,
                         groundtruth_csv: Path = P.GROUND_TRUTH_CSV,
                         grid_size: int = P.GRID_SIZE) -> pd.DataFrame:
    sat = collect_all_satellite(image_root, grid_size=grid_size)
    meta = parse_filename(sat["Imagename"])
    sat = pd.concat([sat.reset_index(drop=True), meta.reset_index(drop=True)], axis=1)
    sat = sat.dropna(subset=["location", "timepoint", "experiment", "range", "row"])

    sat = _add_temporal_deltas(sat, grid_size=grid_size)

    gt = pd.read_csv(groundtruth_csv)
    for c in ["range", "row", "experiment"]:
        gt[c] = gt[c].astype(str)
        sat[c] = sat[c].astype(str)

    merged = sat.merge(
        gt[["location", "experiment", "range", "row", "genotype", "nitrogenTreatment",
            "poundsOfNitrogenPerAcre", "block", "plantingDate", "totalStandCount",
            "daysToAnthesis", "GDDToAnthesis", "yieldPerAcre", "irrigationProvided"]],
        on=["location", "experiment", "range", "row"],
        how="inner",
    )
    merged = merged.dropna(subset=["yieldPerAcre"]).reset_index(drop=True)

    merged["nitrogenTreatment_enc"] = merged["nitrogenTreatment"].map(
        {"Low": 0, "Medium": 1, "High": 2}
    )
    merged["location_enc"] = (merged["location"] == "Lincoln").astype(int)
    merged["timepoint_num"] = merged["timepoint"].str.replace("TP", "").astype(float)

    print(f"\nplots_features: {len(merged)} rows × {merged.shape[1]} cols")
    print(f"  Lincoln  : {(merged.location == 'Lincoln').sum()}")
    print(f"  MOValley : {(merged.location == 'MOValley').sum()}")
    print(f"  Genotypes: {merged.genotype.nunique()}")
    print(f"  Timepoints: {sorted(merged.timepoint.unique())}")
    return merged


def main():
    df = build_plots_features()
    df.to_parquet(P.FEATURES_PARQUET, index=False)
    print(f"\nSaved → {P.FEATURES_PARQUET}")


if __name__ == "__main__":
    main()
