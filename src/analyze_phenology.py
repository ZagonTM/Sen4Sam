"""
Analyse plant greenness (NDVI) by month across the downloaded sample.

For every downloaded patch we compute, per acquisition, the mean NDVI over the
patch's *field* pixels (``parcels != 0``) and the acquisition's Hollstein cloud
fraction. Only cloud-clean acquisitions (cloud <= CLOUD_THRESHOLD) are kept, so a
monthly value reflects the crop, not the clouds.

Writes ``data/phenology_by_month.csv``: one row per (patch, month) with the mean
NDVI of that patch's clean acquisitions in that month. The notebook
(notebooks/00_data_availability.ipynb) aggregates these rows into the by-month
greenness curve that is weighed against availability when the window is chosen.

Usage:
    python src/analyze_phenology.py                # all patches
    python src/analyze_phenology.py --limit 12      # quick timing test
"""

import argparse
import csv
import time
from pathlib import Path

import netCDF4
import numpy as np

from s4a_io import (CLOUD_BANDS, CLOUD_THRESHOLD, hollstein_cloud, read_band,
                    read_dates, read_parcels, resample_to_10m)


def patch_phenology(nc_path):
    """Mean NDVI over field pixels, averaged per month over cloud-clean dates."""
    nc = netCDF4.Dataset(nc_path)
    try:
        dates = read_dates(nc)
        cloud_bands = {b: resample_to_10m(read_band(nc, b)) / 10000.0 for b in CLOUD_BANDS}
        cloud = hollstein_cloud(cloud_bands).mean(axis=(1, 2))
        red = resample_to_10m(read_band(nc, "B04")) / 10000.0
        nir = resample_to_10m(read_band(nc, "B08")) / 10000.0
        parcels = read_parcels(nc)
    finally:
        nc.close()

    mask = parcels != 0
    with np.errstate(divide="ignore", invalid="ignore"):
        ndvi = (nir - red) / (nir + red + 1e-6)

    by_month = {}
    for i, d in enumerate(dates):
        if cloud[i] > CLOUD_THRESHOLD or not mask.any():
            continue
        by_month.setdefault(d.month, []).append(float(ndvi[i][mask].mean()))
    return {m: (float(np.mean(vs)), len(vs)) for m, vs in by_month.items()}


def main():
    parser = argparse.ArgumentParser(description="NDVI-by-month analysis.")
    parser.add_argument("--manifest", default="data/manifest.csv")
    parser.add_argument("--out", default="data/phenology_by_month.csv")
    parser.add_argument("--limit", type=int, default=0, help="only N patches (0 = all)")
    args = parser.parse_args()

    # the whole drawn sample, so the analysis describes it (not just the kept subset)
    with open(args.manifest) as fh:
        rows = list(csv.DictReader(fh))
    if args.limit:
        rows = rows[:args.limit]

    started = time.time()
    out_rows = []
    for i, row in enumerate(rows, 1):
        by_month = patch_phenology(row["file"])
        for month, (ndvi, n) in sorted(by_month.items()):
            out_rows.append({"country": row["country"], "tile": row["tile"],
                             "patch": row["patch"], "month": month,
                             "ndvi": round(ndvi, 4), "n_clean": n})
        if i % 50 == 0 or i == len(rows):
            rate = (time.time() - started) / i
            print(f"[{i}/{len(rows)}] {rate:.2f} s/patch "
                  f"(elapsed {time.time() - started:.0f}s)", flush=True)

    out_path = Path(args.out)
    with open(out_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["country", "tile", "patch",
                                                "month", "ndvi", "n_clean"])
        writer.writeheader()
        writer.writerows(out_rows)
    print(f"\nWrote {out_path}  ({len(out_rows)} patch-month rows)")


if __name__ == "__main__":
    main()
