"""
Analyse the downloaded Sen4AgriNet sample: availability, cloud and fields.

Reads the raw patches listed in the manifest and writes two CSVs:

  * <out>/availability_acquisitions.csv — one row per (patch, acquisition):
    date, month, cloud fraction. This is the granular cloud/date data.
  * <out>/availability_patches.csv — one row per patch: total acquisitions,
    per-month acquisition counts, field count and field coverage.

The notebook (notebooks/00_data_availability.ipynb) turns these into plots and the
decision table. Keeping the numbers here makes them reproducible and citable.

Usage:
    python src/analyze_availability.py                 # all patches
    python src/analyze_availability.py --limit 12       # quick timing test
"""

import argparse
import csv
import time
from pathlib import Path

import netCDF4
import numpy as np

from s4a_io import (CLOUD_BANDS, hollstein_cloud, read_band, read_dates,
                    read_parcels, resample_to_10m)

MONTH_LABELS = [f"M{m}" for m in range(1, 13)]


def analyse_patch(nc_path):
    """Return (acquisition rows, patch row) for one patch."""
    nc = netCDF4.Dataset(nc_path)
    try:
        dates = read_dates(nc)
        bands = {b: resample_to_10m(read_band(nc, b)) / 10000.0 for b in CLOUD_BANDS}
        cloud = hollstein_cloud(bands).mean(axis=(1, 2)).tolist()
        parcels = read_parcels(nc)
    finally:
        nc.close()

    acquisitions = [{"date": d.date().isoformat(), "month": d.month, "cloud": round(c, 4)}
                    for d, c in zip(dates, cloud)]

    counts = {label: 0 for label in MONTH_LABELS}
    for d in dates:
        counts[f"M{d.month}"] += 1

    patch = {
        "n_acquisitions": len(dates),
        "n_fields": len(np.unique(parcels)) - 1,
        "field_coverage": round(float((parcels != 0).mean()), 4),
        **counts,
    }
    return acquisitions, patch


def main():
    parser = argparse.ArgumentParser(description="Availability/cloud/field analysis.")
    parser.add_argument("--manifest", default="data/manifest.csv")
    parser.add_argument("--out-dir", default="data")
    parser.add_argument("--limit", type=int, default=0, help="only N patches (0 = all)")
    args = parser.parse_args()

    # the whole drawn sample, so the analysis describes it (not just the kept subset)
    with open(args.manifest) as fh:
        rows = list(csv.DictReader(fh))
    if args.limit:
        rows = rows[:args.limit]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    acq_fh = open(out_dir / "availability_acquisitions.csv", "w", newline="")
    patch_fh = open(out_dir / "availability_patches.csv", "w", newline="")
    acq_cols = ["country", "tile", "patch", "date", "month", "cloud"]
    patch_cols = ["country", "tile", "patch", "n_acquisitions", "n_fields",
                  "field_coverage"] + MONTH_LABELS
    acq_writer = csv.DictWriter(acq_fh, fieldnames=acq_cols)
    patch_writer = csv.DictWriter(patch_fh, fieldnames=patch_cols)
    acq_writer.writeheader()
    patch_writer.writeheader()

    started = time.time()
    for i, row in enumerate(rows, 1):
        acquisitions, patch = analyse_patch(row["file"])
        for a in acquisitions:
            acq_writer.writerow({"country": row["country"], "tile": row["tile"],
                                 "patch": row["patch"], **a})
        patch_writer.writerow({"country": row["country"], "tile": row["tile"],
                               "patch": row["patch"], **patch})
        if i % 50 == 0 or i == len(rows):
            rate = (time.time() - started) / i
            print(f"[{i}/{len(rows)}] {rate:.2f} s/patch "
                  f"(elapsed {time.time() - started:.0f}s)", flush=True)

    acq_fh.close()
    patch_fh.close()
    print(f"\nWrote {out_dir}/availability_acquisitions.csv and availability_patches.csv")


if __name__ == "__main__":
    main()
