"""
Preprocessing: downloaded Sen4AgriNet patches -> SAM 2 inputs + ground truth.

Reads data/manifest.csv (written by download_s4a.py), keeps the preselected patches
(kept == 1), and for each one writes to data/processed/:

  <patch>_rgb.png      RGB uint8 image from the chosen May date
                       (B04/B03/B02, /10000, 2-98% percentile stretch)
  <patch>_parcels.npy  field id per pixel (0 = non-field)
  <patch>_labels.npy   crop type per pixel

    python src/download_process_data/preprocessing.py --manifest data/manifest.csv --out-dir data/processed
"""

import argparse
import datetime
from pathlib import Path

import netCDF4
import numpy as np

from data_helpers import (CLOUD_THRESHOLD, MONTHS, RGB_BANDS, choose_acquisition,
                          read_band, read_dates, read_labels, read_parcels,
                          window_cloud_fractions)

# --- Building the SAM 2 input image -------------------------------------------

def build_rgb(nc, index):
    """RGB uint8 image from one acquisition: B04/B03/B02, /10000, 2-98% stretch."""
    channels = [read_band(nc, band)[index] / 10000.0 for band in RGB_BANDS]
    rgb = np.stack(channels, axis=-1)
    stretched = np.zeros_like(rgb)
    for c in range(3):
        low, high = np.percentile(rgb[..., c], (2, 98))
        stretched[..., c] = np.clip((rgb[..., c] - low) / (high - low + 1e-6), 0, 1)
    return (stretched * 255).astype(np.uint8)


def read_patch(nc_path, date=None, months=MONTHS, cloud_threshold=CLOUD_THRESHOLD,
               seed_key=None):
    """Read one patch -> the SAM 2 image, the ground truth, and the date used.

    If `date` is given (a datetime.date from the manifest) that acquisition is
    used; otherwise the least-cloudy window acquisition is chosen.
    """
    nc = netCDF4.Dataset(nc_path)
    try:
        dates = read_dates(nc)
        if date is None:
            month_idx, cloud = window_cloud_fractions(nc, dates, months)
            clean = [(dates[i], c) for i, c in zip(month_idx, cloud)
                     if c <= cloud_threshold]
            used = choose_acquisition(clean, seed_key or str(nc_path))
            index = dates.index(used)
        else:
            index = next(i for i, d in enumerate(dates) if d.date() == date)
            used = dates[index]
        image = build_rgb(nc, index)
        parcels = read_parcels(nc)
        labels = read_labels(nc)
    finally:
        nc.close()
    return {"image": image, "parcels": parcels, "labels": labels, "date": used}


# --- CLI: manifest -> processed files ------------------------------------------

def save_processed(record, patch_name, out_dir):
    """Save the RGB image (.png) and the ground truth (.npy) of one patch."""
    from PIL import Image
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    Image.fromarray(record["image"]).save(out_dir / f"{patch_name}_rgb.png")
    np.save(out_dir / f"{patch_name}_parcels.npy", record["parcels"])
    np.save(out_dir / f"{patch_name}_labels.npy", record["labels"])


def main():
    import csv
    parser = argparse.ArgumentParser(description="Build SAM 2 inputs from downloaded patches.")
    parser.add_argument("--manifest", default="data/manifest.csv")
    parser.add_argument("--out-dir", default="data/processed")
    parser.add_argument("--limit", type=int, default=0, help="process only N patches (0 = all)")
    args = parser.parse_args()

    with open(args.manifest) as fh:
        rows = [r for r in csv.DictReader(fh) if r["kept"] == "1"]
    if args.limit:
        rows = rows[:args.limit]

    for i, row in enumerate(rows, 1):
        date = datetime.date.fromisoformat(row["chosen_date"])
        record = read_patch(row["file"], date=date)
        save_processed(record, row["patch"], args.out_dir)
        print(f"[{i}/{len(rows)}] {row['patch']} <- {record['date'].date()} "
              f"({row['n_fields']} fields)")
    print(f"Saved {len(rows)} patches to {args.out_dir}")


if __name__ == "__main__":
    main()
