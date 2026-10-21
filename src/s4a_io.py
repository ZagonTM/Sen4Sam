"""
Read Sen4AgriNet patches and build SAM 2 inputs + ground truth.

This module is the I/O + preprocessing half of the pipeline:

  * parse acquisition dates and read bands (resampled to the 10 m grid),
  * compute Hollstein cloud fractions and pick the single image date,
  * build the RGB uint8 image (B04/B03/B02, /10000, 2-98% percentile stretch),
  * read the ground-truth maps: `parcels` (field ids, 0 = non-field) and
    `labels` (crop type).

The download script (src/download_s4a.py) imports the selection helpers from here,
so there is a single implementation of the cloud logic.

CLI: turn downloaded patches into model-ready files.
    python src/s4a_io.py --manifest data/manifest.csv --out-dir data/processed
"""

import argparse
import datetime
import random
import re
from pathlib import Path

import netCDF4
import numpy as np

# --- Configuration -------------------------------------------------------------

MONTHS = (5,)           # May: peak crop greenness with ~full patch coverage (see docs/data.md)
CLOUD_THRESHOLD = 0.05  # an acquisition is "clean" if its cloud fraction <= this

# Bands: the Hollstein cloud formula needs these seven (mixed 10/20/60 m grids);
# the SAM 2 input image uses the three 10 m visible bands.
CLOUD_BANDS = ["B01", "B03", "B05", "B06", "B07", "B10", "B11"]
RGB_BANDS = ["B04", "B03", "B02"]  # red, green, blue


# --- Dates and bands -----------------------------------------------------------

def time_origin(time_var):
    """Parse the acquisition times' origin from the netCDF time units."""
    units = getattr(time_var, "units", "")
    match = re.search(r"seconds since ([\d\- :]+)", units)
    return datetime.datetime.strptime(match.group(1).strip(), "%Y-%m-%d %H:%M:%S")


def read_dates(nc):
    """Return the list of acquisition datetimes of a patch."""
    time_var = nc["B02"].variables["time"]
    origin = time_origin(time_var)
    return [origin + datetime.timedelta(seconds=int(t)) for t in np.asarray(time_var[:])]


def resample_to_10m(array, size=366):
    """Nearest-neighbour upsample so a band shares the 10 m grid."""
    factor = size // array.shape[1]
    return np.repeat(np.repeat(array, factor, axis=1), factor, axis=2)


def read_band(nc, name):
    """Read one band's full time series as float32."""
    return np.asarray(nc[name].variables[name][:]).astype(np.float32)


# --- Cloud and date selection --------------------------------------------------

def hollstein_cloud(bands):
    """Boolean cloud mask (Hollstein et al., 2016) for one date. Bands in [0, 1]."""
    with np.errstate(divide="ignore", invalid="ignore"):
        return (
            (bands["B03"] >= 0.319) & (bands["B05"] / bands["B11"] < 4.33) & (
                ((bands["B11"] - bands["B10"] < 0.255) & (bands["B06"] - bands["B07"] < -0.016)) |
                ((bands["B11"] - bands["B10"] >= 0.255) & (bands["B01"] >= 0.3))
            )
        )


def window_cloud_fractions(nc, dates, months=MONTHS):
    """Cloud fraction of every acquisition inside `months`.

    Returns (indices, cloud_fractions), aligned.
    """
    month_idx = [i for i, d in enumerate(dates) if d.month in months]
    if not month_idx:
        return [], []
    bands = {}
    for name in CLOUD_BANDS:
        bands[name] = resample_to_10m(read_band(nc, name)[month_idx]) / 10000.0
    return month_idx, hollstein_cloud(bands).mean(axis=(1, 2)).tolist()


def choose_acquisition(clean_dates, seed_key):
    """Pick the one acquisition that represents a patch.

    `clean_dates` is a list of (date, cloud_fraction) pairs. We keep the least
    cloudy ones and pick one at random among them (random, so we do not invent a
    bias between equally good dates; seeded, so it is reproducible).
    """
    if not clean_dates:
        return None
    best = min(cloud for _, cloud in clean_dates)
    ties = sorted(date for date, cloud in clean_dates if cloud <= best + 1e-6)
    return random.Random(seed_key).choice(ties)


def patch_info(nc_path, months=MONTHS, cloud_threshold=CLOUD_THRESHOLD):
    """Read a patch once: acquisition count, window dates, cloud, field stats."""
    nc = netCDF4.Dataset(nc_path)
    try:
        dates = read_dates(nc)
        _, cloud = window_cloud_fractions(nc, dates, months)
        window_dates = [d for d in dates if d.month in months]
        parcels = read_parcels(nc)
    finally:
        nc.close()
    return {
        "n_acquisitions": len(dates),
        "dates": window_dates,
        "cloud": cloud,
        "n_fields": len(np.unique(parcels)) - 1,
        "field_coverage": float((parcels != 0).mean()),
    }


# --- Ground truth --------------------------------------------------------------

def read_parcels(nc):
    """Per-pixel field id map (0 = non-field)."""
    return np.asarray(nc["parcels"].variables["parcels"][:])


def read_labels(nc):
    """Per-pixel crop-type map."""
    return np.asarray(nc["labels"].variables["labels"][:])


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
