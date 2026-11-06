"""
Shared helpers for reading Sen4AgriNet patches (.nc files).

Used by download_s4a.py, analyze_availability.py, analyze_phenology.py and
preprocessing.py, so there is a single implementation of the reading and cloud logic:

  * parse acquisition dates and read bands (resampled to the 10 m grid),
  * compute Hollstein cloud fractions and pick the single image date,
  * read the ground-truth maps: `parcels` (field ids, 0 = non-field) and
    `labels` (crop type).

Library only: nothing to run here. To build data/processed/, run src/download_process_data/preprocessing.py.
"""

import datetime
import random
import re

import netCDF4
import numpy as np

# --- Configuration -------------------------------------------------------------

MONTHS = (5,)           # May: peak crop greenness with ~full patch coverage (see docs/data_en.md)
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
