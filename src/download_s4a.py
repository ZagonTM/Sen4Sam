"""
Download a reproducible random subset of Sen4AgriNet patches.

The selection is fully seeded, so re-running this script reproduces the same tiles
and the same patches:

  1. N_TILES_PER_COUNTRY tiles are drawn at random from each country's tile pool
     (seed = TILE_SEED).
  2. Inside every chosen tile, a random sample of patches is drawn (seed derived
     from SAMPLE_SEED and the tile name).
  3. Every downloaded patch is checked with the Hollstein cloud algorithm
     (Hollstein et al., 2016): the cloud fraction of each acquisition in the
     MONTHS window (default May) is computed, and the single image is taken
     from the least cloudy acquisition (chosen at random among equally good dates).
     A patch is eligible if it has at least MIN_CLEAN_DATES acquisitions at
     <= CLOUD_THRESHOLD.
  4. Up to KEEP eligible patches per tile are flagged ``kept``; the whole drawn sample
     stays on disk, so the analysis can describe it (and the notebook shows how the
     kept subset is derived from it). Each kept patch yields exactly one image later on.
  5. A manifest CSV records every patch and its measurements.

The reading/cloud/preprocessing helpers live in src/s4a_io.py.

Usage:
    python src/download_s4a.py                      # default settings
    python src/download_s4a.py --patches 20 --keep 10 --workers 4   # small test
"""

import argparse
import csv
import random
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent))  # so s4a_io imports work
from s4a_io import CLOUD_THRESHOLD, MONTHS, choose_acquisition, patch_info

# --- Configuration (edit here; the CLI can override the numeric ones) ----------

REPO = "orion-ai-lab/S4A"
YEAR = 2019

# Candidate tiles per country (2019); the tile name is the MGRS id.
TILE_POOLS = {
    "ES": ["31TBF", "31TCF", "31TCG", "31TDF", "31TDG"],
    "FR": ["31TCJ", "31TDK", "31TCL", "31TDM", "31UCP", "31UDR"],
}
COUNTRY_ORDER = ["ES", "FR"]

N_TILES_PER_COUNTRY = 3
TILE_SEED = 42          # seed for the random tile draw (reproducible)
SAMPLE_SEED = 42        # seed for the per-tile patch sampling (reproducible)
MIN_CLEAN_DATES = 1     # a patch is eligible if it has at least this many clean dates

API = "https://huggingface.co/api/datasets"
RESOLVE = "https://huggingface.co/datasets"


# --- Tile and patch selection --------------------------------------------------

def select_tiles(seed=TILE_SEED):
    """Reproducibly pick N_TILES_PER_COUNTRY tiles for each country."""
    rng = random.Random(seed)
    picked = {}
    for country in COUNTRY_ORDER:
        picked[country] = rng.sample(TILE_POOLS[country], N_TILES_PER_COUNTRY)
    return picked


def list_patches(tile, year=YEAR):
    """Return every patch file path of a tile (from the HuggingFace API)."""
    url = f"{API}/{REPO}/tree/main/data/{year}/{tile}"
    response = requests.get(url, timeout=60)
    response.raise_for_status()
    return sorted(row["path"] for row in response.json()
                  if row.get("type") == "file" and row["path"].endswith(".nc"))


def sample_patches(patch_paths, n, seed):
    """Random sample of n patches; the seed makes the choice reproducible."""
    rng = random.Random(seed)
    return rng.sample(patch_paths, min(n, len(patch_paths)))


# --- Downloading ---------------------------------------------------------------

def download_patch(relpath, dest_dir):
    """Download one patch file into dest_dir; return the local path."""
    dest = Path(dest_dir) / Path(relpath).name
    if dest.exists():
        return dest
    url = f"{RESOLVE}/{REPO}/resolve/main/{relpath}"
    tmp = dest.with_suffix(".part")
    with requests.get(url, stream=True, timeout=180) as response:
        response.raise_for_status()
        with open(tmp, "wb") as fh:
            for chunk in response.iter_content(chunk_size=1 << 20):
                fh.write(chunk)
    tmp.rename(dest)
    return dest


def download_many(relpaths, dest_dir, workers=8):
    """Download several patches in parallel (network-bound, so threads help)."""
    def one(relpath):
        try:
            return download_patch(relpath, dest_dir)
        except Exception as exc:  # keep going if a single download fails
            print(f"    ! failed {relpath}: {exc}")
            return None
    with ThreadPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(one, relpaths))


# --- Manifest ------------------------------------------------------------------

def write_manifest(path, rows):
    """Write the per-patch manifest as CSV (one row per downloaded patch)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = ["country", "tile", "patch", "file", "n_acquisitions", "n_window_dates",
               "n_clean_dates", "chosen_date", "mean_cloud", "n_fields",
               "field_coverage", "eligible", "kept"]
    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nWrote manifest: {path}  ({len(rows)} patches)")


# --- Main ----------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Download a reproducible Sen4AgriNet subset.")
    parser.add_argument("--out-dir", default="data/raw/s4a", help="where kept patches are stored")
    parser.add_argument("--manifest", default="data/manifest.csv", help="manifest CSV path")
    parser.add_argument("--patches", type=int, default=150, help="candidate patches sampled per tile")
    parser.add_argument("--keep", type=int, default=148, help="max eligible patches kept per tile")
    parser.add_argument("--months", default="5",
                        help="comma-separated acquisition months to choose from")
    parser.add_argument("--cloud-threshold", type=float, default=CLOUD_THRESHOLD)
    parser.add_argument("--min-dates", type=int, default=MIN_CLEAN_DATES)
    parser.add_argument("--workers", type=int, default=8, help="parallel downloads")
    parser.add_argument("--seed", type=int, default=TILE_SEED, help="tile-draw seed")
    args = parser.parse_args()
    months = tuple(int(m) for m in args.months.split(","))

    tiles = select_tiles(args.seed)
    print(f"Selected tiles (seed {args.seed}):")
    for country, tile_list in tiles.items():
        print(f"  {country}: {', '.join(tile_list)}")

    rows = []
    for country, tile_list in tiles.items():
        for tile in tile_list:
            print(f"\n== {country} / {tile} ==")
            candidates = sample_patches(list_patches(tile), args.patches,
                                        f"{SAMPLE_SEED}:{tile}")
            dest_dir = Path(args.out_dir) / str(YEAR) / tile
            dest_dir.mkdir(parents=True, exist_ok=True)
            files = download_many(candidates, dest_dir, args.workers)

            kept = 0
            for relpath, path in zip(candidates, files):
                if path is None:
                    continue
                info = patch_info(path, months, args.cloud_threshold)
                clean = [(d, c) for d, c in zip(info["dates"], info["cloud"])
                         if c <= args.cloud_threshold]
                eligible = len(clean) >= args.min_dates
                keep = eligible and kept < args.keep
                chosen = choose_acquisition(clean, f"{SAMPLE_SEED}:{tile}:{Path(relpath).stem}")

                rows.append({
                    "country": country, "tile": tile,
                    "patch": Path(relpath).stem, "file": str(path),
                    "n_acquisitions": info["n_acquisitions"],
                    "n_window_dates": len(info["dates"]),
                    "n_clean_dates": len(clean),
                    "chosen_date": chosen.strftime("%Y-%m-%d") if chosen else "",
                    "mean_cloud": round(float(np.mean(info["cloud"])) if info["cloud"] else 0.0, 4),
                    "n_fields": info["n_fields"],
                    "field_coverage": round(info["field_coverage"], 3),
                    "eligible": int(eligible), "kept": int(keep),
                })
                if keep:
                    kept += 1
            print(f"  kept {kept} of {len(files)} downloaded patches")

    write_manifest(args.manifest, rows)


if __name__ == "__main__":
    main()
