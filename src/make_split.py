"""
Fixed train / val / test split of the 888 kept patches.

The split is by PATCH (parcels of one image never end up on both sides) and stratified
by tile, so every tile contributes the same share to each set. Seeded and deterministic:
the same manifest always gives the same split.

Writes results/split.csv with columns: patch, country, tile, split  (train | val | test).

    python src/make_split.py
    python src/make_split.py --manifest data/manifest.csv --out results/split.csv
"""

import argparse
import csv
from collections import Counter
from pathlib import Path

import numpy as np

SEED = 42
FRACTIONS = (0.70, 0.15, 0.15)  # train, val, test


def make_split(rows, seed=SEED, fractions=FRACTIONS):
    """rows: manifest rows (dicts) with `patch`, `tile`, `country`. Returns split rows."""
    by_tile = {}
    for row in rows:
        by_tile.setdefault(row["tile"], []).append(row)

    out = []
    for tile in sorted(by_tile):
        group = sorted(by_tile[tile], key=lambda r: r["patch"])  # order-independent
        rng = np.random.RandomState(seed)
        order = rng.permutation(len(group))
        n_train = round(fractions[0] * len(group))
        n_val = round(fractions[1] * len(group))
        for rank, idx in enumerate(order):
            name = "train" if rank < n_train else "val" if rank < n_train + n_val else "test"
            out.append({"patch": group[idx]["patch"], "country": group[idx]["country"],
                        "tile": tile, "split": name})
    return sorted(out, key=lambda r: r["patch"])


def main():
    parser = argparse.ArgumentParser(description="Train/val/test split by patch, stratified by tile.")
    parser.add_argument("--manifest", default="data/manifest.csv")
    parser.add_argument("--out", default="results/split.csv")
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    with open(args.manifest) as fh:
        rows = [r for r in csv.DictReader(fh) if r["kept"] == "1"]
    split = make_split(rows, seed=args.seed)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["patch", "country", "tile", "split"])
        writer.writeheader()
        writer.writerows(split)

    print(f"wrote {out} ({len(split)} patches)")
    counts = Counter((r["tile"], r["split"]) for r in split)
    for tile in sorted({r["tile"] for r in split}):
        print(f"  {tile}: " + "  ".join(f"{s}={counts[(tile, s)]}" for s in ("train", "val", "test")))
    totals = Counter(r["split"] for r in split)
    print("  total: " + "  ".join(f"{s}={totals[s]}" for s in ("train", "val", "test")))


if __name__ == "__main__":
    main()
