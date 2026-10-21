"""
Evaluate prompt-free SAM 2 against the Sen4AgriNet ground truth (parcels).

For every processed patch we run the automatic mask generator (no prompts) and score
its masks against the parcel map:

  * n_masks, image coverage, decoder calls (points_per_side**2 = the "cost of the prompt")
  * best IoU per real field: each parcel is matched to its best-overlapping SAM 2 mask,
    reported overall and split into the smaller / larger half of fields (tiny fields are
    dominated by boundary effects)
  * masks-on-field fraction: how much of a produced mask actually lands on fields
  * GT-vs-image alignment: edge strength on parcel boundaries vs. elsewhere — a sanity
    check that an IoU deficit is the model's, not a georeferencing artifact

Outputs: data/eval_patches.csv (one row per patch) and data/eval_summary.csv
(per tile / country / total).

Usage:
    python src/evaluate_masks.py --limit 30      # quick test
    python src/evaluate_masks.py                 # all patches
"""

import argparse
import csv
import time
from pathlib import Path

import numpy as np
from PIL import Image

# The model imports live in main() so the metric functions below can be imported
# (and tested) without loading torch.

MIN_PARCEL_AREA = 20  # ignore smaller parcels (pure boundary noise)


def load_image(path):
    """Load an RGB image as the uint8 HxWx3 array SAM 2 expects."""
    return np.array(Image.open(path).convert("RGB"))


def generate_masks(generator, image):
    """Run the prompt-free generator; return an (N, H, W) boolean stack (N may be 0)."""
    records = generator.generate(image)
    if not records:
        return np.zeros((0, *image.shape[:2]), bool)
    return np.stack([r["segmentation"] for r in records]).astype(bool)


def best_iou_per_parcel(masks, parcels, min_area=MIN_PARCEL_AREA):
    """Best IoU of each field with any SAM 2 mask.

    Returns (parcel_ids, areas, best_ious) for parcels of at least `min_area` pixels.
    """
    unique, inverse = np.unique(parcels, return_inverse=True)
    inverse = inverse.ravel()
    areas = np.bincount(inverse, minlength=unique.size)

    best = np.zeros(unique.size)
    for mask in masks:
        flat = mask.ravel()
        inter = np.bincount(inverse, weights=flat, minlength=unique.size)
        union = float(flat.sum()) + areas - inter
        best = np.maximum(best, inter / np.maximum(union, 1e-9))

    keep = areas >= min_area
    if unique.size and unique[0] == 0:
        keep[0] = False  # 0 is background, not a field
    return unique[keep], areas[keep], best[keep]


def boundary_alignment(image, parcels):
    """Mean image-gradient on parcel boundaries divided by that elsewhere."""
    gray = image.astype(float).mean(axis=2)
    grad = np.hypot(*np.gradient(gray))
    boundary = np.zeros_like(parcels, bool)
    for axis in (0, 1):
        diff = np.diff(parcels.astype(np.int64), axis=axis) != 0
        sl = [slice(None)] * 2
        sl[axis] = slice(1, None)
        boundary[tuple(sl)] |= diff
    return float(grad[boundary].mean() / (grad[~boundary].mean() + 1e-9))


def evaluate_patch(image, parcels, generator):
    """All metrics for one patch."""
    masks = generate_masks(generator, image)
    height, width = parcels.shape
    union = masks.any(axis=0) if len(masks) else np.zeros((height, width), bool)

    _, areas, ious = best_iou_per_parcel(masks, parcels)
    on_field = [float((m & (parcels != 0)).sum() / max(m.sum(), 1)) for m in masks]

    # split fields at the patch's median field size
    if len(areas):
        big = areas >= np.median(areas)
        small_iou = float(ious[~big].mean()) if (~big).any() else float("nan")
        large_iou = float(ious[big].mean()) if big.any() else float("nan")
    else:
        small_iou = large_iou = float("nan")

    return {
        "n_masks": len(masks),
        "coverage": round(float(union.mean()), 4),
        "n_fields": int(len(areas)),
        "mean_best_iou": round(float(ious.mean()), 4) if len(ious) else 0.0,
        "median_best_iou": round(float(np.median(ious)), 4) if len(ious) else 0.0,
        "frac_iou_gt_0.5": round(float((ious > 0.5).mean()), 4) if len(ious) else 0.0,
        "small_mean_iou": round(small_iou, 4) if small_iou == small_iou else "",
        "large_mean_iou": round(large_iou, 4) if large_iou == large_iou else "",
        "mask_on_field_fraction": round(float(np.mean(on_field)), 4) if on_field else 0.0,
        "alignment": round(boundary_alignment(image, parcels), 3),
    }


def aggregate(group, name, key="tile"):
    """One summary row for a group of patches."""
    return {
        key: name,
        "patches": len(group),
        "mean_n_masks": round(float(np.mean([r["n_masks"] for r in group])), 1),
        "mean_coverage": round(float(np.mean([r["coverage"] for r in group])), 4),
        "mean_best_iou": round(float(np.mean([r["mean_best_iou"] for r in group])), 4),
        "mean_frac_iou_gt_0.5": round(float(np.mean([r["frac_iou_gt_0.5"] for r in group])), 4),
        "mean_mask_on_field": round(float(np.mean([r["mask_on_field_fraction"] for r in group])), 4),
    }


def summarise(rows, key):
    """Per-group rows (grouped by `key`) plus an ALL row."""
    groups = {}
    for row in rows:
        groups.setdefault(row[key], []).append(row)
    return [aggregate(g, name, key) for name, g in sorted(groups.items())] + \
           [aggregate(rows, "ALL", key)]


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {path} ({len(rows)} rows)")


def main():
    parser = argparse.ArgumentParser(description="Evaluate prompt-free SAM 2 on Sen4AgriNet.")
    parser.add_argument("--manifest", default="data/manifest.csv")
    parser.add_argument("--processed", default="data/processed")
    parser.add_argument("--out-patches", default="data/eval_patches.csv")
    parser.add_argument("--out-summary", default="data/eval_summary.csv")
    parser.add_argument("--limit", type=int, default=0, help="only N patches (0 = all)")
    parser.add_argument("--points-per-side", type=int, default=16,
                        help="prompt grid; decoder calls = this squared")
    parser.add_argument("--pred-iou", type=float, default=0.5, help="mask quality filter")
    parser.add_argument("--stability", type=float, default=0.8, help="stability filter")
    parser.add_argument("--checkpoint", default=None)
    args = parser.parse_args()

    from sam2_auto_segment import build_mask_generator, build_model, get_device

    with open(args.manifest) as fh:
        rows = [r for r in csv.DictReader(fh) if r["kept"] == "1"]
    if args.limit:
        rows = rows[:args.limit]

    device = get_device()
    model = build_model(checkpoint=args.checkpoint) if args.checkpoint else build_model()
    generator = build_mask_generator(model, points_per_side=args.points_per_side,
                                     pred_iou_thresh=args.pred_iou,
                                     stability_score_thresh=args.stability)
    print(f"device {device} | points_per_side {args.points_per_side} "
          f"({args.points_per_side ** 2} decoder calls) | pred_iou {args.pred_iou} "
          f"| stability {args.stability}")

    results = []
    started = time.time()
    for i, row in enumerate(rows, 1):
        image = load_image(Path(args.processed) / f"{row['patch']}_rgb.png")
        parcels = np.load(Path(args.processed) / f"{row['patch']}_parcels.npy")
        metrics = evaluate_patch(image, parcels, generator)
        results.append({"country": row["country"], "tile": row["tile"],
                        "patch": row["patch"], "date": row["chosen_date"], **metrics})
        if i % 10 == 0 or i == len(rows):
            rate = (time.time() - started) / i
            print(f"[{i}/{len(rows)}] {rate:.1f} s/patch, "
                  f"ETA {(len(rows) - i) * rate / 60:.1f} min", flush=True)

    write_csv(args.out_patches, results)
    summary = summarise(results, "tile")
    write_csv(args.out_summary, summary)

    print("\nper-tile summary:")
    for row in summary:
        print(f"  {row['tile']}: masks {row['mean_n_masks']} | "
              f"coverage {row['mean_coverage']} | mean IoU {row['mean_best_iou']} | "
              f"fields@IoU>0.5 {row['mean_frac_iou_gt_0.5']} | "
              f"mask-on-field {row['mean_mask_on_field']}")


if __name__ == "__main__":
    main()
