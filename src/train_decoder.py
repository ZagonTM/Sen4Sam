"""
Fine-tune part of the SAM 2 mask decoder on Sen4AgriNet parcels (Phase B), without notebooks.

Trainable: sam_mask_decoder.transformer.layers[1] + iou_prediction_head (encoder and the rest frozen).
Data:      parcels of `train` patches (results/split.csv); `val` patches for early stopping.

Ablation switches (the point of this script):
  --prompt  centroid | random | grid
                                centroid: click = parcel centroid (notebook 07)
                                random:   click = a point drawn uniformly inside the parcel, new every epoch
                                          (closer to what the point grid of the automatic generator does)
                                grid:     click = the actual points of the generator's points_per_side grid
                                          (--grid-pps, default 16) that fall inside a parcel; target = that
                                          parcel. Large parcels get more clicks, exactly as in the generator.
                                All clicks are in 366 px patch coordinates (pixel j covers [j, j+1)) and are
                                mapped to SAM's 1024 frame with the same transform the generator uses
                                (predictor._transforms.transform_coords(normalize=True, orig_hw=(366, 366))).
  --tokens  0 | multi           mask loss on output token 0 (notebook 07)  or on tokens 1-3, best of 3
                                (the tokens the automatic generator actually uses)
  --select-by loss | generator  pick the best epoch / early stop by the validation LOSS (notebook 07)  or by
                                the real metric: mean best IoU of the automatic generator on val patches

The IoU head is always trained against the true IoU of all 4 masks (MSE, weight --iou-weight).

Writes to --out: best_trainable.pt (only the trained tensors), history.csv, config.json.
Evaluate on the test split with:
    python src/evaluate_masks.py --weights <out>/best_trainable.pt --split-file results/split.csv --split test ...

    python src/train_decoder.py --out runs/v2 --prompt random --tokens multi --select-by generator
"""

import argparse
import csv
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from evaluate_masks import evaluate_patch
from sam2_auto_segment import autocast_context, build_mask_generator, build_model

MIN_AREA = 20       # same minimum parcel size as notebook 05a / evaluate_masks.py
ORIG_HW = (366, 366)


# ---------------------------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------------------------

def read_split(path):
    with open(path) as fh:
        return {r["patch"]: r["split"] for r in csv.DictReader(fh)}


def load_rgb(processed, patch):
    return np.array(Image.open(Path(processed) / f"{patch}_rgb.png").convert("RGB"))


def load_parcels(processed, patch):
    return np.load(Path(processed) / f"{patch}_parcels.npy")


def grid_points(n_per_side, hw=ORIG_HW):
    """The generator's point grid in patch pixel coordinates (x, y), same as SAM2AutomaticMaskGenerator."""
    from sam2.utils.amg import build_point_grid
    return build_point_grid(n_per_side) * np.array(hw)[None, ::-1]


def build_table(processed, patches, per_patch, seed, prompt="centroid", grid_pps=16):
    """Training / validation rows: patch, parcel id, click (cx, cy) in patch pixels.

    centroid / random: one row per parcel (cx, cy = centroid; random draws its click later).
    grid: one row per generator grid point that lands on a parcel (target = that parcel).
    Pixel j covers [j, j+1) in SAM's coordinate convention, so a pixel centre is j + 0.5.
    """
    rng = np.random.RandomState(seed)
    rows = []
    for patch in patches:
        parcels = load_parcels(processed, patch)
        ids, areas = np.unique(parcels, return_counts=True)
        valid = {int(i) for i, a in zip(ids, areas) if i != 0 and a >= MIN_AREA}
        if not valid:
            continue
        if prompt == "grid":
            cand = []
            for x, y in grid_points(grid_pps, parcels.shape):
                pid = int(parcels[int(y), int(x)])
                if pid in valid:
                    cand.append({"patch": patch, "parcel_id": pid, "cx": float(x), "cy": float(y)})
            if not cand:
                continue
            size = len(cand) if per_patch <= 0 else min(per_patch, len(cand))
            rows += [cand[k] for k in rng.choice(len(cand), size=size, replace=False)]
            continue
        ids = sorted(valid)
        size = len(ids) if per_patch <= 0 else min(per_patch, len(ids))
        for pid in rng.choice(ids, size=size, replace=False):
            ys, xs = np.where(parcels == pid)
            rows.append({"patch": patch, "parcel_id": int(pid),
                         "cx": float(xs.mean()) + 0.5, "cy": float(ys.mean()) + 0.5})
    return rows


def pick_patches(split, name, n, seed):
    pool = sorted(p for p, s in split.items() if s == name)
    if n <= 0 or n >= len(pool):
        return pool
    return list(np.random.RandomState(seed).choice(pool, size=n, replace=False))


# ---------------------------------------------------------------------------------------------
# Model pieces
# ---------------------------------------------------------------------------------------------

def dice_loss(pred, target, eps=1e-6):
    pred, target = pred.reshape(-1), target.reshape(-1)
    inter = (pred * target).sum()
    return 1 - (2 * inter + eps) / (pred.sum() + target.sum() + eps)


class Trainer:
    def __init__(self, args, device_model):
        self.args = args
        self.model = device_model
        self.predictor = None
        self.cache = {}        # patch -> encoder features + parcel map (GPU / RAM)
        self.image_pe = None

    def setup(self):
        from sam2.sam2_image_predictor import SAM2ImagePredictor
        model = self.model
        for p in model.parameters():
            p.requires_grad = False
        self.last_block = model.sam_mask_decoder.transformer.layers[1]
        self.iou_head = model.sam_mask_decoder.iou_prediction_head
        for p in list(self.last_block.parameters()) + list(self.iou_head.parameters()):
            p.requires_grad = True
        self.trainable = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
        self.predictor = SAM2ImagePredictor(model)
        self.image_pe = model.sam_prompt_encoder.get_dense_pe()
        n = sum(p.numel() for _, p in self.trainable)
        total = sum(p.numel() for p in model.parameters())
        print(f"trainable parameters: {n:,} ({100 * n / total:.2f}% of {total:,})")

    def features(self, patch):
        if patch not in self.cache:
            image = load_rgb(self.args.processed, patch)
            with autocast_context(self.model.device, enabled=self.args.amp):
                self.predictor.set_image(image)       # no_grad inside; frozen encoder -> bf16 is safe
            # decoder training runs in float32: cast the cached features back
            self.cache[patch] = {
                "image_embed": self.predictor._features["image_embed"].float(),
                "high_res_feats": [h.float() for h in self.predictor._features["high_res_feats"]],
                "parcels": load_parcels(self.args.processed, patch),
            }
        return self.cache[patch]

    def state(self):
        return {n: p.detach().cpu().clone() for n, p in self.trainable}

    def load_state(self, state):
        with torch.no_grad():
            for n, p in self.trainable:
                p.copy_(state[n].to(p.device))

    def step_loss(self, row, point):
        """Loss and stats for one parcel and one click (x, y in patch pixels)."""
        f = self.features(row["patch"])
        embed = f["image_embed"]
        coords = self.predictor._transforms.transform_coords(
            torch.tensor([[[point[0], point[1]]]], dtype=torch.float32, device=embed.device),
            normalize=True, orig_hw=ORIG_HW)
        labels = torch.tensor([[1]], dtype=torch.int64, device=embed.device)
        sparse, dense = self.model.sam_prompt_encoder(points=(coords, labels), boxes=None, masks=None)
        masks, iou_pred, _, _ = self.model.sam_mask_decoder.predict_masks(
            image_embeddings=embed, image_pe=self.image_pe,
            sparse_prompt_embeddings=sparse, dense_prompt_embeddings=dense,
            repeat_image=False, high_res_features=f["high_res_feats"])

        gt = torch.tensor((f["parcels"] == row["parcel_id"]).astype(np.float32),
                          device=embed.device)[None, None]
        gt = F.interpolate(gt, size=masks.shape[-2:], mode="nearest")

        if self.args.tokens == "0":
            dice = dice_loss(torch.sigmoid(masks[:, 0:1]), gt)
        else:                                              # tokens 1-3, best of three
            probs = torch.sigmoid(masks[:, 1:4])
            dice = torch.stack([dice_loss(probs[:, i:i + 1], gt) for i in range(3)]).min()

        with torch.no_grad():
            binary = (masks > 0).float()
            inter = (binary * gt).sum(dim=(-2, -1))
            union = binary.sum(dim=(-2, -1)) + gt.sum(dim=(-2, -1)) - inter
            true_iou = inter / union.clamp(min=1.0)        # (1, 4)
        iou_loss = F.mse_loss(iou_pred, true_iou)

        loss = dice + self.args.iou_weight * iou_loss
        # stats stay on the device; they are read once per epoch
        best_iou = true_iou[:, 1:4].max() if self.args.tokens == "multi" else true_iou[0, 0]
        return loss, torch.stack([loss.detach(), dice.detach(), iou_loss.detach(), best_iou])

    def click(self, row, rng):
        if self.args.prompt in ("centroid", "grid"):
            return row["cx"], row["cy"]
        # random: uniform over the parcel's area = random pixel + uniform offset inside that pixel
        parcels = self.features(row["patch"])["parcels"]
        ys, xs = np.nonzero(parcels == row["parcel_id"])
        k = rng.randrange(len(xs))
        return float(xs[k]) + rng.random(), float(ys[k]) + rng.random()


def mean_stats(rows_stats):
    s = torch.stack(rows_stats).mean(dim=0).tolist()
    return dict(zip(["loss", "dice", "iou_loss", "mask_iou"], s))


# ---------------------------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--processed", default="data/processed")
    ap.add_argument("--split-file", default="results/split.csv")
    ap.add_argument("--out", required=True)
    ap.add_argument("--prompt", choices=["centroid", "random", "grid"], default="centroid")
    ap.add_argument("--grid-pps", type=int, default=16, help="grid density for --prompt grid")
    ap.add_argument("--gen-pps", type=int, default=16, help="points_per_side of the generator metric")
    ap.add_argument("--fp32", dest="amp", action="store_false",
                    help="disable bf16 autocast for the frozen encoder / generator on CUDA")
    ap.add_argument("--tokens", choices=["0", "multi"], default="0")
    ap.add_argument("--select-by", choices=["loss", "generator"], default="loss")
    ap.add_argument("--iou-weight", type=float, default=1.0)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--min-delta", type=float, default=1e-3)
    ap.add_argument("--train-patches", type=int, default=200, help="0 = all train patches")
    ap.add_argument("--parcels-per-patch", type=int, default=30, help="0 = all parcels")
    ap.add_argument("--val-patches", type=int, default=30)
    ap.add_argument("--val-parcels-per-patch", type=int, default=50)
    ap.add_argument("--gen-val-patches", type=int, default=20,
                    help="val patches for the generator metric (--select-by generator)")
    ap.add_argument("--accum", type=int, default=1, help="parcels per optimizer step")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    split = read_split(args.split_file)
    train_patches = pick_patches(split, "train", args.train_patches, args.seed)
    val_patches = pick_patches(split, "val", args.val_patches, args.seed)
    gen_patches = pick_patches(split, "val", args.gen_val_patches, args.seed + 1)
    train_rows = build_table(args.processed, train_patches, args.parcels_per_patch, args.seed,
                             args.prompt, args.grid_pps)
    val_rows = build_table(args.processed, val_patches, args.val_parcels_per_patch, args.seed,
                           args.prompt, args.grid_pps)
    print(f"train: {len(train_rows)} parcels / {len(train_patches)} patches | "
          f"val: {len(val_rows)} parcels / {len(val_patches)} patches | generator val: {len(gen_patches)} patches")

    model = build_model()
    trainer = Trainer(args, model)
    trainer.setup()
    optimizer = torch.optim.Adam([p for _, p in trainer.trainable], lr=args.lr)
    # same operating point as evaluate_masks.py; shares `model`, so it always sees the current weights
    generator = build_mask_generator(model, points_per_side=args.gen_pps, pred_iou_thresh=0.5,
                                     stability_score_thresh=0.8)
    assert generator.predictor.model is model
    print(f"device {model.device} | bf16 autocast (encoder/generator) "
          f"{args.amp and str(model.device).startswith('cuda')}")

    # validation clicks are fixed once, so every epoch is scored on identical prompts
    val_rng = random.Random(args.seed)
    val_points = [trainer.click(r, val_rng) for r in val_rows]

    def validate():
        stats = []
        with torch.no_grad():
            for row, point in zip(val_rows, val_points):
                stats.append(trainer.step_loss(row, point)[1])
        return mean_stats(stats)

    def generator_metric():
        scores = []
        for patch in gen_patches:
            m = evaluate_patch(load_rgb(args.processed, patch),
                               load_parcels(args.processed, patch), generator, amp=args.amp)
            scores.append(m["mean_best_iou"])
        return float(np.mean(scores))

    def score_of(val_stats, gen_score):
        # higher is better
        return gen_score if args.select_by == "generator" else -val_stats["loss"]

    history = []
    v = validate()
    g = generator_metric() if args.select_by == "generator" else float("nan")
    history.append({"epoch": 0, **{f"val_{k}": x for k, x in v.items()}, "val_gen_mean_best_iou": g})
    print(f"epoch 0 (untrained): val_loss={v['loss']:.4f} val_mask_iou={v['mask_iou']:.4f} gen_iou={g:.4f}")
    best_score, best_epoch, best_state, bad = score_of(v, g), 0, trainer.state(), 0

    train_rng = random.Random(args.seed + 1)
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        order = list(range(len(train_rows)))
        train_rng.shuffle(order)
        stats, n_in_step = [], 0
        optimizer.zero_grad()
        for idx in order:
            row = train_rows[idx]
            loss, s = trainer.step_loss(row, trainer.click(row, train_rng))
            (loss / args.accum).backward()
            n_in_step += 1
            if n_in_step == args.accum:
                optimizer.step()
                optimizer.zero_grad()
                n_in_step = 0
            stats.append(s)
        if n_in_step:
            optimizer.step()
            optimizer.zero_grad()

        tr = mean_stats(stats)
        v = validate()
        g = generator_metric() if args.select_by == "generator" else float("nan")
        history.append({"epoch": epoch, **{f"train_{k}": x for k, x in tr.items()},
                        **{f"val_{k}": x for k, x in v.items()}, "val_gen_mean_best_iou": g})

        score = score_of(v, g)
        improved = score > best_score + args.min_delta
        if improved:
            best_score, best_epoch, best_state, bad = score, epoch, trainer.state(), 0
            torch.save(best_state, out / "best_trainable.pt")
        else:
            bad += 1
        print(f"epoch {epoch}: train_loss={tr['loss']:.4f} val_loss={v['loss']:.4f} "
              f"val_mask_iou={v['mask_iou']:.4f} gen_iou={g:.4f} | {time.time() - t0:.0f}s"
              f"{' <- best' if improved else ''}", flush=True)
        if bad >= args.patience:
            print(f"early stopping after {args.patience} epochs without improvement; best epoch {best_epoch}")
            break

    torch.save(best_state, out / "best_trainable.pt")
    with open(out / "history.csv", "w", newline="") as fh:
        keys = list(history[0].keys())
        for h in history:
            for k in h:
                if k not in keys:
                    keys.append(k)
        writer = csv.DictWriter(fh, fieldnames=keys)
        writer.writeheader()
        writer.writerows(history)
    with open(out / "config.json", "w") as fh:
        json.dump({**vars(args), "best_epoch": best_epoch, "epochs_run": len(history) - 1,
                   "train_parcels": len(train_rows), "train_patches": len(train_patches)}, fh, indent=2)
    print(f"done. best epoch {best_epoch}; weights: {out / 'best_trainable.pt'}")


if __name__ == "__main__":
    main()
