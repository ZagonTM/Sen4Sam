# Changes to the Phase B training setup (8–9 Nov 2026)

What was changed, why, and how it was checked. Cell numbers refer to the notebooks as they are after these changes.

| # | Change | Commit | Files |
|---|---|---|---|
| 1 | Prompt coordinates scaled to SAM's 1024 px frame | `1cda35d` | `06`, `07` |
| 2 | GPU check cell | `ba3fa89` | `06`, `07` |
| 3 | Train/val/test split by patch, used in the notebooks | `e12bd46` | `src/make_split.py`, `results/split.csv`, `05a`, `06`, `07` |
| 4 | Baseline evaluation of the untrained model on the test set | `e12bd46` | `07` |
| 5 | IoU loss for the IoU prediction head | `f369da0` | `07` |
| 6 | Validation set, early stopping, curves and final test evaluation | not committed yet | `07` |

Not part of these changes: `README.md` (one line added for `make_split.py`) and `CLAUDE.md` (new file). Neither is committed yet.

---

## 1. Prompt coordinates (`06`, `07`)

**Problem.** The click position of each parcel (`cx`, `cy`, the centroid) is in pixels of the 366×366 patch. SAM's prompt encoder expects coordinates in the 1024×1024 frame to which `set_image` resizes every image. The notebooks passed the 366 px values directly, so the click landed in the wrong place (for example the pixel (51.8, 37.4) was interpreted as (51.8, 37.4) of 1024 instead of (145.0, 104.6)). This is the likely reason for the Dice loss of about 0.998 in NB06 and NB07.

**Change.** In `06` (cell 13) and `07` (cell 14):

```python
point_coords = predictor._transforms.transform_coords(
    torch.tensor([[[row["cx"], row["cy"]]]], dtype=torch.float32,
                 device=image_embed.device),
    normalize=True,
    orig_hw=(366, 366)
)
```

`transform_coords` divides x by the width and y by the height and multiplies by 1024. This is the same call that `predictor.predict()` uses internally. `orig_hw` is fixed to `(366, 366)` because NB07 caches the encoder features per patch, and `predictor._orig_hw` would then belong to the last encoded image. All processed patches are 366×366.

**Check (frozen model, 40 parcels from 2 patches, centroid prompt, IoU against the parcel):**

| Variant | Mean IoU |
|---|---|
| Old code (366 px coordinates) | 0.045 (median 0.000) |
| Fixed code | 0.440 |
| Official `predictor.predict()` (reference) | 0.450 |

The manual path and the official path give nearly the same mask (median IoU between them 0.96).

**Other inputs.** Image: uint8 `H×W×3` array, which `set_image` converts to [0,1], resizes to 1024×1024 (plain stretch, no padding) and normalises with ImageNet mean/std. Labels: `(1,1)` int64 with value 1. Ground-truth mask: resized with nearest neighbour to the 256×256 logit grid. All consistent.

---

## 2. GPU check cell (`06`, `07`)

A new code cell (cell 2, directly after the setup cell that clones the repo) checks `torch.cuda.is_available()`. If there is no GPU it raises an error that tells you to change the Colab runtime type to a T4 GPU. Set `REQUIRE_GPU = False` in that cell to run without a GPU anyway.

No other code change was needed for GPU use: `build_model()` already picks CUDA first, and every tensor in the loop is created on `image_embed.device`. A run with a CPU runtime was about 25 times slower for the encoder (7–9 s per image against about 0.3 s).

---

## 3. Train/val/test split

**New script `src/make_split.py`**, output `results/split.csv` with columns `patch, country, tile, split`.

- Split by **patch**, so parcels of one image never end up in both training and test.
- 70 / 15 / 15 % per tile, seed 42. Every tile gives 104 train, 22 val and 22 test patches. Total: 624 / 132 / 132.
- Deterministic: running it twice gives identical files. Run with `python src/make_split.py`.

**Use in the notebooks** (the split file is read from `results/split.csv` in the cloned repo, or from `Sen4SAM/split.csv` on Drive):

| Notebook | Change |
|---|---|
| `05a` (cell 18) | The 50 patches for the parcel table are drawn only from `train` patches. |
| `06` (cell 8) | `gt_parcel_dataset.csv` is filtered to `train` patches before the sample of 200 parcels. |
| `07` (cell 6) | Same filter before `TRAIN_SAMPLES` parcels are drawn. |

The `val` set is created but **not used yet**. `03` and `04` still pick their single analysis patch from all 888 patches. They only analyse the IoU head and do not train the decoder.

**Check.** An assertion in a local test confirmed that the training dataframe contained only `train` patches and the evaluated patches only `test` patches.

---

## 4. Baseline on the test set (`07`)

New cells between the model/optimizer cell and the training loop (cells 9–11):

- `evaluate_model_on_patches(model, patches_df)`: runs the automatic mask generator (`points_per_side=16`, `pred_iou_thresh=0.5`, `stability_score_thresh=0.8`, the same settings as `results/eval_patches.csv`) and `evaluate_masks.evaluate_patch` on each patch. Returns a dataframe with the same columns as the baseline CSV.
- `summarize_eval(df, title)`: prints mean best IoU, median best IoU, fraction of fields with IoU > 0.5, mask-on-field fraction and mask count, overall and by country.
- The untrained model is evaluated on all `test` patches. The result is stored in `baseline_test` and saved to Drive as `eval_test_untrained.csv`.
- A control cell compares `mean_best_iou` of these patches with `results/eval_patches.csv`. The difference should be about 0.
- `EVAL_LIMIT` (default `None`) evaluates only the first N test patches, for quick checks.

The model is evaluated before the loop, so the weights are still the original ones. After training, the same function call on the same `test_patches` gives the comparison.

**Check.** On 3 test patches the in-notebook result was identical to the committed baseline (mean best IoU 0.0824, maximum difference per patch 0.0).

---

## 5. IoU loss (`07`)

**Problem.** The IoU prediction head was unfrozen, but the loss was Dice on the mask only. `iou_pred` was computed and never used, so the head received no gradient.

**Change.** Config cell: `IOU_LOSS_WEIGHT = 1.0`. Loop (cell 14):

```python
dice = dice_loss(torch.sigmoid(masks[:, 0:1]), gt_mask)      # token 0, as before

with torch.no_grad():
    pred_bin = (masks > 0).float()                           # all 4 masks
    inter = (pred_bin * gt_mask).sum(dim=(-2, -1))
    union = pred_bin.sum(dim=(-2, -1)) + gt_mask.sum(dim=(-2, -1)) - inter
    true_iou = inter / union.clamp(min=1.0)                  # (1, 4)

iou_loss = F.mse_loss(iou_pred, true_iou)
loss = dice + IOU_LOSS_WEIGHT * iou_loss
```

- One combined loss, one `backward()`, one optimizer step.
- The Dice part reaches `layers[1]`. The IoU part reaches the IoU head and also `layers[1]`, because the IoU token passes through the transformer.
- The target `true_iou` carries no gradient.
- The loss uses all 4 outputs of the IoU head (one per mask token). The automatic generator filters on outputs 1–3, so training only output 0 would leave those unsupervised.
- The first-step check also prints the IoU head's gradient norm. Step and epoch lines show `dice` and `iou_loss` separately.

**Check** (12 parcels, 8 epochs, local test): IoU head gradient norm at the first step 0.51 (before: none). IoU loss 0.113 → 0.004, Dice 0.52 → 0.34. Both decrease together.

---

## 6. Validation, early stopping, full train set, final evaluation (`07`)

Added on top of the committed NB07 without touching the existing loop structure. The first attempt, a rewrite of the training cells, was discarded in favour of these additions.

**Config (cell 4).** `EPOCHS = 100` (maximum), `LR` unchanged at `1e-4`, `N_TRAIN_PATCHES = 200`, `TRAIN_PARCELS_PER_PATCH = 30`, `TRAIN_SAMPLES = None` (optional cap on the total, none by default). New: `PATIENCE = 5`, `MIN_DELTA = 1e-3`, `N_VAL_PATCHES = 30`, `VAL_PARCELS_PER_PATCH = 50`, `LOG_EVERY = 500`.

**Training data (cell 6).** The notebook no longer reads `gt_parcel_dataset.csv` (that file came from only 50 random patches and may not exist in your Drive). It builds the parcel table itself from the parcel maps of `train` patches, with the same definition as NB05a (centroid prompt, minimum area 20 px): 200 random train patches (of 624) with up to 30 parcels each, about 5,500 parcels (tile counts 22–42 per tile). Set `N_TRAIN_PATCHES = None` for all 624 patches (about 14 MB of GPU memory per patch for the cached encoder features) and `TRAIN_PARCELS_PER_PATCH = None` for all parcels. A new check at the start of the cell stops with a clear message if `Drive/Sen4SAM/processed/processed/` is missing. The data is not downloaded: it has to be in your Drive.

**Validation table (cell 6).** 30 random `val` patches (seeded), up to 50 parcels each, same centroid prompt and minimum area (20 px) as training, built from the parcel maps.

**Validation function (new cells 14 and 15).** `evaluate_validation()` computes the same loss (Dice + IoU) as the training loop without gradients, plus `mask_iou` (IoU of the token-0 mask against the parcel). It also defines `get_trainable_state()` and `set_trainable_state()` to save and restore the trained weights.

**Training loop (cell 16).** The original loop is kept. Changes:
- Epoch 0: the untrained model is validated once, as the starting point for early stopping.
- After each epoch: validation, and a history entry (train and validation loss, Dice, IoU loss, mask IoU).
- **Early stopping**: the best epoch is the one with the lowest validation loss. If it does not improve by `MIN_DELTA` for `PATIENCE` epochs in a row, the loop stops. The weights of the best epoch are kept (written to `sam2_decoder_best_trainable.pt` at each new best) and restored after the loop. If validation never improves, the untrained weights are restored.
- Train `mask_iou` is accumulated next to the existing running losses.
- Progress is printed every `LOG_EVERY` parcels instead of for each parcel.
- After the loop, `training_history.csv` is saved. The existing cells that follow (loss plot, example image, `sam2_decoder_finetuned.pt` with the full `state_dict`) are unchanged and now save the best-epoch weights.

**New cells at the end.**
- *Curves* (cell 23): train against validation for the total loss, Dice loss, IoU-head loss and mask IoU, with the best epoch marked. Saved as `training_curves.png`. Pixel accuracy is not used because the background dominates.
- *Final evaluation on the test set* (cell 25): the best-epoch model runs through the same generator and metrics as the baseline (`evaluate_model_on_patches`, same 132 test patches). It prints the untrained and fine-tuned means, the same by country, and a paired per-patch comparison with a bootstrap 95% interval for the change in mean best IoU, plus the number of patches that improve and get worse. Figure: `test_comparison.png`. Saved: `eval_test_finetuned.csv`.
- The old "Resultados preliminares" text from the first run stays, with a note that it must be replaced after the final run.

The test set is used only for the baseline (cell 10) and the final evaluation (cell 25). It does not influence any decision.

**Local test** (6 epochs, 80 train parcels, 4 test patches, tiny configuration): training, validation, early stopping (best epoch 4, stopped at epoch 6), curves and the final comparison ran through. A forced run without improvement stopped after 2 epochs and restored the untrained weights. A leakage assertion confirmed that the train and validation patches do not overlap and come from their split. Not tested: CUDA and the real Drive paths.

**Learning rate check (local, not part of the repository).** 749 train parcels, 2 epochs, validation Dice loss (0.678 untrained): lr 1e-4 gave 0.356, 3e-5 gave 0.350, 1e-5 gave 0.366. Differences are within what one seed can resolve, so `1e-4` was left as the notebook's value.

---

## Decided not to change

| Item | Consequence |
|---|---|
| Mask loss on token 0 only (no multimask change) | The generator uses mask tokens 1–3 (`multimask_output=True`). Token 0 is trained directly; tokens 1–3 only change through the shared `layers[1]`. State this in the report. |
| Checkpoint saved as a plain `state_dict` (no `{"model": ...}` wrapper) | `evaluate_masks.py --checkpoint` cannot load it. Evaluate in the same session with `evaluate_model_on_patches`, or load it with `model.load_state_dict(torch.load(...))` into a fresh `build_model()`. |
| Centroid prompt (no point guaranteed to be inside the parcel) | For curved parcels the centroid may lie outside, which gives a wrong training label for these few parcels. Training uses ground-truth prompts, the test uses a grid: state this limitation. |

## Still open

- The notebook has not yet been run end to end on Colab. The "Resultados" text at the end of NB07 is a placeholder.
- Training uses ground-truth centroid prompts, the test uses the automatic grid. Improvement on the validation loss does not guarantee improvement on the test metrics.
- Cached encoder features take about 14 MB of GPU memory per patch (train and validation patches).

## Before running on Colab

1. Make sure `results/split.csv` and `src/make_split.py` are pushed. The notebooks clone the repo from GitHub. For `05a`, which does not clone the repo, copy `split.csv` to Drive as `Sen4SAM/split.csv`.
2. Set the Colab runtime to GPU. Commit and push the new NB07 first.
3. NB07 needs `Sen4SAM/processed/processed/` in your Drive. It does not need `gt_parcel_dataset.csv` any more (NB06 still does).
