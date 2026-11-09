# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

University project (Deep Learning 1INF52, PUCP): prompt-free field segmentation of Sentinel-2 imagery with a frozen SAM 2.1 (`hiera_small`), evaluated against Sen4AgriNet parcel maps. Phase A = instrumented teardown of the frozen model; Phase B = adapt the mask decoder (`transformer.layers[1]` + `iou_prediction_head`, ~3.7% of params, encoder frozen); Phase C = domain-shift analysis (SA-1B → Sen4AgriNet, not started). `docs/decisions_en.md` holds the decision tree and rationale.

## Setup and commands

```bash
conda env create -f environment.yaml && conda activate sen4sam
# weights (not committed) -> checkpoints/sam2.1_hiera_small.pt
# https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_small.pt
```

Run everything from the repo root. There is no test suite or linter.

```bash
python src/evaluate_masks.py --limit 30        # baseline eval; drop --limit for all 888 patches (slow)
python src/download_process_data/download_s4a.py                                   # ~24 GB -> data/raw/s4a, data/manifest.csv
python src/download_process_data/preprocessing.py --manifest data/manifest.csv --out-dir data/processed
```

Keep download/preprocessing defaults: seeds are fixed so every run yields the same 888 patches. Quickstart shortcut: copy `data/processed/` and `data/manifest.csv` from the shared Drive folder linked in `README.md`.

## Architecture

- `src/sam2_auto_segment.py`: `build_model()` (device auto-picked CUDA → MPS → CPU; eval mode) and `build_mask_generator()`. Operating point: `points_per_side=16, pred_iou_thresh=0.5, stability_score_thresh=0.8`. Notebooks and scripts import from here by putting `src/` on `sys.path`.
- `src/teardown.py`: reusable instrumentation (params, MACs via forward hooks, activation capture, effective receptive field, timing). Used by notebook 01.
- `src/make_split.py` -> `results/split.csv` (patch, country, tile, split): 70/15/15 by patch, stratified by tile, seed 42 (104/22/22 per tile). Training uses `train` only; final evaluation uses `test` only. NB05a, 06 and 07 filter on it.
- `src/evaluate_masks.py`: baseline. For each ground-truth parcel (area ≥ `MIN_PARCEL_AREA`) takes the best IoU over generated masks; scored only on field pixels (`parcels != 0`); writes `data/eval_patches.csv` and `data/eval_summary.csv`.
- `src/download_process_data/`: `data_helpers.py` is the shared library; `download_s4a.py` and `preprocessing.py` are the two pipeline steps; `analyze_*.py` only regenerate the CSVs behind notebook 00.
- Per-patch data in `data/processed/`: `<patch>_rgb.png` (366×366 uint8, May, B04/B03/B02 stretched), `<patch>_parcels.npy` (instance ids, 0 = background), `<patch>_labels.npy`. `data/manifest.csv` column `kept==1` selects the 888 patches (148 per tile × 6 tiles).

## Repository layout

- `results/`: `split.csv` (input, root), `baseline/` (eval_patches/eval_summary/eval_baseline.log/explanation.md), `v1_nb07/` (notebook-07 outputs), `phase_c/` (from `analysis/phase_c.py`), `v2/` (pod experiment outputs).
- `submission/abstract/` and `submission/poster/`: `draft.md` + `build_*.py` (rebuild with `uv run --with python-docx --with pillow python submission/abstract/build_abstract.py` and `uv run --with python-pptx --with qrcode --with pillow python submission/poster/build_poster.py`).
- `course/` (gitignored): assignment PDF, `templates/` for WAIMLAp templates.

## Notebooks

- `00`, `01` exist in `_en` and `_es`; `02`–`06` are Spanish-only and run on **Colab** (clone repo, mount Drive at `/content/drive/MyDrive/Sen4SAM`, unzip `processed.zip`). They are not runnable locally as-is; loaders (`load_rgb`, `load_parcels`) are duplicated per notebook.
- Phase B flow: 03 builds a (pred_iou, true_iou) dataset → 04 post-hoc IoU calibration (null result, only 30 samples from one patch) → 05a builds a per-parcel dataset (id, area, centroid prompt) → 05b/06 reimplement the differentiable path (`predictor.set_image` → `_features["image_embed"/"high_res_feats"]` → `sam_prompt_encoder` → `sam_mask_decoder.predict_masks`) and test a Dice-loss backward step. Real multi-epoch training has not been run.
- Prompt coordinates are in the native 366 px patch space while SAM resizes to 1024; masks from `predict_masks` are 256×256 logits. Check this when training (initial Dice loss was ~0.998).

## Conventions

- Keep docs bilingual (`*_en.md` / `*_es.md`) when editing `docs/`.
- `checkpoints/` and `data/` are not committed.
