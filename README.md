# SAM 2 × Sen4AgriNet

Deep Learning, 1INF52, PUCP

## The data

Six Sentinel-2 tiles (3 in Spain, 3 in France, **chosen at random** to avoid bias),
148 patches each, one cloud-free May image per patch → **888 patches**, with
ground-truth field maps. Full reasoning and numbers: [`notebooks/00_en_data_availability.ipynb`](notebooks/00_en_data_availability.ipynb).

## Repository layout

```
src/
  sam2_auto_segment.py     build SAM 2 model + automatic mask generator
  download_process_data/   Download and preprocessing of Sen4AgriNet data
    download_s4a.py          1. download - creates: data/raw/s4a/ and data/manifest.csv
    preprocessing.py         2. processing - creates: subdataset in RGB format
    analyze_availability.py  optional: availability / cloud / field-coverage analysis
    analyze_phenology.py     optional: NDVI-by-month (growing state) analysis
    data_helpers.py          shared library for download and processing
  evaluate_masks.py        baseline: SAM 2 masks vs parcels -> CSVs
  make_split.py            fixed train/val/test split by patch (seed 42, per tile) -> results/split.csv
  teardown.py              Phase A instrumentation (params, MACs, features, ERF, timing)
notebooks/   narrative, experiments, plots — English and Spanish versions
             (00_{en,es}_data_availability, 01_{en,es}_base_model_teardown)
docs/        data_{en,es}.md (data decisions) + decisions_{en,es}.md (full decision log)
checkpoints/ SAM 2 weights (not committed — download separately)
data/        patches + derived CSVs/images (not committed — produced by the scripts)
```

## Setup

Dependencies are pinned in [`environment.yaml`](environment.yaml)

```bash
conda env create -f environment.yaml
conda activate sen4sam
```

Then download the SAM 2 weights into `checkpoints/` — e.g. `sam2.1_hiera_small.pt` from
`https://dl.fbaipublicfiles.com/segment_anything_2/092824/`.

The device is chosen automatically (CUDA → Apple MPS → CPU), so the same code runs
locally and on Colab.


## How to run

Run everything from the repo root, after the [Setup](#setup) above.

### Quickstart

You only need `data/processed/` and `data/manifest.csv`. Get them by either:

- copying them from the [shared folder](https://drive.google.com/drive/folders/1QfVGnmcKq7cZ68suQgPiK6QL5QCSK0dt?usp=share_link) (into `data/`), or
- rebuilding them (next section; the download is ~24 GB).

Then check that everything works:

```bash
python src/evaluate_masks.py --limit 30          # quick test; drop --limit for all 888 patches (long)
```

### Rebuild the processed data from scratch

Two steps. All seeds are fixed and the defaults give the same 888 patches every time, so
keep the defaults.

| # | command | writes | note |
|---|---|---|---|
| 1 | `python src/download_process_data/download_s4a.py` | `data/raw/s4a/`, `data/manifest.csv` | ~24 GB, slow |
| 2 | `python src/download_process_data/preprocessing.py --manifest data/manifest.csv --out-dir data/processed` | `data/processed/<patch>_rgb.png`, `_parcels.npy`, `_labels.npy` | one May RGB image + field maps per patch |

### Optional: analysis behind the data decisions

Only needed to re-create the plots in `notebooks/00_*_data_availability.ipynb`. Not needed
for modelling. They need the raw files from step 1.

```bash
python src/download_process_data/analyze_availability.py               # -> data/availability_*.csv
python src/download_process_data/analyze_phenology.py                  # -> data/phenology_by_month.csv
```

## Documentation — where to read what

| file | what it is |
|---|---|
| `docs/data_en.md` / `data_es.md` | the data + preprocessing decisions, with the measured numbers |
| `docs/decisions_en.md` / `decisions_es.md` | the full decision log (options, evidence, caveats) + decision tree |
| `notebooks/00_{en,es}_data_availability.ipynb` | the measurements behind the data decisions |
| `notebooks/01_{en,es}_base_model_teardown.ipynb` | Phase A: params, compute cost, features, ERF, latency |

## Credits

Sen4AgriNet (Sykas et al., 2022), SAM 2 (Ravi et al., 2024), SAM (Kirillov et al.,
2023), Hollstein et al. (2016) for the cloud method. Dataset and model licences apply.
