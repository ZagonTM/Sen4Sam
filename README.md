# SAM 2 × Sen4AgriNet

Deep Learning, 1INF52, PUCP

# What we do — three phases

- **A · Open the model up.** Parameter and compute cost per block, intermediate
  features, receptive field, latency and memory → `notebooks/01_base_model_teardown.ipynb`.
- **B · Intervene once, measure once.** One change inside the mask decoder (hypothesis
  written *before* the experiment), one controlled ablation against the frozen baseline.
- **C · Test under domain shift.** Evaluate on regions the model has not seen, and
  report honestly what breaks and what does not work.

## The data, in one line

Six Sentinel-2 tiles (3 in Spain, 3 in France, **chosen at random** to avoid bias),
148 patches each, one cloud-free May image per patch → **888 patches**, with
ground-truth field maps. Full reasoning and numbers: [`notebooks/00_data_availability.ipynb`](notebooks/00_data_availability.ipynb).

## Repository layout

```
src/         computation (importable modules + command-line scripts)
  sam2_auto_segment.py     build the SAM 2 model + the automatic mask generator
  download_s4a.py          reproducible (seeded) download of Sen4AgriNet patches
  s4a_io.py                read a patch -> one May RGB image + field maps
  analyze_availability.py  availability / cloud / field-coverage analysis -> CSVs
  analyze_phenology.py     NDVI-by-month (growing state) analysis -> CSV
  evaluate_masks.py        baseline: SAM 2 masks vs parcels -> CSVs
  teardown.py              Phase A instrumentation (params, MACs, features, ERF, timing)
notebooks/   narrative, experiments, plots (00_data_availability, 01_base_model_teardown)
docs/        data.md (data + preprocessing decisions) + decisions.md (full decision log)
checkpoints/ SAM 2 weights (not committed — download separately)
data/        patches + derived CSVs/images (not committed — produced by the scripts)
```

## Setup

Dependencies are pinned in [`environment.yaml`](environment.yaml): Python 3.12 plus
torch, torchvision, `sam2` (from git), numpy, pandas, matplotlib, scipy, tifffile,
pillow, netCDF4, requests.

```bash
conda env create -f environment.yaml
conda activate sen4sam
```

Then download the SAM 2 weights into `checkpoints/` — e.g. `sam2.1_hiera_small.pt` from
`https://dl.fbaipublicfiles.com/segment_anything_2/092824/`.

The device is chosen automatically (CUDA → Apple MPS → CPU), so the same code runs
locally and on Colab.

### Run on Colab

Colab has no MPS but provides a CUDA GPU (choose Runtime → GPU); the code detects it. One
setup cell:

```python
!git clone https://github.com/ZagonTM/Sen4Sam.git
%cd Sen4Sam
!pip install "git+https://github.com/facebookresearch/sam2.git" netCDF4
!mkdir -p checkpoints && wget -q -P checkpoints \
    https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_small.pt
```

Then run the scripts or open the notebooks as below. The full download (900 patches,
~24 GB) is heavy for a free runtime — use `--limit` / `--patches` to try a subset first.

## How to run

```bash
# data (reproducible: same tiles / patches / dates every time, all seeds fixed)
python src/download_s4a.py                       # -> data/raw/s4a/ + data/manifest.csv
python src/s4a_io.py --manifest data/manifest.csv --out-dir data/processed
python src/analyze_availability.py               # -> data/availability_*.csv
python src/analyze_phenology.py                  # -> data/phenology_by_month.csv

# baseline evaluation (needs the processed images above)
python src/evaluate_masks.py --limit 30          # quick test
python src/evaluate_masks.py                     # all 888 patches (long)
```

## Documentation — where to read what

| file | what it is |
|---|---|
| `docs/data.md` | the data + preprocessing decisions, with the measured numbers |
| `docs/decisions.md` | the full decision log (options, evidence, caveats) + decision tree |
| `notebooks/00_data_availability.ipynb` | the measurements behind the data decisions |
| `notebooks/01_base_model_teardown.ipynb` | Phase A: params, compute cost, features, ERF, latency |

## Status

- [x] SAM 2 setup — frozen model, prompt-free mask generator
- [x] Reproducible, seeded data download (900 drawn / 888 used)
- [x] Patch → one May RGB image + ground truth
- [x] Data availability + phenology analysis (cloud, field coverage, NDVI by month)
- [x] Decision log (`docs/decisions.md`)
- [x] **Phase A — instrumented teardown** (params, MACs, features, ERF, latency)
- [ ] Baseline evaluation — SAM 2 masks vs the field maps (running)
- [ ] Phase B — mask-decoder intervention + ablation
- [ ] Phase C — domain-shift analysis + honest failure report

## Credits

Sen4AgriNet (Sykas et al., 2022), SAM 2 (Ravi et al., 2024), SAM (Kirillov et al.,
2023), Hollstein et al. (2016) for the cloud method. Dataset and model licences apply.
