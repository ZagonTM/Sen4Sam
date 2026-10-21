# Data — Sen4AgriNet (Sentinel-2)

The dataset we use, and the decisions behind the subset we train and evaluate on.
Every measured number below is produced by `src/analyze_availability.py` /
`src/analyze_phenology.py` and shown in `notebooks/00_data_availability.ipynb`.

## The data

[Sen4AgriNet](https://github.com/Orion-AI-Lab/S4A) is a Sentinel-2 **time-series**
benchmark for agricultural monitoring (Sykas et al., 2022), built for crop
classification and semantic segmentation.

- **Sensor:** Sentinel-2 **L1C** (top-of-atmosphere), digital numbers ÷10000. Only
  scenes with **<10% cloud** are downloaded — a scene-level filter, not pixel-level, so
  a patch can still contain clouds.
- **Coverage:** 2019 and 2020; two regions — Catalonia (ES) and Occitanie (FR). Labels
  come from farmer declarations (LPIS), harmonised into ~168 crop classes.
- **Sub-dataset used:** the **Patches Assembled Dataset (PAD)** — image patches with
  pixel-wise labels. (The other one, OAD, is per-parcel statistics, unusable here.)
- **Access:** HuggingFace `orion-ai-lab/S4A`.

**One patch** is one netCDF file (`{year}_{tile}_patch_{i}_{j}.nc`) on a 10 m grid
(366×366 px, ~3.7×3.7 km), inside a ~110×110 km MGRS **tile** that holds hundreds of
patches. It stores:

- 13 bands `B01..B12, B8A`, each a time series of the year's acquisitions (~14–70);
- `parcels` — per-pixel **field id** (0 = background) → instance ground truth;
- `labels` — per-pixel **crop type** → semantic ground truth (not used in our
  instance-level evaluation);

Both label maps are **annual**, so they are valid for any image built from that year.

## Decisions

- **Dataset → PAD.** The only candidate shipping pixel-level labels aligned to the imagery.
- **Year → 2019.** 2020 contains only Catalonia, which would remove the ES/FR comparison.
- **Tiles → 3 per country, drawn at random (seed 42):** ES `31TBF`, `31TDG`, `31TCG`;
  FR `31TCL`, `31TDK`, `31UCP`. Random avoids biasing towards good-looking tiles.
- **Patches → 150 drawn per tile (900), 148 kept per tile = 888** (444 ES + 444 FR),
  random (seed 42), with **no field-coverage filter**: a realistic mix of field-rich and
  near-empty patches (coverage p10 0.018 / median 0.409 / p90 0.791 — notebook §3).
- **One image per patch, not a median.** Cloud is negligible — median acquisition
  0.000, only 4.6% above 5% (notebook §1) — so there is nothing to composite; and a May
  median would cover just 31% of patches (notebook §4).
- **Window → May.** The greenest month that still covers (almost) every patch: field NDVI
  peaks in **May (0.58)** and is back to the winter baseline in **July (0.38)**; July
  covers every patch but May all but two (99.8%), while June is 75% and April 61%
  (notebook §1–§2). **2 drawn patches (both `31TCL`) have no clean May acquisition, so
  every tile is trimmed to the same 148 → 888 used** (notebook §4). The image is the
  least-cloudy May acquisition (≤5%), random among ties.
- **Input image → `B04/B03/B02`** (true colour), ÷10000, per-band 2nd–98th percentile
  stretch → `H×W×3` uint8. SAM 2 expects 3-channel RGB; dropping the other 10 bands is
  deliberate and part of the domain-shift story.

Rejected alternatives: a multi-month median; a wider window (May–August or the whole
year) — measured, it spreads the date across the season for no greenness gain; June
(green but 75% coverage); July (full coverage but past the greenness peak).

## Preprocessing

Turn one patch into one RGB image plus its ground truth:

1. Download randomly sampled patches (HuggingFace).
2. Select the least-cloudy **May** acquisition (≤5% cloud).
3. `B04/B03/B02` → ÷10000 → 2nd–98th percentile stretch → uint8.
4. Save the RGB image alongside `parcels` and `labels`. Metrics are computed only over
   field pixels (`parcels ≠ 0`).

## Caveats

- A stretched satellite view fed to a natural-image model is itself the **domain shift**
  (SA-1B → Sentinel-2) — the Phase C point, not a bug.
- Each patch uses its own least-cloudy date, so patches span the whole month (May 1–31).
- The cloud gate mildly favours clearer sites — stated, not hidden.

## References

- Sykas, D., Sdraka, M., Zografakis, D., Papoutsis, I. (2022). *A Sentinel-2
  Multiyear, Multicountry Benchmark Dataset for Crop Classification and
  Segmentation With Deep Learning.* IEEE JSTARS. doi:10.1109/JSTARS.2022.3164771.
- Drusch, M. et al. (2012). *Sentinel-2: ESA's Optical High-Resolution Mission
  for GMES Operational Services.* Remote Sensing of Environment. doi:10.1016/j.rse.2011.11.026.
- Hollstein, A., Segl, K., Guanter, L., Brell, M., Enesco, M. (2016).
  *Ready-to-Use Methods for the Detection of Clouds, Cirrus, Snow, Shadow, Water
  and Clear Sky Pixels in Sentinel-2 MSI Images.* Remote Sensing 8(8):666.
  doi:10.3390/rs8080666.
- Kirillov, A. et al. (2023). *Segment Anything.* ICCV. arXiv:2304.02643.
- Ravi, N. et al. (2024). *SAM 2: Segment Anything in Images and Videos.*
  arXiv:2408.00714.
