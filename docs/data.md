# Data — Sen4AgriNet (Sentinel-2)

We use [Sen4AgriNet](https://github.com/Orion-AI-Lab/S4A) (Sykas et al., 2022), a
Sentinel-2 **time-series** benchmark for agricultural segmentation. Every measured number
below comes from `notebooks/00_data_availability.ipynb`.

## The dataset: basic information

The Sen4AgriNet dataset is based on Sentinel-2 data:

- Spatially it includes regions in Occitanie, France (FR) and Catalonia, Spain (ES).
- The years 2019 and 2020 are covered, but 2020 only includes data for Spain.
- We work with the **sub-dataset Patches Assembled Dataset (PAD)**: image patches with
  pixel-wise labels. (The other one, OAD, is per-parcel statistics, unusable for
  segmentation.)
- **One patch** = one netCDF file on a 10 m grid (366×366 px, inside a ~110×110 km tile of
  hundreds of patches).
- For each patch: 13 bands `B01..B12, B8A` as a time series (~14–70 dates), plus
  `parcels` (per-pixel field id → instance ground truth) and `labels` (crop type →
  semantic ground truth, unused here). Both label maps are annual.

## Data selection
To understand the data selection process please read: `notebooks/00_data_availability.ipynb`.
**Here only a short summary is given.**

The following selections from the base dataset were made:
- Timespan: Only 2019 images: 2020 does not include France and would create a spatially unbalanced dataset.
- Tiles & patches: 3 tiles per country drawn at random, 150 patches each = 900 drawn.
  - Random to avoid bias and ensure real-world conditions
  - Necessary as the whole dataset is too large to process in Google Colab
- We include one acquisition (date) per patch. All of them from the month of May as it combined two important requirements:
  - Cloud coverage (Hollstein et al., 2016) has to be <5 % for each individual patch, as SAM 2 (Ravi et al., 2024) works with one input image only (no time series).
  - Seasonality plays an important role in agricultural land-use analysis. Seasonal patterns are beyond the scope of this study, we focus on the active growing season only to reduce variability and facilitate error analysis.
- **Field coverage:** no filter — keep a realistic mix of agricultural and
  non-agricultural patches.
- **Result:** **888 patches** (148 per tile, 444 ES + 444 FR), one image each.

## Preprocessing

Turn one patch into one RGB image plus its ground truth: pick the **least-cloudy May**
acquisition, build `B04/B03/B02` → ÷10000 → 2nd–98th percentile stretch → uint8, and save
it with `parcels` and `labels`. Metrics are computed only over field pixels
(`parcels ≠ 0`).

## Caveats

- Applying satellite-based imagery to a natural-image model is the **domain shift** (SA-1B, Kirillov et al., 2023 → Sentinel-2) asked for in Phase C.
- Each patch keeps its own least-cloudy May date, so patches span the month (May 1–31).

## References

- Sykas, D., Sdraka, M., Zografakis, D., Papoutsis, I. (2022). *A Sentinel-2
  Multiyear, Multicountry Benchmark Dataset for Crop Classification and
  Segmentation With Deep Learning.* IEEE JSTARS. doi:10.1109/JSTARS.2022.3164771.
- Hollstein, A., Segl, K., Guanter, L., Brell, M., Enesco, M. (2016).
  *Ready-to-Use Methods for the Detection of Clouds, Cirrus, Snow, Shadow, Water
  and Clear Sky Pixels in Sentinel-2 MSI Images.* Remote Sensing 8(8):666.
  doi:10.3390/rs8080666.
- Kirillov, A. et al. (2023). *Segment Anything.* ICCV. arXiv:2304.02643.
- Ravi, N. et al. (2024). *SAM 2: Segment Anything in Images and Videos.*
  arXiv:2408.00714.
