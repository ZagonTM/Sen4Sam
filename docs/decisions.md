# Decision log — SAM 2 × Sen4AgriNet

Every design decision, the options we weighed, and the evidence behind the choice.
Measured numbers come from `notebooks/00_data_availability.ipynb` (data) and
`src/evaluate_masks.py` (baseline). Each entry ends with its caveat, so the weak
points are on the record too.

## The decision tree at a glance

```
Segment individual farm fields in Sentinel-2 imagery with SAM 2, without human prompts
│
├── 1. DATA SCOPE
│   ├─ which dataset?        → Sen4AgriNet PAD        (needs pixel-level labels)
│   ├─ which year?           → 2019                   (only year with ES *and* FR)
│   ├─ which tiles?          → 3 per country, RANDOM (seed 42)
│   ├─ which patches?        → 148 per tile, random  (seed 42), no coverage filter
│   ├─ which month?          → MAY                    (greenest month covering every patch)
│   ├─ one date or median?   → ONE real acquisition   (not a composite)
│   ├─ cloud rule?           → ≤5% (Hollstein), take the least cloudy
│   └─ input image?          → B04/B03/B02, ÷10000, 2–98% stretch → uint8
│
├── 2. MODEL & TASK
│   ├─ which model?          → SAM 2.1 hiera_small, FROZEN
│   ├─ prompted?             → NO — prompt-free automatic generator (the task)
│   └─ operating point?      → points_per_side 16, pred_iou 0.5, stability 0.8
│
├── 3. EVALUATION
│   ├─ metric?               → instance: best IoU per real field
│   ├─ where measured?       → only field pixels (parcels ≠ 0)
│   └─ guards?               → stratify by field size + GT-alignment sanity check
│
├── 4. PHASE B (planned)
│   └─ intervention?         → ?
│
└── 5. PHASE C
    └─ domain shift?         → SA-1B → Sen4AgriNet
```

## 1. Data scope

### 1.1 Which dataset → Sen4AgriNet, Patches Assembled Dataset (PAD)
- **Options:** Sen4AgriNet vs generic Sentinel-2 tiles vs synthetic benchmarks.
- **Decision:** PAD. It is the only candidate shipping **pixel-level** ground truth
  (`parcels`, `labels`) aligned to Sentinel-2 time series — segmentation needs spatial labels.
- **Caveat:** the different sub-dataset (OAD) is per-parcel statistics, unusable here.

### 1.2 Which year → 2019
- **Options:** 2019; 2020; both.
- **Decision:** 2019. 2020 contains **only Catalonia**, which would delete the cross-country
  comparison.
- **Caveat:** one year limits seasonal/crop-year diversity.

### 1.3 Which tiles → 3 per country, drawn at random (seed 42)
- **Options:** (a) rank tiles by field count/acquisitions and keep the best; (b) draw at random.
- **Decision:** (b) random — ES {31TBF, 31TDG, 31TCG}, FR {31TCL, 31TDK, 31UCP}. Picking the
  best-looking tiles would bias the results in the model's favour.
- **Evidence:** random selection is the honest choice; ranking was explicitly rejected. Because the
  tiles are random, no exhaustive per-tile profiling was needed.
- **Caveat:** the draw handed us **31TDG, which is largely non-agricultural** — median field
  coverage 0.061 vs 0.73 for 31UCP (`00_data_availability.ipynb`, §3). We keep it (that is what a
  random draw does) and report it rather than quietly swapping it out.

### 1.4 Which patches → 148 per tile, random, no field-coverage filter
- **Options:** (a) keep only patches with enough fields; (b) random sample, keep all.
- **Decision:** (b). Field coverage is **recorded and reported as a distribution, never used to
  exclude patches**.
- **Evidence:** coverage spans p10 0.018 / median 0.408 / p90 0.788 — a real-world mix. Filtering
  would make the metrics optimistic and delete useful "no field here" negatives.
- **Caveat:** the mild cloud/date gate still biases the sample slightly toward clearer locations.
- **Note:** 150 candidates are sampled per tile, but only **148** are kept — the May window
  leaves 2 patches (both `31TCL`) uncovered, so every tile is trimmed to the same 148 for balance.

### 1.5 Which month → May
- **Options:** April; May; June; July; a wider window (May–August); a monthly median.
- **Decision:** **May** — the greenest month that still covers (almost) every patch.
- **Evidence** (`00_data_availability.ipynb`, §1 & §2): field NDVI peaks in **May (0.58)** and has
  fallen back to the winter baseline by July (0.38). Coverage (share of patches with ≥1 clean date)
  and greenness together:

  | month | patches with ≥1 clean date | mean field NDVI |
  |---|---|---|
  | April | 61% | 0.57 |
  | **May** | **100%** | **0.58** |
  | June | 75% | 0.43 |
  | July | 100% | 0.38 |
  | August | 100% | 0.37 |

  Only May and July reach ~100% coverage, so May wins on greenness at equal coverage. May has 1,911
  acquisitions overall, 82% of them *exactly* cloud-free. A single month also keeps the imagery
  phenologically consistent across patches, which a multi-month window cannot.
- **Caveat:** a May per-pixel median would cover only 31% of patches (fewer clean dates per patch
  than July), and May is slightly cloudier than July (95% vs 98% of acquisitions ≤5% cloud).

### 1.6 One date or a median → one real acquisition
- **Options:** (a) monthly median; (b) one acquisition (least cloudy, random among ties).
- **Decision:** (b). SAM 2 consumes a **single image**; a median would only replace a clean
  observation with a synthetic one.
- **Evidence:** cloud is **negligible** — median cloud fraction 0.000, only 4.6% of acquisitions
  exceed 5%. A median is a cloud-rejection tool we simply do not need. A per-pixel median across
  dates also mixes phenology (different pixels sampling different dates) and is not the dataset's
  own convention (theirs is a median *within* a month).
- **Caveat:** patches do not all come from the same day (May dates span 1–31).

### 1.7 Cloud rule → Hollstein ≤5%, pick the least cloudy
- **Options:** ≤5%; ≤1%; exactly 0.
- **Decision:** ≤5%, using the dataset's own Hollstein formula, and then take the least cloudy
  acquisition (random among ties).
- **Evidence:** requiring *exactly* 0 would leave only 838/888 patches covered (94%), and a single
  false-positive pixel (a roof edge, bare soil) turns a visually clean image into "0.2% cloud".
  82% of May acquisitions are exactly 0 anyway.
- **Caveat:** the gate mildly favours clearer sites.

### 1.8 Input image → B04/B03/B02, ÷10000, 2–98% percentile stretch
- **Options:** true colour vs false colour (NIR) vs other band sets.
- **Decision:** true colour `B04/B03/B02` (the native 10 m bands), divided by 10000 (L1C
  reflectance), stretched per band by its 2nd–98th percentile to uint8.
- **Rationale:** SAM 2 expects 3-channel RGB. Throwing away the other 10 bands *is* part of the
  domain-shift story. The stretch matters because raw reflectance is dark and low-contrast, and
  SAM 2 was trained on natural images.
- **Caveat:** no NIR means we drop information that would help vegetation — deliberate, and it is
  exactly the kind of cost the project examines.

## 2. Model and task

### 2.1 Which model → SAM 2.1 `hiera_small`, frozen
- **Options:** SAM 2 `tiny` / `small` / `base+` / `large`.
- **Decision:** `hiera_small` (46.06 M params) as the main model; `tiny` (38.96 M) for fast
  iteration. Both weights are local.
- **Evidence:** the four sizes differ **only** in `image_encoder.trunk` — mask decoder, prompt
  encoder and memory modules are identical — so the decoder analysis is size-invariant.
- **Caveat:** the frozen model is the baseline, not an adapted one; adaptation is Phase B.

### 2.2 Prompted or prompt-free → prompt-free (automatic generator)
- **Options:** human/point prompts; the automatic mask generator.
- **Decision:** prompt-free. It is literally the assigned task ("adapt without human prompts") and
  the source of the "cost of the prompt" measure: `points_per_side²` decoder calls.

### 2.3 Operating point → relaxed thresholds
- **Options:** SAM 2 defaults (`pred_iou` 0.8 / stability 0.95) vs a relaxed setting.
- **Decision:** `points_per_side`=16 (256 decoder calls), `pred_iou`=0.5, stability=0.8.
- **Evidence:** at the **defaults the model returns 0 masks** on our imagery — itself direct
  evidence of the domain shift, and useless as a baseline.
- **Caveat:** the operating point changes the numbers; a threshold sweep is a natural follow-up.

## 3. Evaluation

### 3.1 Metric → instance matching, best IoU per field
- **Options:** pixel accuracy; semantic IoU; per-field best match.
- **Decision:** for each real field (`parcels`), find its best-overlapping SAM 2 mask and record
  IoU. SAM 2 returns an unordered mask *list*, not a partition, so instance matching is the honest
  framing. Reported: mean/median best IoU, share of fields at IoU>0.5, plus mask count, coverage
  and mask-on-field fraction.
- **Caveat:** SAM 2 masks may nest/overlap, so "best match" flatters a model that over-segments.

### 3.2 Where measured → field pixels only
- **Decision:** restrict metrics to `parcels ≠ 0`. Filtering *which patches* we keep (never) is a
  different decision from restricting *where* we measure (yes).

### 3.3 Guards → size stratification + alignment sanity check
- **Decision:** split fields at each patch's median size, and record an image-alignment score
  (edge strength on parcel boundaries vs elsewhere) with every patch.
- **Evidence:** alignment is 1.2–2.2× everywhere and the best-fit shift is (0,0) in 8/12 patches
  (±1 px otherwise), so a low IoU is the **model's** doing, not a georeferencing artifact.
  Larger fields score ~3× better than smaller ones — a finding, not noise.

## 4. Phase B — the intervention (planned)

- **Decision:** adapt the **mask decoder** (candidate: unfreeze its last *k* layers), leaving the
  image encoder frozen; the controlled ablation is frozen vs unfrozen last *k* layers.
- **Why the decoder:** it is the graded core of the assignment and it is small (~9% of params), so
  it is the cheapest place to test a hypothesis.
- **Caveat:** a classification head bolted on top would bypass the decoder and forfeit the graded
  core — not an option.

## 5. Phase C — domain shift and honesty

- **Decision:** the **primary** domain shift is SA-1B (natural photos) → Sen4AgriNet (Sentinel-2);
  it is present in every experiment, not a separate step. The **ES vs FR** split is relabelled a
  *within-domain cross-region robustness check*.
- **Evidence:** the frozen model returns 0 masks at default thresholds and matches only 1 of 30
  real fields at IoU>0.5 — the shift is measured, not assumed.

## Where AI assistance was used (explicitly graded)

- **Used for:** drafting and refactoring the `src/` scripts (download, I/O, availability analysis,
  evaluator) and the notebook scaffolding; running the measurements; drafting wording in these
  docs.
- **Decided by us (the team):** the data scope, the month choice, the one-date rule, the metric
  design and the Phase B intervention. Every such decision is listed above with the evidence we
  based it on.
- **Verification:** real runs back every number (888-patch download, the availability CSVs, the
  baseline evaluation), and `python tests/test_s4a.py` pins the data layer's behaviour.
