# Decision log — SAM 2 × Sen4AgriNet

Every design decision, the options we weighed, and the evidence behind the choice.
Measured numbers come from `notebooks/00_en_data_availability.ipynb` (data) and
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
│   ├─ which month?          → MAY                    (greenest month covering (almost) every patch)
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
├── 4. PHASE B
│   ├─ what is trained?      → transformer.layers[1] + iou_prediction_head (1.71 M, ~3.7%)
│   ├─ loss?                 → Dice (token 0) + MSE on the IoU head
│   ├─ training prompt?      → parcel centroid
│   ├─ stopping rule?        → early stopping on val loss (v1); on generator IoU (v2)
│   └─ v1 result?            → val improves, real test collapses (0.156 → 0.064)
│
└── 5. PHASE C
    └─ domain shift?         → SA-1B → Sen4AgriNet
```
## 4. Phase B — the intervention

### 4.1 Decisions taken (v1, notebook 07)
- **What is trained:** only `transformer.layers[1]` of the mask decoder plus
  `iou_prediction_head` (1.71 M params, ~3.7% of the model); image encoder and the rest of the
  decoder stay frozen. Why: the decoder is the graded core of the task, it is small, and it is
  the cheapest place to test a hypothesis; an external classification head would bypass it and
  is not an option.
- **Loss:** Dice on output token 0 + MSE of the IoU head against the true IoU of all 4 masks.
  Why: Dice handles class imbalance (small fields); the MSE keeps the IoU head calibrated while
  the masks move.
- **Training prompt:** parcel centroid (scaled to SAM's 1024 px frame), 200 train patches x up
  to 30 parcels. Why: simplest deterministic prompt with a label available.
- **Stopping:** early stopping on validation loss; best epoch = 4.
- **v1 result (honest):** on validation with centroid prompts it improved: Dice loss
  0.704 -> 0.342, mask IoU 0.312 -> 0.530, IoU-head MSE 0.099 -> 0.027. On the **real test with
  the automatic generator** (132 patches) it got **much worse**: mean best IoU 0.156 -> 0.064
  (paired delta -0.092, 95% bootstrap CI [-0.103, -0.083]); fields with IoU > 0.5: 14.2% ->
  5.9%; masks per patch 78.7 -> 41.1; image coverage 0.59 -> 0.10; mask-on-field unchanged
  (0.58 -> 0.59); 128 of 132 patches worse, all tiles and both countries.
- **Interpretation:** masks disappear (recall collapse) rather than getting worse. The
  validation loss could not warn because it measures the same prompt type that was trained.
  Hypotheses, **not verified**: (1) the IoU head became pessimistic for grid points off parcel
  centres and the `pred_iou_thresh=0.5` filter drops them; (2) tokens 1-3 (used by the
  generator) got worse because only token 0 was trained; (3) the stability filter drops more
  masks.
- **Caveat:** the real bottleneck may be prompt density, not the decoder: the median parcel is
  ~10x10 px and a 16x16 grid on 366 px puts a point every ~23 px, so a median parcel gets a
  click ~20% of the time (arithmetic, not a measurement).

### 4.2 Phase B v2 ablation — expectations written BEFORE results (timestamp 2026-11-08 22:15 UTC-5)

All runs are evaluated on the 132 test patches with the automatic generator. Trained runs are
selected by the generator's mean best IoU on val (except `A_loss_select`). The "Result:" lines
are filled in later, without editing the prediction.

**Diagnostics on the untrained model**

1. **Filters off** (`--pred-iou 0 --stability 0`).
   - Prediction: mean IoU rises above 0.161 and masks and coverage increase; for the v1 model
     with filters off, IoU recovers largely toward the baseline level.
   - Reason: if IoU recovers, filtering (IoU head / stability) is at fault; if it stays low,
     the masks themselves got worse.
   - Result:
2. **`points_per_side` sweep 16 / 32 / 64** (first `SWEEP_LIMIT` test patches).
   - Prediction: 32 and 64 raise recall and mean IoU, especially for small fields and Spain;
     time grows roughly quadratically (~4x and ~16x versus 16).
   - Reason: at 16 a median parcel gets a click ~20% of the time; more points cut the prompt
     cost in coverage, with diminishing returns.
   - Result:

**Trained runs**

3. **A_centroid_tok0** (centroid prompt, loss on token 0).
   - Prediction: modest or no gain over baseline (~0.16); better than v1 because selection
     uses the generator.
   - Reason: still trains only token 0 with prompts the generator does not use.
   - Result:
4. **B_centroid_multi** (centroid, loss on tokens 1-3, best of three).
   - Prediction: somewhat better than A.
   - Reason: the generator uses tokens 1-3, so training matches use.
   - Result:
5. **C_random_tok0** (random points, token 0).
   - Prediction: better than A, worse than D.
   - Reason: random points mimic the generator grid (off-centre and background points), but
     token 0 is still mismatched.
   - Result:
6. **D_random_multi** (random points, tokens 1-3).
   - Prediction: **best** of the four and above baseline.
   - Reason: only run matching both prompt distribution and tokens to what the generator does
     at inference.
   - Result:
7. **A_loss_select** (as A but stopped on val loss; replicates v1).
   - Prediction: **worst**; reproduces the collapse (test IoU < 0.10, few masks).
   - Reason: val loss cannot see the generator failure.
   - Result:
8. **E_grid_multi** (added 2026-11-08 22:20 UTC-5, before any v2 result; training clicks are the
   real 16x16 generator grid points that land on a parcel, tokens 1-3).
   - Prediction: about equal to D, possibly slightly better; still limited by grid coverage
     (only ~32% of parcels get a grid point at pps=16, Phase C), so the gain over baseline is small.
   - Reason: matches the generator's prompt distribution exactly, but cannot help parcels
     that receive no grid point.
   - Result:
9. **Follow-up, added 2026-11-08 22:35 UTC-5 after seeing that A-E all kept epoch 0** (every
   config lowered the generator metric on val after one epoch at lr 3e-5): E_grid_multi and
   D_random_multi at lr 1e-5 and 3e-6.
   - Prediction: lower lr reduces the damage but does not beat the untrained model by more than
     noise (|Δ| < 0.01 mean best IoU); E_grid stays closer to baseline than D.
   - Reason: if dense supervision itself shifts the IoU head / masks away from the generator's
     operating point, a smaller step only slows the shift.
   - Result:

What each result would refute: if D does not beat A, prompt/token alignment is not the factor;
if the grid sweep beats every trained run, the bottleneck is prompt density, not the decoder.

## AI assistance (explicitly disclosed)

- **Claude Code (Anthropic)** was used as a coding tool: `src/train_decoder.py`, the new
  options of `src/evaluate_masks.py`, the RunPod scripts (`scripts/`), the experiment runner,
  analysis scripts, and drafting of these documents.
- **Humans:** chose the topic, ran notebooks 00-07, and must verify every decision; no member
  should defend something they cannot explain.
