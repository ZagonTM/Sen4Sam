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
├── 4. PHASE B (planned)
│   └─ intervention?         → ?
│
└── 5. PHASE C
    └─ domain shift?         → SA-1B → Sen4AgriNet
```