# Poster content: The cost of the prompt in SAM

<!-- PDF (E4): "Presentación en WAIMLAP: póster, 18-20 de noviembre de 2026", 4 points. The PDF gives no poster size, language, or section list; check the WAIMLAP template. English used. Fill {{...}} when v2 results arrive. -->

## Panel 1. Title and team
**The cost of the prompt in SAM: prompt-free field segmentation of Sentinel-2 with a frozen SAM 2.1**
Deep Learning 1INF52, PUCP. Authors: {{AUTHORS}}. Code: github.com/ZagonTM/Sen4Sam

## Panel 2. Question and data
- Can SAM segment parcels with no human prompt in a new domain (satellite)?
- Model: SAM 2.1 `hiera_small`, frozen; automatic generator, 16x16 point grid.
- Data: Sen4AgriNet, 888 patches (366x366 px, RGB), 6 tiles, Spain and France. Split 70/15/15 by patch, seed 42; test = 132 patches.
- Metric: best IoU per ground-truth parcel (area >= 20 px) over generated masks.

## Panel 3. Baseline (Phase A)
- Mean best IoU **0.161**; **14.6%** of fields IoU > 0.5.
- Spain 0.085 vs France 0.220 (field-weighted).
- Suggested figure: `results/phase_c/fig_iou_by_country.png`

## Panel 4. Intervention (Phase B v1): it improved, then it broke
- Trained `transformer.layers[1]` + IoU head (1.71 M params, 3.7%), Dice loss, centroid prompts.
- Validation (centroid prompts): mask IoU 0.312 to 0.530.
- Test (automatic generator): 0.156 to **0.064**; paired delta -0.092, CI [-0.103, -0.083]; masks per patch 78.7 to 41.1; 128/132 patches worse.
- Message: training prompt is not the inference prompt; validation loss could not see it.
- Suggested figures: `results/training_curves.png`, `results/test_comparison.png`, `results/phase_c/fig_paired_delta_hist.png`

## Panel 5. Why: prompt density (Phase C)
- Median parcel about 113 px (11x11); grid spacing at 16x16 is 22.9 px.
- Parcels with at least one grid point: **32%** (ES 24%, FR 40%); 20-49 px parcels: 6%. At 32x32: 68%. At 64x64: **95%**.
- Per-patch IoU vs median parcel area: Spearman **0.87** (confounded with country).
- Suggested figure: `results/phase_c/fig_parcels_hit_vs_pps.png`
- Caveat: coverage is not IoU.

## Panel 6. Controlled ablation (v2; expectations written before results)
- Factors: prompt (centroid / random / grid) x tokens (0 / 1-3); selection by generator IoU.
- Best trained variant on test: {{RESULT_D_IOU}} (baseline 0.156; v1 0.064).
- Grid sweep: pps 16/32/64 gives IoU {{PPS16_IOU}} / {{PPS32_IOU}} / {{PPS64_IOU}}; time per patch {{PPS16_TIME}} / {{PPS32_TIME}} / {{PPS64_TIME}}.
- Filters off: {{FILTERS_OFF_VERDICT}}.
- Verdict: {{OVERALL_VERDICT}}. Pre-registered predictions that failed are reported as failed.

## Panel 7. Takeaways and limits
- Decoder-only adaptation attacked a bottleneck that may sit in the prompt grid.
- Validate with the deployment pipeline, not a proxy prompt.
- Limits: single seed; per-patch CSV aggregates; 22 test patches per tile; ES/FR gap mixes size with crop and radiometry shift.
- Negative results are reported as findings.

## Panel 8. Reproducibility and AI use
- Fixed seeds, split in `results/split.csv`, decision log `docs/decisions_en.md`.
- AI assistance (Claude Code) for code and drafting; humans verify every decision.
- References: SAM 2 (Ravi et al., 2024); Sen4AgriNet (Sykas et al., 2022). {{CHECK_LICENSES}}
