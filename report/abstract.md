# The cost of the prompt in SAM: prompt-free field segmentation of Sentinel-2 with a frozen SAM 2.1

<!-- Requirements (PDF, section 6/8): "resumen extendido (4-8 pág.) según la plantilla de WAIMLAP"; "informe final (4-8 pág., IEEE/LNCS)". The PDF gives no word limit or language for a short abstract and no abstract section list. Written in English; translate if the template requires Spanish. Placeholders {{...}} are filled when v2 results arrive. -->

## Abstract

Promptable segmenters such as SAM need a prompt, but remote-sensing pipelines have no human to supply one. We ask what the prompt costs when a frozen SAM 2.1 (`hiera_small`) segments agricultural parcels automatically with a regular point grid. We use 888 Sentinel-2 patches (366x366 px, RGB) from Sen4AgriNet, six tiles in Spain (ES) and France (FR), and score each ground-truth parcel by its best IoU over all generated masks.

**Baseline (Phase A/B).** The frozen model reaches a mean best IoU of 0.161; only 14.6% of fields have IoU > 0.5. Spain is much worse than France (field-weighted IoU 0.085 vs 0.220).

**Intervention (Phase B, v1).** We adapted the mask decoder (`transformer.layers[1]` and the IoU head, 1.71 M parameters, about 3.7% of the model) with a Dice loss and parcel-centroid prompts. On validation, with centroid prompts, it improved (mask IoU 0.312 to 0.530). On the real test, with the automatic generator on 132 held-out patches, it collapsed: mean best IoU 0.156 to 0.064 (paired delta -0.092, 95% bootstrap CI [-0.103, -0.083]; 128 of 132 patches worse), and masks per patch fell from 78.7 to 41.1. The training prompt did not match the inference prompt, and the validation loss could not detect this. This is a negative result.

**Failure analysis (Phase C).** The median parcel covers about 113 px (roughly 11x11 px), while a 16x16 grid has a point every 22.9 px. Only 32% of parcels receive any grid point (ES 24%, FR 40%; parcels of 20-49 px: 6%). A 64x64 grid reaches 95%. Per-patch IoU correlates with median parcel area at Spearman 0.87 (confounded with country). The prompt density caps recall before any decoder quality enters: this is the cost of the prompt. Coverage is not IoU, and this part of the analysis comes from existing evaluation CSVs without new model runs.

**Controlled ablation (v2, expectations pre-registered in `docs/decisions_en.md`).** We vary the training prompt (centroid / random / grid) and the loss tokens (token 0 / tokens 1-3), select checkpoints by the generator's own IoU, and add two diagnostics on the frozen model: filters off, and a grid sweep (16/32/64 points per side) with runtime. Results: the best trained variant reaches test IoU {{RESULT_D_IOU}} against 0.156 untrained; the 64-point grid reaches IoU {{PPS64_IOU}} at {{PPS64_TIME}} per patch; with the IoU and stability filters off, {{FILTERS_OFF_VERDICT}}. Taken together, these results {{OVERALL_VERDICT}}.

**Limitations.** One seed; per-patch rather than per-field aggregates in the CSV analysis; 22 test patches per tile; coverage counts a point inside a parcel, not a correct mask; ES/FR differences mix parcel size with crop and radiometry shift.

*AI assistance (Claude Code) was used for code and drafting, as recorded in the decision log.*

**Keywords:** SAM 2.1, automatic segmentation, Sentinel-2, domain shift, prompt density, negative results.
