"""Build the WAIMLAp 2026 A0 poster "The cost of the prompt in SAM 2" from the group's poster.

Run from the repository root:

    uv run --with python-pptx --with qrcode --with pillow python submission/poster/build_poster.py

Base: submission/poster/group_original.pptx (official WAIMLAp template filled by the group). Header, footer, logos,
panel boxes, fonts and colours are kept; panel text is replaced in place (the first run's formatting is reused) and the
panels/pictures are repositioned. Numbers come from submission/numbers.json: a value like "{{PPS64_IOU}}" is pending
and is rendered as "[PPS64_IOU]"; the pending keys are printed. Phase A cost numbers are constants below, taken from
notebooks/01_en_base_model_teardown.ipynb (sections 1, 3 and 8).
Figures: results/phase_c/fig_parcels_hit_vs_pps.png; v1 val/test chart drawn natively from numbers.json;
results/figures/pps_sweep.png is inserted if it exists, otherwise a labelled placeholder box is drawn.
Output: submission/poster/Poster_SAM2_Cost_of_the_Prompt.pptx (export the PDF from PowerPoint).
"""
import copy
import json
import re
from pathlib import Path

from PIL import Image
from pptx import Presentation
from pptx.chart.data import CategoryChartData
from pptx.dml.color import RGBColor
from pptx.enum.chart import XL_CHART_TYPE, XL_LABEL_POSITION, XL_LEGEND_POSITION, XL_TICK_LABEL_POSITION
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.oxml.ns import qn
from pptx.util import Inches, Pt

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
TEMPLATE = HERE / "group_original.pptx"
OUT = HERE / "Poster_SAM2_Cost_of_the_Prompt.pptx"
NUMBERS = json.loads((ROOT / "submission/numbers.json").read_text())
FIG_HIT = ROOT / "results/phase_c/fig_parcels_hit_vs_pps.png"
FIG_SWEEP = ROOT / "results/figures/pps_sweep.png"
REPO_URL = "https://github.com/ZagonTM/Sen4Sam"

PENDING = []


def N(key, fmt="{:.3f}"):
    """Number from numbers.json; pending '{{X}}' values render as '[X]'."""
    v = NUMBERS[key]
    if isinstance(v, str):
        m = re.fullmatch(r"\{\{(\w+)\}\}", v.strip())
        if m:
            PENDING.append(key)
            return f"[{m.group(1)}]"
        return v
    return fmt.format(v)


def P(key):
    """Fraction as percent."""
    return N(key, "{:.0%}") if not isinstance(NUMBERS[key], str) else N(key)


# Phase A constants (notebook 01, frozen SAM 2.1 hiera_small)
ENC_GMAC, DEC_GMAC_PER_PROMPT = 134, 1.4

# ----------------------------------------------------------------------------------------------------------------- CONTENT
TITLE = "The Cost of the Prompt: Prompt-Free SAM 2 on Sentinel-2 Fields"
AFFIL = "Pontificia Universidad Católica del Perú (PUCP), Lima, Peru"

MOTIVATION = [
    "• SAM 2 [1] segments anything from **one human click**. Mapping thousands of fields needs it **prompt-free**: "
    "the automatic generator clicks a regular point grid instead.",
    "• Sentinel-2 parcels are far from SA-1B [2] objects: 10 m pixels, a median parcel of "
    f"**{N('parcel_median_px', '{:d}')} px (≈ 11 × 11 px)**.",
    "",
    "**Question:** how well does frozen, prompt-free SAM 2 find agricultural parcels, and what limits it: "
    "the model or the prompt?",
]
OBJECTIVES = [
    "• **Teardown** of frozen SAM 2.1: parameters, compute per block, latency.",
    "• **Baseline** of the automatic generator against parcel ground truth.",
    "• **Adapt** the mask decoder (encoder frozen) and test it with the deployment pipeline.",
    "• **Diagnose** the gap: domain shift (Spain vs. France) and prompt-grid density.",
]
METHOD = [
    f"• **Data:** Sen4AgriNet [3], {NUMBERS['parcels_total']:,} parcels (≥ 20 px) in 888 Sentinel-2 patches "
    "(366 × 366 px, RGB, 10 m, May), 6 tiles, Spain and France. Split 70/15/15 by patch, stratified by tile, seed 42; "
    "test = 132 patches.",
    "• **Model:** SAM 2.1 Hiera-Small; automatic generator, 16 × 16 grid (pps = 16), predicted-IoU ≥ 0.5, "
    "stability ≥ 0.8.",
    "• **Metric:** for each true parcel, best IoU over all generated masks; mean over parcels.",
    "• **Adaptation (v1):** train decoder transformer layer 2 + IoU head (1.71 M params, 3.7 %), Dice loss, "
    "one centroid prompt per parcel.",
    "• **Ablation (v2):** prompt (centroid / random / grid) × output tokens (0 / 1–3), checkpoint chosen by "
    "generator IoU; grid sweep pps 16 / 32 / 64.",
]
CAPTION_DATA = "Fig. 1. A test patch (RGB) and its parcel map; the 6 Sentinel-2 tiles lie in Catalonia and France."

RES_A = [
    "**A. The prompt dominates compute.** The encoder costs "
    f"{ENC_GMAC} GMAC per image, the decoder {DEC_GMAC_PER_PROMPT} GMAC per prompt: a 16 × 16 grid (256 prompts) "
    "already costs **2.6 × the encoder**, 32 × 32 ≈ 10 ×.",
    f"**Frozen baseline:** mean best IoU **{N('baseline_all_mean_best_iou')}**; "
    f"{P('baseline_all_frac_iou_gt_05')} of parcels with IoU > 0.5; Spain {N('baseline_es_mean_best_iou')} ≪ "
    f"France {N('baseline_fr_mean_best_iou')}. Masks often merge neighbouring parcels.",
]
RES_B = [
    "**B. Fine-tuning helped the proxy, hurt the pipeline.** Validation with centroid prompts rose "
    f"{N('v1_val_mask_iou_before')} → {N('v1_val_mask_iou_after')}; the automatic generator on test fell "
    f"{N('v1_test_iou_untrained')} → **{N('v1_test_iou_finetuned')}** (paired Δ {N('v1_test_delta')}, 95 % CI "
    f"{N('v1_test_delta_ci')}; masks per patch {N('v1_masks_per_patch_before', '{:.1f}')} → "
    f"{N('v1_masks_per_patch_after', '{:.1f}')}). **Training prompt ≠ inference prompt.**",
]
CAPTION_B = "Fig. 2. Mean best IoU before / after decoder fine-tuning (v1), on the proxy and on the real pipeline."
RES_C = [
    f"**C. The grid misses the fields.** At pps = 16 (a point every 22.9 px) only **{P('grid_hit_pps16')}** of parcels "
    f"get a grid point (Spain {P('grid_hit_pps16_es')}, France {P('grid_hit_pps16_fr')}; 20–49 px parcels "
    f"{P('grid_hit_pps16_small_20_49px')}); pps = 32: {P('grid_hit_pps32')}, pps = 64: **{P('grid_hit_pps64')}**. "
    f"Patch IoU tracks parcel size (Spearman ρ = {N('spearman_iou_vs_median_parcel_area', '{:.2f}')}; "
    "confounded with country). Coverage is an upper bound, not IoU.",
]
CAPTION_C = "Fig. 3. Fraction of true parcels hit by ≥ 1 grid point, by parcel size (all 888 patches)."
RES_D = [
    f"**D. Denser grid, better masks (v2).** On {NUMBERS['pps_sweep_n_patches']} test patches (Spain), IoU rises "
    f"{N('pps16_test_iou')} → {N('pps32_test_iou')} → **{N('pps64_test_iou')}** for pps 16 / 32 / 64, at "
    f"{N('pps16_sec_per_patch', '{:.2f}')} → {N('pps32_sec_per_patch', '{:.2f}')} → "
    f"{N('pps64_sec_per_patch', '{:.2f}')} s per patch. Filters off (pps = 16): {N('filters_off_test_iou')}.",
    "**Decoder ablation:** with checkpoints chosen by generator IoU, no trained variant beat the frozen model "
    f"(best = epoch 0, {N('best_v2_test_iou')}); choosing by validation loss gives {N('v2_loss_select_test_iou')}.",
]
CAPTION_D = ("Fig. 4. Mean best IoU (left) and seconds per patch (right, log scale; shared pod GPU) vs. grid density, "
             "first 40 test patches.")

CONCLUSIONS = [
    "• **The bottleneck is prompt density, not the decoder:** a 16 × 16 grid reaches only "
    f"{P('grid_hit_pps16')} of parcels; a 64 × 64 grid raises IoU {N('pps16_test_iou')} → {N('pps64_test_iou')}, "
    "while no decoder fine-tune beat the frozen model.",
    "• **Validate with the deployment pipeline:** tuning on centroid prompts improved validation "
    f"({N('v1_val_mask_iou_before')} → {N('v1_val_mask_iou_after')}) and broke the automatic generator "
    f"({N('v1_test_iou_untrained')} → {N('v1_test_iou_finetuned')}).",
    "• **Removing the human prompt has a price:** every grid point is a decoder pass, so the better grid costs "
    f"{N('pps16_sec_per_patch', '{:.2f}')} → {N('pps64_sec_per_patch', '{:.2f}')} s per patch.",
]
LIMITS = ("Limits: one seed; per-patch aggregates; grid sweep on 40 Spanish test patches; the Spain–France gap mixes parcel size "
          "with crop and radiometric shift.")
REFERENCES = [
    "[1] N. Ravi et al., “SAM 2: Segment Anything in Images and Videos,” arXiv:2408.00714, 2024.",
    "[2] A. Kirillov et al., “Segment Anything,” in Proc. IEEE/CVF ICCV, pp. 4015–4026, 2023.",
    "[3] D. Sykas et al., “A Sentinel-2 Multiyear, Multicountry Benchmark Dataset for Crop Classification and "
    "Segmentation With Deep Learning,” IEEE JSTARS, vol. 15, pp. 2937–2953, 2022.",
]
ACK = [
    "Course Deep Learning (1INF52), PUCP.",
    "Compute: rented GPUs on RunPod.",
    "Model: Meta AI SAM 2.1; data: Sen4AgriNet.",
    "Code written with AI assistance (Claude Code); every result checked by the authors.",
]
CONTACT = [
    "**Contact:** andre.rubio@pucp.edu.pe",
    "**Code, splits and results:**",
    "github.com/ZagonTM/Sen4Sam",
]

# -------------------------------------------------------------------------------------------------------------- HELPERS
INK, MUTED = RGBColor(0x28, 0x28, 0x2D), RGBColor(0x50, 0x50, 0x5A)
prs = Presentation(TEMPLATE)
slide = prs.slides[0]
by_id = {sh.shape_id: sh for sh in slide.shapes}
PURPLE = by_id[89].fill.fore_color.rgb if by_id[89].fill.type == 1 else RGBColor(0x4B, 0x2E, 0x83)
BODY_RPR = copy.deepcopy(by_id[119].text_frame.paragraphs[0].runs[1]._r.find(qn("a:rPr")))   # Arial, 28282D


def geom(sh, x, y, w=None, h=None):
    sh.left, sh.top = Inches(x), Inches(y)
    if w is not None:
        sh.width = Inches(w)
    if h is not None:
        sh.height = Inches(h)


def remove(sh):
    sh._element.getparent().remove(sh._element)


def _run(p, text, rpr, size, bold, italic, color):
    r = p.add_run()
    r.text = text
    old = r._r.find(qn("a:rPr"))
    if old is not None:
        r._r.remove(old)
    new = copy.deepcopy(rpr)
    r._r.insert(0, new)
    f = r.font
    f.size, f.bold, f.italic = Pt(size), bold, italic
    if color is not None:
        f.color.rgb = color


def fill(sh, lines, size=24, color=None, italic=False, align=PP_ALIGN.LEFT, space_after=8, bullet_indent=0.32, rpr=None):
    """Replace the text of a shape in place; formatting comes from its first run (or rpr). '**x**' = bold,
    lines starting with '• ' become hanging-indent bullets."""
    tf = sh.text_frame
    tf.word_wrap = True
    if rpr is None:
        r0 = next((r for p in tf.paragraphs for r in p.runs), None)
        rpr = r0._r.find(qn("a:rPr")) if r0 is not None and r0._r.find(qn("a:rPr")) is not None else BODY_RPR
        rpr = copy.deepcopy(rpr)
    if rpr.find(qn("a:latin")) is None:
        for tag in ("a:latin", "a:ea", "a:cs"):
            rpr.append(rpr.makeelement(qn(tag), {"typeface": "Arial"}))
    body = tf._txBody.find(qn("a:bodyPr"))
    for child in list(body):
        body.remove(child)
    body.append(body.makeelement(qn("a:noAutofit"), {}))
    for p in list(tf.paragraphs)[1:]:
        p._p.getparent().remove(p._p)
    first = tf.paragraphs[0]
    for child in list(first._p):
        if child.tag != qn("a:pPr"):
            first._p.remove(child)                  # runs, line breaks, endParaRPr
    for i, line in enumerate(lines):
        p = first if i == 0 else tf.add_paragraph()
        pPr = p._p.get_or_add_pPr()
        for b in [e for t in ("a:buClr", "a:buSzPts", "a:buFont", "a:buChar", "a:buNone") for e in pPr.findall(qn(t))]:
            pPr.remove(b)
        p.alignment = align
        p.space_after = Pt(space_after)
        if line.startswith("• "):
            pPr.set("marL", str(Inches(bullet_indent)))
            pPr.set("indent", str(-Inches(bullet_indent)))
            pPr.append(pPr.makeelement(qn("a:buChar"), {"char": "•"}))
            line = line[2:]
        else:
            pPr.set("marL", "0"); pPr.set("indent", "0")
            pPr.append(pPr.makeelement(qn("a:buNone"), {}))
        if not line:
            p.space_after = Pt(2)
            continue
        for k, part in enumerate(re.split(r"\*\*", line)):
            if part:
                _run(p, part, rpr, size, k % 2 == 1, italic, color)


def textbox(x, y, w, h, lines, **kw):
    tb = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    kw.setdefault("rpr", BODY_RPR)
    fill(tb, lines, **kw)
    return tb


def picture(path, x, y, w=None, h=None):
    with Image.open(path) as im:
        ratio = im.height / im.width
    if h is not None and w is None:
        w = h / ratio
    slide.shapes.add_picture(str(path), Inches(x), Inches(y), width=Inches(w))
    return y + w * ratio


def move_picture(sh, x, y, w):
    ratio = sh.height / sh.width
    geom(sh, x, y, w, w * ratio)
    return y + w * ratio


def section(ids, x, y, w, h, body_top=0.95):
    """Move one template panel: rounded box, header bar, header text, body text."""
    box, bar, head, body = (by_id[i] for i in ids)
    geom(box, x, y, w, h)
    geom(bar, x, y, w)
    geom(head, x + 0.35, y + 0.08)
    geom(body, x + 0.35, y + body_top, w - 0.7, h - body_top - 0.25)
    return body


def placeholder(x, y, w, h, label):
    s = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(x), Inches(y), Inches(w), Inches(h))
    s.fill.solid(); s.fill.fore_color.rgb = RGBColor(0xF4, 0xF1, 0xF8)
    s.line.color.rgb = PURPLE; s.line.width = Pt(3); s.line.dash_style = 4
    s.shadow.inherit = False
    s.text_frame.vertical_anchor = MSO_ANCHOR.MIDDLE
    fill(s, [label], size=26, color=PURPLE, align=PP_ALIGN.CENTER, rpr=BODY_RPR)
    return y + h


def bar_chart(x, y, w, h):
    cd = CategoryChartData()
    cd.categories = ["Validation (centroid prompts)", "Test (automatic generator)"]
    cd.add_series("Frozen SAM 2.1", (NUMBERS["v1_val_mask_iou_before"], NUMBERS["v1_test_iou_untrained"]))
    cd.add_series("Fine-tuned decoder (v1)", (NUMBERS["v1_val_mask_iou_after"], NUMBERS["v1_test_iou_finetuned"]))
    gf = slide.shapes.add_chart(XL_CHART_TYPE.COLUMN_CLUSTERED, Inches(x), Inches(y), Inches(w), Inches(h), cd)
    ch = gf.chart
    ch.has_legend = True; ch.legend.position = XL_LEGEND_POSITION.TOP; ch.legend.include_in_layout = False
    ch.legend.font.size = Pt(20)
    ch.font.size = Pt(20); ch.font.name = "Arial"
    va = ch.value_axis
    va.maximum_scale, va.minimum_scale = 0.6, 0.0
    va.has_major_gridlines = False
    va.tick_label_position = XL_TICK_LABEL_POSITION.NONE
    va.format.line.fill.background()
    ch.category_axis.tick_labels.font.size = Pt(20)
    for s, c in zip(ch.plots[0].series, (RGBColor(0x8C, 0x8C, 0x96), PURPLE)):
        s.format.fill.solid(); s.format.fill.fore_color.rgb = c
    pl = ch.plots[0]
    pl.gap_width = 60
    pl.has_data_labels = True
    dl = pl.data_labels
    dl.number_format, dl.number_format_is_linked = "0.000", False
    dl.position = XL_LABEL_POSITION.OUTSIDE_END
    dl.font.size = Pt(20); dl.font.bold = True
    for ser in pl.series:                              # literal text: decimal point independent of locale
        for i, v in enumerate(ser.values):
            tf = ser.points[i].data_label.text_frame
            tf.text = f"{v:.3f}"
            tf.paragraphs[0].runs[0].font.size = Pt(22); tf.paragraphs[0].runs[0].font.bold = True
            ser.points[i].data_label.position = XL_LABEL_POSITION.OUTSIDE_END
    return y + h


# --------------------------------------------------------------------------------------------------------------- LAYOUT
# header: new title, fixed affiliation; authors kept
fill(by_id[86], [TITLE], size=40, align=PP_ALIGN.CENTER, space_after=0)
by_id[86].text_frame.paragraphs[0].runs[0].font.bold = True
aff = by_id[87].text_frame.paragraphs[1]
for r in aff.runs[1:]:
    r._r.getparent().remove(r._r)
if aff.runs:
    aff.runs[0].text = AFFIL

L, R, W = 1.20, 16.88, 15.03
Y0, YEND = 6.42, 43.85                    # below header; above footer (44.19 in)

# left column ---------------------------------------------------------------------------------------------------------
remove(by_id[129])                        # second motivation text box (merged into MOTIVATION)
mb = section((88, 89, 90, 91), L, Y0, W, 6.35)
geom(by_id[128], L + 0.3, Y0 + 1.0, 3.2, 1.8)            # SAM 2 logo, top-left of the panel
geom(mb, L + 3.6, Y0 + 0.9, W - 3.95, 5.2)
fill(mb, MOTIVATION[:1], size=24, align=PP_ALIGN.LEFT)
tb = textbox(L + 0.35, Y0 + 3.05, W - 0.7, 3.2, MOTIVATION[1:], size=24)

ob = section((92, 93, 94, 95), L, 13.17, W, 4.45)
fill(ob, OBJECTIVES, size=24, space_after=6)

remove(by_id[125])                        # "Project phases" box (content folded into METHOD/OBJECTIVES)
mt = section((96, 97, 98, 99), L, 18.02, W, 12.3)
yi, s = 18.02 + 0.95, 4.3
move_picture(by_id[124], L + 0.55, yi, s)               # RGB patch
geom(by_id[126], L + 0.55 + s + 0.35, yi, s * 425 / 418, s)  # parcel map (matplotlib axes)
by_id[127].width = Inches(s * 567 / 516 * 0.98); by_id[127].height = Inches(s * 0.98)
geom(by_id[127], L + W - 0.55 - s * 567 / 516, yi)       # tile map
textbox(L + 0.45, yi + s + 0.05, W - 0.9, 0.6, [CAPTION_DATA], size=19, color=MUTED, italic=True, space_after=0)
geom(mt, L + 0.35, yi + s + 0.75, W - 0.7, 30.32 - (yi + s + 0.75) - 0.2)
fill(mt, METHOD, size=23, space_after=6)

cb = section((104, 105, 106, 107), L, 30.77, W, 7.6)
fill(cb, CONCLUSIONS + ["", LIMITS], size=24, space_after=8)
lim = cb.text_frame.paragraphs[-1]
for r in lim.runs:
    r.font.size = Pt(20); r.font.italic = True; r.font.color.rgb = MUTED

rf = section((108, 109, 110, 111), L, 38.82, W, YEND - 38.82)
fill(rf, REFERENCES, size=21, space_after=6)
by_id[110].left = Inches(L + 0.35)

# right column: results ------------------------------------------------------------------------------------------------
RES_H = 29.5
rb = section((100, 101, 102, 103), R, Y0, W, RES_H)
x0, wfig = R + 0.4, W - 0.8
y = Y0 + 0.9
geom(rb, R + 0.35, y, W - 0.7, 2.6)
fill(rb, RES_A, size=23, space_after=6)
y += 2.75
textbox(R + 0.35, y, W - 0.7, 1.6, RES_B, size=23); y += 1.7
y = bar_chart(R + 0.9, y, W - 1.8, 5.3)
textbox(x0, y, wfig, 0.5, [CAPTION_B], size=19, color=MUTED, italic=True, space_after=0); y += 0.65
textbox(R + 0.35, y, W - 0.7, 2.0, RES_C, size=23); y += 2.05
y = picture(FIG_HIT, R + 2.5, y, w=W - 5.0)
textbox(x0, y, wfig, 0.5, [CAPTION_C], size=19, color=MUTED, italic=True, space_after=0); y += 0.65
textbox(R + 0.35, y, W - 0.7, 2.1, RES_D, size=23); y += 2.2
if FIG_SWEEP.exists():
    y = picture(FIG_SWEEP, R + 0.9, y, w=W - 1.8)
else:
    PENDING.append("results/figures/pps_sweep.png")
    y = placeholder(R + 1.6, y, W - 3.2, 3.6, "[FIGURE PENDING]\nresults/figures/pps_sweep.png\nIoU and s/patch vs. pps 16 / 32 / 64")
textbox(x0, y, wfig, 0.9, [CAPTION_D], size=19, color=MUTED, italic=True, space_after=0)
y += 0.5
RESULTS_END = y + 0.6

# right column: acknowledgments and contact
YA = Y0 + RES_H + 0.45
ab = section((112, 113, 114, 115), R, YA, W, 2.9, body_top=0.9)
by_id[114].text_frame.paragraphs[0].runs[0].text = "ACKNOWLEDGMENTS"
for r in by_id[114].text_frame.paragraphs[0].runs[1:]:
    r._r.getparent().remove(r._r)
fill(ab, [" • ".join(ACK[:3]), ACK[3]], size=22, space_after=2, rpr=BODY_RPR)
YC = YA + 2.9 + 0.45
cc = section((116, 117, 118, 119), R, YC, W, YEND - YC)
qs = min(YEND - YC - 1.15, 3.8)
geom(cc, R + 0.35, YC + 1.0, W - qs - 1.1, qs)
fill(cc, CONTACT, size=28, space_after=6)
geom(by_id[123], R + W - qs - 0.4, YC + 0.85, qs, qs)    # group's QR -> github.com/ZagonTM/Sen4Sam.git

# document properties (shown as PDF title/author)
prs.core_properties.title = TITLE
prs.core_properties.author = "André Rubio, Johanna Zagon, Pierre Lavergne"
prs.core_properties.subject = "WAIMLAp 2026 poster"
prs.save(OUT)
print("saved", OUT)
print("results content ends at", round(RESULTS_END, 2), "in; results panel ends at", round(Y0 + RES_H, 2))
print("PENDING:", ", ".join(dict.fromkeys(PENDING)) or "none")
