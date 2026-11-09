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
FIG_SWEEP = HERE / "fig_pps_sweep.png"
FIG_FILT = ROOT / "results/figures/filters_off.png"
FIG_CURVES = ROOT / "results/figures/val_gen_curves.png"
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
    "• **Diagnose** the prompt: grid density and SAM's fixed mask filters.",
    "• **Adapt** the mask decoder (encoder frozen) and test it with the deployment pipeline.",
]
METHOD = [
    f"• **Data:** Sen4AgriNet [3], {NUMBERS['parcels_total']:,} parcels (≥ 20 px) in 888 Sentinel-2 patches "
    "(366 × 366 px, RGB, 10 m, May), 6 tiles, Spain and France. Split 70/15/15 by patch, stratified by tile, seed 42; "
    "test = 132 patches.",
    "• **Model:** SAM 2.1 Hiera-Small; automatic generator, 16 × 16 grid (pps = 16), filters from SA-1B: "
    "predicted IoU ≥ 0.5, stability ≥ 0.8.",
    "• **Metric:** for each true parcel, best IoU over all generated masks; mean over parcels; paired bootstrap 95 % CI.",
    "• **Adaptation:** train decoder transformer layer 2 + IoU head (1.71 M params, 3.7 %), Dice (+ IoU-head) loss. "
    "v1: centroid prompts. v2: prompt (centroid / random / grid) × output tokens, checkpoint chosen by generator IoU on val.",
    "• **Pre-registered:** expectations written before each run (D = random prompts predicted best; "
    "E_noiou won).",
]
CAPTION_DATA = "Fig. 1. A test patch (RGB) and its parcel map; the 6 Sentinel-2 tiles lie in Catalonia and France."

RES_A = [
    "**A. Frozen baseline and cost.** The encoder costs "
    f"{ENC_GMAC} GMAC per image, the decoder {DEC_GMAC_PER_PROMPT} GMAC per prompt (16 × 16 grid = 2.6 × the encoder). "
    f"Automatic mode on test: mean best IoU **{N('v1_test_iou_untrained')}** (132 patches, pps 16); "
    f"Spain {N('baseline_es_mean_best_iou')} ≪ France {N('baseline_fr_mean_best_iou')}.",
]
RES_B = [
    f"**B. Prompt density.** A 16 × 16 grid reaches only **{P('grid_hit_pps16')}** of parcels (32 × 32: "
    f"{P('grid_hit_pps32')}, 64 × 64: {P('grid_hit_pps64')}). pps 32 on all 132 test patches: "
    f"**{N('pps32_full132_iou')}** (Δ {N('pps32_full132_delta_ci')}) at 5–9 × the time; pps 64: "
    f"{N('pps64_test_iou')} vs {N('pps16_test_iou')} on 40 Spanish patches.",
]
CAPTION_B = ("Fig. 2. Test mean best IoU (left) and seconds per patch (right, log; shared pod GPU) vs. grid density, "
             "on all 132 test patches (pps 16/32) and 40 Spanish test patches (pps 16/32/64).")
RES_C = [
    "**C. SAM's filters, calibrated on SA-1B, discard good masks.** Filters off (pps 16): "
    f"**{N('filters_off_test_iou')}** (Δ {N('filters_off_delta_ci')}), "
    f"{N('filters_off_masks_per_patch', '{:.0f}')} vs {N('filters_on_masks_per_patch', '{:.0f}')} masks per patch; "
    f"share of mask area on fields {N('untrained_mask_on_field', '{:.2f}')} → "
    f"{N('filters_off_mask_on_field', '{:.2f}')}: a precision trade-off.",
]
RES_D = [
    "**D. Fine-tuning.** v1 (centroid prompts) raised prompted mask IoU on val "
    f"{N('v1_val_mask_iou_before')} → {N('v1_val_mask_iou_after')} but the generator on test collapsed to "
    f"**{N('v1_test_iou_finetuned')}**. v2: training the IoU head lowers the generator metric every time "
    "(selection kept epoch 0) although masks improve (filters off: E_loss "
    f"{N('e_loss_test_nofilter_iou')} vs {N('filters_off_test_iou')}). **IoU-head loss off** (grid prompts, "
    f"tokens 1–3): test **{N('e_noiou_test_iou', '{:.3f}')}**, Δ +0.085 [+0.080, +0.091], the first fine-tune "
    f"that beats frozen SAM at default filters (filters off alone: 0.262). Negative clicks + step-level selection: {N('f_iouoff_test_iou')}.",
]
CAPTION_C = "Fig. 3. Test mean best IoU with SAM's filters on / off (m/p = masks per patch)."
CAPTION_V1 = "Fig. 4. v1: val bars = prompted mask IoU; test bars = automatic generator, mean best IoU."
CAPTION_D = ("Fig. 5. Validation generator IoU during training (12 val patches). Every run that trains the IoU head "
             "falls below the frozen model; with the IoU-head loss off it rises.")

CONCLUSIONS = [
    f"• **The cost of the prompt** is three things: grid density ({P('grid_hit_pps16')} of parcels reached at pps 16), "
    f"filters calibrated on another domain (off: {N('v1_test_iou_untrained')} → {N('filters_off_test_iou')}), and "
    f"the train/inference prompt mismatch (v1: {N('v1_test_iou_finetuned')}).",
    "• **Do not retrain the IoU head under fixed thresholds**, or recalibrate the thresholds: with its loss off, "
    f"the decoder beats default frozen SAM ({N('v1_test_iou_untrained')} → {N('e_noiou_test_iou')}).",
    "• **Validate on the deployment pipeline:** proxy gains (val prompted IoU) predicted the opposite of test.",
]
LIMITS = ("Limits: one seed; 150 training patches; pps 64 on 40 Spanish patches only; timings on a shared GPU.")
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
    cd.categories = ["prompted mask IoU (val)", "automatic generator, mean best IoU (test)"]
    cd.add_series("Frozen SAM 2.1", (NUMBERS["v1_val_mask_iou_before"], NUMBERS["v1_test_iou_untrained"]))
    cd.add_series("Fine-tuned (v1)", (NUMBERS["v1_val_mask_iou_after"], NUMBERS["v1_test_iou_finetuned"]))
    gf = slide.shapes.add_chart(XL_CHART_TYPE.COLUMN_CLUSTERED, Inches(x), Inches(y), Inches(w), Inches(h), cd)
    ch = gf.chart
    ch.has_legend = True; ch.legend.position = XL_LEGEND_POSITION.TOP; ch.legend.include_in_layout = False
    ch.legend.font.size = Pt(15)
    ch.font.size = Pt(15); ch.font.name = "Arial"
    va = ch.value_axis
    va.maximum_scale, va.minimum_scale = 0.6, 0.0
    va.has_major_gridlines = False
    va.tick_label_position = XL_TICK_LABEL_POSITION.NONE
    va.format.line.fill.background()
    ch.category_axis.tick_labels.font.size = Pt(14)
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
            tf.paragraphs[0].runs[0].font.size = Pt(16); tf.paragraphs[0].runs[0].font.bold = True
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
FS, CS = 22, 17
def para(lines, h):
    global y
    textbox(R + 0.35, y, W - 0.7, h, lines, size=FS, space_after=4); y += h + 0.1
def cap(x, w, text, h):
    textbox(x, y, w, h, [text], size=CS, color=MUTED, italic=True, space_after=0)
geom(rb, R + 0.35, y, W - 0.7, 1.25)
fill(rb, RES_A, size=FS, space_after=4)
y += 1.35
para(RES_B, 1.55)
y = picture(FIG_SWEEP, R + 0.8, y, w=W - 1.6)
cap(x0, wfig, CAPTION_B, 0.7); y += 0.8
para(RES_C, 1.25)
para(RES_D, 2.75)
hw = (W - 1.0) / 2
yf = picture(FIG_FILT, R + 0.4, y, w=hw)
bar_chart(R + 0.6 + hw, y, hw, yf - y)
y = yf
cap(R + 0.4, hw, CAPTION_C, 0.7); cap(R + 0.6 + hw, hw, CAPTION_V1, 0.7); y += 0.85
y = picture(FIG_CURVES, R + 0.4, y, w=W - 0.8)
cap(x0, wfig, CAPTION_D, 0.7); y += 0.7
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
