"""Build the WAIMLAp 2026 extended abstract (max. 2 pages incl. references) on top of the group's draft.

Run from the repository root:

    uv run --with python-docx --with pillow python submission/abstract/build_abstract.py

Base document: submission/abstract/group_original.docx (official WAIMLAp template: header, page setup, fonts; title,
authors, affiliations and e-mails are kept as they are). The script replaces the body text, adds Table 1 and Fig. 1 and
writes submission/abstract/Abstract_SAM2_Cost_of_the_Prompt.docx.

All result numbers come from submission/numbers.json. Values that are still pending ("{{KEY}}") are rendered visibly as
"[KEY]" and listed at the end of the run. Phase A teardown facts (TEARDOWN below) are copied from the outputs of
notebooks/01_en_base_model_teardown.ipynb.

Export the PDF from Word (the template font Aptos is not available in LibreOffice):

    osascript -e 'tell application "Microsoft Word"' -e 'open POSIX file "/abs/path/Abstract_SAM2_Cost_of_the_Prompt.docx"' \
      -e 'save as active document file name "/abs/path/Abstract_SAM2_Cost_of_the_Prompt.pdf" file format format PDF' \
      -e 'close active document saving no' -e 'end tell'
"""
import copy
import json
import re
from pathlib import Path

from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn
from docx.shared import Cm, Pt
from docx.text.paragraph import Paragraph

ROOT = Path(__file__).resolve().parents[2]
TEMPLATE = ROOT / "submission/abstract/group_original.docx"
NUMBERS = ROOT / "submission/numbers.json"
FIGURE = ROOT / "results/figures/filters_off.png"
OUT = ROOT / "submission/abstract/Abstract_SAM2_Cost_of_the_Prompt.docx"

# ------------------------------------------------------------------------------------------------------------- NUMBERS
N = json.loads(NUMBERS.read_text())
PENDING = []


def v(key, nd=3, pct=False, unit="", pd=0):
    """Format numbers.json[key]; pending values ('{{X}}') become a visible '[X]'."""
    x = N[key]
    if isinstance(x, str) and x.startswith("{{"):
        tag = x.strip("{}")
        if tag not in PENDING:
            PENDING.append(tag)
        return f"[{tag}]"
    if isinstance(x, str):
        return x.replace("-", "−")
    if pct:
        return f"{x * 100:.{pd}f} %"
    s = f"{x:,.{nd}f}" if isinstance(x, float) else f"{x:,}"
    return s.replace("-", "−") + unit


# Phase A facts, verified against the outputs of notebooks/01_en_base_model_teardown.ipynb
TEARDOWN = {
    "enc_share": "74.5 %",          # image_encoder 34.317 M params
    "dec_share": "9.2 %",           # sam_mask_decoder 4.215 M params
    "enc_gmacs": "133.6",           # image encoder, per image (1024 px)
    "dec_gmacs": "1.37",            # mask decoder, per prompt
    "ratio16": "2.6",               # decoder/encoder MACs at pps=16 (256 prompts)
    "ratio32": "10.5",              # ... at pps=32 (1024 prompts)
}
T = TEARDOWN

# ------------------------------------------------------------------------------------------------------------- CONTENT
TITLE = "The Cost of the Prompt: Evaluating SAM 2 for Prompt-Free Agricultural Field Segmentation"

INTRO = [
    "Segment Anything (SAM) [2] and SAM 2 [1] segment natural images well when a user clicks on the object. Remote-sensing "
    "pipelines have no user, so the prompt must be generated automatically, usually as a regular grid of points. We ask "
    "*what this prompt costs* when a frozen SAM 2.1 segments agricultural parcels in Sentinel-2 images [4], and whether "
    "adapting the mask decoder can pay that cost. We find a weak frozen baseline, split the cost of the prompt into "
    "three measurable parts (prompt density, a quality filter tuned on another domain, and the mismatch between training "
    "and inference prompts), and show that a decoder fine-tune helps only once the last two are removed.",
]
METHOD = [
    "**Data.** We use the Patches Assembled Dataset of Sen4AgriNet [3]: six Sentinel-2 tiles, three in Spain (ES) and three "
    "in France (FR), seeded random sampling, cloud-free May 2019 acquisitions [5]. This gives 888 RGB patches of "
    f"366 × 366 px (444 per country) with {v('parcels_total')} parcels of at least 20 px (median {v('parcel_median_px')} px, "
    "about 11 × 11 px). Split by patch, stratified by tile: 624 train / 132 validation / 132 test.",
    "**Model and metric.** Frozen SAM 2.1 with a Hiera-Small encoder [6] and the automatic mask generator (16 × 16 point "
    "grid, predicted-IoU threshold 0.5, stability threshold 0.8). Each ground-truth parcel is scored by its best IoU over "
    "all generated masks; we report the mean over parcels (mean best IoU) and the fraction of parcels with IoU > 0.5.",
    f"**Phase A: teardown.** Forward hooks measure parameters and MACs. The image encoder holds {T['enc_share']} of the "
    f"parameters and the mask decoder {T['dec_share']}. The encoder costs {T['enc_gmacs']} GMACs once per image, the "
    f"decoder {T['dec_gmacs']} GMACs per prompt; at 16 × 16 points the decoder already costs {T['ratio16']}× the encoder, "
    f"at 32 × 32 {T['ratio32']}×. In prompt-free mode the prompt, not the backbone, dominates compute.",
    "**Phase B: decoder adaptation.** We train only the second two-way transformer layer and the IoU head of the mask "
    "decoder (1.71 M parameters, 3.7 % of the model) with a Dice loss, encoder frozen. v1 trained on parcel-centroid "
    "prompts and selected the checkpoint by validation loss. v2 is a controlled ablation with expectations written before "
    "the results: training prompt (centroid, random point, generator grid point) × loss on token 0 or on tokens 1–3, "
    "selected by the generator's own validation IoU, plus a v1 replicate selected by loss.",
    "**Phase C: failure analysis.** We overlay the generator grid on the true parcel masks and count the parcels that "
    "receive at least one point, by size and country; on the frozen model we sweep 16/32/64 points per side with runtime, "
    "and evaluate with both filters off.",
]
RESULTS = [
    f"**Frozen baseline.** On the 132 test patches the frozen generator reaches a mean best IoU of "
    f"{v('v1_test_iou_untrained')} (all 888 patches: {v('baseline_all_mean_best_iou')}; only "
    f"{v('baseline_all_frac_iou_gt_05', pct=True, pd=1)} of parcels exceed IoU 0.5). Spain is much worse than France "
    f"({v('baseline_es_mean_best_iou')} vs. {v('baseline_fr_mean_best_iou')}).",
    f"**Cost 1: prompt density.** A 16 × 16 grid places a point every 22.9 px, so only "
    f"{v('grid_hit_pps16', pct=True)} of parcels receive any point (ES {v('grid_hit_pps16_es', pct=True)}, "
    f"FR {v('grid_hit_pps16_fr', pct=True)}; parcels of 20–49 px: {v('grid_hit_pps16_small_20_49px', pct=True)}). A 32 × 32 "
    f"grid raises test IoU to {v('pps32_full132_iou')} (Δ {v('pps32_full132_delta_ci')}) at 5–9× the runtime; on "
    f"{v('pps_sweep_n_patches')} Spanish patches 64 × 64 gives {v('pps64_test_iou')} vs. {v('pps16_test_iou')}.",
    f"**Cost 2: a filter tuned on another domain.** The generator keeps a mask only if predicted IoU ≥ 0.5 and "
    f"stability ≥ 0.8, thresholds set on SA-1B. With both off, IoU rises to {v('filters_off_test_iou')} "
    f"(Δ {v('filters_off_delta_ci')}) with {v('filters_off_masks_per_patch', nd=0)} instead of "
    f"{v('filters_on_masks_per_patch', nd=0)} masks per patch, at lower precision (mask pixels on fields "
    f"{v('untrained_mask_on_field', nd=2)} → {v('filters_off_mask_on_field', nd=2)}).",
    f"**Cost 3: training prompt ≠ inference prompt.** v1 raised centroid-prompt validation IoU "
    f"({v('v1_val_mask_iou_before')} → {v('v1_val_mask_iou_after')}) but the generator fell to "
    f"{v('v1_test_iou_finetuned')}. In v2, every configuration that also trained the IoU head lowered the generator's "
    "validation IoU at all three learning rates, so selection kept epoch 0. "
    f"With filters off, however, the loss-selected grid model (E_loss) beats the frozen masks "
    f"({v('e_loss_test_nofilter_iou')} vs. {v('filters_off_test_iou')}): the masks get better, and the recalibrated "
    "IoU head under the fixed 0.5 threshold discards them. With the IoU-head loss off (grid prompts, tokens 1–3, "
    f"E_noiou) the fine-tune beats the frozen model for the first time (Δ {v('e_noiou_test_delta_ci_3dp')}); "
    f"negative clicks (F) reach {v('f_iouoff_test_iou')}. Expectations written before the results were refuted on: D "
    "best, trained masks equal with filters off (+0.042), F gain ≤ +0.02 (+0.057).",
]
TABLE_CAPTION = ("**Table 1.** Test results (132 patches). Δ: paired difference in mean best IoU to the frozen 16 × 16 "
                 "generator, 95 % bootstrap CI. Trained rows use the 16 × 16 generator with filters on.")
TABLE = [["Model", "Mean best IoU", "Masks / patch", "Δ [95 % CI]"],
         ["Frozen, 16 × 16 (default)", v("v1_test_iou_untrained"), v("filters_on_masks_per_patch", nd=1), "—"],
         ["Frozen, filters off", v("filters_off_test_iou"), v("filters_off_masks_per_patch", nd=1),
          v("filters_off_delta_ci")],
         ["Frozen, 32 × 32", v("pps32_full132_iou"), v("pps32_full132_masks_per_patch", nd=1),
          v("pps32_full132_delta_ci")],
         ["v1: centroid, loss-selected", v("v1_test_iou_finetuned"), v("v1_masks_per_patch_after", nd=1),
          v("v1_test_delta_ci_full")],
         ["E_loss: grid, IoU loss on", v("e_loss_test_iou"), v("e_loss_masks_per_patch", nd=1),
          v("e_loss_test_delta_ci")],
         ["E_noiou: grid, IoU loss off", v("e_noiou_test_iou"), v("e_noiou_masks_per_patch", nd=1),
          v("e_noiou_test_delta_ci_3dp")],
         ["F: + negatives, IoU loss off", v("f_iouoff_test_iou"), v("f_iouoff_masks_per_patch", nd=1),
          v("f_iouoff_test_delta_ci")]]
FIGURE_CAPTION = ("**Fig. 1.** Test mean best IoU with SAM's quality filters on and off (labels: masks per patch). "
                  "Trained masks beat frozen ones once the filter stops discarding them.")
CONCLUSIONS = [
    f"Prompt-free SAM 2.1 transfers poorly to Sentinel-2 parcels ({v('v1_test_iou_untrained')}). The cost of the "
    "prompt has three parts: prompt density, a quality filter calibrated on another domain, and a mismatch between "
    "training and inference prompts. With matched prompts and the IoU head left alone, adapting 3.7 % of "
    f"the parameters gives +{v('e_noiou_test_delta', nd=3)}, close to a 32 × 32 grid while keeping the 16 × 16 prompts. Each "
    "failure improved a proxy: validate on the deployment pipeline, not the proxy. "
    "**Limitations:** one seed; 150 training patches; 64 × 64 only on 40 patches; shared-GPU timings are noisy; RGB, "
    "one month.",
]
REFERENCES = [
    "[1] N. Ravi et al., “SAM 2: Segment Anything in Images and Videos,” arXiv:2408.00714, 2024.",
    "[2] A. Kirillov et al., “Segment Anything,” in Proc. IEEE/CVF ICCV, pp. 4015–4026, 2023.",
    "[3] D. Sykas, M. Sdraka, D. Zografakis and I. Papoutsis, “A Sentinel-2 Multiyear, Multicountry Benchmark Dataset for "
    "Crop Classification and Segmentation With Deep Learning,” IEEE J. Sel. Topics Appl. Earth Observ. Remote Sens., "
    "vol. 15, pp. 2937–2953, 2022.",
    "[4] M. Drusch et al., “Sentinel-2: ESA's Optical High-Resolution Mission for GMES Operational Services,” Remote Sens. "
    "Environ., vol. 120, pp. 25–36, 2012.",
    "[5] A. Hollstein, K. Segl, L. Guanter, M. Brell and M. Enesco, “Ready-to-Use Methods for the Detection of Clouds, "
    "Cirrus, Snow, Shadow, Water and Clear Sky Pixels in Sentinel-2 MSI Images,” Remote Sens., vol. 8, no. 8, p. 666, 2016.",
    "[6] C. Ryali et al., “Hiera: A Hierarchical Vision Transformer without the Bells-and-Whistles,” in Proc. ICML, 2023.",
]

# -------------------------------------------------------------------------------------------------------------- HELPERS
doc = Document(TEMPLATE)
P = list(doc.paragraphs)


def set_text(p, text, size=None):
    """Replace the runs of paragraph p, keeping the first run's formatting; **bold** and *italic* spans are supported."""
    base = copy.deepcopy(p.runs[0]._r) if p.runs else None
    for r in p._p.findall(qn("w:r")):
        p._p.remove(r)
    for part in re.split(r"(\*\*.+?\*\*|\*[^*]+?\*)", text):
        if not part:
            continue
        r = p.add_run(part.strip("*") if part.startswith("*") else part)
        if base is not None:
            rpr = base.find(qn("w:rPr"))
            if rpr is not None:
                rpr = copy.deepcopy(rpr)
                for tag in ("w:b", "w:bCs", "w:i", "w:iCs"):   # bold/italic come only from the markup
                    for e in rpr.findall(qn(tag)):
                        rpr.remove(e)
                r._r.insert(0, rpr)
        if part.startswith("**"):
            r.bold = True
        elif part.startswith("*"):
            r.italic = True
        if size:
            r.font.size = Pt(size)
    return p


def insert_after(p, text, size=None, like=None):
    """New paragraph after p, formatted like `like` (default: p itself)."""
    new = copy.deepcopy((like or p)._p)
    p._p.addnext(new)
    return set_text(Paragraph(new, p._parent), text, size)


def remove(p):
    p._p.getparent().remove(p._p)


def fill(body_idx, lines, size=None):
    """Write `lines` into the paragraphs P[body_idx[0]] (template) and following; delete the other old body paragraphs."""
    first = P[body_idx[0]]
    for i in body_idx[1:]:
        remove(P[i])
    last = set_text(first, lines[0], size)
    for line in lines[1:]:
        last = insert_after(last, line, size)
    return last


# --------------------------------------------------------------------------------------------------------------- BUILD
# Group draft layout: 1 title | 3-7 authors, affiliations, e-mails (kept) | 9 INTRO | 10-11 | 12 METHOD | 13-15 |
# 16-19 empty | 20 "3. PRELIMINARY FINDINGS" | 21-22 | 23 CONCLUSIONS | 24 empty | 25 | 26 empty | 27 REFERENCES |
# 28 empty | 29-35 references [1]-[7] | 36 empty
assert P[20].text.startswith("3. PRELIMINARY") and P[27].text.startswith("REFERENCES"), "group draft layout changed"
set_text(P[1], f"**{TITLE}**")
fill([10, 11], INTRO)
fill([13, 14, 15], METHOD)
for i in (16, 17, 18, 19, 24, 28):
    remove(P[i])
set_text(P[20], "**3. RESULTS**")
fill([21, 22], RESULTS)
fill([25], CONCLUSIONS)
fill(list(range(29, 36)), REFERENCES, size=9)
for p in doc.paragraphs:
    if re.match(r"^\[\d\] ", p.text):
        p.paragraph_format.space_after = Pt(1)
while not doc.paragraphs[-1].text.strip():
    remove(doc.paragraphs[-1])

# Table 1 after the "Cost 3" paragraph (caption above the table)
paying = next(p for p in doc.paragraphs if p.text.startswith("Cost 3"))
cap = insert_after(paying, TABLE_CAPTION, size=8.5)
cap.paragraph_format.space_after = Pt(2)
cap.paragraph_format.keep_with_next = True
tbl = doc.add_table(rows=len(TABLE), cols=len(TABLE[0]))
tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
tbl.autofit = False
widths = [Cm(5.0), Cm(2.6), Cm(2.6), Cm(4.0)]
for i, row in enumerate(TABLE):
    for j, val in enumerate(row):
        c = tbl.cell(i, j)
        c.width = widths[j]
        p = c.paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.paragraph_format.space_after = Pt(0)
        run = p.add_run(val)
        run.font.size = Pt(8)
        run.font.name = "Aptos"
        run.bold = i == 0
tblPr = tbl._tbl.tblPr
borders = tblPr.makeelement(qn("w:tblBorders"), {})
for edge in ("top", "bottom", "insideH"):
    borders.append(borders.makeelement(qn(f"w:{edge}"), {qn("w:val"): "single", qn("w:sz"): "4", qn("w:color"): "808080"}))
anchor = next((tblPr.find(qn(t)) for t in ("w:shd", "w:tblLayout", "w:tblCellMar", "w:tblLook")
               if tblPr.find(qn(t)) is not None), None)
anchor.addprevious(borders) if anchor is not None else tblPr.append(borders)
cap._p.addnext(tbl._tbl)
spacer = copy.deepcopy(cap._p)
tbl._tbl.addnext(spacer)
sp = Paragraph(spacer, cap._parent)
for r_ in sp._p.findall(qn("w:r")):
    sp._p.remove(r_)
sp.paragraph_format.space_after = Pt(0)
sp.paragraph_format.line_spacing = Pt(6)
for i, row in enumerate(tbl.rows):
    trPr = row._tr.get_or_add_trPr()
    trPr.append(trPr.makeelement(qn("w:cantSplit"), {}))
    if i < len(tbl.rows) - 1:
        for c in row.cells:
            c.paragraphs[0].paragraph_format.keep_with_next = True

# Fig. 1 after "Cost 2" (filters on vs off)
grid = next(p for p in doc.paragraphs if p.text.startswith("Cost 2"))
figp = insert_after(grid, "")
figp.alignment = WD_ALIGN_PARAGRAPH.CENTER
figp.paragraph_format.space_after = Pt(0)
figp.paragraph_format.keep_with_next = True
figp.add_run().add_picture(str(FIGURE), width=Cm(5.8))
fcap = insert_after(figp, FIGURE_CAPTION, size=8.5, like=grid)
fcap.alignment = WD_ALIGN_PARAGRAPH.CENTER
fcap.paragraph_format.space_after = Pt(6)

# Tighter body spacing than the group draft (line 1.43, 11 pt before/after) so the text fits the 2-page limit.
for p in doc.paragraphs:
    pf = p.paragraph_format
    if pf.line_spacing and pf.line_spacing > 1.2:
        pf.line_spacing = 1.15
        pf.space_before = Pt(0)
        pf.space_after = Pt(4)
    pf.widow_control = True
    pf.keep_together = bool(re.match(r"^\[\d+\] ", p.text))
    if re.match(r"^(\d\. [A-Z]|REFERENCES)", p.text):
        pf.keep_with_next = True

# Drop the empty paragraph(s) before the REFERENCES heading (space is tight on page 2).
refs = next(p for p in doc.paragraphs if p.text.startswith("REFERENCES"))
while refs._p.getprevious() is not None and refs._p.getprevious().tag == qn("w:p") and \
        not Paragraph(refs._p.getprevious(), refs._parent).text.strip():
    refs._p.getparent().remove(refs._p.getprevious())
refs.paragraph_format.space_before = Pt(6)

doc.core_properties.title = TITLE
doc.core_properties.author = "Johanna Zagon, Andre Rubio, Pierre Lavergne"
doc.core_properties.subject = "WAIMLAp 2026 extended abstract"

doc.save(OUT)
print("saved", OUT)
print("pending keys:", ", ".join(PENDING) if PENDING else "none")
