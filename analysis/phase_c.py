"""Phase C: domain-shift / failure analysis from existing results (no model runs).
Run from repo root: python analysis/phase_c.py"""
import os, numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = "results/phase_c"; os.makedirs(OUT, exist_ok=True)
MIN_AREA = 20
BINS = [(20, 50), (50, 100), (100, 200), (200, 500), (500, 10**9)]
BLAB = ["20-49", "50-99", "100-199", "200-499", ">=500"]
PPS = [16, 32, 64]
rng = np.random.default_rng(0)

base = pd.read_csv("results/baseline/eval_patches.csv")
unt = pd.read_csv("results/v1_nb07/eval_test_untrained.csv")
fin = pd.read_csv("results/v1_nb07/eval_test_finetuned.csv")
man = pd.read_csv("data/manifest.csv"); man = man[man.kept == 1]
split = pd.read_csv("results/split.csv")

# ---- 1. tables (per-patch values only: CSVs have no per-field IoU) ----
def agg(df, keys):
    # field-weighted mean of per-patch means; also unweighted patch mean
    def f(g):
        w = g.n_fields
        return pd.Series({
            "n_patches": len(g), "n_fields": int(w.sum()),
            "mean_best_iou_patchmean": g.mean_best_iou.mean(),
            "mean_best_iou_fieldweighted": np.average(g.mean_best_iou, weights=w),
            "pct_fields_iou_gt_0.5": 100*np.average(g["frac_iou_gt_0.5"], weights=w),
            "mask_on_field": g.mask_on_field_fraction.mean(),
            "n_masks_mean": g.n_masks.mean()})
    return df.groupby(keys).apply(f, include_groups=False).reset_index()
def allrow(df, name):
    d = df.copy(); d["country"] = "ALL"; d["tile"] = "ALL"
    return d
tabs = []
for name, df in [("baseline_all888", base), ("test_untrained", unt), ("test_finetuned", fin)]:
    t = pd.concat([agg(df, ["country"]).assign(tile="ALL"), agg(df, ["country", "tile"]),
                   agg(allrow(df, name), ["country"]).assign(tile="ALL")])
    t.insert(0, "run", name); tabs.append(t)
tab = pd.concat(tabs); tab.round(4).to_csv(f"{OUT}/table_iou_by_country_tile.csv", index=False)

# paired delta on test
pair = unt.merge(fin, on=["patch", "country", "tile"], suffixes=("_u", "_f"))
pair["delta"] = pair.mean_best_iou_f - pair.mean_best_iou_u
bs = [pair.delta.values[rng.integers(0, len(pair), len(pair))].mean() for _ in range(5000)]
ci = np.percentile(bs, [2.5, 97.5])
dstat = pair.groupby("country").delta.agg(["mean", "count", lambda x: (x < 0).mean()])
dstat.columns = ["mean_delta", "n", "frac_worse"]
dstat.to_csv(f"{OUT}/table_paired_delta_by_country.csv")

# ---- 2. parcel sizes + grid hit analysis ----
H = W = 366
rows = []
for _, r in man.iterrows():
    p = np.load(f"data/processed/{r.patch}_parcels.npy")
    ids, cnt = np.unique(p[p != 0], return_counts=True)
    keep = cnt >= MIN_AREA
    hit = {}
    for n in PPS:
        c = (np.arange(n) + 0.5) / n * np.array([W])[0]   # SAM: (i+0.5)/n of image side
        px = np.clip(np.floor(c).astype(int), 0, W - 1)
        gy, gx = np.meshgrid(px, px, indexing="ij")
        hit[n] = set(np.unique(p[gy, gx]).tolist())
    for i, a in zip(ids[keep], cnt[keep]):
        rows.append(dict(patch=r.patch, country=r.country, tile=r.tile, area=int(a),
                         **{f"hit{n}": int(i in hit[n]) for n in PPS}))
par = pd.DataFrame(rows)
par["bin"] = pd.cut(par.area, [b[0] for b in BINS] + [10**9], right=False, labels=BLAB)
par.drop(columns=[]).to_csv(f"{OUT}/parcels_hit.csv", index=False)
size = par.area.describe(percentiles=[.1, .25, .5, .75, .9]).round(1)
size_c = par.groupby("country").area.describe(percentiles=[.25, .5, .75]).round(1)
size_c.to_csv(f"{OUT}/table_parcel_area_by_country.csv")
hit_rows = []
for grp, g in [("ALL", par)] + list(par.groupby("country")):
    for n in PPS:
        hit_rows.append(dict(group=grp, pps=n, spacing_px=round(W/n, 1), bin="all",
                             n=len(g), frac_hit=g[f"hit{n}"].mean()))
        for b in BLAB:
            s = g[g.bin == b]
            hit_rows.append(dict(group=grp, pps=n, spacing_px=round(W/n, 1), bin=b,
                                 n=len(s), frac_hit=s[f"hit{n}"].mean()))
hit = pd.DataFrame(hit_rows); hit.round(4).to_csv(f"{OUT}/table_fraction_parcels_hit.csv", index=False)
med = par.area.median()
# naive arithmetic: median-size square parcel side s -> P(hit)~ (s/spacing)^2 capped at 1
arith = pd.DataFrame([dict(pps=n, spacing_px=W/n, median_side_px=np.sqrt(med),
                           naive_p_hit=min(1, (np.sqrt(med)/(W/n))**2)) for n in PPS])
arith.round(4).to_csv(f"{OUT}/table_prompt_density_arithmetic.csv", index=False)

# link per-patch baseline IoU to patch-level median parcel area
pm = par.groupby("patch").area.median().rename("median_area")
pm_hit = par.groupby("patch").hit16.mean().rename("frac_hit16")
bj = base.merge(pm, on="patch").merge(pm_hit, on="patch")
corr = bj[["mean_best_iou", "median_area", "frac_hit16"]].corr(method="spearman").round(3)
corr.to_csv(f"{OUT}/table_spearman_patch_iou_vs_size_hit.csv")

# ---- 4. figures ----
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
# fig1
fig, ax = plt.subplots(1, 2, figsize=(9, 3.8))
for a, col, ttl in [(ax[0], "mean_best_iou", "Mean best IoU per patch"),
                    (ax[1], "frac_iou_gt_0.5", "Fraction of fields with IoU > 0.5")]:
    data = [base[base.country == c][col] for c in ["ES", "FR"]]
    bp = a.boxplot(data, tick_labels=["ES", "FR"], showmeans=True, patch_artist=True)
    for p_, c in zip(bp["boxes"], ["#d95f02", "#1b9e77"]): p_.set_facecolor(c); p_.set_alpha(.6)
    a.set_title(ttl); a.set_ylabel(col)
fig.suptitle("Baseline (frozen SAM 2.1, pps=16), 888 patches (per-patch values)")
fig.tight_layout(); fig.savefig(f"{OUT}/fig_iou_by_country.png", dpi=150); plt.close(fig)
# fig2
fig, ax = plt.subplots(figsize=(7, 4.2))
g = hit[(hit.group == "ALL") & (hit.bin != "all")]
for n, c in zip(PPS, ["#d95f02", "#7570b3", "#1b9e77"]):
    s = g[g.pps == n]; ax.plot(s.bin, s.frac_hit, "o-", color=c, label=f"pps={n} (spacing {W/n:.1f} px)")
ax.set_xlabel("Parcel area (px, 366x366 patch)"); ax.set_ylabel("Fraction of parcels with >=1 grid point")
ax.set_ylim(0, 1.02); ax.legend(frameon=False); ax.set_title("Prompt-grid coverage of parcels (888 patches)")
fig.tight_layout(); fig.savefig(f"{OUT}/fig_parcels_hit_vs_pps.png", dpi=150); plt.close(fig)
# fig3
fig, ax = plt.subplots(figsize=(7, 4))
ax.hist(pair.delta, bins=30, color="#7570b3"); ax.axvline(0, color="k", lw=1)
ax.axvline(pair.delta.mean(), color="r", ls="--", label=f"mean {pair.delta.mean():.3f}")
ax.set_xlabel("Per-patch delta mean best IoU (finetuned - untrained)"); ax.set_ylabel("Patches")
ax.set_title(f"Paired delta, 132 test patches ({(pair.delta<0).sum()} worse)"); ax.legend(frameon=False)
fig.tight_layout(); fig.savefig(f"{OUT}/fig_paired_delta_hist.png", dpi=150); plt.close(fig)

# ---- 5. summary ----
def md(df):
    fmt = lambda x: f"{x:.3f}" if isinstance(x, float) else str(x)
    L = ["| " + " | ".join(map(str, df.columns)) + " |", "|" + "---|" * len(df.columns)]
    L += ["| " + " | ".join(fmt(x) for x in r) + " |" for r in df.itertuples(index=False)]
    return "\n".join(L)
sel = tab[tab.tile == "ALL"][["run", "country", "n_patches", "n_fields", "mean_best_iou_fieldweighted", "pct_fields_iou_gt_0.5", "mask_on_field", "n_masks_mean"]]
tl = tab[(tab.tile != "ALL")][["run", "country", "tile", "n_patches", "mean_best_iou_fieldweighted", "pct_fields_iou_gt_0.5", "mask_on_field"]]
hp = hit[hit.group == "ALL"].pivot(index="bin", columns="pps", values="frac_hit").reindex(["all"] + BLAB)
hn = hit[(hit.group == "ALL")&(hit.pps==16)].set_index("bin").n.reindex(["all"] + BLAB)
hp.insert(0, "n_parcels", hn.values)
hc = hit[(hit.bin == "all")].pivot(index="group", columns="pps", values="frac_hit")
h16 = lambda c, b="all": hit[(hit.group == c) & (hit.pps == 16) & (hit.bin == b)].frac_hit.iloc[0]
h = lambda n, b="all", c="ALL": hit[(hit.group == c) & (hit.pps == n) & (hit.bin == b)].frac_hit.iloc[0]
def v(run, c, col): return tab[(tab.run == run) & (tab.country == c) & (tab.tile == "ALL")][col].iloc[0]
S = f"""# Phase C: domain shift and cost of the prompt (from existing CSVs, no model runs)

Generated by `analysis/phase_c.py`. **Granularity:** the eval CSVs hold only per-patch aggregates (mean best IoU, fraction IoU>0.5, mask-on-field), not per-field IoU. All IoU tables are therefore per-patch values; the "mean" columns are either the plain patch mean or weighted by `n_fields`. Fields = parcels with area >= {MIN_AREA} px.

## 1. IoU by country and tile
{md(sel.reset_index(drop=True))}

By tile:

{md(tl.reset_index(drop=True))}

Paired fine-tuned minus untrained on 132 test patches: mean delta {pair.delta.mean():.4f}, 95% bootstrap CI [{ci[0]:.4f}, {ci[1]:.4f}], {(pair.delta<0).sum()}/{len(pair)} patches worse.

{md(dstat.reset_index())}

## 2. Parcel size and prompt density
Parcels (area >= {MIN_AREA}) over the 888 kept patches: n={len(par)}, median area {med:.0f} px (about {np.sqrt(med):.0f}x{np.sqrt(med):.0f} px), mean {par.area.mean():.0f}, p10 {par.area.quantile(.1):.0f}, p90 {par.area.quantile(.9):.0f}.

{md(size_c.reset_index())}

Grid spacing on 366 px: pps16 = {W/16:.1f} px, pps32 = {W/32:.1f} px, pps64 = {W/64:.1f} px. SAM grid points at (i+0.5)/n of the side, overlaid on the true parcel masks (empirical fraction of parcels with >=1 point):

{md(hp.reset_index())}

By country (all sizes):

{md(hc.reset_index())}

Naive arithmetic for a median-size square parcel (P ~ (side/spacing)^2): {', '.join(f'pps{int(r.pps)}: {r.naive_p_hit:.2f}' for r in arith.itertuples())}.

Spearman correlation across the 888 baseline patches:

{md(corr.reset_index())}

## 3. Interpretation
- Baseline is poor everywhere: mean best IoU {v('baseline_all888','ES','mean_best_iou_fieldweighted'):.3f} (ES) vs {v('baseline_all888','FR','mean_best_iou_fieldweighted'):.3f} (FR); ES is clearly worse. Fields with IoU>0.5: {v('baseline_all888','ES','pct_fields_iou_gt_0.5'):.1f}% (ES) vs {v('baseline_all888','FR','pct_fields_iou_gt_0.5'):.1f}% (FR).
- At pps=16 only {h(16)*100:.1f}% of parcels receive any grid point (ES {h(16,c='ES')*100:.1f}%, FR {h(16,c='FR')*100:.1f}%); for parcels <100 px it is {h(16,BLAB[0])*100:.1f}% / {h(16,BLAB[1])*100:.1f}%. So most small fields cannot be segmented by this prompt grid regardless of the decoder: an upper bound on recall from the prompt, before any model quality enters.
- Raising pps to 32 / 64 raises that coverage to {h(32)*100:.1f}% / {h(64)*100:.1f}%. This is a measurement of coverage, not of IoU; whether SAM then produces good masks at those points is untested here.
- ES parcels are smaller than FR ones (see the per-country size table), which is consistent with part of the ES/FR gap. Hypothesis: size and prompt density explain a large share of the gap, but other shifts (crop types, texture, tile radiometry) are not separated by this analysis; the per-patch correlations above are only suggestive and confounded.
- Fine-tuning made the automatic pipeline worse on test: mean best IoU {unt.mean_best_iou.mean():.3f} -> {fin.mean_best_iou.mean():.3f} ({(pair.delta<0).sum()}/{len(pair)} patches worse, CI excludes zero). This is a negative result for the Phase B intervention as run.
- The fine-tuned model produces fewer masks ({unt.n_masks.mean():.1f} -> {fin.n_masks.mean():.1f} per patch) while mask-on-field is about unchanged ({unt.mask_on_field_fraction.mean():.3f} -> {fin.mask_on_field_fraction.mean():.3f}). Hypothesis (not verified here): the IoU/stability filters discard masks, i.e. recall collapses; a filters-off evaluation would test it.
- Even a perfect decoder is capped by prompt coverage; the decoder-only adaptation attacks the wrong bottleneck for small fields. This is an argument from the coverage numbers, to be confirmed with a pps sweep run on the model.
- Limitations: per-patch aggregation hides per-field behaviour; coverage counts a point inside the parcel, not whether SAM returns the parcel mask; test split has 22 patches per tile, so tile-level numbers are noisy.
"""
open(f"{OUT}/summary.md", "w").write(S)
print(S[:6000])
