"""Aggregate pod outputs in results/v2 -> tables, figures, numbers.json, summary.md.
Idempotent; tolerant of missing files (prints MISSING list)."""
import glob, json, os
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = "results/v2"; TAB = "results/tables"; FIG = "results/figures"
os.makedirs(TAB, exist_ok=True); os.makedirs(FIG, exist_ok=True)
MISSING = []
RUNS = ["A_centroid_tok0", "B_centroid_multi", "C_random_tok0", "D_random_multi", "E_grid_multi", "A_loss_select"]
IOU = "mean_best_iou"
CB = {"ALL": "#000000", "ES": "#D55E00", "FR": "#0072B2"}


def rd(p):
    if os.path.exists(p):
        try:
            return pd.read_csv(p)
        except Exception as e:
            MISSING.append(f"{p} (unreadable: {e})")
            return None
    MISSING.append(p)
    return None


def first(*pats):
    for p in pats:
        g = sorted(glob.glob(p))
        if g:
            return g[0]
    return None


def boot(d, n=2000, seed=0):
    d = np.asarray(d, float)
    if len(d) == 0:
        return (np.nan, np.nan, np.nan)
    rng = np.random.default_rng(seed)
    m = d[rng.integers(0, len(d), (n, len(d)))].mean(1)
    return d.mean(), np.percentile(m, 2.5), np.percentile(m, 97.5)


def delta(a, b):
    """paired mean(a-b) over common patches, with CI."""
    if a is None or b is None:
        return (np.nan,) * 3 + (0,)
    m = a[["patch", IOU]].merge(b[["patch", IOU]], on="patch", suffixes=("_a", "_b"))
    return boot(m[IOU + "_a"] - m[IOU + "_b"]) + (len(m),)


def fmt_ci(t):
    return "" if np.isnan(t[0]) else f"{t[0]:+.4f} [{t[1]:+.4f}, {t[2]:+.4f}]"


def stats(p):
    if p is None or len(p) == 0:
        return dict(n=0, iou=np.nan, frac=np.nan, masks=np.nan, sec=np.nan)
    return dict(n=len(p), iou=p[IOU].mean(), frac=p["frac_iou_gt_0.5"].mean(), masks=p.n_masks.mean(), sec=p.gen_seconds.mean())


# ---------- load
split = rd("results/split.csv")
test_patches = set(split[split.split == "test"].patch) if split is not None else None
base = rd(f"{R}/diag/untrained_patches.csv")
nf_path = first(f"{R}/diag/*nofilter*patches*.csv", f"{R}/diag/*no_filter*patches*.csv")
base_nf = rd(nf_path) if nf_path else (MISSING.append(f"{R}/diag/untrained_nofilter_patches.csv") or None)

hist, cfg, tp, tnf = {}, {}, {}, {}
for sub, label in [("", "lr3e-5"), ("lr1e-5/", "lr1e-5"), ("lr3e-6/", "lr3e-6")]:
    for r in RUNS:
        d = f"{R}/{sub}runs/{r}"
        if not os.path.isdir(d):
            if sub == "":
                MISSING.append(d)
            continue
        key = (r, label)
        hist[key] = rd(f"{d}/history.csv")
        try:
            cfg[key] = json.load(open(f"{d}/config.json"))
        except Exception:
            cfg[key] = {}
            MISSING.append(f"{d}/config.json")
        tp[key] = rd(f"{d}/test_patches.csv")
        pn = first(f"{d}/test_nofilter*patches*.csv")
        tnf[key] = rd(pn) if pn else (MISSING.append(f"{d}/test_nofilter_patches.csv") or None)
if not any(k[1] != "lr3e-5" for k in hist):
    MISSING.append(f"{R}/lr1e-5/runs, {R}/lr3e-6/runs (low-lr follow-up)")

# ---------- 1 ablation
rows = []
for key in hist:
    r, lr = key
    h, c, p = hist[key], cfg[key], tp[key]
    be = c.get("best_epoch")
    g = None
    if h is not None and "val_gen_mean_best_iou" in h:
        g = h.set_index("epoch").val_gen_mean_best_iou
    def ge(e):
        return g.get(e, np.nan) if g is not None and e is not None else np.nan
    s, sn = stats(p), stats(tnf[key])
    dl = delta(p, base)
    dn = delta(tnf[key], base_nf)
    rows.append(dict(run=r, lr=c.get("lr", lr), prompt=c.get("prompt"), tokens=c.get("tokens"), select_by=c.get("select_by"),
                     best_epoch=be, epochs_run=c.get("epochs_run"),
                     val_gen_iou_ep0=ge(0), val_gen_iou_ep1=ge(1),
                     val_gen_iou_best=ge(be), test_iou_filters_on=s["iou"], test_iou_filters_off=sn["iou"],
                     frac_gt05=s["frac"], masks_per_patch=s["masks"], n_test=s["n"],
                     delta_vs_untrained=dl[0], delta_ci_lo=dl[1], delta_ci_hi=dl[2],
                     delta_nofilter_vs_untrained_nofilter=dn[0], delta_nf_ci_lo=dn[1], delta_nf_ci_hi=dn[2]))
bs = stats(base)
rows.insert(0, dict(run="untrained", lr=np.nan, prompt="grid16", tokens="-", select_by="-", best_epoch=0,
                    val_gen_iou_ep0=np.nan, test_iou_filters_on=bs["iou"], test_iou_filters_off=stats(base_nf)["iou"],
                    frac_gt05=bs["frac"], masks_per_patch=bs["masks"], n_test=bs["n"]))
abl = pd.DataFrame(rows)
abl.to_csv(f"{TAB}/ablation.csv", index=False, float_format="%.4f")

# ---------- 2 pps sweep
def country_map():
    return dict(zip(split.patch, split.country)) if split is not None else {}

def with_country(p):
    if p is None:
        return None
    p = p.copy()
    if "country" not in p:
        p["country"] = p.patch.map(country_map())
    return p

pps = {}  # (pps, subset) -> df
for k in (16, 32, 64):
    p = rd(f"{R}/diag/untrained_pps{k}_patches.csv")
    if p is not None:
        pps[(k, "first40")] = with_country(p)
pps[(16, "full132")] = with_country(base)
for k in (32, 64):
    parts, miss = [], []
    for s in range(4):
        f = f"{R}/sweep_full/pps{k}_shard{s}/eval_patches.csv"
        if os.path.exists(f):
            parts.append(pd.read_csv(f))
        else:
            miss.append(f)
    if miss:
        MISSING.extend(miss)
    if parts:
        pps[(k, "full132")] = with_country(pd.concat(parts).drop_duplicates("patch"))

prow = []
for (k, sub), p in sorted(pps.items(), key=lambda x: (x[0][1], x[0][0])):
    if p is None:
        continue
    ref = pps.get((16, sub))
    if sub == "full132" and test_patches and len(p) < 132:
        pass
    for grp in ("ALL", "ES", "FR"):
        q = p if grp == "ALL" else p[p.country == grp]
        s = stats(q)
        rq = None if ref is None else (ref if grp == "ALL" else ref[ref.country == grp])
        d = delta(q, rq) if k != 16 else (np.nan,) * 4
        prow.append(dict(subset=sub, pps=k, group=grp, n_patches=s["n"], mean_best_iou=s["iou"], frac_gt05=s["frac"],
                         masks_per_patch=s["masks"], sec_per_patch=s["sec"], delta_vs_pps16=d[0], ci_lo=d[1], ci_hi=d[2],
                         n_paired=d[3]))
pp = pd.DataFrame(prow)
pp.to_csv(f"{TAB}/pps_sweep.csv", index=False, float_format="%.4f")

# ---------- 3 filters off
frow = []
def frow_add(name, on, off):
    so, sf = stats(on), stats(off)
    d = delta(off, on)
    frow.append(dict(model=name, n_on=so["n"], n_off=sf["n"], iou_on=so["iou"], iou_off=sf["iou"],
                     masks_on=so["masks"], masks_off=sf["masks"], delta_off_minus_on=d[0], ci_lo=d[1], ci_hi=d[2]))
frow_add("untrained", base, base_nf)
for key in hist:
    if key[1] == "lr3e-5":
        frow_add(key[0], tp[key], tnf[key])
fo = pd.DataFrame(frow)
fo.to_csv(f"{TAB}/filters_off.csv", index=False, float_format="%.4f")

def verdict():
    u = fo[fo.model == "untrained"].iloc[0]
    if np.isnan(u.iou_off):
        return "UNKNOWN: filters-off evals not synced yet."
    t = fo[(fo.model != "untrained") & fo.iou_off.notna()]
    s = f"Untrained: filters off {u.iou_off:.3f} vs on {u.iou_on:.3f} (masks {u.masks_off:.1f} vs {u.masks_on:.1f}). "
    if len(t):
        s += "Trained runs: mean filters-off IoU %.3f vs on %.3f. " % (t.iou_off.mean(), t.iou_on.mean())
        gap_on = (t.iou_on - u.iou_on).mean(); gap_off = (t.iou_off - u.iou_off).mean()
        s += f"Mean gap to untrained: {gap_on:+.3f} with filters, {gap_off:+.3f} without. "
        s += ("Filters explain the drop (gap mostly closes)." if abs(gap_off) < 0.5 * abs(gap_on)
              else "Filters do NOT explain the drop (gap persists without them); the masks themselves degrade.")
    return s

# ---------- numbers.json
def put(d, k, v):
    if isinstance(d.get(k), str) and d[k].startswith("{{"):
        d[k] = v
    elif k not in d:
        d[k] = v

nj = json.load(open("submission/numbers.json"))
used = {}
ALLFULL = all((k, "full132") in pps and len(pps[(k, "full132")]) >= 132 for k in (16, 32, 64))
for k in (16, 32, 64):
    sub = "full132" if ALLFULL else "first40"
    p = pps.get((k, sub))
    if p is None:
        continue
    s = stats(p); used[k] = (sub, s["n"])
    put(nj, f"pps{k}_test_iou", round(s["iou"], 3)); put(nj, f"pps{k}_sec_per_patch", round(s["sec"], 2))
if used:
    ns = {v[1] for v in used.values()}
    nj["pps_sweep_n_patches"] = {f"pps{k}": v[1] for k, v in used.items()} if len(ns) > 1 else ns.pop()
if not np.isnan(stats(base_nf)["iou"]):
    put(nj, "filters_off_test_iou", round(stats(base_nf)["iou"], 3))
main = abl[(abl.run != "untrained") & (abl.lr == 3e-5) & (abl.run != "A_loss_select")] if len(abl) > 1 else abl.iloc[0:0]
best_run = best_iou = None
if len(main):
    allep0 = (main.best_epoch == 0).all()
    if allep0:
        best_run, best_iou = "none (all runs kept epoch 0 = untrained)", bs["iou"]
    else:
        b = main.sort_values("test_iou_filters_on", ascending=False).iloc[0]
        best_run, best_iou = b.run, b.test_iou_filters_on
    put(nj, "best_v2_run", best_run); put(nj, "best_v2_test_iou", round(float(best_iou), 3))
ls = abl[abl.run == "A_loss_select"]
if len(ls) and not np.isnan(ls.test_iou_filters_on.iloc[0]):
    put(nj, "v2_loss_select_test_iou", round(float(ls.test_iou_filters_on.iloc[0]), 3))
json.dump(nj, open("submission/numbers.json", "w"), indent=2)

# ---------- figures
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
sub = "full132" if all((k, "full132") in pps and len(pps[(k, "full132")]) >= 132 for k in (16, 32, 64)) else "first40"
q = pp[pp["subset"] == sub]
if len(q):
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.8))
    for g in ("ALL", "ES", "FR"):
        z = q[q.group == g].sort_values("pps")
        ax[0].plot(z.pps, z.mean_best_iou, "o-", color=CB[g], label=g)
    ax[0].set_xscale("log", base=2); ax[0].set_xticks([16, 32, 64]); ax[0].set_xticklabels(["16", "32", "64"])
    ax[0].set_xlabel("points_per_side"); ax[0].set_ylabel("mean best IoU"); ax[0].legend(frameon=False)
    ax[0].set_title(f"IoU vs prompt density ({sub}, n={int(q.n_patches.max())})")
    z = q[q.group == "ALL"].sort_values("pps")
    ax[1].plot(z.pps, z.sec_per_patch, "s-", color="#009E73")
    ax[1].set_xscale("log", base=2); ax[1].set_yscale("log"); ax[1].set_xticks([16, 32, 64]); ax[1].set_xticklabels(["16", "32", "64"])
    ax[1].set_xlabel("points_per_side"); ax[1].set_ylabel("sec / patch (log)"); ax[1].set_title("Generation cost (pod GPU, may be shared)")
    fig.tight_layout(); fig.savefig(f"{FIG}/pps_sweep.png", dpi=150); plt.close(fig)

fig, ax = plt.subplots(figsize=(7, 4.2))
pal = ["#0072B2", "#E69F00", "#009E73", "#CC79A7", "#D55E00", "#56B4E9"]
ls_ = {"lr3e-5": "-", "lr1e-5": "-.", "lr3e-6": ":"}
u0 = None
for i, r in enumerate(RUNS):
    for lrl in ("lr3e-5", "lr1e-5", "lr3e-6"):
        h = hist.get((r, lrl))
        if h is None or "val_gen_mean_best_iou" not in h:
            continue
        ax.plot(h.epoch, h.val_gen_mean_best_iou, ls_[lrl], marker="o", ms=3, color=pal[i],
                label=r if lrl == "lr3e-5" else f"{r} ({lrl})")
        if u0 is None:
            u0 = h.val_gen_mean_best_iou.iloc[0]
if u0 is not None:
    ax.axhline(u0, color="k", ls="--", lw=1, label=f"untrained (epoch 0) = {u0:.3f}")
ax.set_xlabel("epoch"); ax.set_ylabel("val generator mean best IoU"); ax.set_title("Generator metric on val (12 patches) per epoch")
ax.legend(frameon=False, fontsize=7, ncol=2); fig.tight_layout(); fig.savefig(f"{FIG}/ablation_val_curves.png", dpi=150); plt.close(fig)

# ---------- summary.md
def md(df):
    if df is None or not len(df):
        return "_no data_"
    df = df.copy()
    for c in df.columns:
        if df[c].dtype.kind == "f":
            df[c] = df[c].map(lambda x: "" if pd.isna(x) else f"{x:.4f}")
    df = df.fillna("")
    h = "| " + " | ".join(df.columns) + " |\n|" + "---|" * len(df.columns) + "\n"
    return h + "\n".join("| " + " | ".join(str(v) for v in r) + " |" for r in df.values)

def g(run, lr="lr3e-5"):
    r = abl[(abl.run == run) & (abl.lr == {"lr3e-5": 3e-5, "lr1e-5": 1e-5, "lr3e-6": 3e-6}[lr])]
    return r.iloc[0] if len(r) else None

def pv(k, grp="ALL", s=sub):
    z = pp[(pp.pps == k) & (pp.group == grp) & (pp["subset"] == s)]
    return z.iloc[0] if len(z) else None

B = []
# 1 filters
B.append("1. Filters off (pred 1): " + verdict())
# 2 pps
p16, p32, p64 = pv(16), pv(32), pv(64)
if p16 is not None and p32 is not None and p64 is not None:
    ok = p32.mean_best_iou > p16.mean_best_iou and p64.mean_best_iou > p32.mean_best_iou
    es = pv(64, "ES"); fr = pv(64, "FR"); es16 = pv(16, "ES"); fr16 = pv(16, "FR")
    B.append(f"2. pps sweep ({sub}): IoU {p16.mean_best_iou:.3f} / {p32.mean_best_iou:.3f} / {p64.mean_best_iou:.3f}; "
             f"sec/patch {p16.sec_per_patch:.1f} / {p32.sec_per_patch:.1f} / {p64.sec_per_patch:.1f}. "
             f"Gain prediction {'CONFIRMED' if ok else 'REFUTED'}; ES gain at 64 {es.mean_best_iou - es16.mean_best_iou:+.3f} vs FR {fr.mean_best_iou - fr16.mean_best_iou:+.3f} "
             f"(ES bigger? {'yes' if es.mean_best_iou - es16.mean_best_iou > fr.mean_best_iou - fr16.mean_best_iou else 'no'}). "
             f"Time ratio vs 16: x{p32.sec_per_patch / p16.sec_per_patch:.1f}, x{p64.sec_per_patch / p16.sec_per_patch:.1f} (predicted ~4x, ~16x; timings from a possibly shared GPU, noisy).")
else:
    B.append("2. pps sweep: incomplete data.")
# 3-7
def line(n, run, pred):
    r = g(run)
    if r is None:
        return f"{n}. {run}: no data. Prediction: {pred}."
    return (f"{n}. {run}: best_epoch={r.best_epoch}, val gen IoU ep0 {r.val_gen_iou_ep0:.3f} -> ep1 {r.val_gen_iou_ep1:.3f}, "
            f"test IoU {r.test_iou_filters_on:.3f} (delta vs untrained {fmt_ci((r.delta_vs_untrained, r.delta_ci_lo, r.delta_ci_hi))}), "
            f"masks/patch {r.masks_per_patch:.1f}. Prediction ({pred}): ")
ordr = {r: (g(r).test_iou_filters_on if g(r) is not None else np.nan) for r in RUNS}
A_, B_, C_, D_, E_, L_ = (ordr[r] for r in RUNS)
best4 = max([A_, B_, C_, D_], key=lambda x: -1 if np.isnan(x) else x)
tail = {
    "A_centroid_tok0": ("modest/no gain ~0.16", lambda: "CONFIRMED (no gain)" if abs(A_ - bs["iou"]) < 0.01 or A_ <= bs["iou"] else "REFUTED (gain)"),
    "B_centroid_multi": ("better than A", lambda: "CONFIRMED" if B_ > A_ else "REFUTED"),
    "C_random_tok0": ("better than A, worse than D", lambda: "CONFIRMED" if A_ < C_ < D_ else "REFUTED"),
    "D_random_multi": ("best of four and above baseline", lambda: "CONFIRMED" if D_ == best4 and D_ > bs["iou"] else "REFUTED"),
    "A_loss_select": ("worst, test IoU < 0.10", lambda: "CONFIRMED" if L_ < 0.10 and L_ <= min(A_, B_, C_, D_) else "REFUTED"),
    "E_grid_multi": ("about equal to D, small gain over baseline", lambda: "CONFIRMED" if abs(E_ - D_) < 0.01 and E_ - bs["iou"] < 0.02 else "REFUTED"),
}
for n, run in zip(range(3, 9), ["A_centroid_tok0", "B_centroid_multi", "C_random_tok0", "D_random_multi", "A_loss_select", "E_grid_multi"]):
    pred, fn = tail[run]
    l = line(n, run, pred)
    r_ = g(run)
    if r_ is None:
        res = ""
    elif r_.best_epoch == 0 and run != "A_loss_select":
        res = (f"NOT TESTABLE on test: epoch 0 kept, so test = untrained. On val after 1 epoch the run fell to {r_.val_gen_iou_ep1:.3f} "
               f"from {r_.val_gen_iou_ep0:.3f}; so 'no gain' holds but any 'better than baseline/other run' claim is REFUTED (no run improves; ep1 val order is a weak proxy).")
    else:
        res = fn()
    B.append(l + res)
# follow-up
fu = abl[(abl.lr < 3e-5) & abl.run.isin(["D_random_multi", "E_grid_multi"])]
if len(fu):
    parts = [f"{r.run}@{r.lr:g}: delta {r.delta_vs_untrained:+.4f}, best_epoch {r.best_epoch}" for r in fu.itertuples()]
    ok = (fu.delta_vs_untrained.abs() < 0.01).all()
    B.append("9. Low-lr follow-up: " + "; ".join(parts) + f". Prediction (|delta|<0.01): {'CONFIRMED' if ok else 'REFUTED'} so far (partial).")
else:
    B.append("9. Low-lr follow-up: no data yet.")

txt = f"""# v2 results summary (auto-generated by analysis/v2_analysis.py)

## Ablation
{md(abl)}

## points_per_side sweep
{md(pp)}

## Filters on vs off
{md(fo)}

Verdict: {verdict()}

## Pre-registered predictions vs results
""" + "\n".join(f"- {b}" for b in B) + "\n\n## Missing at generation time\n" + "\n".join(f"- {m}" for m in MISSING) + "\n"
open(f"{R}/summary.md", "w").write(txt)

print(abl[["run", "lr", "best_epoch", "val_gen_iou_ep0", "val_gen_iou_ep1", "test_iou_filters_on", "delta_vs_untrained", "delta_ci_lo", "delta_ci_hi", "masks_per_patch"]].to_string())
print(pp[pp.group == "ALL"].to_string())
print("MISSING:", *MISSING, sep="\n  ")
