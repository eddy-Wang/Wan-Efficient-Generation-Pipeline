"""Aggregate M2 (Self Forcing) results into tables and figures.

Input  (rsynced from the cluster):  results/outputs/<config>/<pid>_<seed>.json   generation records
                                    results/analysis/<config>/<pid>_<seed>.json  evaluation records
                                    results/outputs/subsets.json
Output: tables/*.csv, tables/*.md, figures/*.png
"""
import argparse, glob, json, os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

INK, INK2, GRID = "#0b0b0b", "#52514e", "#e5e4df"
BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"
FPS = 16

plt.rcParams.update({
    "font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True, "figure.dpi": 100, "savefig.dpi": 200,
    "legend.frameon": False,
})


def load(results):
    gen, vid, seg = [], [], []
    for p in glob.glob(os.path.join(results, "outputs", "*", "*.json")):
        r = json.load(open(p))
        if r.get("schema") != "genai-m2-v1":
            continue
        c, perf = r["config"], r["performance"]
        gen.append(dict(config=r["config_name"], pid=r["prompt_id"], steps=len(c["steps"]), L=c["L"], sink=c["sink"],
                        chunk=c["chunk"], ctx=c["ctx"], frames=r["video"]["frames"],
                        video_s=r["video"]["frames"] / FPS, gen_s=perf["generation_seconds"],
                        diffusion_ms=perf["diffusion_ms"], vae_ms=perf["vae_ms"], init_ms=perf["init_ms"],
                        peak_gib=perf["peak_torch_allocated_bytes"] / 2**30, block_ms=perf["block_ms"],
                        gpu=perf["gpu_name"]))
    for p in glob.glob(os.path.join(results, "analysis", "*", "*.json")):
        r = json.load(open(p))
        s = pd.DataFrame(r["segments"])
        vid.append(dict(config=r["config"], pid=r["prompt_id"], clip=r["frame_mean_clip_cosine_x100"],
                        brightness=s.brightness.mean(), saturation=s.saturation.mean(),
                        sharpness=s.sharpness.mean(), motion=s.motion.mean()))
        for d in r["segments"]:
            seg.append(dict(config=r["config"], pid=r["prompt_id"], **d))
    return pd.DataFrame(gen), pd.DataFrame(vid), pd.DataFrame(seg)


def se(x):
    x = np.asarray(x, float)
    return 1.96 * x.std(ddof=1) / np.sqrt(len(x)) if len(x) > 1 else 0.0


def md_table(df, path, floatfmt="{:.2f}"):
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(floatfmt.format(v) if isinstance(v, (float, np.floating)) else str(v)
                                       for v in r.values) + " |")
    open(path, "w", encoding="utf-8").write("\n".join(lines) + "\n")


def paired_delta(base, other, col, n_boot=4000, seed=0):
    m = base.merge(other, on="pid", suffixes=("_b", "_o"))
    if len(m) < 3:
        return np.nan, np.nan, np.nan, len(m)
    d = (m[col + "_o"] - m[col + "_b"]).to_numpy()
    rng = np.random.default_rng(seed)
    bs = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(n_boot)]
    return d.mean(), np.percentile(bs, 2.5), np.percentile(bs, 97.5), len(d)


def finish(fig, path, rect=(0, 0, 1, 1)):
    fig.tight_layout(rect=rect)
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--out", default=".")
    ap.add_argument("--keep-smoke", action="store_true", help="include smoke_* configs (for testing only)")
    a = ap.parse_args()
    tdir, fdir = os.path.join(a.out, "tables"), os.path.join(a.out, "figures")
    os.makedirs(tdir, exist_ok=True)
    os.makedirs(fdir, exist_ok=True)
    gen, vid, seg = load(a.results)
    if not a.keep_smoke:
        gen, vid, seg = [d[~d.config.str.startswith("smoke")] for d in (gen, vid, seg)]
    subsets = json.load(open(os.path.join(a.results, "outputs", "subsets.json")))
    sub30 = set(subsets["sub30"])
    d = gen.merge(vid, on=["config", "pid"], how="left")

    # ---- table 1: performance, all configs
    perf = d.groupby("config").agg(
        n=("pid", "count"), steps=("steps", "first"), window=("L", "first"), sink=("sink", "first"),
        chunk=("chunk", "first"), ctx=("ctx", "first"), video_s=("video_s", "first"),
        gen_s=("gen_s", "mean"), gen_s_sd=("gen_s", "std"), peak_gib=("peak_gib", "max"),
        diffusion_ms=("diffusion_ms", "mean"), vae_ms=("vae_ms", "mean")).reset_index()
    perf["realtime_factor"] = perf.video_s / perf.gen_s  # >1 means faster than playback
    perf["vae_share_pct"] = 100 * perf.vae_ms / (perf.diffusion_ms + perf.vae_ms)
    perf.to_csv(os.path.join(tdir, "m2_performance.csv"), index=False)
    md_table(perf.drop(columns=["diffusion_ms", "vae_ms"]), os.path.join(tdir, "m2_performance.md"))

    # ---- table 2: quality + paired deltas vs main on the same 30 prompts
    short = d[(d.frames == 81)]
    base = short[(short.config == "m2_main") & short.pid.isin(sub30)]
    rows = []
    for cfgname, g in short.groupby("config"):
        g30 = g[g.pid.isin(sub30)]
        if len(g30) == 0 or cfgname.startswith("m2_seed"):
            continue
        row = dict(config=cfgname, n=len(g30), clip=g30["clip"].mean(), clip_sd=g30["clip"].std(),
                   motion=g30.motion.mean(), sharpness=g30.sharpness.mean(), gen_s=g30.gen_s.mean())
        for col in ["clip", "motion", "sharpness", "gen_s"]:
            m, lo, hi, _ = paired_delta(base, g30, col)
            row[f"d_{col}"], row[f"d_{col}_lo"], row[f"d_{col}_hi"] = m, lo, hi
        rows.append(row)
    q = pd.DataFrame(rows)
    q.to_csv(os.path.join(tdir, "m2_ablation_quality.csv"), index=False)
    if len(q):
        md_table(q[["config", "n", "clip", "motion", "sharpness", "gen_s", "d_clip", "d_clip_lo", "d_clip_hi"]],
                 os.path.join(tdir, "m2_ablation_quality.md"))

    # ---- table 3: drift (long videos)
    long = seg[seg.config.str.startswith("m2_long")]
    if len(long):
        rows = []
        for cfgname, g in long.groupby("config"):
            row = dict(config=cfgname, n=g.pid.nunique())
            for col in ["clip_text_x100", "clip_to_first", "brightness", "saturation", "sharpness", "motion"]:
                piv = g.pivot(index="pid", columns="segment", values=col)
                first, last = piv.iloc[:, :2].mean(axis=1), piv.iloc[:, -2:].mean(axis=1)
                dd = (last - first)
                row[f"{col}_first2"], row[f"{col}_last2"] = first.mean(), last.mean()
                row[f"{col}_delta"], row[f"{col}_delta_ci"] = dd.mean(), se(dd)
            rows.append(row)
        dr = pd.DataFrame(rows)
        dr.to_csv(os.path.join(tdir, "m2_drift.csv"), index=False)
        md_table(dr[["config", "n"] + [c for c in dr.columns if c.endswith("_delta") or c.endswith("_delta_ci")]],
                 os.path.join(tdir, "m2_drift.md"), "{:.3f}")

    # ---- figure 1: sampling steps
    def stat(df, col):
        return df[col].mean(), se(df[col])

    names = {1: "m2_steps1", 2: "m2_steps2", 3: "m2_steps3"}
    pts = {}
    for k, nme in names.items():
        g = short[(short.config == nme) & short.pid.isin(sub30)]
        if len(g):
            pts[k] = g
    if pts:
        pts[4] = base
        fig, ax = plt.subplots(1, 3, figsize=(9.2, 2.9))
        for i, (col, lab) in enumerate([("gen_s", "generation time per 5 s video (s)"),
                                        ("clip", "CLIP text alignment (x100)"),
                                        ("motion", "motion (mean abs frame diff)")]):
            ks = sorted(pts)
            m = [stat(pts[k], col)[0] for k in ks]
            e = [stat(pts[k], col)[1] for k in ks]
            ax[i].errorbar(ks, m, yerr=e, color=BLUE, lw=2, marker="o", ms=7, capsize=3, mec="white", mew=1.5)
            ax[i].set_xticks(ks)
            ax[i].set_xlabel("denoising steps per chunk")
            ax[i].set_ylabel(lab)
        fig.suptitle("Fewer denoising steps: speed vs quality (30 prompts, mean ± 95% CI)", x=0.01, ha="left",
                     fontsize=10)
        finish(fig, os.path.join(fdir, "fig_steps.png"))

    # ---- figure 2: attention window / sink
    win_cfg = {6: "m2_win6", 9: "m2_win9", 12: "m2_win12", 15: "m2_win15"}
    wp = {}
    for L, nme in win_cfg.items():
        g = short[(short.config == nme) & short.pid.isin(sub30)]
        if len(g):
            wp[L] = g
    if wp:
        wp[21] = base  # 21 latent frames = whole clip visible = global attention
        sink = short[(short.config == "m2_win9_sink3") & short.pid.isin(sub30)]
        fig, ax = plt.subplots(1, 3, figsize=(9.2, 2.9))
        for i, (col, lab) in enumerate([("clip", "CLIP text alignment (x100)"),
                                        ("motion", "motion (mean abs frame diff)"),
                                        ("gen_s", "generation time per 5 s video (s)")]):
            ks = sorted(wp)
            ax[i].errorbar(ks, [stat(wp[k], col)[0] for k in ks], yerr=[stat(wp[k], col)[1] for k in ks],
                           color=BLUE, lw=2, marker="o", ms=7, capsize=3, mec="white", mew=1.5, label="no sink")
            if len(sink):
                ax[i].errorbar([9.25], [stat(sink, col)[0]], yerr=[stat(sink, col)[1]], color=ORANGE, marker="D",
                               ms=7, capsize=3, mec="white", mew=1.5, ls="none", label="window 9 + sink 3")
            ax[i].set_xticks(sorted(wp))
            ax[i].set_xticklabels([str(k) if k != 21 else "21\n(global)" for k in sorted(wp)])
            ax[i].set_xlabel("attention window (latent frames)")
            ax[i].set_ylabel(lab)
        h, l = ax[0].get_legend_handles_labels()
        fig.legend(h, l, loc="lower center", ncol=2, fontsize=8.5, bbox_to_anchor=(0.5, 0.0))
        fig.suptitle("Sliding KV window on 5 s clips (30 prompts, mean ± 95% CI)", x=0.01, ha="left", fontsize=10)
        finish(fig, os.path.join(fdir, "fig_window.png"), rect=(0, 0.08, 1, 1))

    # ---- figure 3: chunk size
    ch = {1: "m2_chunk1", 3: None, 7: "m2_chunk7"}
    cp = {}
    for k, nme in ch.items():
        g = base if nme is None else short[(short.config == nme) & short.pid.isin(sub30)]
        if len(g):
            cp[k] = g
    if len(cp) > 1:
        fig, ax = plt.subplots(1, 3, figsize=(9.2, 2.7))
        for i, (col, lab) in enumerate([("clip", "CLIP text alignment (x100)"), ("motion", "motion"),
                                        ("gen_s", "generation time (s)")]):
            ks = sorted(cp)
            ax[i].errorbar([str(k) for k in ks], [stat(cp[k], col)[0] for k in ks],
                           yerr=[stat(cp[k], col)[1] for k in ks], ls="none", marker="o", ms=8, capsize=3, lw=2,
                           color=BLUE, mec="white", mew=1.5)
            j = ks.index(3)
            ax[i].plot([str(3)], [stat(cp[3], col)[0]], marker="o", ms=8, color=INK2, mec="white", mew=1.5, ls="none", zorder=5)
            ax[i].set_xlabel("frames per chunk (3 = trained)")
            ax[i].set_ylabel(lab)
        fig.suptitle("Chunk size (30 prompts, mean ± 95% CI between prompts)", x=0.01, ha="left",
                     fontsize=10)
        finish(fig, os.path.join(fdir, "fig_chunk.png"))

    # ---- figures 4a/4b: drift curves on 20 s videos
    metrics = [("clip_text_x100", "CLIP text alignment (x100)"), ("clip_to_first", "CLIP similarity to first 2 s"),
               ("brightness", "mean brightness (0-255)"), ("saturation", "mean saturation (0-255)"),
               ("sharpness", "sharpness (Laplacian variance)"), ("motion", "motion (mean abs frame diff)")]

    n_drift = long[long.config == "m2_long_w21"].pid.nunique() if len(long) else 0

    def drift_figure(order, title, fname):
        present = [o for o in order if o[0] in set(long.config)] if len(long) else []
        if len(present) < 2:
            return
        fig, axes = plt.subplots(2, 3, figsize=(10.5, 6.0))
        for ax, (col, lab) in zip(axes.flat, metrics):
            for cname, label, color, mk, ls in present:
                g = long[long.config == cname].groupby("segment")[col]
                m, s = g.mean(), g.apply(se)
                t = m.index * 2.0
                ax.plot(t, m.values, color=color, lw=2, marker=mk, ms=5, ls=ls, label=label)
                ax.fill_between(t, m - s, m + s, color=color, alpha=0.12, lw=0)
            ax.set_xlabel("time (s)")
            ax.set_ylabel(lab)
        h, l = axes.flat[0].get_legend_handles_labels()
        fig.legend(h, l, loc="lower center", ncol=2, fontsize=8.5, bbox_to_anchor=(0.5, 0.0))
        fig.suptitle(title, x=0.01, ha="left", fontsize=10.5)
        finish(fig, os.path.join(fdir, fname), rect=(0, 0.09, 1, 1))

    drift_figure([("m2_long_w21", "window 21", BLUE, "o", "-"),
                  ("m2_long_w21_sink3", "window 21 + sink 3", ORANGE, "s", "-"),
                  ("m2_long_w12", "window 12", AQUA, "^", "--"),
                  ("m2_long_w12_sink3", "window 12 + sink 3", YELLOW, "D", "--")],
                 f"Window and sink vs drift over a 20 s rollout ({n_drift} prompts, 2 s segments, mean ± 95% CI)",
                 "fig_drift_window.png")
    ramp = ["#9ec5f0", "#4f93de", "#1b4f93"]  # one hue, light to dark = low to high noise level
    drift_figure([("m2_long_w21_sink3", "window 21 + sink 3 (clean context)", INK2, "s", "--"),
                  ("m2_long_w21_sink3_ctx25", "+ re-noised context, t=25", ramp[0], "o", "-"),
                  ("m2_long_w21_sink3_ctx50", "+ re-noised context, t=50", ramp[1], "^", "-"),
                  ("m2_long_w21_sink3_ctx100", "+ re-noised context, t=100", ramp[2], "D", "-")],
                 f"Re-noising the KV context vs drift (window 21 + sink 3; {n_drift} prompts, mean ± 95% CI)",
                 "fig_drift_context.png")

    # ---- table 4: paired drift comparisons (per-prompt change from first to last 4 s)
    if len(long):
        def change(cfgname, col):
            piv = long[long.config == cfgname].pivot(index="pid", columns="segment", values=col)
            return (piv.iloc[:, -2:].mean(axis=1) - piv.iloc[:, :2].mean(axis=1)).rename("v").reset_index()

        pairs = [(c, "m2_long_w21") for c in sorted(set(long.config)) if c != "m2_long_w21"]
        pairs += [(c, "m2_long_w21_sink3") for c in sorted(set(long.config))
                  if c.startswith("m2_long_w21_sink3_ctx")]
        rows = []
        for other, basecfg in pairs:
            if basecfg not in set(long.config):
                continue
            for col, _ in metrics:
                m, lo, hi, n = paired_delta(change(basecfg, col), change(other, col), "v")
                rows.append(dict(config=other, baseline=basecfg, metric=col, diff_of_change=m, ci_lo=lo,
                                 ci_hi=hi, n=n, significant=bool(lo > 0 or hi < 0)))
        pd.DataFrame(rows).to_csv(os.path.join(tdir, "m2_drift_paired.csv"), index=False)
        md_table(pd.DataFrame(rows), os.path.join(tdir, "m2_drift_paired.md"), "{:.3f}")

    # ---- figure 5: per-chunk latency vs chunk index (cost of the KV cache)
    series = [("m2_main", "global attention, 5 s clip", INK2, "o", "-"),
              ("m2_long_w21", "window 21, 20 s clip", BLUE, "s", "-"),
              ("m2_long_w12", "window 12, 20 s clip", AQUA, "^", "--")]
    pl = [s for s in series if s[0] in set(gen.config)]
    if len(pl) > 1:
        fig, ax = plt.subplots(figsize=(6.4, 3.2))
        for cname, label, color, mk, ls in pl:
            blocks = np.array([b for b in gen[gen.config == cname].block_ms if len(b)], dtype=object)
            n = min(len(b) for b in blocks)
            arr = np.array([b[:n] for b in blocks], float)
            ax.plot(np.arange(n), arr.mean(0), color=color, lw=2, marker=mk, ms=5, ls=ls, label=label)
        ax.set_xlabel("chunk index (3 latent frames = 12 video frames each)")
        ax.set_ylabel("time per chunk (ms)")
        ax.legend(fontsize=8.5)
        ax.set_title("Per-chunk generation latency", loc="left", fontsize=10)
        finish(fig, os.path.join(fdir, "fig_chunk_latency.png"))

    # ---- seeds: how much do results move when only the seed changes?
    if "seed10" in subsets and "m2_seed2_main" in set(gen.config):
        ten = set(subsets["seed10"])
        names = ["m2_main", "m2_seed2_main", "m2_seed3_main"]
        sd = pd.concat([d[(d.config == n) & d.pid.isin(ten)].assign(seed_idx=i + 1) for i, n in enumerate(names)])
        rows = []
        for col in ["clip", "motion", "sharpness", "gen_s"]:
            piv = sd.pivot(index="pid", columns="seed_idx", values=col)
            within = float(np.sqrt((piv.var(axis=1, ddof=1)).mean()))   # seed-to-seed sd inside one prompt
            between = float(piv.mean(axis=1).std(ddof=1))               # sd of prompt means
            rows.append(dict(metric=col, mean_seed1=piv[1].mean(), mean_seed2=piv[2].mean(), mean_seed3=piv[3].mean(),
                             within_prompt_sd=within, between_prompt_sd=between))
        sdf = pd.DataFrame(rows)
        sdf.to_csv(os.path.join(tdir, "m2_seed_variance.csv"), index=False)
        md_table(sdf, os.path.join(tdir, "m2_seed_variance.md"), "{:.3f}")
        piv = sd.pivot(index="pid", columns="seed_idx", values="clip")
        order_ids = piv.mean(axis=1).sort_values().index
        fig, ax = plt.subplots(figsize=(7.6, 3.2))
        for k, (color, mk) in enumerate([(BLUE, "o"), (ORANGE, "s"), (AQUA, "^")], start=1):
            ax.plot(range(len(order_ids)), piv.loc[order_ids, k].values, ls="none", marker=mk, ms=7, color=color,
                    mec="white", mew=1.2, label=f"seed {k}")
        ax.set_xticks(range(len(order_ids)))
        ax.set_xticklabels([i.replace("hard_", "h_").replace("vb_", "v") for i in order_ids], rotation=40, ha="right",
                           fontsize=7.5)
        ax.set_ylabel("CLIP text alignment (x100)")
        ax.set_title("Same prompt, three seeds (10 prompts)", loc="left", fontsize=10)
        ax.legend(fontsize=8.5, ncol=3, loc="upper left")
        finish(fig, os.path.join(fdir, "fig_seeds_clip.png"))

        # does the effect of a setting hold for every seed? (change over the 20 s rollout, paired over prompts)
        def chg(cfgname, col):
            g = seg[(seg.config == cfgname) & seg.pid.isin(ten)]
            if not len(g):
                return None
            piv2 = g.pivot(index="pid", columns="segment", values=col)
            return piv2.iloc[:, -2:].mean(axis=1) - piv2.iloc[:, :2].mean(axis=1)

        eff = []
        for k, sfx in enumerate(["", "seed2_", "seed3_"], start=1):
            nm = lambda c: ("m2_long_" + c) if k == 1 else f"m2_{sfx}long_{c}"
            for target, label in (("w21_sink3", "window 21 + sink 3 minus window 21"),
                                  ("w12_sink3", "window 12 + sink 3 minus window 21")):
                for col in ["clip_to_first", "motion", "brightness"]:
                    a_, b_ = chg(nm("w21"), col), chg(nm(target), col)
                    if a_ is None or b_ is None:
                        continue
                    eff.append(dict(effect=label, metric=col, seed=k, mean_diff=float((b_ - a_).mean()),
                                    n=int(len(a_))))
        if eff:
            e = pd.DataFrame(eff).pivot(index=["effect", "metric"], columns="seed", values="mean_diff").reset_index()
            e.columns = [str(c) if not isinstance(c, int) else f"seed{c}" for c in e.columns]
            e.to_csv(os.path.join(tdir, "m2_seed_effects.csv"), index=False)
            md_table(e, os.path.join(tdir, "m2_seed_effects.md"), "{:.3f}")

    # ---- streaming benchmark
    bpath = os.path.join(a.results, "..", "bench", "bench_stream.json")
    if os.path.exists(bpath):
        b = json.load(open(bpath))
        rows = []
        labels = {"A": "A  bf16, whole-clip decode", "B": "B  fp16, streaming decode", "C": "C  fp16, streaming, light VAE",
                  "D": "D  B + torch.compile", "E": "E  C + torch.compile"}
        for k, v in b["configs"].items():
            if "error" in v:
                rows.append(dict(config=labels.get(k, k), status="failed: " + v["error"][:60]))
                continue
            vs = v["videos"]
            row = dict(config=labels.get(k, k), status="ok", n=len(vs),
                       total_s=np.mean([x["total_s"] for x in vs]), realtime_factor=np.mean([x["realtime_factor"] for x in vs]),
                       peak_gib=np.mean([x["peak_gib"] for x in vs]), warmup_s=v.get("warmup_seconds"))
            if "first_frame_s" in vs[0]:
                row.update(first_frame_s=np.mean([x["first_frame_s"] for x in vs]),
                           steady_chunk_s=np.mean([x["steady_block_s"] for x in vs]))
                ps = [x["psnr_vs_full_vae_db"] for x in vs if "psnr_vs_full_vae_db" in x]
                if ps:
                    row["psnr_vs_full_vae_db"] = float(np.mean(ps))
            rows.append(row)
        bdf = pd.DataFrame(rows)
        if "n" in bdf:
            bdf["n"] = bdf["n"].astype("Int64")
        bdf.to_csv(os.path.join(tdir, "m2_streaming.csv"), index=False)
        md_table(bdf, os.path.join(tdir, "m2_streaming.md"), "{:.2f}")
        ok = bdf[bdf.status == "ok"]
        if "steady_chunk_s" in ok and ok.steady_chunk_s.notna().any():
            st = ok[ok.steady_chunk_s.notna()]
            fig, ax = plt.subplots(1, 2, figsize=(9.0, 3.2))
            x = np.arange(len(st))
            ax[0].bar(x, st.steady_chunk_s, color=BLUE, width=0.6)
            ax[0].axhline(0.75, color=INK, lw=1.5, ls="--", label="real-time limit: 0.75 s per chunk")
            ax[0].legend(loc="upper right", fontsize=8.5)
            ax[1].bar(x, st.first_frame_s, color=BLUE, width=0.6)
            for axx, col in ((ax[0], "steady_chunk_s"), (ax[1], "first_frame_s")):
                for xi, v in zip(x, st[col]):
                    axx.text(xi, v, f"{v:.2f}", ha="center", va="bottom", fontsize=8, color=INK)
                axx.set_xticks(x)
                axx.set_xticklabels([c.split("  ")[0] for c in st.config])
            ax[0].set_ylabel("time per 3-frame chunk, steady state (s)")
            ax[1].set_ylabel("time to first frame (s)")
            ax[0].set_xlabel("configuration (see table)")
            ax[1].set_xlabel("configuration (see table)")
            fig.suptitle("Streaming generation on one L40S", x=0.01, ha="left", fontsize=10)
            finish(fig, os.path.join(fdir, "fig_streaming.png"))

    # ---- VBench (pinned custom_input evaluator, five per-video dimensions)
    vdir = os.path.join(a.results, "vbench")
    VD = ["subject_consistency", "motion_smoothness", "dynamic_degree", "imaging_quality", "aesthetic_quality"]
    vrows = []
    for setname in ("main", "abl", "clips"):
        pth = os.path.join(vdir, f"{setname}_eval_results.json")
        if not os.path.exists(pth):
            continue
        raw = json.load(open(pth))
        per = {}
        for dim in VD:
            if dim not in raw:
                continue
            for r in raw[dim][1]:
                per.setdefault(os.path.basename(r["video_path"])[:-4], {})[dim] = (
                    float(r["video_results"]) / (100.0 if dim == "imaging_quality" else 1.0))  # imaging_quality in 0-1
        for stem, vals in per.items():
            parts = stem.split("__")
            cfgname, segname, rest = (parts[0], parts[1], parts[2]) if setname == "clips" else (parts[0], "full", parts[1])
            pid, _seed = rest.rsplit("_", 1)
            vrows.append(dict(config=cfgname, segment=segname, pid=pid, seed=_seed, **vals))
    vdf = pd.DataFrame(vrows)
    if len(vdf):
        vdf.to_csv(os.path.join(tdir, "m2_vbench_per_video.csv"), index=False)
        vfull = vdf[vdf.segment == "full"]
        have = [c for c in VD if c in vdf.columns]
        # main table: the 130 main videos (the row the group table needs)
        vm = vfull[vfull.config == "m2_main"]
        if len(vm):
            t = pd.DataFrame([dict(dimension=c, n=int(vm[c].notna().sum()), mean=vm[c].mean(), sd=vm[c].std(ddof=1))
                              for c in have])
            t.to_csv(os.path.join(tdir, "m2_vbench_main.csv"), index=False)
            md_table(t, os.path.join(tdir, "m2_vbench_main.md"), "{:.4f}")
        # ablations: per config mean and paired difference from the default on the same 30 prompts
        b30 = vfull[(vfull.config == "m2_main") & vfull.pid.isin(sub30)]
        rows = []
        for cfgname, g in vfull.groupby("config"):
            if cfgname.startswith("m2_seed"):
                continue
            g30 = g[g.pid.isin(sub30)]
            if not len(g30):
                continue
            row = dict(config=cfgname, n=len(g30))
            for c in have:
                row[c] = g30[c].mean()
                if cfgname != "m2_main":
                    m, lo, hi, _ = paired_delta(b30, g30, c)
                    row[c + "_diff"], row[c + "_lo"], row[c + "_hi"] = m, lo, hi
            rows.append(row)
        if rows:
            vt = pd.DataFrame(rows)
            vt.to_csv(os.path.join(tdir, "m2_vbench_ablation.csv"), index=False)
            md_table(vt[["config", "n"] + have + [c + "_diff" for c in have if c + "_diff" in vt]],
                     os.path.join(tdir, "m2_vbench_ablation.md"), "{:.4f}")
        # drift: first 4 s vs last 4 s of each 20 s video
        vcl = vdf[vdf.segment.isin(["first4s", "last4s"])]
        if len(vcl):
            ch = {}
            for cfgname, g in vcl.groupby("config"):
                piv = {c: g.pivot(index="pid", columns="segment", values=c) for c in have}
                ch[cfgname] = {c: (piv[c]["last4s"] - piv[c]["first4s"]).dropna() for c in have}
            rows = []
            for cfgname, d_ in ch.items():
                if cfgname.startswith("m2_seed"):
                    continue
                row = dict(config=cfgname, n=len(d_[have[0]]))
                g = vcl[vcl.config == cfgname]
                for c in have:
                    row[c + "_first4s"] = g[g.segment == "first4s"][c].mean()
                    row[c + "_last4s"] = g[g.segment == "last4s"][c].mean()
                    row[c + "_change"] = d_[c].mean()
                    row[c + "_change_ci"] = se(d_[c])
                rows.append(row)
            vd = pd.DataFrame(rows)
            vd.to_csv(os.path.join(tdir, "m2_vbench_drift.csv"), index=False)
            md_table(vd[["config", "n"] + [c + "_change" for c in have] + [c + "_change_ci" for c in have]],
                     os.path.join(tdir, "m2_vbench_drift.md"), "{:.4f}")
            prow = []
            pairs = [(c, "m2_long_w21") for c in sorted(ch) if c != "m2_long_w21" and not c.startswith("m2_seed")]
            pairs += [(c, "m2_long_w21_sink3") for c in sorted(ch) if c.startswith("m2_long_w21_sink3_ctx")]
            for other, basecfg in pairs:
                if basecfg not in ch:
                    continue
                for c in have:
                    m, lo, hi, n = paired_delta(ch[basecfg][c].rename("v").reset_index(),
                                                ch[other][c].rename("v").reset_index(), "v")
                    prow.append(dict(config=other, baseline=basecfg, metric=c, diff_of_change=m, ci_lo=lo, ci_hi=hi,
                                     n=n, significant=bool(lo > 0 or hi < 0)))
            pd.DataFrame(prow).to_csv(os.path.join(tdir, "m2_vbench_drift_paired.csv"), index=False)
            md_table(pd.DataFrame(prow), os.path.join(tdir, "m2_vbench_drift_paired.md"), "{:.4f}")
            # figure: change from first to last 4 s, one panel per dimension
            lab = {"imaging_quality": "imaging quality (0-1)", "aesthetic_quality": "aesthetic quality (0-1)",
                   "subject_consistency": "subject consistency", "dynamic_degree": "share of dynamic clips",
                   "motion_smoothness": "motion smoothness"}
            names = {"m2_long_w21": "window 21", "m2_long_w21_sink3": "window 21 + sink 3", "m2_long_w12": "window 12",
                     "m2_long_w12_sink3": "window 12 + sink 3", "m2_long_w21_ctx50": "window 21, re-noised t=50",
                     "m2_long_w21_sink3_ctx25": "w21 + sink, re-noised t=25",
                     "m2_long_w21_sink3_ctx50": "w21 + sink, re-noised t=50",
                     "m2_long_w21_sink3_ctx100": "w21 + sink, re-noised t=100"}
            shown = [k for k in names if k in set(vd.config)]
            show_dims = [c for c in ["imaging_quality", "aesthetic_quality", "subject_consistency", "dynamic_degree"] if c in have]
            if shown and show_dims:
                fig, axes = plt.subplots(1, len(show_dims), figsize=(2.9 * len(show_dims) + 2.2, 3.6), sharey=True)
                axes = np.atleast_1d(axes)
                y = np.arange(len(shown))[::-1]
                for ax, c in zip(axes, show_dims):
                    sub = vd.set_index("config").loc[shown]
                    ax.errorbar(sub[c + "_change"].values, y, xerr=sub[c + "_change_ci"].values, ls="none", marker="o",
                                ms=7, capsize=3, lw=2, color=BLUE, mec="white", mew=1.5)
                    ax.axvline(0, color=INK2, lw=1)
                    ax.set_xlabel(f"{lab[c]}: last 4 s minus first 4 s")
                    ax.grid(axis="y", visible=False)
                axes[0].set_yticks(y)
                axes[0].set_yticklabels([names[k] for k in shown], fontsize=8.5)
                fig.suptitle("VBench change over a 20 s rollout (mean ± 95% CI over prompts)", x=0.01, ha="left",
                             fontsize=10)
                finish(fig, os.path.join(fdir, "fig_vbench_drift.png"))

    # ---- flat per-video data for the repository (one row per video / per 2 s segment)
    ddir = os.path.join(a.out, "data")
    os.makedirs(ddir, exist_ok=True)
    g_out = gen.copy()
    g_out["block_ms"] = g_out.block_ms.apply(lambda b: json.dumps([round(x, 1) for x in b]))
    g_out.drop(columns=["video_s"]).to_csv(os.path.join(ddir, "per_video_generation.csv"), index=False)
    vid.to_csv(os.path.join(ddir, "per_video_clip_stats.csv"), index=False)
    seg.to_csv(os.path.join(ddir, "per_segment_stats.csv"), index=False)

    print("tables:", sorted(os.listdir(tdir)))
    print("figures:", sorted(os.listdir(fdir)))


if __name__ == "__main__":
    main()
