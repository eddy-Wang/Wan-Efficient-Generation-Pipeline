"""M1 (Wan2.1-1.3B, the teacher) and its comparison with M2 (Self Forcing), on the same prompts, seed and evaluators.

Input:  <m1>/outputs/<config>/*.json  (run_m1.py records)   <m1>/analysis/<config>/*.json (analyze_m2.py)
        <m1>/vbench/m1_main_eval_results.json, m1_abl_eval_results.json
        <m2>: the M2 results directory used by aggregate_m2.py (outputs/, analysis/, vbench/)
Output: tables/m1_*.csv|md, figures/fig_m1_*.png
"""
import argparse, glob, json, os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

INK, INK2, GRID = "#0b0b0b", "#52514e", "#e5e4df"
BLUE, ORANGE = "#2a78d6", "#eb6834"
VD = ["subject_consistency", "motion_smoothness", "dynamic_degree", "imaging_quality", "aesthetic_quality"]
SHORT = {"subject_consistency": "subject consistency", "motion_smoothness": "motion smoothness",
         "dynamic_degree": "dynamic degree", "imaging_quality": "imaging quality", "aesthetic_quality": "aesthetic quality",
         "clip": "CLIP x100"}
plt.rcParams.update({
    "font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True, "savefig.dpi": 200, "legend.frameon": False,
})


def records(root, schema):
    rows = []
    for p in glob.glob(os.path.join(root, "outputs", "*", "*.json")):
        r = json.load(open(p))
        if r.get("schema") != schema:
            continue
        steps = r["config"]["steps"]
        rows.append(dict(config=r["config_name"], pid=r["prompt_id"], seed=r["seed"],
                         steps=len(steps) if isinstance(steps, list) else steps,
                         gen_s=r["performance"]["generation_seconds"],
                         peak_gib=r["performance"]["peak_torch_allocated_bytes"] / 2**30))
    return pd.DataFrame(rows)


def clip_scores(root):
    rows = []
    for p in glob.glob(os.path.join(root, "analysis", "*", "*.json")):
        r = json.load(open(p))
        rows.append(dict(config=r["config"], pid=r["prompt_id"], clip=r["frame_mean_clip_cosine_x100"]))
    return pd.DataFrame(rows)


def vbench(root, names):
    rows = {}
    for name in names:
        p = os.path.join(root, "vbench", f"{name}_eval_results.json")
        if not os.path.exists(p):
            continue
        raw = json.load(open(p))
        for dim in VD:
            if dim not in raw:
                continue
            for r in raw[dim][1]:
                stem = os.path.basename(r["video_path"])[:-4]
                cfgname, rest = stem.split("__")[0], stem.split("__")[-1]
                pid = rest.rsplit("_", 1)[0]
                rows.setdefault((cfgname, pid), {})[dim] = float(r["video_results"]) / (100.0 if dim == "imaging_quality" else 1.0)
    return pd.DataFrame([dict(config=k[0], pid=k[1], **v) for k, v in rows.items()])


def paired(a, b, col, n_boot=4000, seed=0):
    """mean of (b - a) over shared prompts, with bootstrap 95% interval."""
    m = a[["pid", col]].merge(b[["pid", col]], on="pid", suffixes=("_a", "_b")).dropna()
    if len(m) < 3:
        return np.nan, np.nan, np.nan, len(m)
    d = (m[col + "_b"] - m[col + "_a"]).to_numpy()
    rng = np.random.default_rng(seed)
    bs = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(n_boot)]
    return d.mean(), np.percentile(bs, 2.5), np.percentile(bs, 97.5), len(d)


def md_table(df, path, fmt="{:.4f}"):
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(fmt.format(v) if isinstance(v, (float, np.floating)) and not pd.isna(v)
                                       else str(v) for v in r.values) + " |")
    open(path, "w", encoding="utf-8").write("\n".join(lines) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--m1", required=True)
    ap.add_argument("--m2", required=True)
    ap.add_argument("--out", default=".")
    a = ap.parse_args()
    tdir, fdir = os.path.join(a.out, "tables"), os.path.join(a.out, "figures")
    os.makedirs(tdir, exist_ok=True)
    os.makedirs(fdir, exist_ok=True)

    g1, g2 = records(a.m1, "genai-m1-v1"), records(a.m2, "genai-m2-v1")
    c1, c2 = clip_scores(a.m1), clip_scores(a.m2)
    v1, v2 = vbench(a.m1, ["m1_main", "m1_abl"]), vbench(a.m2, ["main", "abl"])
    d1 = g1.merge(c1, on=["config", "pid"], how="left").merge(v1, on=["config", "pid"], how="left")
    d2 = g2.merge(c2, on=["config", "pid"], how="left").merge(v2, on=["config", "pid"], how="left")
    d1 = d1[d1.seed == 20261027]
    d2 = d2[d2.seed == 20261027]
    sub30 = json.load(open(os.path.join(a.m2, "outputs", "subsets.json")))["sub30"]
    d1.to_csv(os.path.join(tdir, "m1_per_video.csv"), index=False)

    metrics = [m for m in ["clip"] + VD if m in d1.columns and m in d2.columns]

    # ---- M1 main row (130 prompts) and paired M2 - M1 on the same 130 prompts
    m1m, m2m = d1[d1.config == "m1_main"], d2[d2.config == "m2_main"]
    if len(m1m):
        row = dict(method="M1 Wan2.1-1.3B", n=len(m1m), steps=50, cfg=6.0, gen_s=m1m.gen_s.mean(), gen_s_sd=m1m.gen_s.std(),
                   peak_gib=m1m.peak_gib.max(), **{c: m1m[c].mean() for c in metrics})
        row2 = dict(method="M2 Self Forcing", n=len(m2m), steps=4, cfg=0.0, gen_s=m2m.gen_s.mean(), gen_s_sd=m2m.gen_s.std(),
                    peak_gib=m2m.peak_gib.max(), **{c: m2m[c].mean() for c in metrics})
        main_tab = pd.DataFrame([row, row2])
        main_tab.to_csv(os.path.join(tdir, "m1_vs_m2_main.csv"), index=False)
        md_table(main_tab, os.path.join(tdir, "m1_vs_m2_main.md"))
        prow = []
        for c in metrics + ["gen_s"]:
            m, lo, hi, n = paired(m1m, m2m, c)
            prow.append(dict(metric=c, m1_mean=m1m[c].mean(), m2_mean=m2m[c].mean(), diff_m2_minus_m1=m, ci_lo=lo, ci_hi=hi,
                             n=n, significant=bool(lo > 0 or hi < 0)))
        pt = pd.DataFrame(prow)
        pt.to_csv(os.path.join(tdir, "m1_vs_m2_paired.csv"), index=False)
        md_table(pt, os.path.join(tdir, "m1_vs_m2_paired.md"))
        # by prompt source
        src = lambda df: df.assign(source=["hard" if p.startswith("hard_") else "vbench" for p in df.pid])
        bys = pd.concat([src(m1m).assign(method="M1"), src(m2m).assign(method="M2")]).groupby(["method", "source"])[metrics].mean()
        bys.reset_index().to_csv(os.path.join(tdir, "m1_vs_m2_by_source.csv"), index=False)
        md_table(bys.reset_index(), os.path.join(tdir, "m1_vs_m2_by_source.md"))

    # ---- equal-budget comparison on the 30-prompt subset: teacher with few steps vs the distilled student
    cfgs1 = {4: "m1_steps4", 8: "m1_steps8", 16: "m1_steps16", 50: "m1_main"}
    cfgs2 = {1: "m2_steps1", 2: "m2_steps2", 3: "m2_steps3", 4: "m2_main"}
    base = d2[(d2.config == "m2_main") & d2.pid.isin(sub30)]
    rows = []
    for method, cfgs, df in (("M1", cfgs1, d1), ("M2", cfgs2, d2)):
        for k, nm in cfgs.items():
            g = df[(df.config == nm) & df.pid.isin(sub30)]
            if not len(g):
                continue
            r = dict(method=method, steps=k, n=len(g), gen_s=g.gen_s.mean(), **{c: g[c].mean() for c in metrics})
            for c in ["aesthetic_quality", "imaging_quality", "dynamic_degree", "subject_consistency", "clip"]:
                if c in metrics and method == "M2" and k == 4:
                    r[c + "_vs_m2_4steps"] = "reference"
                elif c in metrics:
                    m, lo, hi, _ = paired(base, g, c)
                    r[c + "_vs_m2_4steps"] = f"{m:+.3f} [{lo:+.3f}, {hi:+.3f}]" + ("*" if (lo > 0 or hi < 0) else "")
            rows.append(r)
    if rows:
        bt = pd.DataFrame(rows)
        bt.to_csv(os.path.join(tdir, "m1_budget.csv"), index=False)
        md_table(bt, os.path.join(tdir, "m1_budget.md"))

        # figure: speed-quality frontier (one panel per measure, time on a log axis)
        shown = [c for c in ["aesthetic_quality", "imaging_quality", "dynamic_degree", "clip"] if c in bt.columns]
        fig, axes = plt.subplots(1, len(shown), figsize=(3.0 * len(shown) + 0.6, 3.3))
        axes = np.atleast_1d(axes)
        for ax, c in zip(axes, shown):
            for method, color, mk in (("M1", ORANGE, "s"), ("M2", BLUE, "o")):
                s = bt[bt.method == method].sort_values("gen_s")
                if not len(s):
                    continue
                ax.plot(s.gen_s, s[c], color=color, lw=2, marker=mk, ms=7, mec="white", mew=1.5,
                        label="M1 Wan2.1 (teacher)" if method == "M1" else "M2 Self Forcing")
                for _, r in s.iterrows():
                    ax.annotate(f"{int(r.steps)}", (r.gen_s, r[c]), textcoords="offset points", xytext=(5, 4),
                                fontsize=7.5, color=INK2)
            ax.set_xscale("log")
            ax.set_xlabel("generation time per 5 s clip (s, log scale)")
            ax.set_ylabel(SHORT[c])
        h, l = axes[0].get_legend_handles_labels()
        fig.legend(h, l, fontsize=8.5, loc="upper right", ncol=2, bbox_to_anchor=(1.0, 1.0))
        fig.suptitle("Speed against quality: teacher with fewer steps vs distilled student (30 prompts; labels = steps)",
                     x=0.01, ha="left", fontsize=10)
        fig.tight_layout(rect=(0, 0, 1, 0.93))
        fig.savefig(os.path.join(fdir, "fig_m1_vs_m2_budget.png"), bbox_inches="tight", facecolor="white")
        plt.close(fig)
    print("tables:", sorted(f for f in os.listdir(tdir) if f.startswith("m1_")))
    print("figures:", sorted(f for f in os.listdir(fdir) if f.startswith("fig_m1")))


if __name__ == "__main__":
    main()
