"""Generate RESULTS.md (the alignment document for teammates) from the tables written by aggregate_m2.py.

All numbers in tables come from the csv files, never typed by hand. Run after aggregate_m2.py:  python build_docs.py <results dir with tables/ and data/> [output.md]
"""
import json
import os
import sys

import numpy as np
import pandas as pd

OUT = sys.argv[1] if len(sys.argv) > 1 else "."
DOC = sys.argv[2] if len(sys.argv) > 2 else os.path.join(OUT, "RESULTS.md")
T = os.path.join(OUT, "tables")
D = os.path.join(OUT, "data")
VD = ["subject_consistency", "motion_smoothness", "dynamic_degree", "imaging_quality", "aesthetic_quality"]
SHORT = {"subject_consistency": "subject consistency", "motion_smoothness": "motion smoothness",
         "dynamic_degree": "dynamic degree", "imaging_quality": "imaging quality", "aesthetic_quality": "aesthetic quality",
         "clip": "CLIP x100"}


def md(df, fmt="{:.3f}"):
    cols = list(df.columns)
    out = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, r in df.iterrows():
        out.append("| " + " | ".join(fmt.format(v) if isinstance(v, (float, np.floating)) and not pd.isna(v)
                                     else ("" if (isinstance(v, float) and pd.isna(v)) else str(v)) for v in r.values) + " |")
    return "\n".join(out)


def ci(m, lo, hi, nd=3):
    return f"{m:+.{nd}f} [{lo:+.{nd}f}, {hi:+.{nd}f}]"


def sig(lo, hi):
    return "*" if (lo > 0 or hi < 0) else ""


gen = pd.read_csv(os.path.join(D, "per_video_generation.csv"))
clipdf = pd.read_csv(os.path.join(D, "per_video_clip_stats.csv"))
perf = pd.read_csv(os.path.join(T, "m2_performance.csv"))
vmain = pd.read_csv(os.path.join(T, "m2_vbench_main.csv")).set_index("dimension")
vabl = pd.read_csv(os.path.join(T, "m2_vbench_ablation.csv"))
qabl = pd.read_csv(os.path.join(T, "m2_ablation_quality.csv")).set_index("config")
vdrift = pd.read_csv(os.path.join(T, "m2_vbench_drift.csv"))
vdp = pd.read_csv(os.path.join(T, "m2_vbench_drift_paired.csv"))
cdrift = pd.read_csv(os.path.join(T, "m2_drift.csv"))
cdp = pd.read_csv(os.path.join(T, "m2_drift_paired.csv"))
seedv = pd.read_csv(os.path.join(T, "m2_seed_variance.csv"))
seede = pd.read_csv(os.path.join(T, "m2_seed_effects.csv"))
vper = pd.read_csv(os.path.join(T, "m2_vbench_per_video.csv"))
m1_main_p = os.path.join(T, "m1_vs_m2_main.csv")
HAS_M1 = os.path.exists(m1_main_p)
if HAS_M1:
    m1main = pd.read_csv(m1_main_p)
    m1pair = pd.read_csv(os.path.join(T, "m1_vs_m2_paired.csv"))
    m1src = pd.read_csv(os.path.join(T, "m1_vs_m2_by_source.csv"))
    m1bud = pd.read_csv(os.path.join(T, "m1_budget.csv"))
stream_path = os.path.join(T, "m2_streaming.csv")
stream = pd.read_csv(stream_path) if os.path.exists(stream_path) else None

main = gen[gen.config == "m2_main"]
mclip = clipdf[clipdf.config == "m2_main"]["clip"]
pm = perf.set_index("config").loc["m2_main"]

# ---- M2 row for the group table
row = {"method": "M2 Self Forcing", "resolution": "832x480", "frames": 81, "fps": 16, "steps": 4, "CFG": "none (distilled)",
       "offload": "no", "gen_s (mean, sd)": f"{main.gen_s.mean():.2f} ({main.gen_s.std():.2f})",
       "peak torch alloc (GiB)": f"{main.peak_gib.max():.1f}"}
for d in VD:
    row[SHORT[d]] = f"{vmain.loc[d, 'mean']:.4f}"
row["CLIP x100"] = f"{mclip.mean():.2f}"
group_row = md(pd.DataFrame([row]))

split = vper[vper.config == "m2_main"].copy()
split["source"] = ["hard (self-authored, 30)" if p.startswith("hard_") else "VBench subset (100)" for p in split.pid]
by_src = split.groupby("source")[VD].mean().reset_index().rename(columns=SHORT)
by_src_md = md(by_src, "{:.4f}")

# ---- efficiency
eff_names = [("m2_main", "default: 4 steps (n=130)"), ("m2_steps3", "3 steps"), ("m2_steps2", "2 steps"),
             ("m2_steps1", "1 step"), ("m2_win6", "window 6"), ("m2_win9", "window 9"), ("m2_win12", "window 12"),
             ("m2_win15", "window 15"), ("m2_chunk1", "chunk 1"), ("m2_chunk7", "chunk 7"),
             ("m2_long_w21", "20 s, window 21"), ("m2_long_w12", "20 s, window 12")]
pidx = perf.set_index("config")
eff = pd.DataFrame([dict(setting=lab, n=int(pidx.loc[c, "n"]), video_s=pidx.loc[c, "video_s"],
                         gen_s=pidx.loc[c, "gen_s"], realtime_factor=pidx.loc[c, "realtime_factor"],
                         peak_GiB=pidx.loc[c, "peak_gib"], vae_share_pct=pidx.loc[c, "vae_share_pct"])
                    for c, lab in eff_names if c in pidx.index])
eff_md = md(eff, "{:.2f}")

# ---- ablations: VBench + CLIP paired differences vs the default on the same 30 prompts
names = {"m2_steps1": "1 step", "m2_steps2": "2 steps", "m2_steps3": "3 steps", "m2_chunk1": "chunk 1", "m2_chunk7": "chunk 7",
         "m2_win6": "window 6", "m2_win9": "window 9", "m2_win12": "window 12", "m2_win15": "window 15",
         "m2_win9_sink3": "window 9 + sink 3"}
rows = []
for c, lab in names.items():
    r = vabl[vabl.config == c]
    if not len(r):
        continue
    r = r.iloc[0]
    out = {"change from default (4 steps, chunk 3, global attention)": lab}
    for d in ["subject_consistency", "dynamic_degree", "imaging_quality", "aesthetic_quality"]:
        out[SHORT[d]] = ci(r[d + "_diff"], r[d + "_lo"], r[d + "_hi"]) + sig(r[d + "_lo"], r[d + "_hi"])
    q = qabl.loc[c]
    out["CLIP x100"] = ci(q["d_clip"], q["d_clip_lo"], q["d_clip_hi"], 2) + sig(q["d_clip_lo"], q["d_clip_hi"])
    rows.append(out)
abl_md = md(pd.DataFrame(rows))
b = vabl[vabl.config == "m2_main"].iloc[0]
abl_base = ", ".join(f"{SHORT[d]} {b[d]:.4f}" for d in VD) + f", CLIP {qabl.loc['m2_main', 'clip']:.2f}"

# ---- drift
dn = {"m2_long_w21": "window 21", "m2_long_w21_sink3": "window 21 + sink 3", "m2_long_w12": "window 12",
      "m2_long_w12_sink3": "window 12 + sink 3", "m2_long_w21_ctx50": "window 21, re-noised t=50",
      "m2_long_w21_sink3_ctx25": "w21 + sink 3, re-noised t=25", "m2_long_w21_sink3_ctx50": "w21 + sink 3, re-noised t=50",
      "m2_long_w21_sink3_ctx100": "w21 + sink 3, re-noised t=100"}
rows = []
for c, lab in dn.items():
    v = vdrift[vdrift.config == c]
    k = cdrift[cdrift.config == c]
    if not len(v) or not len(k):
        continue
    v, k = v.iloc[0], k.iloc[0]
    rows.append({"configuration": lab, "n": int(v.n),
                 "dynamic degree first 4 s": f"{v.dynamic_degree_first4s:.2f}", "last 4 s": f"{v.dynamic_degree_last4s:.2f}",
                 "aesthetic change": f"{v.aesthetic_quality_change:+.3f}", "imaging change": f"{v.imaging_quality_change:+.3f}",
                 "subject consistency change": f"{v.subject_consistency_change:+.3f}",
                 "CLIP similarity to first 2 s, change": f"{k.clip_to_first_delta:+.3f}",
                 "brightness change": f"{k.brightness_delta:+.1f}"})
drift_md = md(pd.DataFrame(rows))


def paired_rows(base_cfg, others, metrics, src, key):
    rr = []
    for o in others:
        for m_ in metrics:
            r = src[(src.config == o) & (src.baseline == base_cfg) & (src.metric == m_)]
            if len(r):
                r = r.iloc[0]
                rr.append((dn.get(o, o), m_, ci(r.diff_of_change, r.ci_lo, r.ci_hi) + sig(r.ci_lo, r.ci_hi)))
    return rr


vm_ = ["dynamic_degree", "aesthetic_quality", "subject_consistency", "imaging_quality"]
tabs = []
for base_cfg, others in (("m2_long_w21", ["m2_long_w21_sink3", "m2_long_w12", "m2_long_w12_sink3", "m2_long_w21_ctx50"]),
                         ("m2_long_w21_sink3", ["m2_long_w21_sink3_ctx25", "m2_long_w21_sink3_ctx50", "m2_long_w21_sink3_ctx100"])):
    data = {}
    for o in others:
        data[dn[o]] = {}
        for m_ in vm_:
            r = vdp[(vdp.config == o) & (vdp.baseline == base_cfg) & (vdp.metric == m_)]
            if len(r):
                r = r.iloc[0]
                data[dn[o]][SHORT[m_]] = ci(r.diff_of_change, r.ci_lo, r.ci_hi) + sig(r.ci_lo, r.ci_hi)
        r = cdp[(cdp.config == o) & (cdp.baseline == base_cfg) & (cdp.metric == "clip_to_first")]
        if len(r):
            r = r.iloc[0]
            data[dn[o]]["CLIP similarity to first 2 s"] = ci(r.diff_of_change, r.ci_lo, r.ci_hi) + sig(r.ci_lo, r.ci_hi)
    df = pd.DataFrame(data).T.reset_index().rename(columns={"index": f"vs {dn.get(base_cfg, base_cfg)}"})
    tabs.append(md(df))
paired_w21_md, paired_sink_md = tabs

# ---- seeds
seed_md = md(seedv, "{:.3f}")
seede2 = seede.copy()
seed_eff_md = md(seede2, "{:.3f}")

# ---- streaming
if stream is not None:
    stream_md = md(stream, "{:.2f}")
else:
    stream_md = "_not available_"

m1_section = m1_summary = ""
if HAS_M1:
    r1 = m1main[m1main.method.str.startswith("M1")].iloc[0]
    rowm1 = {"method": "M1 Wan2.1-1.3B", "resolution": "832x480", "frames": 81, "fps": 16, "steps": 50, "CFG": "6.0 (Wan negative prompt)",
             "offload": "no", "gen_s (mean, sd)": f"{r1.gen_s:.2f} ({r1.gen_s_sd:.2f})", "peak torch alloc (GiB)": f"{r1.peak_gib:.1f}"}
    for d in VD:
        rowm1[SHORT[d]] = f"{r1[d]:.4f}"
    rowm1["CLIP x100"] = f"{r1['clip']:.2f}"
    rows_ = [rowm1, row]
    both_md = md(pd.DataFrame(rows_))
    pr = m1pair.copy()
    pr["M2 minus M1 [95% interval]"] = [ci(a_, b_, c_, 4) + sig(b_, c_) for a_, b_, c_ in zip(pr.diff_m2_minus_m1, pr.ci_lo, pr.ci_hi)]
    pr = pr[["metric", "m1_mean", "m2_mean", "M2 minus M1 [95% interval]", "n"]]
    pr["metric"] = [SHORT.get(m, m) if m != "gen_s" else "generation seconds" for m in pr.metric]
    pair_md = md(pr, "{:.4f}")
    src_md = md(m1src.rename(columns=SHORT), "{:.4f}")
    bcols = ["method", "steps", "n", "gen_s"] + [c for c in ["aesthetic_quality", "imaging_quality", "dynamic_degree",
                                                            "subject_consistency", "clip"] if c in m1bud.columns]
    bud_md = md(m1bud[bcols].rename(columns=SHORT), "{:.3f}")
    vcols = ["method", "steps"] + [c for c in m1bud.columns if c.endswith("_vs_m2_4steps")]
    bud_vs_md = md(m1bud[vcols].rename(columns={c: SHORT.get(c.replace("_vs_m2_4steps", ""), c) for c in vcols}))
    m1_summary = (f"\n- M1 (Wan2.1-1.3B, the teacher, measured the same way) takes {r1.gen_s:.1f} s per clip, so M2 is "
                  f"{r1.gen_s / main.gen_s.mean():.1f}x faster with the same CLIP alignment, higher VBench aesthetic and imaging quality and "
                  f"lower dynamic degree; the teacher with fewer steps does not reach M2's quality at M2's cost (section 4.7).")
    m1_section = f"""### 4.7 M1 (Wan2.1-1.3B, the teacher) and M1 against M2

M1 was generated with the same prompts, seed, GPU type, timing scope, memory scope and encoder as M2 (Wan2.1 code revision
9737cba9c1c3c4d04b33fcad41c111989865d315, the revision in P1's example config; 50 UniPC steps, CFG 6.0, shift 8.0, Wan's default
negative prompt, bf16, no offload), and scored with the same VBench and CLIP setup.

The two rows for the group table:

{both_md}

Paired over the same 130 prompts (`*` = interval excludes 0):

{pair_md}

By prompt source:

{src_md}

Equal budget: the teacher run with fewer UniPC steps against the distilled student with fewer steps, on the 30-prompt subset
(figure `fig_m1_vs_m2_budget.png`):

{bud_md}

Difference from M2 with 4 steps (its default) on the same prompts:

{bud_vs_md}

How to read this (interpretation, written by hand from the tables above):

- **Speed.** M1 takes {r1.gen_s:.1f} s per 5 s clip (sd {r1.gen_s_sd:.2f} s over 130 videos), M2 {main.gen_s.mean():.2f} s: M2 is
  **{r1.gen_s / main.gen_s.mean():.1f}x faster** at about the same peak memory ({r1.peak_gib:.1f} against {main.peak_gib.max():.1f} GiB).
  M1 time is linear in steps: about 4.5 s per UniPC step (two network passes for CFG) plus about 9 s of fixed cost
  (text encoding and VAE decoding); the 4/8/16/50-step times fit this within 0.1 s.
- **Quality at full budget (130 prompts, paired).** Text alignment is the same (CLIP difference not significant). M2 scores
  higher on aesthetic quality (+0.044) and imaging quality (+0.035) and slightly higher on subject consistency and motion
  smoothness, but **M1 is clearly more dynamic** (dynamic degree 0.84 against 0.72, -0.12 for M2). Higher consistency and
  smoothness go together with less motion, so those two gains should not be read as better temporal quality. Higher VBench
  aesthetic and imaging scores for a DMD-distilled student than for its teacher are plausible (distribution matching tends to
  favour sharp, high-contrast, mode-like samples), but VBench is not a human judgement and we ran no user study.
- **Is distillation needed, or would fewer teacher steps do? It is needed.** M1 with 4 steps (26.8 s, still 2.2x slower than
  M2 because of CFG and full-clip attention) scores far lower (and was a blur in the example we inspected): aesthetic -0.20, imaging -0.28, CLIP -3.1 against M2 with
  4 steps. With 8 steps (44.7 s, 3.7x M2's time) M1 is still clearly worse on aesthetic and imaging quality; with 16 steps
  (80.5 s, 6.6x) it gains CLIP (+1.4) but still loses aesthetic (-0.07) and imaging (-0.05) quality. No M1 setting we ran
  matches M2 at M2's cost; even M2 at 2 steps (9.4 s) beats M1 at 16 steps on aesthetic and imaging quality (though not on
  dynamic degree or CLIP).
- **Caveats.** One seed. The 30-prompt subset slightly favours M1 on CLIP (+0.95, significant there) while all 130 prompts show
  no difference, so subset-level CLIP differences of about one point are within prompt-selection noise (see 4.5).
"""

doc = f"""# M2 (Self Forcing) and M1 (Wan2.1-1.3B): results and alignment notes

This document is for the group: it says exactly how M2 was run and measured so that M1, M3 and M4 can be reported the
same way, and it gives all M2 and M1 results. Every table and number below is generated from the csv files in `results/` by
`m2_self_forcing/code/build_docs.py`; only the interpretation bullets are written by hand. Videos are not in this repository (see section 7).

## 1. Summary

- M2 is Self Forcing: Wan2.1-1.3B turned into a chunk-wise autoregressive model with a KV cache and distilled to 4 steps
  without CFG. On one L40S it makes a 5 s clip (832x480, 81 frames) in **{main.gen_s.mean():.2f} s**, peak torch memory
  **{main.peak_gib.max():.1f} GiB**, which is **{pm.realtime_factor:.2f}x** real time (slower than playback on this GPU in the
  offline setting, see section 4.6).
- VBench (pinned custom_input evaluator, 130 main videos): subject consistency {vmain.loc['subject_consistency','mean']:.4f},
  motion smoothness {vmain.loc['motion_smoothness','mean']:.4f}, dynamic degree {vmain.loc['dynamic_degree','mean']:.4f},
  imaging quality {vmain.loc['imaging_quality','mean']:.4f}, aesthetic quality {vmain.loc['aesthetic_quality','mean']:.4f}.
- Fewer steps are much faster but VBench shows the cost that CLIP does not: 1 step loses aesthetic quality and dynamics clearly
  (section 4.2). A sink of 3 frames is the useful fix for drift in 20 s videos (section 4.3). Our re-noised-context idea did
  **not** help (section 4.4).{m1_summary}

## 2. Protocol to align on

| item | what M2 did | what the other methods should do |
|---|---|---|
| prompts | the group file `prompts_expanded_v1.jsonl`, field `prompt` (expanded); 130 prompts | same file, same field, no extra expansion |
| seed | 20261027, set again before every video | same seed; record the seed per video |
| resolution / frames / fps | native: 832x480, 81 frames, 16 fps (5.06 s) | keep each model's native values and report them; do not change fps to fake equal duration |
| steps / CFG / negative prompt | 4 steps (1000, 750, 500, 250); no CFG and no negative prompt (guidance is distilled into the weights) | report the real values; M1 and M3 use CFG and Wan's negative prompt, this difference is a property of the methods |
| precision / offload | bf16; no offload | report; M3 needs offload on 48 GB (it ran out of memory in VAE decoding without it), so its time includes offload |
| GPU | one NVIDIA L40S (node TC2N03), no other load | L40S only; record `gpu_name` |
| timing scope | `generation_seconds` = one call of the pipeline: text encoding + cache initialisation + denoising + VAE decoding. Excludes model loading and mp4 export | same scope; P1's wrapper uses the same definition |
| warm-up | one untimed video at the start of the process, then `torch.cuda.synchronize()` before and after each timed video | same |
| memory | peak PyTorch allocation per video (`reset_peak_memory_stats` before each video), not nvidia-smi | same |
| video encoding | imageio, libx264, quality=8, macro_block_size=1 (identical to Wan's own `save_video`), about 6.5 Mbps | **same encoder settings.** A first M2 run used torchvision's default (about 1 Mbps) and was discarded: bitrate alone changed the sharpness statistic (445 vs 538 on the same prompts) |
| VBench | pinned source revision fd18b3d055cb0fc6f066ca90fe2c3c8cbb698490, `custom_input`, five dimensions below; weights byte-identical to P1's recorded hashes (9 of 9 files) | same command; imaging quality is reported on 0-1 (VBench's per-video value divided by 100), dynamic degree is the share of videos judged dynamic |
| CLIP | OpenAI ViT-B/32 (revision 3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268), 16 uniform frames, original caption (`original_prompt`), 100 x max(0, mean cosine) | same (this is the group definition from P1) |

VBench was run on whole folders (each model loaded once). Checked on three videos: scoring the folder and scoring each video alone
gave identical per-video values for all five dimensions (maximum absolute difference 0.0), so it is equivalent to P1's
one-video-at-a-time call.

## 3. The M2 row for the group table

{group_row}

By prompt source (same videos):

{by_src_md}

The self-authored hard prompts score lower on aesthetic quality and subject consistency; keep that in mind when comparing
averages over different prompt mixes.

## 4. Results

### 4.1 Speed and memory (5 s clips unless stated; L40S)

{eff_md}

`realtime_factor` = video seconds / generation seconds; above 1 means faster than playback. `vae_share_pct` = share of
(denoising + VAE decoding) time spent in VAE decoding. n=130 for the first row and n=30 for all others.
For comparison, M1 (Wan2.1-1.3B, 50 steps with CFG, same scope and GPU type) is in section 4.7.

### 4.2 Steps, chunk size, window and sink on 5 s clips (30 prompts, paired)

Default on these 30 prompts: {abl_base}. Cells are the mean difference from the default with a 95% bootstrap interval over
prompts; `*` marks intervals that exclude 0.

{abl_md}

- 1 step is clearly worse on every VBench dimension shown, while CLIP text alignment does not change: CLIP cannot see this loss.
- 2 and 3 steps lose little (aesthetic quality about -0.02 and -0.01); 2 steps saves 23% of the time (9.37 s against 12.19 s).
- Chunk 1 (not the trained 3) is clearly worse and 13% slower. Chunk 7 is close to the default.
- A window shorter than the clip, with or without sink, changes almost nothing on 5 s clips (window 6 loses a little subject consistency).

### 4.3 Drift in 20 s rollouts (30 prompts)

The model was trained on 5 s clips; longer videos use the rolling cache. Each 20 s video is cut into its first 4 s and last 4 s and
both clips are scored by VBench; "change" is last minus first. Dynamic degree is the share of clips judged dynamic.
**Subject consistency must be read together with dynamic degree**: a video that slowly freezes becomes more "consistent" frame to
frame, so a higher value late in the video is not an improvement.

{drift_md}

Paired differences of the change, against window 21 (`*` = interval excludes 0):

{paired_w21_md}

- Without a sink the drift is mostly a loss of motion (dynamic degree falls from about 0.8 to 0.5 for window 21) and of aesthetic quality.
- A sink of 3 latent frames halves the fall in dynamic degree for window 21 (-0.33 to -0.17), but for that setting the interval of the
  difference only just touches 0 ([0.000, +0.333]). Window 12 + sink 3 is the best setting, and clearly better than window 21 (dynamic degree
  falls by only 0.10, aesthetic quality by 0.009); it is also 13% faster and 2.4 GiB smaller than window 21.
- A window of 12 without a sink is worse than window 21 on CLIP text alignment and on similarity to the first 2 s.

### 4.4 Our modification: re-noised KV context (negative result)

Upstream `context_noise` only relabels the timestep of the cache-refresh pass. We noise the clean chunk to level t before the
refresh pass (own implementation, `GenProxy` in `run_m2.py`). Against window 21 + sink 3 with a clean context:

{paired_sink_md}

Against the clean-context sink setting, subject consistency and CLIP similarity to the first 2 s are significantly lower at all
three levels; aesthetic quality is significantly lower at t=25 and imaging quality at t=50. Dynamic degree is not significantly
different. We do not claim an improvement: the model was never trained with noisy context.

### 4.5 Seeds

Ten prompts, three seeds (20261027, 20261028, 20261029); `within_prompt_sd` is the seed-to-seed standard deviation inside a prompt,
`between_prompt_sd` the standard deviation of prompt means.

{seed_md}

Same prompt, different seed: CLIP changes by about 1.3 points, comparable to the effect of the largest settings change we measured on
CLIP; single-video differences below that are noise. The effect of a sink on 20 s videos (difference of the change over the rollout,
per seed):

{seed_eff_md}

The sign is the same for all three seeds.

### 4.6 Streaming (real-time) benchmark

The authors' demo decodes each 3-frame chunk as soon as it is ready (fp16, block-wise VAE decoder, optional light TAEHV decoder,
optional `torch.compile`). We replicated that loop on the L40S. A = our standard offline setting. `first_frame_s` = text encoding +
cache initialisation + first chunk (denoising, cache refresh, decoding). A 3-latent-frame chunk is 0.75 s of video, so
real-time needs `steady_chunk_s` below 0.75. `psnr_vs_full_vae_db` compares the streamed pixels with the standard whole-clip VAE decode of
the same latents.

{stream_md}

A-C were measured in one job and D-E in a second job on the same GPU type, four timed videos each after one untimed warm-up; the
`torch.compile` cost (about 255 s, paid in D's warm-up) is a one-time cost that E reuses.

- Streaming in the demo's default configuration (B) is not faster than our offline setting (A). Block-wise decoding with the
  standard VAE costs about 0.77 s per 12-frame chunk in steady state, more than the denoising of that chunk (about 0.91 s; the cache refresh adds 0.23 s).
- The light VAE (C) removes almost all of the decoding time (about 0.05 s per chunk): 7.36 s for the clip (0.69x real time) and the first frame
  after 0.80 s. The price is fidelity: PSNR against the standard decode is 28.4 dB on average and 23.6 to 34.7 dB between videos (B: 37.0 dB).
- `torch.compile` helps by 2.3 s with the standard decoder (12.71 s to 10.45 s) and by 0.7 s with the light decoder (7.36 s to 6.69 s).
- The best combination (E) reaches **0.76x real time**, first frame after 0.69 s, and 0.98 s per chunk in steady state against the
  0.75 s of video that each chunk contains. **Real time was not reached on this GPU.** With decoding almost free, denoising plus cache
  refresh (about 0.82 s plus 0.21 s once the cache is full) is the remaining limit. To play the 5 s clip without stalls a viewer would need about 2.2 s
  of start-up buffering (the last chunk is ready at 6.69 s and is played from 4.5 s after start).
- Not tested: FP8 quantisation, a TensorRT decoder, other GPUs. The authors' real-time claim refers to different hardware and
  options, so this result neither confirms nor refutes it.

{m1_section}

## 5. How to read these numbers

- All comparisons inside M2 are paired (same prompts, same seed). With 30 prompts, differences below about 0.5 CLIP points
  or 0.02 on a VBench dimension are not resolved; the interval in each cell says what is resolved.
- One seed per prompt except in section 4.5.
- VBench here is `custom_input` on our own prompts, per video; it is not the official leaderboard protocol.
- Dynamic degree, motion smoothness and the CLIP-based statistics measure different things: a frozen video scores high on smoothness and
  subject consistency. Judge drift with dynamic degree and aesthetic quality.
- The 20 s videos were scored as two 4 s clips (frames 0-63 and the last 64), re-encoded at near-lossless quality (imageio quality=10);
  their absolute imaging/aesthetic values are therefore not comparable with the 5 s videos.

## 6. Settings table (what each configuration name means)

`m2_main`: default. `m2_steps{{1,2,3}}`: denoising steps (timesteps 1000; 1000,500; 1000,750,500). `m2_chunk{{1,7}}`: frames per chunk.
`m2_win{{6,9,12,15}}`: attention window in latent frames (21 = whole 5 s clip = global attention). `m2_win9_sink3`: window 9 plus the first
3 latent frames always kept. `m2_long_*`: 20 s rollouts (81 latent frames, 321 video frames): `w21`/`w12` = window, `sink3` = sink of
3 latent frames, `ctxNN` = re-noised context at level NN (0-1000 scale). `m2_seed2_*`, `m2_seed3_*`: same settings, seeds 20261028 and
20261029, 10 prompts.

## 7. Files

- `results/tables/*.md|csv`: every table (`m2_vbench_*`, `m2_ablation_quality`, `m2_drift*`, `m2_performance`, `m2_seed_*`, `m2_streaming`).
- `results/figures/*.png`: all figures.
- `results/data/per_video_generation.csv`, `per_video_clip_stats.csv`, `per_segment_stats.csv` and `results/tables/m2_vbench_per_video.csv`: one row per video or segment.
- `results/data/m2_main_manifest.csv`: the 130 main videos with seed, settings, timing, memory and SHA256, so that a copy of the video
  files can be verified. The videos themselves (about 0.6 GB for the main set) are not in git.
- `results/raw/outputs`, `results/raw/analysis`: the per-video generation records and per-video scores (json) from which everything is computed;
  `results/raw/vbench/*.json`: raw VBench output; `results/bench/bench_stream.json`: raw streaming benchmark. To rebuild all tables and figures:
  `python m2_self_forcing/code/aggregate_m2.py --results results/raw --out results` and
  `python m2_self_forcing/code/build_docs.py results RESULTS.md`.
- `m2_self_forcing/code/`: generation, scoring, aggregation and benchmark scripts with a README.
- `report/M2_report_section.md`: the draft of the M2 part of the report.
"""
open(DOC, "w", encoding="utf-8").write(doc)
print("RESULTS.md written", len(doc.split()), "words")
