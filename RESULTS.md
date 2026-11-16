# M2 (Self Forcing) and M1 (Wan2.1-1.3B): results and alignment notes

This document is for the group: it says exactly how M2 was run and measured so that M1, M3 and M4 can be reported the
same way, and it gives all M2 and M1 results. Every table and number below is generated from the csv files in `results/` by
`m2_self_forcing/code/build_docs.py`; only the interpretation bullets are written by hand. Videos are not in this repository (see section 7).

## 1. Summary

- M2 is Self Forcing: Wan2.1-1.3B turned into a chunk-wise autoregressive model with a KV cache and distilled to 4 steps
  without CFG. On one L40S it makes a 5 s clip (832x480, 81 frames) in **12.19 s**, peak torch memory
  **23.3 GiB**, which is **0.42x** real time (slower than playback on this GPU in the
  offline setting, see section 4.6).
- VBench (pinned custom_input evaluator, 130 main videos): subject consistency 0.9591,
  motion smoothness 0.9850, dynamic degree 0.7154,
  imaging quality 0.6931, aesthetic quality 0.6069.
- Fewer steps are much faster but VBench shows the cost that CLIP does not: 1 step loses aesthetic quality and dynamics clearly
  (section 4.2). A sink of 3 frames is the useful fix for drift in 20 s videos (section 4.3). Our re-noised-context idea did
  **not** help (section 4.4).
- M1 (Wan2.1-1.3B, the teacher, measured the same way) takes 232.9 s per clip, so M2 is 19.1x faster with the same CLIP alignment, higher VBench aesthetic and imaging quality and lower dynamic degree; the teacher with fewer steps does not reach M2's quality at M2's cost (section 4.7).

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

| method | resolution | frames | fps | steps | CFG | offload | gen_s (mean, sd) | peak torch alloc (GiB) | subject consistency | motion smoothness | dynamic degree | imaging quality | aesthetic quality | CLIP x100 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| M2 Self Forcing | 832x480 | 81 | 16 | 4 | none (distilled) | no | 12.19 (0.03) | 23.3 | 0.9591 | 0.9850 | 0.7154 | 0.6931 | 0.6069 | 31.27 |

By prompt source (same videos):

| source | subject consistency | motion smoothness | dynamic degree | imaging quality | aesthetic quality |
|---|---|---|---|---|---|
| VBench subset (100) | 0.9628 | 0.9849 | 0.7200 | 0.6887 | 0.6195 |
| hard (self-authored, 30) | 0.9469 | 0.9855 | 0.7000 | 0.7075 | 0.5652 |

The self-authored hard prompts score lower on aesthetic quality and subject consistency; keep that in mind when comparing
averages over different prompt mixes.

## 4. Results

### 4.1 Speed and memory (5 s clips unless stated; L40S)

| setting | n | video_s | gen_s | realtime_factor | peak_GiB | vae_share_pct |
|---|---|---|---|---|---|---|
| default: 4 steps (n=130) | 130 | 5.06 | 12.19 | 0.42 | 23.32 | 41.67 |
| 3 steps | 30 | 5.06 | 10.78 | 0.47 | 23.32 | 47.18 |
| 2 steps | 30 | 5.06 | 9.37 | 0.54 | 23.32 | 54.34 |
| 1 step | 30 | 5.06 | 7.95 | 0.64 | 23.32 | 64.11 |
| window 6 | 30 | 5.06 | 10.71 | 0.47 | 19.30 | 47.52 |
| window 9 | 30 | 5.06 | 11.24 | 0.45 | 20.10 | 45.28 |
| window 12 | 30 | 5.06 | 11.62 | 0.44 | 20.91 | 43.78 |
| window 15 | 30 | 5.06 | 11.95 | 0.42 | 21.71 | 42.53 |
| chunk 1 | 30 | 5.06 | 13.74 | 0.37 | 23.32 | 36.98 |
| chunk 7 | 30 | 5.06 | 12.83 | 0.39 | 23.32 | 39.59 |
| 20 s, window 21 | 30 | 20.06 | 54.29 | 0.37 | 23.89 | 37.09 |
| 20 s, window 12 | 30 | 20.06 | 47.15 | 0.43 | 21.48 | 42.76 |

`realtime_factor` = video seconds / generation seconds; above 1 means faster than playback. `vae_share_pct` = share of
(denoising + VAE decoding) time spent in VAE decoding. n=130 for the first row and n=30 for all others.
For comparison, M1 (Wan2.1-1.3B, 50 steps with CFG, same scope and GPU type) is in section 4.7.

### 4.2 Steps, chunk size, window and sink on 5 s clips (30 prompts, paired)

Default on these 30 prompts: subject consistency 0.9413, motion smoothness 0.9816, dynamic degree 0.7667, imaging quality 0.6757, aesthetic quality 0.5920, CLIP 30.77. Cells are the mean difference from the default with a 95% bootstrap interval over
prompts; `*` marks intervals that exclude 0.

| change from default (4 steps, chunk 3, global attention) | subject consistency | dynamic degree | imaging quality | aesthetic quality | CLIP x100 |
|---|---|---|---|---|---|
| 1 step | -0.059 [-0.079, -0.041]* | -0.300 [-0.500, -0.100]* | -0.037 [-0.068, -0.004]* | -0.121 [-0.146, -0.097]* | -0.04 [-0.81, +0.74] |
| 2 steps | -0.007 [-0.014, +0.000] | -0.067 [-0.233, +0.100] | -0.010 [-0.028, +0.008] | -0.022 [-0.041, -0.003]* | +0.38 [-0.06, +0.87] |
| 3 steps | -0.005 [-0.009, -0.002]* | -0.067 [-0.200, +0.067] | -0.014 [-0.026, -0.003]* | -0.012 [-0.020, -0.004]* | +0.16 [-0.10, +0.44] |
| chunk 1 | -0.057 [-0.076, -0.035]* | -0.367 [-0.600, -0.133]* | -0.028 [-0.052, -0.002]* | -0.022 [-0.051, +0.007] | -1.27 [-2.20, -0.41]* |
| chunk 7 | -0.012 [-0.025, +0.002] | +0.133 [+0.000, +0.300] | -0.000 [-0.021, +0.023] | +0.008 [-0.010, +0.027] | +0.46 [-0.08, +1.01] |
| window 6 | -0.012 [-0.019, -0.004]* | -0.033 [-0.133, +0.067] | -0.001 [-0.009, +0.006] | -0.003 [-0.012, +0.006] | -0.17 [-0.42, +0.06] |
| window 9 | -0.002 [-0.005, +0.001] | -0.067 [-0.167, +0.000] | +0.004 [-0.002, +0.009] | +0.001 [-0.004, +0.007] | +0.02 [-0.16, +0.19] |
| window 12 | -0.001 [-0.003, +0.002] | -0.033 [-0.100, +0.000] | -0.000 [-0.004, +0.004] | -0.001 [-0.004, +0.003] | -0.01 [-0.15, +0.12] |
| window 15 | -0.002 [-0.003, -0.001]* | -0.033 [-0.100, +0.000] | +0.001 [-0.002, +0.003] | +0.000 [-0.003, +0.003] | +0.01 [-0.08, +0.09] |
| window 9 + sink 3 | +0.001 [-0.001, +0.003] | +0.000 [+0.000, +0.000] | +0.003 [+0.000, +0.006]* | +0.001 [-0.003, +0.004] | +0.04 [-0.07, +0.17] |

- 1 step is clearly worse on every VBench dimension shown, while CLIP text alignment does not change: CLIP cannot see this loss.
- 2 and 3 steps lose little (aesthetic quality about -0.02 and -0.01); 2 steps saves 23% of the time (9.37 s against 12.19 s).
- Chunk 1 (not the trained 3) is clearly worse and 13% slower. Chunk 7 is close to the default.
- A window shorter than the clip, with or without sink, changes almost nothing on 5 s clips (window 6 loses a little subject consistency).

### 4.3 Drift in 20 s rollouts (30 prompts)

The model was trained on 5 s clips; longer videos use the rolling cache. Each 20 s video is cut into its first 4 s and last 4 s and
both clips are scored by VBench; "change" is last minus first. Dynamic degree is the share of clips judged dynamic.
**Subject consistency must be read together with dynamic degree**: a video that slowly freezes becomes more "consistent" frame to
frame, so a higher value late in the video is not an improvement.

| configuration | n | dynamic degree first 4 s | last 4 s | aesthetic change | imaging change | subject consistency change | CLIP similarity to first 2 s, change | brightness change |
|---|---|---|---|---|---|---|---|---|
| window 21 | 30 | 0.80 | 0.47 | -0.038 | +0.028 | +0.018 | -0.110 | -14.2 |
| window 21 + sink 3 | 30 | 0.80 | 0.63 | -0.022 | +0.026 | -0.000 | -0.066 | -8.0 |
| window 12 | 30 | 0.80 | 0.37 | -0.032 | +0.018 | +0.021 | -0.147 | -11.4 |
| window 12 + sink 3 | 30 | 0.80 | 0.70 | -0.009 | +0.027 | -0.000 | -0.068 | -7.6 |
| window 21, re-noised t=50 | 30 | 0.90 | 0.70 | -0.065 | -0.030 | -0.016 | -0.161 | -10.6 |
| w21 + sink 3, re-noised t=25 | 30 | 0.83 | 0.77 | -0.044 | +0.007 | -0.026 | -0.125 | -3.3 |
| w21 + sink 3, re-noised t=50 | 30 | 0.90 | 0.77 | -0.039 | -0.001 | -0.038 | -0.107 | -1.4 |
| w21 + sink 3, re-noised t=100 | 30 | 0.90 | 0.83 | -0.030 | +0.014 | -0.025 | -0.094 | -6.0 |

Paired differences of the change, against window 21 (`*` = interval excludes 0):

| vs window 21 | dynamic degree | aesthetic quality | subject consistency | imaging quality | CLIP similarity to first 2 s |
|---|---|---|---|---|---|
| window 21 + sink 3 | +0.167 [+0.000, +0.333] | +0.016 [-0.000, +0.033] | -0.019 [-0.030, -0.010]* | -0.001 [-0.021, +0.017] | +0.044 [+0.032, +0.057]* |
| window 12 | -0.100 [-0.267, +0.067] | +0.005 [-0.017, +0.029] | +0.003 [-0.005, +0.010] | -0.010 [-0.034, +0.012] | -0.037 [-0.064, -0.013]* |
| window 12 + sink 3 | +0.233 [+0.033, +0.433]* | +0.029 [+0.012, +0.045]* | -0.018 [-0.034, -0.005]* | -0.001 [-0.016, +0.015] | +0.042 [+0.019, +0.064]* |
| window 21, re-noised t=50 | +0.133 [-0.067, +0.367] | -0.027 [-0.054, -0.003]* | -0.034 [-0.053, -0.018]* | -0.058 [-0.105, -0.012]* | -0.051 [-0.074, -0.028]* |

- Without a sink the drift is mostly a loss of motion (dynamic degree falls from about 0.8 to 0.5 for window 21) and of aesthetic quality.
- A sink of 3 latent frames halves the fall in dynamic degree for window 21 (-0.33 to -0.17), but for that setting the interval of the
  difference only just touches 0 ([0.000, +0.333]). Window 12 + sink 3 is the best setting, and clearly better than window 21 (dynamic degree
  falls by only 0.10, aesthetic quality by 0.009); it is also 13% faster and 2.4 GiB smaller than window 21.
- A window of 12 without a sink is worse than window 21 on CLIP text alignment and on similarity to the first 2 s.

### 4.4 Our modification: re-noised KV context (negative result)

Upstream `context_noise` only relabels the timestep of the cache-refresh pass. We noise the clean chunk to level t before the
refresh pass (own implementation, `GenProxy` in `run_m2.py`). Against window 21 + sink 3 with a clean context:

| vs window 21 + sink 3 | dynamic degree | aesthetic quality | subject consistency | imaging quality | CLIP similarity to first 2 s |
|---|---|---|---|---|---|
| w21 + sink 3, re-noised t=25 | +0.100 [-0.133, +0.333] | -0.022 [-0.042, -0.004]* | -0.026 [-0.042, -0.012]* | -0.019 [-0.053, +0.011] | -0.059 [-0.075, -0.043]* |
| w21 + sink 3, re-noised t=50 | +0.033 [-0.200, +0.267] | -0.016 [-0.041, +0.006] | -0.037 [-0.063, -0.014]* | -0.027 [-0.052, -0.004]* | -0.041 [-0.059, -0.024]* |
| w21 + sink 3, re-noised t=100 | +0.100 [-0.133, +0.333] | -0.008 [-0.025, +0.008] | -0.024 [-0.041, -0.009]* | -0.012 [-0.039, +0.012] | -0.027 [-0.040, -0.015]* |

Against the clean-context sink setting, subject consistency and CLIP similarity to the first 2 s are significantly lower at all
three levels; aesthetic quality is significantly lower at t=25 and imaging quality at t=50. Dynamic degree is not significantly
different. We do not claim an improvement: the model was never trained with noisy context.

### 4.5 Seeds

Ten prompts, three seeds (20261027, 20261028, 20261029); `within_prompt_sd` is the seed-to-seed standard deviation inside a prompt,
`between_prompt_sd` the standard deviation of prompt means.

| metric | mean_seed1 | mean_seed2 | mean_seed3 | within_prompt_sd | between_prompt_sd |
|---|---|---|---|---|---|
| clip | 30.391 | 30.271 | 30.111 | 1.280 | 3.845 |
| motion | 8.611 | 9.177 | 8.373 | 2.653 | 6.200 |
| sharpness | 595.206 | 716.883 | 484.179 | 235.408 | 369.956 |
| gen_s | 12.181 | 12.168 | 12.189 | 0.030 | 0.026 |

Same prompt, different seed: CLIP changes by about 1.3 points, comparable to the effect of the largest settings change we measured on
CLIP; single-video differences below that are noise. The effect of a sink on 20 s videos (difference of the change over the rollout,
per seed):

| effect | metric | seed1 | seed2 | seed3 |
|---|---|---|---|---|
| window 12 + sink 3 minus window 21 | brightness | 8.494 | 8.790 | 9.332 |
| window 12 + sink 3 minus window 21 | clip_to_first | 0.031 | 0.037 | 0.048 |
| window 12 + sink 3 minus window 21 | motion | 3.089 | 5.240 | 7.100 |
| window 21 + sink 3 minus window 21 | brightness | 7.755 | 5.973 | 8.214 |
| window 21 + sink 3 minus window 21 | clip_to_first | 0.032 | 0.016 | 0.037 |
| window 21 + sink 3 minus window 21 | motion | 3.587 | 5.552 | 7.264 |

The sign is the same for all three seeds.

### 4.6 Streaming (real-time) benchmark

The authors' demo decodes each 3-frame chunk as soon as it is ready (fp16, block-wise VAE decoder, optional light TAEHV decoder,
optional `torch.compile`). We replicated that loop on the L40S. A = our standard offline setting. `first_frame_s` = text encoding +
cache initialisation + first chunk (denoising, cache refresh, decoding). A 3-latent-frame chunk is 0.75 s of video, so
real-time needs `steady_chunk_s` below 0.75. `psnr_vs_full_vae_db` compares the streamed pixels with the standard whole-clip VAE decode of
the same latents.

| config | status | n | total_s | realtime_factor | peak_gib | warmup_s | first_frame_s | steady_chunk_s | psnr_vs_full_vae_db |
|---|---|---|---|---|---|---|---|---|---|
| A  bf16, whole-clip decode | ok | 4 | 12.06 | 0.42 | 23.48 | 11.96 |  |  |  |
| B  fp16, streaming decode | ok | 4 | 12.71 | 0.40 | 25.13 | 14.26 | 1.84 | 1.80 | 36.97 |
| C  fp16, streaming, light VAE | ok | 4 | 7.36 | 0.69 | 25.00 | 7.31 | 0.80 | 1.08 | 28.40 |
| D  B + torch.compile | ok | 4 | 10.45 | 0.48 | 27.76 | 255.30 | 1.53 | 1.48 | 36.68 |
| E  C + torch.compile | ok | 4 | 6.69 | 0.76 | 25.00 | 6.64 | 0.69 | 0.98 | 28.25 |

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

### 4.7 M1 (Wan2.1-1.3B, the teacher) and M1 against M2

M1 was generated with the same prompts, seed, GPU type, timing scope, memory scope and encoder as M2 (Wan2.1 code revision
9737cba9c1c3c4d04b33fcad41c111989865d315, the revision in P1's example config; 50 UniPC steps, CFG 6.0, shift 8.0, Wan's default
negative prompt, bf16, no offload), and scored with the same VBench and CLIP setup.

The two rows for the group table:

| method | resolution | frames | fps | steps | CFG | offload | gen_s (mean, sd) | peak torch alloc (GiB) | subject consistency | motion smoothness | dynamic degree | imaging quality | aesthetic quality | CLIP x100 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| M1 Wan2.1-1.3B | 832x480 | 81 | 16 | 50 | 6.0 (Wan negative prompt) | no | 232.94 (0.31) | 24.4 | 0.9470 | 0.9779 | 0.8385 | 0.6578 | 0.5627 | 31.49 |
| M2 Self Forcing | 832x480 | 81 | 16 | 4 | none (distilled) | no | 12.19 (0.03) | 23.3 | 0.9591 | 0.9850 | 0.7154 | 0.6931 | 0.6069 | 31.27 |

Paired over the same 130 prompts (`*` = interval excludes 0):

| metric | m1_mean | m2_mean | M2 minus M1 [95% interval] | n |
|---|---|---|---|---|
| CLIP x100 | 31.4935 | 31.2660 | -0.2275 [-0.7286, +0.2820] | 130 |
| subject consistency | 0.9470 | 0.9591 | +0.0121 [+0.0045, +0.0195]* | 130 |
| motion smoothness | 0.9779 | 0.9850 | +0.0072 [+0.0052, +0.0091]* | 130 |
| dynamic degree | 0.8385 | 0.7154 | -0.1231 [-0.2000, -0.0462]* | 130 |
| imaging quality | 0.6578 | 0.6931 | +0.0353 [+0.0211, +0.0499]* | 130 |
| aesthetic quality | 0.5627 | 0.6069 | +0.0443 [+0.0303, +0.0585]* | 130 |
| generation seconds | 232.9374 | 12.1940 | -220.7433 [-220.7976, -220.6879]* | 130 |

By prompt source:

| method | source | CLIP x100 | subject consistency | motion smoothness | dynamic degree | imaging quality | aesthetic quality |
|---|---|---|---|---|---|---|---|
| M1 | hard | 32.3579 | 0.9390 | 0.9823 | 0.8333 | 0.6735 | 0.5451 |
| M1 | vbench | 31.2341 | 0.9495 | 0.9765 | 0.8400 | 0.6530 | 0.5679 |
| M2 | hard | 32.7544 | 0.9469 | 0.9855 | 0.7000 | 0.7075 | 0.5652 |
| M2 | vbench | 30.8194 | 0.9628 | 0.9849 | 0.7200 | 0.6887 | 0.6195 |

Equal budget: the teacher run with fewer UniPC steps against the distilled student with fewer steps, on the 30-prompt subset
(figure `fig_m1_vs_m2_budget.png`):

| method | steps | n | gen_s | aesthetic quality | imaging quality | dynamic degree | subject consistency | CLIP x100 |
|---|---|---|---|---|---|---|---|---|
| M1 | 4 | 30 | 26.772 | 0.390 | 0.391 | 0.700 | 0.909 | 27.631 |
| M1 | 8 | 30 | 44.670 | 0.493 | 0.590 | 0.767 | 0.933 | 31.134 |
| M1 | 16 | 30 | 80.533 | 0.518 | 0.628 | 0.800 | 0.938 | 32.204 |
| M1 | 50 | 30 | 232.911 | 0.551 | 0.666 | 0.800 | 0.944 | 31.726 |
| M2 | 1 | 30 | 7.946 | 0.471 | 0.639 | 0.467 | 0.882 | 30.733 |
| M2 | 2 | 30 | 9.372 | 0.570 | 0.666 | 0.700 | 0.934 | 31.156 |
| M2 | 3 | 30 | 10.781 | 0.580 | 0.661 | 0.700 | 0.936 | 30.934 |
| M2 | 4 | 30 | 12.188 | 0.592 | 0.676 | 0.767 | 0.941 | 30.774 |

Difference from M2 with 4 steps (its default) on the same prompts:

| method | steps | aesthetic quality | imaging quality | dynamic degree | subject consistency | CLIP x100 |
|---|---|---|---|---|---|---|
| M1 | 4 | -0.202 [-0.237, -0.168]* | -0.284 [-0.338, -0.226]* | -0.067 [-0.233, +0.067] | -0.033 [-0.056, -0.008]* | -3.142 [-4.957, -1.315]* |
| M1 | 8 | -0.099 [-0.131, -0.070]* | -0.086 [-0.122, -0.049]* | +0.000 [-0.167, +0.167] | -0.009 [-0.028, +0.013] | +0.360 [-0.853, +1.687] |
| M1 | 16 | -0.074 [-0.107, -0.044]* | -0.048 [-0.083, -0.011]* | +0.033 [-0.167, +0.233] | -0.004 [-0.024, +0.018] | +1.431 [+0.185, +2.752]* |
| M1 | 50 | -0.042 [-0.071, -0.013]* | -0.010 [-0.030, +0.013] | +0.033 [-0.100, +0.167] | +0.003 [-0.014, +0.022] | +0.953 [+0.063, +1.853]* |
| M2 | 1 | -0.121 [-0.147, -0.097]* | -0.037 [-0.068, -0.003]* | -0.300 [-0.500, -0.100]* | -0.059 [-0.077, -0.042]* | -0.040 [-0.811, +0.740] |
| M2 | 2 | -0.022 [-0.042, -0.003]* | -0.010 [-0.028, +0.009] | -0.067 [-0.233, +0.100] | -0.007 [-0.015, +0.001] | +0.382 [-0.061, +0.866] |
| M2 | 3 | -0.012 [-0.020, -0.004]* | -0.014 [-0.027, -0.003]* | -0.067 [-0.200, +0.067] | -0.005 [-0.009, -0.002]* | +0.160 [-0.103, +0.441] |
| M2 | 4 | reference | reference | reference | reference | reference |

How to read this (interpretation, written by hand from the tables above):

- **Speed.** M1 takes 232.9 s per 5 s clip (sd 0.31 s over 130 videos), M2 12.19 s: M2 is
  **19.1x faster** at about the same peak memory (24.4 against 23.3 GiB).
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

`m2_main`: default. `m2_steps{1,2,3}`: denoising steps (timesteps 1000; 1000,500; 1000,750,500). `m2_chunk{1,7}`: frames per chunk.
`m2_win{6,9,12,15}`: attention window in latent frames (21 = whole 5 s clip = global attention). `m2_win9_sink3`: window 9 plus the first
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
