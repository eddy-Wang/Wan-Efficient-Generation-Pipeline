# M2: Self Forcing (autoregressive streaming distillation)

*Draft of the M2 part of the group report. All numbers come from our own runs on one NVIDIA L40S (48 GB); the full tables are in `RESULTS.md` and `results/tables/`, figures in `results/figures/`. M1 (the teacher) was measured with the same protocol; see `M1_report_section.md`.*

## 1. Method

**Model.** Self Forcing (Huang et al., 2025) starts from Wan2.1-T2V-1.3B, a bidirectional video diffusion transformer that denoises all 21 latent frames of a clip together, over 50 steps with classifier-free guidance. Self Forcing turns it into a *chunk-wise autoregressive* model: the 21 latent frames (81 video frames, 5 s at 16 fps) are produced in 7 chunks of 3 latent frames. Each chunk attends to the cached keys and values of the earlier chunks (KV cache), is denoised in 4 steps (timesteps 1000, 750, 500, 250), and is then passed through the network once more to write its clean keys and values into the cache. The checkpoint is distilled with distribution matching (DMD, Yin et al., CVPR 2024) and trained by *rolling out its own previous chunks*, so the context it sees in training has the same imperfections as at inference; this is Self Forcing's answer to exposure bias. Guidance is distilled into the weights, so inference is a single conditional pass: no negative prompt, no CFG.

**Long videos.** The cache is a rolling window, so generation can continue beyond the 21 frames of training. The model supports a local attention window (in latent frames) and an attention *sink*: the first `s` latent frames stay in the cache when older chunks are evicted.

**What we changed.** The upstream pipeline has a `context_noise` option, but it only changes the timestep label of the cache-refresh pass; the cached chunk stays clean. We implemented a *re-noised context* variant: before the refresh pass, the clean chunk is noised to level t with the scheduler (t = 25, 50, 100 on a 0 to 1000 scale). The idea was that a slightly noisy context should be less sensitive to errors in earlier chunks. This is our own experiment, not a published method.

## 2. Setup

- **Prompts.** The group's 130 expanded prompts (100 sampled from VBench, 30 hard prompts written by the group). Ablations use a fixed random subset of 30 (20 VBench, 10 hard); 20 s rollouts use 30 prompts (the same 30); seed studies use 10 of them.
- **Settings.** 832×480, 16 fps; 81 frames (5.06 s) for the main run and ablations, 321 frames (20.06 s) for rollouts; seed 20261027, set again for every video; EMA weights, bf16.
- **Timing.** `generation_seconds` covers one pipeline call (text encoding, cache initialisation, denoising, VAE decoding); it excludes model loading and file export. One untimed warm-up video starts each process. Memory is the PyTorch peak allocation per video.
- **Metrics.** (i) VBench custom_input mode from a pinned source revision, five per-video dimensions: subject consistency, motion smoothness, dynamic degree (share of videos judged dynamic), imaging quality and aesthetic quality (0-1). The weights are byte-identical to the group's recorded ones, and scoring a folder gave exactly the same per-video values as scoring one video at a time. (ii) CLIP text alignment (ViT-B/32, 16 frames, original caption, ×100), the group's definition. (iii) For drift, each 20 s video is cut into its first and last 4 s and both clips are scored by VBench; we also compute, per 2 s segment, CLIP similarity to the first segment, brightness, saturation, sharpness and frame difference. Comparisons between configurations are *paired* (same prompt, same seed) with a bootstrap 95% interval over prompts.
- **Encoding.** All videos use the encoder settings of Wan's own writer (libx264, quality 8, about 6.5 Mbps). A first run used a lower default bitrate (about 1 Mbps) and was discarded, because bitrate alone changed the sharpness statistic (445 versus 538 on the same prompts).

## 3. Results

### 3.1 Speed and memory

| Setting | Video | Generation time | Real-time factor* | Peak memory | VAE share of time |
|---|---|---|---|---|---|
| 4 steps (default), n=130 | 5.06 s | 12.19 s (sd 0.03) | 0.42 | 23.3 GiB | 42% |
| 3 steps | 5.06 s | 10.78 s | 0.47 | 23.3 GiB | 47% |
| 2 steps | 5.06 s | 9.37 s | 0.54 | 23.3 GiB | 54% |
| 1 step | 5.06 s | 7.95 s | 0.64 | 23.3 GiB | 64% |
| window 6 / 9 / 12 / 15 | 5.06 s | 10.71 / 11.24 / 11.62 / 11.95 s | 0.47 to 0.42 | 19.3 / 20.1 / 20.9 / 21.7 GiB | 48% to 43% |
| 20 s, window 21 | 20.06 s | 54.29 s | 0.37 | 23.9 GiB | 37% |
| 20 s, window 12 | 20.06 s | 47.15 s | 0.43 | 21.5 GiB | 43% |

*Real-time factor = video seconds / generation seconds; above 1 means faster than playback. n=30 except the first row.

- For the same architecture, Wan2.1-1.3B (M1: 832×480, 81 frames, 50 steps with CFG) takes 232.94 s per video (sd 0.31, 130 videos, same timing scope and GPU type), so the default Self Forcing setting is **19.1× faster** per video. M1 runs 100 full-clip network evaluations per video (50 steps × 2 for CFG); M2 runs 35 evaluations, each on a 3-frame chunk with cached context (7 chunks × (4 denoising passes + 1 cache-refresh pass)).
- Each denoising step costs about 1.4 s; the remaining 6.5 s is a fixed cost dominated by VAE decoding (about 5 s). **After distillation the VAE decoder, not the transformer, is the largest single cost: 42% of the time at 4 steps and 64% at 1 step.** Reducing steps further has sharply diminishing returns.
- Per-chunk latency rises for the first seven chunks while the cache fills (about 0.7 s to 1.35 s) and then stays flat for the remaining twenty; a window of 12 frames caps it at about 1.02 s. Cost per new frame does not grow with video length, and memory stays at 21 to 24 GiB for a 20 s video.

### 3.2 Steps, chunk size and window on 5 s clips

Default on these 30 prompts: subject consistency 0.9413, dynamic degree 0.7667, imaging quality 0.6757, aesthetic quality 0.5920, CLIP 30.77. Mean paired difference from the default [95% interval]; `*` = interval excludes 0.

| Change | Subject consistency | Dynamic degree | Imaging quality | Aesthetic quality | CLIP ×100 |
|---|---|---|---|---|---|
| 1 step | -0.059 [-0.079, -0.041]* | -0.300 [-0.500, -0.100]* | -0.037 [-0.068, -0.004]* | -0.121 [-0.146, -0.097]* | -0.04 [-0.81, 0.74] |
| 2 steps | -0.007 [-0.014, +0.000] | -0.067 [-0.233, +0.100] | -0.010 [-0.028, +0.008] | -0.022 [-0.041, -0.003]* | +0.38 [-0.06, 0.87] |
| 3 steps | -0.005 [-0.009, -0.002]* | -0.067 [-0.200, +0.067] | -0.014 [-0.026, -0.003]* | -0.012 [-0.020, -0.004]* | +0.16 [-0.10, 0.44] |
| chunk 1 frame | -0.057 [-0.076, -0.035]* | -0.367 [-0.600, -0.133]* | -0.028 [-0.052, -0.002]* | -0.022 [-0.051, +0.007] | -1.27 [-2.20, -0.41]* |
| chunk 7 frames | -0.012 [-0.025, +0.002] | +0.133 [0.000, +0.300] | -0.000 [-0.021, +0.023] | +0.008 [-0.010, +0.027] | +0.46 [-0.08, 1.01] |
| window 6 | -0.012 [-0.019, -0.004]* | -0.033 [-0.133, +0.067] | -0.001 [-0.009, +0.006] | -0.003 [-0.012, +0.006] | -0.17 [-0.42, 0.06] |
| window 9 + sink 3 | +0.001 [-0.001, +0.003] | 0.000 [0.000, 0.000] | +0.003 [+0.000, +0.006]* | +0.001 [-0.003, +0.004] | +0.04 [-0.07, 0.17] |

- **CLIP text alignment does not detect the loss from fewer steps; VBench does.** One step leaves CLIP unchanged but lowers aesthetic quality by 0.12, imaging quality, dynamic degree and subject consistency, all clearly. Two and three steps lose little (aesthetic quality about -0.02 and -0.01), so 2 steps, 23% faster than the default (9.37 s against 12.19 s), is the sensible trade-off point.
- Chunk size 1, which is not the trained size, is clearly worse on four of five measures and 13% slower. Chunk 7 is close to the default.
- A window shorter than the clip, with or without sink, changes almost nothing on 5 s clips (window 6 loses 0.012 of subject consistency); it saves memory (19.3 against 23.3 GiB at window 6) and matters when videos exceed the training length.

### 3.3 Long-horizon drift (20 s, 30 prompts)

Each 20 s video is scored as its first 4 s and its last 4 s; "change" is last minus first. **Subject consistency must be read together with dynamic degree**: a video that slowly freezes becomes more consistent from frame to frame, so a higher value late in the video is not an improvement (for window 21 it rises by 0.018 while the share of dynamic clips falls from 0.80 to 0.47).

| Configuration | Dynamic degree, first 4 s → last 4 s | Aesthetic change | CLIP similarity to first 2 s, change |
|---|---|---|---|
| window 21 | 0.80 → 0.47 | -0.038 | -0.110 |
| window 21 + sink 3 | change -0.17 (difference from window 21: +0.167 [0.000, +0.333]) | -0.022 | -0.066 |
| window 12 | change -0.43 | -0.032 | -0.147 |
| window 12 + sink 3 | change -0.10 | -0.009 | -0.068 |

Paired against window 21 (`*` = interval excludes 0):

- **A short window without a sink drifts more.** Window 12: CLIP text alignment -0.94 [-1.72, -0.22]*, similarity to the first 2 s -0.037 [-0.064, -0.013]*.
- **A sink of 3 latent frames removes most of the drift.** Window 21 + sink: similarity +0.044 [+0.032, +0.057]*, frame-difference ("motion") statistic +3.16 [+2.07, +4.44]*. Window 12 + sink 3 is the best setting: dynamic degree +0.233 [+0.033, +0.433]* and aesthetic quality +0.029 [+0.012, +0.045]* better than window 21, and it is also 13% faster and uses 2.4 GiB less memory. The first chunk plausibly acts as a stable anchor for appearance.
- Even the best setting still drifts: motion decays, brightness falls by about 8 levels, and in the one example we inspected a brick wall bent without a sink while with a sink a second bicycle appeared at 19 s.

### 3.4 Our modification: re-noised context (negative result)

On top of window 21 + sink 3, paired against the same setting with a clean context (30 prompts):

| Noise level t | Subject consistency | Aesthetic quality | CLIP similarity to first 2 s | Dynamic degree |
|---|---|---|---|---|
| 25 | -0.026 [-0.042, -0.012]* | -0.022 [-0.042, -0.004]* | -0.059 [-0.075, -0.043]* | not significant |
| 50 | -0.037 [-0.063, -0.014]* | not significant (imaging quality -0.027 [-0.052, -0.004]*) | -0.041 [-0.059, -0.024]* | not significant |
| 100 | -0.024 [-0.041, -0.009]* | not significant | -0.027 [-0.040, -0.015]* | not significant |

Re-noising does not slow the loss of motion in a way VBench can confirm, and it lowers subject consistency and similarity to the first segment at every level. The effect is not a slow drift: saturation and sharpness are already lower in the first segment. In the example we inspected, the yellow bicycle turns beige and grey at t = 25. The t = 25 and 50 clips are about 30% larger files at the same encoder setting (25 MB against 19 MB), which suggests more high-frequency content, so the extra frame difference is probably partly flicker. We do **not** claim an improvement: the model was never trained on noisy context, and the sink is the effective remedy. Re-noising would probably have to be part of training, as in Diffusion Forcing, to be useful.

### 3.5 Seeds

On ten prompts with three seeds, the seed-to-seed standard deviation of CLIP inside a prompt is 1.28 points, against 3.85 between prompts. Single-video differences of about one point or less are therefore noise, which is why every comparison above is paired over 30 prompts. The effect of a sink on the 20 s videos has the same sign for all three seeds (similarity to the first segment +0.016 to +0.048, frame difference +3.1 to +7.3, brightness loss reduced by 6 to 9 levels).

### 3.6 Streaming

The authors describe real-time streaming. We replicated their demo loop (each 3-frame chunk, 0.75 s of video, is decoded as soon as it is ready; fp16; optional light TAEHV decoder; optional `torch.compile`) on the L40S. Four timed videos per configuration after one warm-up; `first frame` = text encoding + cache initialisation + the first chunk; `steady chunk` = mean time per chunk after the first.

| Configuration | Clip time | Real-time factor | First frame | Steady chunk | PSNR against standard decode |
|---|---|---|---|---|---|
| A bf16, whole-clip decode (our standard) | 12.06 s | 0.42 | not applicable | not applicable | not applicable |
| B fp16, per-chunk decode | 12.71 s | 0.40 | 1.84 s | 1.80 s | 37.0 dB |
| C B + light VAE | 7.36 s | 0.69 | 0.80 s | 1.08 s | 28.4 dB |
| D B + torch.compile | 10.45 s | 0.48 | 1.53 s | 1.48 s | 36.7 dB |
| E C + torch.compile | 6.69 s | 0.76 | 0.69 s | 0.98 s | 28.3 dB |

Real time needs a steady chunk below 0.75 s. **No configuration reached it.** Streaming by itself does not help (B): decoding a chunk with the standard VAE takes about 0.77 s, more than the denoising. The light VAE removes the decoding cost (about 0.05 s per chunk) and gives a first frame after 0.8 s, at a visible fidelity cost (PSNR 28.4 dB on average, 23.6 to 34.7 dB between videos). `torch.compile` costs about 4 minutes once and saves 2.3 s with the standard decoder and 0.7 s with the light one. The best combination (E) runs at 0.76× real time with 0.98 s per chunk. With decoding almost free, denoising plus cache refresh (about 1.0 s per chunk once the cache is full) is the remaining limit on this GPU, and a viewer would need about 2.2 s of start-up buffering to play a 5 s clip without stalls. We did not test FP8 quantisation, a TensorRT decoder or other GPUs, so this does not contradict the authors' real-time claim, which refers to different hardware and options.

## 4. Discussion

**Strengths.** Distillation plus a KV cache gives a model that is 19.1× faster than its teacher per video, with the same CLIP text alignment and higher VBench aesthetic (+0.044) and imaging (+0.035) quality over 130 paired prompts, although with less motion (dynamic degree 0.72 against 0.84); the teacher with 4, 8 or 16 steps cannot match it at comparable cost (see the M1 section); cost per frame is constant in video length; memory stays near 24 GiB for 20 s videos on one GPU.

**Limitations.** (i) The transformer is no longer the bottleneck: VAE decoding takes 42% to 64% of the time. (ii) Drift is reduced by a sink but not removed (motion decay, brightness, geometry). (iii) The model is trained for 3-frame chunks and 21 frames and is brittle outside that (chunk 1). (iv) Guidance is baked in, so there is no CFG scale or negative prompt to adjust. (v) Real-time streaming was not reached on an L40S (best 0.76×), and the light VAE that gets closest costs fidelity. (vi) Aggressive step reduction (1 step) is clearly visible in VBench although CLIP does not show it. (vii) Videos move less than the teacher's (dynamic degree 0.72 against 0.84 over 130 prompts).

**Future work.** Train with re-noised context instead of applying it at inference; periodic re-anchoring of the sink on long videos; FP8 or a faster GPU to close the remaining gap to real time; measure time to first frame in a real streaming client.

## 5. Evidence and limits

One seed per prompt (20261027) except in section 3.5; 30 prompts for ablations and rollouts, so differences below about 0.5 CLIP points, 0.02 on a VBench dimension, or 0.02 in similarity are not resolved. VBench here is custom_input on our own prompts, not the official leaderboard protocol. CLIP-to-first mixes true drift with legitimate scene change (camera motion), and the frame-difference statistic cannot tell motion from flicker. The 20 s videos were scored as two 4 s clips re-encoded at near-lossless quality, so their absolute VBench values are not comparable with the 5 s videos. No human evaluation.

## Reproducibility

Code revision 33593df3e81fa3ec10239271dd2c100facac6de1; Wan2.1-T2V-1.3B base files plus `self_forcing_dmd.pt` (SHA256 starts a0413986d973); PyTorch 2.7.1 (CUDA 12.8), flash-attn 2.8.3; L40S on node TC2N03. The main run was repeated once (after the encoder change); its mean generation time was 12.26 s and 12.19 s. VBench source revision fd18b3d055cb0fc6f066ca90fe2c3c8cbb698490. Scripts: `m2_self_forcing/code/`.
