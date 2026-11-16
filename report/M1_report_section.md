# M1: Wan2.1-T2V-1.3B (the baseline and the teacher of Self Forcing)

*Draft of the M1 part of the group report. All numbers come from our own runs on one NVIDIA L40S (48 GB), measured with exactly the protocol of M2 (same prompts, seed, timing scope, memory scope, encoder and evaluators). Full tables: `RESULTS.md` section 4.7 and `results/tables/m1_*.csv`; figure `results/figures/fig_m1_vs_m2_budget.png`.*

## 1. Method

**Latent space.** Wan2.1 compresses video with Wan-VAE, a causal 3D VAE that reduces time by 4 and each spatial side by 8 (16 latent channels). Causality in time means the first frame is encoded on its own, so 81 frames become 1 + 80/4 = 21 latent frames; at 832×480 each latent frame is 104×60.

**Denoiser.** A diffusion transformer (1.3B parameters: 30 blocks, width 1536, 12 heads). The latent is cut into 1×2×2 patches, which gives 21 × 30 × 52 = 32,760 tokens for a 5 s clip. Every block has *full* spatio-temporal self-attention over all tokens (every frame sees every other frame, so the model is bidirectional in time), cross-attention to the text embedding from the umT5 encoder (multilingual T5), and a timestep embedding that modulates each block through scale and shift parameters predicted by a shared MLP.

**Training objective and sampling.** Wan is trained with flow matching: it predicts the velocity that moves a noisy latent along a straight path from noise to data. At inference an ODE solver integrates this velocity field; the default is UniPC with 50 steps and a timestep shift (8.0 here) that spends more steps at high noise. Classifier-free guidance (scale 6.0) runs the network twice per step, once with the prompt and once with Wan's default negative prompt, so a video costs 100 full-clip network passes.

**Relation to M2.** Self Forcing (M2) is distilled from exactly this model and these weights. M1 is therefore both the quality reference and the cost reference for M2: same architecture and VAE, but bidirectional, 50 steps and CFG, against M2's causal chunks, 4 steps and no CFG.

## 2. Setup

- **Code and weights.** Official Wan2.1 code at revision `9737cba9c1c3c4d04b33fcad41c111989865d315` (the revision in the group's example config), Wan2.1-T2V-1.3B weights, bf16, no offload, single GPU.
- **Settings.** 832×480, 81 frames (5.06 s at 16 fps), 50 UniPC steps, CFG 6.0, shift 8.0, Wan's default negative prompt; seed 20261027 set again for every video.
- **Prompts.** The group's 130 expanded prompts (100 from VBench, 30 hard prompts). The step ablation uses the same 30-prompt subset as the M2 ablations.
- **Measurement.** `generation_seconds` is one `WanT2V.generate` call (text encoding, denoising with CFG, VAE decoding), excluding model loading and file export; one untimed warm-up per process; `cuda.synchronize` around each video; peak PyTorch allocation reset per video. Videos written with Wan's own `cache_video` (libx264, quality 8).
- **Metrics.** VBench (pinned revision, `custom_input`, five dimensions) and CLIP ViT-B/32 text alignment (16 frames, original caption, ×100), identical to M2. Comparisons with M2 are paired over prompts with a bootstrap 95% interval.

## 3. Results

### 3.1 Main run (130 prompts)

| Method | Steps / CFG | Time per 5 s clip | Peak memory | Subject cons. | Motion smooth. | Dynamic degree | Imaging | Aesthetic | CLIP ×100 |
|---|---|---|---|---|---|---|---|---|---|
| M1 Wan2.1-1.3B | 50 / 6.0 | 232.94 s (sd 0.31) | 24.4 GiB | 0.9470 | 0.9779 | 0.8385 | 0.6578 | 0.5627 | 31.49 |
| M2 Self Forcing | 4 / none | 12.19 s (sd 0.03) | 23.3 GiB | 0.9591 | 0.9850 | 0.7154 | 0.6931 | 0.6069 | 31.27 |
| M2 minus M1 [95% interval] | | -220.7 s | | +0.012 [+0.005, +0.020]* | +0.007 [+0.005, +0.009]* | -0.123 [-0.200, -0.046]* | +0.035 [+0.021, +0.050]* | +0.044 [+0.030, +0.059]* | -0.23 [-0.73, +0.28] |

`*` = interval excludes 0. M1 runs at 0.022× real time; M2 is 19.1× faster.

- **Text alignment is the same.** The CLIP difference is not significant over 130 prompts.
- **M1 moves more.** 84% of M1 videos are judged dynamic against 72% for M2. M2's higher subject consistency and motion smoothness go together with this lower motion, so they are not evidence of better temporal quality.
- **M2 scores higher on per-frame quality.** Aesthetic (+0.044) and imaging quality (+0.035) favour the distilled student. This is plausible for DMD distillation, which tends to produce sharp, high-contrast, mode-like samples, but VBench is not human judgement and we did not run a user study.

### 3.2 Cost structure

Time is linear in the number of steps: 26.8 s (4), 44.7 s (8), 80.5 s (16), 232.9 s (50), i.e. about 4.5 s per UniPC step (two passes because of CFG) plus about 9 s of fixed cost (text encoding and VAE decoding). The denoiser is 96% of M1's time. In M2 the denoiser costs about 1.4 s per step and the VAE decoder becomes the largest single cost (42%). Peak memory is almost the same (24.4 against 23.3 GiB); speed, not memory, is what separates the two on a 48 GB GPU.

### 3.3 Is distillation necessary? The teacher with fewer steps (30 prompts)

Paired difference from M2 at its default 4 steps on the same 30 prompts (`*` = interval excludes 0):

| Setting | Time | Aesthetic | Imaging | Dynamic degree | CLIP ×100 |
|---|---|---|---|---|---|
| M1, 4 steps | 26.8 s | -0.202 [-0.237, -0.168]* | -0.284 [-0.338, -0.226]* | -0.067 [-0.233, +0.067] | -3.14 [-4.96, -1.32]* |
| M1, 8 steps | 44.7 s | -0.099 [-0.131, -0.070]* | -0.086 [-0.122, -0.049]* | 0.000 [-0.167, +0.167] | +0.36 [-0.85, +1.69] |
| M1, 16 steps | 80.5 s | -0.074 [-0.107, -0.044]* | -0.048 [-0.083, -0.011]* | +0.033 [-0.167, +0.233] | +1.43 [+0.19, +2.75]* |
| M1, 50 steps | 232.9 s | -0.042 [-0.071, -0.013]* | -0.010 [-0.030, +0.013] | +0.033 [-0.100, +0.167] | +0.95 [+0.06, +1.85]* |
| M2, 2 steps | 9.4 s | -0.022 [-0.042, -0.003]* | -0.010 [-0.028, +0.009] | -0.067 [-0.233, +0.100] | +0.38 [-0.06, +0.87] |

- **Cutting the teacher's steps is not a substitute for distillation.** At 4 steps M1 still takes 2.2× M2's time (CFG doubles every step and each pass covers the whole clip) and collapses: aesthetic -0.20, imaging -0.28, CLIP -3.1; the example we inspected is a blur. Even at 16 steps (6.6× M2's time) M1 remains below M2 on aesthetic and imaging quality.
- **The speed-quality frontiers do not overlap** (figure `fig_m1_vs_m2_budget.png`): every M2 setting from 2 to 4 steps is at least 19× cheaper than M1 at 50 steps, higher on aesthetic quality than every M1 setting we ran, and about equal on imaging quality (0.662 to 0.676 against 0.666). M1 keeps an advantage on motion (dynamic degree) and, from 16 steps on this subset, on CLIP.
- On this subset M1 at 50 steps is +0.95 CLIP above M2, while over all 130 prompts there is no difference; one-point CLIP differences on 30 prompts are within prompt-selection noise.

## 4. Discussion

**Strengths of M1.** Highest motion (dynamic degree) of the two; full bidirectional attention lets every frame see the whole clip, with no autoregressive drift; CFG and the negative prompt remain adjustable at inference; the reference that distilled models are judged against.

**Limitations of M1.** 233 s for 5 s of video on an L40S (0.022× real time) makes it unusable for interactive use; the whole clip must be denoised together, so nothing can be shown before the end and cost grows faster than linearly with length (attention over all tokens); quality falls apart when steps are cut, so the only way to make it fast is to change the model, which is what M2 (and M4 without training) do.

## 5. Evidence and limits

One seed per prompt (20261027); the step ablation uses 30 prompts. Generation time over 130 videos varied by less than 2 s (232.3 to 233.9 s), so the timing is reliable. VBench is `custom_input` on our own prompts, not the leaderboard protocol; no human evaluation.

## Reproducibility

Wan2.1 revision `9737cba9c1c3c4d04b33fcad41c111989865d315`; PyTorch 2.7.1 (CUDA 12.8), flash-attn 2.8.3; L40S on TC2. Scripts: `m1_wan21/code/run_m1.py` (generation), `analyze_m2.py` (CLIP), `vbench_batch.py`, `aggregate_m1.py` (tables and figure).
