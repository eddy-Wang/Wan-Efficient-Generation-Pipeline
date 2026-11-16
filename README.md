# Wan-Efficient-Generation-Pipeline

Efficient text-to-video generation on the Wan family. This repository holds the **M1 (Wan2.1-1.3B baseline)** and **M2 (Self Forcing)**
parts of a group technical review for AI6132 Generative AI for Visual Synthesis (NTU): the code, all results, and the notes the group
uses to report the other methods (M3 Wan2.2-TI2V-5B, M4 training-free acceleration) in the same way.

**Start with [`RESULTS.md`](RESULTS.md).** It gives the measurement protocol to align on, the M2 row for the group table, and every
experiment with its numbers. M1 and the M1-against-M2 comparison are in its section 4.7.

## What M2 is

Self Forcing turns Wan2.1-1.3B into a chunk-wise autoregressive video model with a KV cache and distils it to 4 denoising steps
without classifier-free guidance. We measure it on one NVIDIA L40S (48 GB) and study what its speed costs and how it behaves on
videos longer than it was trained for.

| | |
|---|---|
| Prompts | 130 (100 sampled from VBench, 30 hard prompts), seed 20261027 |
| Videos | 832x480, 81 frames, 16 fps; 20 s rollouts for the drift study |
| Generation | 12.19 s per 5 s clip, 23.3 GiB peak torch memory (L40S) |
| Evaluation | VBench (pinned revision, five dimensions) and a CLIP text-alignment score, with paired comparisons |

Experiments: denoising steps, chunk size, attention window and sink, drift in 20 s rollouts, a re-noised KV context variant
(a negative result), three seeds, and a streaming-speed benchmark (fp16, block-wise decoding, a light VAE, `torch.compile`).

## M1 against M2 in one table

Same 130 prompts, seed, GPU, timing scope and evaluators (VBench pinned revision; CLIP ViT-B/32).

| | Steps / CFG | Time per 5 s clip | Peak memory | Dynamic degree | Imaging | Aesthetic | CLIP x100 |
|---|---|---|---|---|---|---|---|
| M1 Wan2.1-1.3B | 50 / 6.0 | 232.9 s | 24.4 GiB | 0.838 | 0.658 | 0.563 | 31.49 |
| M2 Self Forcing | 4 / none | 12.2 s | 23.3 GiB | 0.715 | 0.693 | 0.607 | 31.27 |

M2 is 19.1x faster with the same text alignment and higher per-frame quality scores, but less motion. Running the teacher with
fewer steps does not close the gap: at 4 steps it still takes 26.8 s and its quality collapses.

## Repository layout

```
RESULTS.md                    results and alignment notes (generated from results/)
report/                       drafts of the M1 and M2 parts of the report
results/tables, figures       every table and figure
results/data                  one row per video / 2 s segment, the manifest of the 130 main videos (with SHA256)
results/raw, raw_m1           per-video generation and score records (json), raw VBench output (M2, M1)
results/bench                 raw streaming benchmark
m2_self_forcing/code          M2 generation, scoring, aggregation and benchmark scripts (see its README)
m1_wan21/code                 M1 generation, evaluation job and M1-against-M2 aggregation (see its README)
```

## Reproducing

The scripts expect the Self Forcing code at commit `33593df3e81fa3ec10239271dd2c100facac6de1`, the Wan2.1-T2V-1.3B files and the
DMD checkpoint, and the group prompt file `prompts_expanded_v1.jsonl`. Python 3.10, PyTorch 2.7.1 (CUDA 12.8), flash-attn 2.8.3
(`m2_self_forcing/code/requirements_m2.txt`). VBench was run from its source at revision
`fd18b3d055cb0fc6f066ca90fe2c3c8cbb698490` in a separate environment (torch 2.7.1, numpy 1.26.4, transformers 4.33.2,
timm 1.0.12, pyiqa 0.1.10). The order is `run_m2.py`, `analyze_m2.py`, `prep_eval_sets.py` and `vbench_batch.py`, `bench_stream.py`,
`aggregate_m2.py`, `build_docs.py`; the Slurm templates show the exact calls. M1 uses the official Wan2.1 code at revision
`9737cba9c1c3c4d04b33fcad41c111989865d315` with the same weights and environment (`m1_wan21/code/README.md`). Tables, figures and
`RESULTS.md` can be rebuilt from `results/raw` and `results/raw_m1` alone (`aggregate_m2.py --results results/raw --out results`,
`aggregate_m1.py --m1 results/raw_m1 --m2 results/raw --out results`, then `build_docs.py results RESULTS.md`).

## Not in this repository

Generated videos (about 0.6 GB for the 130 M2 main videos; verify a copy against the SHA256 in `results/data/m2_main_manifest.csv`), model
weights, the group prompt file and the group evaluation tooling (they belong to the group members who wrote them), and course
materials.
