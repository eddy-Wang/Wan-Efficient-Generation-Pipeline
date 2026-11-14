# M2: Self Forcing experiments (batch runner, evaluation, aggregation)

These scripts run the Self Forcing model (chunk-wise autoregressive video diffusion, DMD-distilled to 4 steps)
for the group's technical review. They do not contain any dataset or model weights.

## Requirements
- A checkout of the Self Forcing code at commit 33593df3e81fa3ec10239271dd2c100facac6de1, with the
  Wan2.1-T2V-1.3B files in `wan_models/Wan2.1-T2V-1.3B/` and the DMD checkpoint in `checkpoints/self_forcing_dmd.pt`.
- Python 3.10 with `requirements_m2.txt`, one 48 GB GPU (all runs used an L40S).
- The group prompt file `prompts_expanded_v1.jsonl` (fields: prompt_id, prompt, original_prompt, source).

## Files
- `run_m2.py`: loads the model once and generates all videos of the chosen suites in one process. Each video is seeded separately,
  so results do not depend on prompt order. A video is skipped if its mp4 and json record exist, so re-submitting continues an interrupted
  run. It records generation time (text encoding, cache initialisation, denoising, VAE decoding; not model loading or file export),
  per-chunk latency and peak torch memory per video. Suites: `main` (130 prompts), `ablate` (steps, window, sink, chunk on 30 prompts),
  `drift` and `drift_more` (20 s rollouts on 20 + 10 prompts: window, sink, re-noised context), `seeds` (two more seeds on 10 prompts).
  Videos are written with the same encoder settings as Wan's own writer (libx264, quality 8).
- `analyze_m2.py`: CLIP text alignment (ViT-B/32, 16 uniform frames, original caption) and per-2-second statistics (CLIP similarity to
  the first segment, brightness, saturation, sharpness, frame difference).
- `prep_eval_sets.py`, `vbench_batch.py`, `vbench_eqcheck.py`: build the folders VBench scores (5 s videos; first and last 4 s of the 20 s
  videos), score a folder with the pinned VBench source (custom_input, five dimensions), and check that folder scoring equals one-video scoring.
- `bench_stream.py`: streaming-speed benchmark replicating the authors' demo loop (fp16, per-chunk decoding, light VAE, `torch.compile`).
- `aggregate_m2.py`: tables (csv/md), figures and per-video csv files from the records, with paired bootstrap comparisons.
- `build_docs.py`: writes `RESULTS.md` from the tables.
- `submit_*.sbatch`: Slurm templates (one GPU, at most 6 hours per job). Edit the project directory in the `cd` line, create a `logs/`
  directory where you submit (Slurm writes the logs there), e.g. `sbatch submit_m2.sbatch main,ablate,drift 330`.

## Usage
```
cd Self-Forcing
python ../m2/run_m2.py --prompts ../m2/prompts_expanded_v1.jsonl --out ../outputs --suites main,ablate,drift,drift_more,seeds --use_ema
python ../m2/analyze_m2.py --videos ../outputs --prompts ../m2/prompts_expanded_v1.jsonl --out ../analysis
# VBench, in its own environment (see the repository README): prep_eval_sets.py, then vbench_batch.py per folder
python ../m2/bench_stream.py --out ../bench/bench_stream.json
python aggregate_m2.py --results <dir with outputs/, analysis/, vbench/> --out <results dir>
python build_docs.py <results dir> RESULTS.md
```

## Notes
- `context_noise` in the upstream pipeline only changes the timestep label of the cache-refresh pass. The re-noised-context variant here is
  implemented in `GenProxy` in `run_m2.py`: the clean chunk is noised to level t with the scheduler before it is written into the KV cache.
- The attention window and sink size are changed at run time by setting `local_attn_size`, `sink_size` and `max_attention_size` on the
  model's attention blocks and re-allocating the KV cache.
- `torch.compile` kernels need a newer `libstdc++` than the cluster's anaconda module provides; `submit_bench.sbatch` puts the gcc runtime first on
  `LD_LIBRARY_PATH`.
