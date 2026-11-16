# M1: Wan2.1-T2V-1.3B baseline (batch runner and comparison with M2)

These scripts generate the M1 videos with the official Wan2.1 code and measure them exactly like M2 (same prompts, seed,
timing scope, memory scope, encoder, VBench and CLIP). They contain no weights or prompts.

## Requirements
- The official Wan2.1 code at revision `9737cba9c1c3c4d04b33fcad41c111989865d315` and the Wan2.1-T2V-1.3B weights
  (the same files Self Forcing uses in `wan_models/Wan2.1-T2V-1.3B/`).
- The same Python environment as M2 (`../../m2_self_forcing/code/requirements_m2.txt`), one 48 GB GPU (L40S).
- The group prompt file `prompts_expanded_v1.jsonl` and `subsets.json` from the M2 run (in this repository as
  `results/data/prompt_subsets.json`), so the step ablation uses the same 30 prompts as the M2 ablations.

## Files
- `run_m1.py`: loads `WanT2V` once and generates the chosen suites in one process: `smoke` (one prompt, 4 steps), `main` (130 prompts,
  50 UniPC steps, CFG 6.0, shift 8.0, Wan's default negative prompt), `steps` (4, 8 and 16 steps on the 30-prompt subset). Seed 20261027 is
  set for every video; one untimed warm-up; `generation_seconds` = one `generate` call (text encoding, denoising with CFG, VAE decoding),
  excluding model loading and export; peak torch memory reset per video; videos written with Wan's `cache_video` (libx264, quality 8).
  Finished videos are skipped and `--budget-minutes` stops cleanly before the 6 h job limit, so re-submitting continues.
- `submit_m1.sbatch`: Slurm template for the generation (`sbatch submit_m1.sbatch main,steps 335`).
- `submit_m1_eval.sbatch`: CLIP and per-segment statistics with `analyze_m2.py`, then VBench with `vbench_batch.py` (both in
  `../../m2_self_forcing/code/`), for the main run and the step ablation.
- `m1_poller.sh`: head-node helper that queues the evaluation after the generation jobs (the cluster allows only two queued jobs).
- `aggregate_m1.py`: M1 tables, paired M2 minus M1 comparison over the 130 prompts, equal-budget comparison on the 30-prompt subset
  and the speed-quality figure.

## Rebuilding the tables from this repository
```
python m1_wan21/code/aggregate_m1.py --m1 results/raw_m1 --m2 results/raw --out results
python m2_self_forcing/code/build_docs.py results RESULTS.md
```
