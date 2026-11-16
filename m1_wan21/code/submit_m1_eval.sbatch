#!/bin/bash
#SBATCH --job-name=m1_eval
#SBATCH --partition=MGPU-TC2
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --exclude=TC2N01,TC2N02
#SBATCH --cpus-per-task=8
#SBATCH --mem=28G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# CLIP + per-segment statistics and VBench for the M1 videos, with the same tools and settings as M2
cd $HOME/sfproj || exit 1
module load anaconda
eval "$(conda shell.bash hook)"
echo "Host: $(hostname)  Job: $SLURM_JOB_ID  Start: $(date)"
conda activate sf
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
python m2/analyze_m2.py --videos outputs_m1 --prompts m2/prompts_expanded_v1.jsonl --out analysis_m1 --configs "m1_*"
mkdir -p eval_sets/m1_main eval_sets/m1_abl
for f in outputs_m1/m1_main/*_20261027.mp4; do ln -sf "$(readlink -f $f)" eval_sets/m1_main/m1_main__$(basename $f); done
for c in m1_steps4 m1_steps8 m1_steps16; do
  for f in outputs_m1/$c/*_20261027.mp4; do [ -e "$f" ] && ln -sf "$(readlink -f $f)" eval_sets/m1_abl/${c}__$(basename $f); done
done
echo "m1_main videos: $(ls eval_sets/m1_main | wc -l)  m1_abl videos: $(ls eval_sets/m1_abl | wc -l)"
conda deactivate
conda activate p1_eval
export OMP_NUM_THREADS=4 VBENCH_CACHE_DIR=$HOME/sfproj/p1/model_cache/vbench HF_HOME=$HOME/sfproj/p1/model_cache/huggingface TORCH_HOME=$HOME/sfproj/p1/model_cache/torch
for S in m1_main m1_abl; do
  python -u m2/vbench_batch.py --src p1/vendor/VBench --videos eval_sets/$S --out vbench_out --name $S
done
echo "End: $(date)"
