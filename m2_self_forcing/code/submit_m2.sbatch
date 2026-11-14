#!/bin/bash
#SBATCH --job-name=m2
#SBATCH --partition=MGPU-TC2
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --exclude=TC2N01,TC2N02
#SBATCH --cpus-per-task=4
#SBATCH --mem=28G
#SBATCH --time=05:55:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# usage: sbatch [--time=HH:MM:SS] submit_m2.sbatch SUITES BUDGET_MINUTES
# Re-submitting continues the work: finished videos are skipped.
SUITES=${1:-main,ablate,drift}
BUDGET=${2:-330}
cd $HOME/sfproj/Self-Forcing || exit 1
module load anaconda
eval "$(conda shell.bash hook)"
conda activate sf
echo "Host: $(hostname)  Job: $SLURM_JOB_ID  Start: $(date)  suites=$SUITES budget=$BUDGET"
nvidia-smi --query-gpu=name,memory.total --format=csv
export TOKENIZERS_PARALLELISM=false
python ../m2/run_m2.py --prompts ../m2/prompts_expanded_v1.jsonl --out ../outputs \
  --suites "$SUITES" --budget-minutes "$BUDGET" --use_ema
RC=$?
echo "generation finished exit=$RC at $(date)"
# score whatever is new (CLIP + drift statistics); already scored videos are skipped
export HF_HUB_OFFLINE=1
python ../m2/analyze_m2.py --videos ../outputs --prompts ../m2/prompts_expanded_v1.jsonl --out ../analysis --configs "m2_*"
echo "End: $(date) exit=$?"
