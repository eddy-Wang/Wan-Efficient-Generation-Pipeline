#!/bin/bash
#SBATCH --job-name=m2_analyze
#SBATCH --partition=MGPU-TC2
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --exclude=TC2N01,TC2N02
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=01:30:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

cd $HOME/sfproj || exit 1
module load anaconda
eval "$(conda shell.bash hook)"
conda activate sf
echo "Host: $(hostname)  Job: $SLURM_JOB_ID  Start: $(date)"
export TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1
python m2/analyze_m2.py --videos outputs --prompts m2/prompts_expanded_v1.jsonl --out analysis --configs "m2_*"
echo "End: $(date) exit=$?"
