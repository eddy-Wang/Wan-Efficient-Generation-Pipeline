#!/bin/bash
#SBATCH --job-name=m1
#SBATCH --partition=MGPU-TC2
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --exclude=TC2N01,TC2N02
#SBATCH --cpus-per-task=4
#SBATCH --mem=28G
#SBATCH --time=05:58:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# usage (from ~/sfproj): sbatch m2/submit_m1.sbatch smoke,main,steps 335   -- re-submitting continues the work
SUITES=${1:-main,steps}
BUDGET=${2:-335}
cd $HOME/sfproj || exit 1
module load anaconda
eval "$(conda shell.bash hook)"
conda activate sf
echo "Host: $(hostname)  Job: $SLURM_JOB_ID  Start: $(date)  suites=$SUITES budget=$BUDGET"
nvidia-smi --query-gpu=name,memory.total --format=csv
python -u m2/run_m1.py --wan_repo $HOME/sfproj/Wan2.1 --ckpt $HOME/sfproj/Self-Forcing/wan_models/Wan2.1-T2V-1.3B \
  --prompts m2/prompts_expanded_v1.jsonl --subsets outputs/subsets.json --out outputs_m1 \
  --suites "$SUITES" --budget-minutes "$BUDGET"
echo "End: $(date) exit=$?"
