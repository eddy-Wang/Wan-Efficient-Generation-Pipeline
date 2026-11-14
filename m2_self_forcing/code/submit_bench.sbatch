#!/bin/bash
#SBATCH --job-name=m2_bench
#SBATCH --partition=MGPU-TC2
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --exclude=TC2N01,TC2N02
#SBATCH --cpus-per-task=8
#SBATCH --mem=28G
#SBATCH --time=02:00:00
# usage: sbatch submit_bench.sbatch [A,B,C,D,E]   (configurations already in the output file are kept)
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

cd $HOME/sfproj/Self-Forcing || exit 1
module load anaconda
eval "$(conda shell.bash hook)"
conda activate sf
echo "Host: $(hostname)  Job: $SLURM_JOB_ID  Start: $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv
export TOKENIZERS_PARALLELISM=false
# torch.compile builds kernels with the cluster gcc; make its newer libstdc++ win over the older anaconda one
export LD_LIBRARY_PATH=/cm/local/apps/gcc/14.2.0/lib64:$LD_LIBRARY_PATH
python ../m2/bench_stream.py --prompts ../m2/prompts_expanded_v1.jsonl --out ../bench/bench_stream.json --n 4 --configs "${1:-A,B,C,D,E}"
echo "End: $(date) exit=$?"
