#!/bin/bash
#SBATCH --job-name=m2_vbench
#SBATCH --partition=MGPU-TC2
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --exclude=TC2N01,TC2N02
#SBATCH --cpus-per-task=8
#SBATCH --mem=28G
#SBATCH --time=05:50:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# usage: sbatch submit_vbench.sbatch "main" "<dimensions>" 1     (third argument 1 = also run the batch-vs-single check)
SETS=${1:-"main"}
DIMS=${2:-"subject_consistency motion_smoothness dynamic_degree imaging_quality aesthetic_quality"}
EQ=${3:-0}
cd $HOME/sfproj || exit 1
module load anaconda
eval "$(conda shell.bash hook)"
echo "Host: $(hostname)  Job: $SLURM_JOB_ID  Start: $(date)  sets=$SETS dims=$DIMS"
nvidia-smi --query-gpu=name,memory.total --format=csv
conda activate sf
python m2/prep_eval_sets.py --procs 6
conda deactivate
conda activate p1_eval
export OMP_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false
export VBENCH_CACHE_DIR=$HOME/sfproj/p1/model_cache/vbench HF_HOME=$HOME/sfproj/p1/model_cache/huggingface TORCH_HOME=$HOME/sfproj/p1/model_cache/torch
if [ "$EQ" = "1" ]; then
  mkdir -p eval_sets/eq_batch; i=0
  for f in $(ls eval_sets/main | head -3); do
    i=$((i+1)); mkdir -p eval_sets/eq_single_$i
    ln -sf "$(readlink -f eval_sets/main/$f)" eval_sets/eq_batch/$f; ln -sf "$(readlink -f eval_sets/main/$f)" eval_sets/eq_single_$i/$f
  done
  python -u m2/vbench_batch.py --src p1/vendor/VBench --videos eval_sets/eq_batch --out vbench_out --name eq_batch --dims $DIMS
  for j in 1 2 3; do python -u m2/vbench_batch.py --src p1/vendor/VBench --videos eval_sets/eq_single_$j --out vbench_out --name eq_single_$j --dims $DIMS; done
  python m2/vbench_eqcheck.py
fi
for S in $SETS; do
  echo "=== $S ==="
  python -u m2/vbench_batch.py --src p1/vendor/VBench --videos eval_sets/$S --out vbench_out --name $S --dims $DIMS
done
echo "End: $(date)"
