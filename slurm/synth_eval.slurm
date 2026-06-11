#!/bin/bash
# Step 3: extract synthetic-image descriptors (GPU) then retrieve vs the Met DB (CPU).
# usage: sbatch slurm/synth_eval.slurm [dataset_root] [out_dir]   (defaults = v1)
#   v2:  sbatch slurm/synth_eval.slurm /mnt/storage_6/project_data/pl0896-03/visart-dataset-v2 data/descriptors/synthetic_v2
#SBATCH --account=pl0896-03
#SBATCH --partition=tesla
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=2:00:00
#SBATCH --job-name=met-synth-eval
#SBATCH --output=logs/%x-%j.out

cd "$SLURM_SUBMIT_DIR"; mkdir -p logs
export SYNTH_DS="${1:-/mnt/storage_6/project_data/pl0896-03/visart-dataset}"
export SYNTH_OUT="${2:-data/descriptors/synthetic}"
export TORCH_HOME="$SLURM_SUBMIT_DIR/data/torch_home"; export PYTHONUNBUFFERED=1
echo "host=$(hostname) ds=$SYNTH_DS out=$SYNTH_OUT start=$(date)"
nvidia-smi --query-gpu=name --format=csv,noheader

echo "=== extract synthetic descriptors (multi-scale) ==="
.venv/bin/python scripts/extract_synthetic.py
echo "=== retrieval vs Met DB (recall@k, all angles + per angle) ==="
.venv/bin/python scripts/eval_synthetic_retrieval.py
echo "end=$(date)"
