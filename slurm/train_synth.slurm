#!/bin/bash
# Clean ablation: train R18-SWSL Con-Syn+Real-closest FROM SWSL for 10 epochs on
# Met studio + synthetic gallery data. Identical recipe to step-1 (slurm/train.slurm) except
# the augmented training set -> directly comparable to step-1 (GAP 35.97, no synthetic).
# usage: sbatch slurm/train_synth.slurm [info=data/gt_aug] [im_root=data/aug] [out=data/models/r18SWSL_scratch_synth]
#   v2:  sbatch slurm/train_synth.slurm data/gt_aug_v2 data/aug_v2 data/models/r18SWSL_scratch_synth_v2
#SBATCH --account=pl0896-03
#SBATCH --partition=tesla
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=3-00:00:00
#SBATCH --job-name=met-scratch-synth
#SBATCH --output=logs/%x-%j.out

cd "$SLURM_SUBMIT_DIR"; mkdir -p logs data/models
INFO="${1:-data/gt_aug}"; IMROOT="${2:-data/aug}"; OUT="${3:-data/models/r18SWSL_scratch_synth}"
export TORCH_HOME="$SLURM_SUBMIT_DIR/data/torch_home"; export PYTHONUNBUFFERED=1
echo "host=$(hostname) info=$INFO im_root=$IMROOT out=$OUT start=$(date)"
nvidia-smi --query-gpu=name --format=csv,noheader

# Paper defaults (10 epochs, sched_step 6, lr 1e-7, ...) — same as step 1; only --info_dir/--im_root differ.
.venv/bin/python -m code.examples.train_contrastive "$OUT" \
    --net r18_sw-sup --pretrained --pairs_type new_pos+new_neg --emb_proj --pca \
    --seed 0 --info_dir "$INFO" --im_root "$IMROOT" --gpuid 0
echo "end=$(date)"
