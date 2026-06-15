#!/bin/bash
# EXP-15 control: ZERO-SHOT DINOv3 ViT-L (frozen, NO fine-tuning, NO projector) evaluated in
# BOTH protocols -- the baseline the three fine-tuned arms are measured against (did FT move the
# needle up or down?). Same mean-pooled-patch representation as the FT arms, minus the trained
# projector, so the only difference vs the FT arms is the fine-tuning itself.
# usage: sbatch --job-name=met-dev-zs slurm/dino_paint_eval_zs.slurm
#SBATCH --account=pl0896-03
#SBATCH --partition=tesla
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=8:00:00
#SBATCH --output=logs/%x-%j.out

cd "$SLURM_SUBMIT_DIR"; mkdir -p logs
export HF_HOME=/mnt/storage_6/project_data/pl0896-03/art-research/.cache/huggingface
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
export TORCH_HOME="$SLURM_SUBMIT_DIR/data/torch_home"

READOUT="${1:-cls}"
SUF=""; [ "$READOUT" != "cls" ] && SUF="_${READOUT}"   # cls dirs stay unsuffixed; patch -> _patch
CLOSED="data/descriptors_dino_paint_zs${SUF}"
FULL="data/descriptors_dino_full_zs${SUF}"
echo "zero-shot DINOv3 eval readout=$READOUT start=$(date)"
nvidia-smi --query-gpu=name --format=csv,noheader

echo "=== [closed] ZS extract over 12,403 painting DB + 148 painting queries (.venv-dino) ==="
.venv-dino/bin/python scripts/extract_dino_ckpt.py --net dinov3_vitl16 --zeroshot --dino-readout "$READOUT" \
    --out-dir "$CLOSED" --info-dir data/gt_paint --im-root data/ \
    || { echo "closed ZS extract failed"; exit 1; }
echo "=== [closed] painting world eval, 2-fold CV (.venv) ==="
.venv/bin/python scripts/eval_paintings_closed.py "$CLOSED" data/gt_paint 512

echo "=== [full] ZS extract over original 397k studio DB + 1,003 real queries (.venv-dino) ==="
.venv-dino/bin/python scripts/extract_dino_ckpt.py --net dinov3_vitl16 --zeroshot --dino-readout "$READOUT" \
    --out-dir "$FULL" --info-dir data/ground_truth --im-root data/ \
    || { echo "full ZS extract failed"; exit 1; }
echo "=== [full] benchmark eval (PCAw 1024->512 + K x tau grid, val-tuned), .venv ==="
.venv/bin/python scripts/eval_fullgrid.py "$FULL" data/ground_truth 512 | tee "$FULL/fullgrid.log"
echo "=== [full] painting slice (148) at fixed K=7 tau=50 ==="
.venv/bin/python scripts/eval_paintings.py "$FULL" data/ground_truth 7 50 512
echo "end=$(date)"
