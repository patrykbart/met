#!/bin/bash
# EXP-15: evaluate a fine-tuned DINOv3 painting model EXACTLY as real-vs-synthetic-mix does,
# in both settings. Eval data is ALWAYS the real painting queries; only the model's training
# data differed across arms. Extraction in .venv-dino (transformers); eval in the proven .venv.
#   1) CLOSED painting world: extract over the 12,403 real painting DB + 148 painting queries
#      (data/gt_paint), then 2-fold-CV eval (val=1) -> GAP- / ACC.
#   2) FULL Met benchmark: extract over the original 397k studio DB + 1,003 real queries
#      (data/ground_truth), then full K x tau grid (val-tuned) -> GAP / GAP- / ACC, plus the
#      painting slice (148) at the FIXED K=7/tau=50 used project-wide for cross-model comparison.
# usage: sbatch --job-name=met-dev-real slurm/dino_paint_eval.slurm data/models/dinov3L_lora_real real lora
#SBATCH --account=pl0896-03
#SBATCH --partition=tesla
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=8:00:00
#SBATCH --output=logs/%x-%j.out

MODELDIR="${1:?usage: slurm/dino_paint_eval.slurm <model_dir> <arm_tag> <lora|head> [cls|patch]}"
ARM="${2:?usage: slurm/dino_paint_eval.slurm <model_dir> <arm_tag> <lora|head> [cls|patch]}"
METHOD="${3:?usage: slurm/dino_paint_eval.slurm <model_dir> <arm_tag> <lora|head> [cls|patch]}"
READOUT="${4:-cls}"
LORA=""; [ "$METHOD" = "lora" ] && LORA="--lora"
SUF=""; [ "$READOUT" != "cls" ] && SUF="_${READOUT}"   # cls dirs stay unsuffixed; patch -> _patch

cd "$SLURM_SUBMIT_DIR"; mkdir -p logs
export HF_HOME=/mnt/storage_6/project_data/pl0896-03/art-research/.cache/huggingface
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
export TORCH_HOME="$SLURM_SUBMIT_DIR/data/torch_home"

CKPT=$(ls $MODELDIR/*_epoch:_3 2>/dev/null | head -1)
CLOSED="data/descriptors_dino_paint_${ARM}${SUF}"
FULL="data/descriptors_dino_full_${ARM}${SUF}"
echo "arm=$ARM method=$METHOD readout=$READOUT lora=[$LORA] ckpt=$CKPT start=$(date)"
[ -z "$CKPT" ] && { echo "ERROR: no epoch-3 checkpoint in $MODELDIR/"; exit 1; }
nvidia-smi --query-gpu=name --format=csv,noheader

echo "=== [closed] extract over 12,403 painting DB + 148 painting queries (.venv-dino) ==="
.venv-dino/bin/python scripts/extract_dino_ckpt.py --net dinov3_vitl16 $LORA --dino-readout "$READOUT" \
    --netpath "$CKPT" --out-dir "$CLOSED" --info-dir data/gt_paint --im-root data/ \
    || { echo "closed extract failed"; exit 1; }
echo "=== [closed] painting world eval, 2-fold CV (.venv) ==="
.venv/bin/python scripts/eval_paintings_closed.py "$CLOSED" data/gt_paint 512

echo "=== [full] extract over original 397k studio DB + 1,003 real queries (.venv-dino) ==="
.venv-dino/bin/python scripts/extract_dino_ckpt.py --net dinov3_vitl16 $LORA --dino-readout "$READOUT" \
    --netpath "$CKPT" --out-dir "$FULL" --info-dir data/ground_truth --im-root data/ \
    || { echo "full extract failed"; exit 1; }
echo "=== [full] benchmark eval (PCAw 1024->512 + K x tau grid, val-tuned), .venv ==="
.venv/bin/python scripts/eval_fullgrid.py "$FULL" data/ground_truth 512 | tee "$FULL/fullgrid.log"
echo "=== [full] painting slice (148) at fixed K=7 tau=50 (project-wide cross-model point) ==="
.venv/bin/python scripts/eval_paintings.py "$FULL" data/ground_truth 7 50 512
echo "end=$(date)"
