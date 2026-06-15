#!/bin/bash
# EXP-15: DINOv3 ViT-L fine-tune on a PAINTING SUBSET, reusing /met's contrastive recipe
# (Embedder + ContrastiveLoss + pair mining + validate), backbone swapped to DINOv3-ViT-L.
# The DINOv3 analogue of slurm/paint_train.slurm: only the data/im_root change between arms;
# eval is always on the real painting queries (slurm/dino_paint_eval.slurm).
#   method=lora -> + LoRA adapters (r=16);  method=head -> frozen backbone, train projector only.
# usage (the three arms):
#   sbatch --job-name=met-dtr-real        slurm/dino_paint_train.slurm data/gt_paint               data/      real        lora
#   sbatch --job-name=met-dtr-synthv2     slurm/dino_paint_train.slurm data/gt_paint_mix_0r100s_v2 data/aug_v2 synthv2     lora
#   sbatch --job-name=met-dtr-synthv2-all slurm/dino_paint_train.slurm data/gt_paint_synthall_v2   data/aug_v2 synthv2_all lora
#SBATCH --account=pl0896-03
#SBATCH --partition=tesla
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=1-00:00:00            # ViT-L @512, 3 epochs; per-epoch full-DB feat extraction dominates
#SBATCH --output=logs/%x-%j.out

INFO="${1:?usage: slurm/dino_paint_train.slurm <info_dir> <im_root> <arm_tag> <lora|head>}"
IMROOT="${2:?usage: slurm/dino_paint_train.slurm <info_dir> <im_root> <arm_tag> <lora|head>}"
ARM="${3:?usage: slurm/dino_paint_train.slurm <info_dir> <im_root> <arm_tag> <lora|head>}"
METHOD="${4:?usage: slurm/dino_paint_train.slurm <info_dir> <im_root> <arm_tag> <lora|head> [cls|patch]}"
READOUT="${5:-cls}"
case "$METHOD" in
  lora) LORA="--lora" ;;
  head) LORA="" ;;
  *) echo "bad METHOD '$METHOD' (want lora|head)"; exit 1 ;;
esac
SUF=""; [ "$READOUT" != "cls" ] && SUF="_${READOUT}"   # cls dirs stay unsuffixed (back-compat); patch -> _patch
OUT="data/models/dinov3L_${METHOD}_${ARM}${SUF}"

cd "$SLURM_SUBMIT_DIR"; mkdir -p logs "$OUT"
# reuse the DINOv3 weights already downloaded by the sibling art-research repo (data, not code)
export HF_HOME=/mnt/storage_6/project_data/pl0896-03/art-research/.cache/huggingface
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
export TORCH_HOME="$SLURM_SUBMIT_DIR/data/torch_home"
echo "host=$(hostname) arm=$ARM method=$METHOD readout=$READOUT info=$INFO im_root=$IMROOT out=$OUT start=$(date)"
nvidia-smi --query-gpu=name --format=csv,noheader

# Paper recipe (Con-Syn+Real-closest mining, FC projector + PCAw) on DINOv3-ViT-L.
# Small physical batch + virtual batching keeps ViT-L @512 within H100 memory; imsize=512 -> 32x32 patches.
# IDENTICAL across the three arms -- only --info_dir / --im_root change.
.venv-dino/bin/python -m code.examples.train_contrastive "$OUT" \
    --net dinov3_vitl16 --pretrained $LORA --emb_proj --pca \
    --pairs_type new_pos+new_neg --imsize 512 \
    --bsize 8 --vbsizemul 8 --epochs 3 --sched_step 2 \
    --seed 0 --dino_readout "$READOUT" --info_dir "$INFO" --im_root "$IMROOT" --gpuid 0
echo "end=$(date)"
