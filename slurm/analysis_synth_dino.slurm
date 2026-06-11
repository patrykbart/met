#!/bin/bash
# CPU job: assemble the real DINOv3 ViT-L reference clouds (reuse art-research feats)
# then analyze synthetic embedding structure (angle vs floor) + domain separation.
# Chained after the synthetic extraction job via --dependency=afterok.
# usage: sbatch slurm/analysis_synth_dino.slurm [synth_root] [dir] [r1_json]   (defaults = v1)
#   v2:  sbatch slurm/analysis_synth_dino.slurm .../visart-dataset-v2 data/synth_dino_v2 \
#          data/descriptors/synthetic_v2/retrieval_summary.json
#SBATCH --account=pl0896-03
#SBATCH --partition=standard
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=2:00:00
#SBATCH --job-name=met-synth-dino-analysis
#SBATCH --output=logs/%x-%j.out

SYNTH_ROOT="${1:-/mnt/storage_6/project_data/pl0896-03/visart-dataset}"
DIR="${2:-data/synth_dino}"
R1JSON="${3:-}"
cd "$SLURM_SUBMIT_DIR"; mkdir -p logs "$DIR"
# keep all caches OFF the 1GB $HOME (matplotlib/fontconfig/etc. -> quota errors there)
export MPLCONFIGDIR="$SLURM_SUBMIT_DIR/data/.mplcache"
export XDG_CACHE_HOME="$SLURM_SUBMIT_DIR/data/.cache"
mkdir -p "$MPLCONFIGDIR" "$XDG_CACHE_HOME"
export PYTHONUNBUFFERED=1
echo "host=$(hostname) root=$SYNTH_ROOT dir=$DIR r1=$R1JSON start=$(date)"

echo "=== assemble real reference features (studio sources + painting queries) ==="
if [ -f "$DIR/real_dino_vitl16_aspect512.npz" ]; then
  echo "real features already exist -> skipping assemble"
else
  .venv-dino/bin/python scripts/assemble_real_dino.py --synth-root "$SYNTH_ROOT" --out-dir "$DIR" \
    || { echo "assemble FAILED"; exit 1; }
fi

echo "=== analyze embedding structure + domain gap ==="
.venv-dino/bin/python scripts/analyze_synth_dino.py --dir "$DIR" ${R1JSON:+--r1-json "$R1JSON"}

echo "end=$(date)"
