# AGENTS.md

This file provides guidance to coding agents (Claude Code, Codex, Antigravity CLI) when working with code in this repository.

> **Status (2026-10-25): this repo is a record.** The SynGallery paper was accepted at VISART 2026 and its
> reproduction code is [SynGallery/syngallery-bench](https://github.com/SynGallery/syngallery-bench). Grant
> `pl0896-03` has expired (every SLURM submission is rejected with `AssocMaxSubmitJobLimit`), and the paths
> below under `/mnt/storage_6/project_data/pl0896-03/` (`met`, `met-dataset`, `visart-dataset`) no longer
> exist. To run anything on Eagle now, follow the research vault's `compute/README.md`: GPU jobs charge
> `pl1373-01` on partition `proxima`, CPU jobs charge `pl0896-01`.

## Overview

Official code for **The Met dataset** (NeurIPS paper) — instance-level recognition / image retrieval over ~224k classes of museum exhibits. The pipeline trains a CNN embedding model with contrastive learning, extracts global descriptors, and classifies query images against the training set with a non-parametric kNN classifier. There is no build system, test suite, or linter config — this is a research codebase.

## Research context (this fork)

This is a **fork of [nikosips/met](https://github.com/nikosips/met)** being extended for a paper
submission to **VISART** (the Vision for Art workshop). Two planned contributions, on top of the
original dataset/benchmark:

1. **A synthetic dataset of phone photos of paintings in a gallery.** The hypothesis is that adding
   this data *by itself* improves recognition. This directly attacks the benchmark's core
   difficulty — the **distribution shift** between studio-condition exhibit images (training) and
   real visitor photos (queries), plus the long tail (60.8% of classes have a single training image).
2. **A new method** (details TBD) that improves the numbers further, on top of the synthetic data.

When making changes, the goal is to **beat the original paper's results** (primary baseline:
R18-SWSL Con-Syn+Real-closest, **GAP 36.1**) using the *same task definition, metrics, and
protocol*. Preserve the GAP/GAP⁻/ACC evaluation so results stay comparable.

**The original paper is the main reference.** Its LaTeX source is in [`reference/`](reference/),
and [`reference/README.md`](reference/README.md) summarizes the task, methods, the full result
tables (numbers to beat), and the crucial `--pairs_type` ⇄ paper-method-name mapping. Read it before
designing experiments or writing up results.

**Progress & results live in [`EXPERIMENTS.md`](EXPERIMENTS.md)** (the running lab notebook — read it
first). As of 2026-06: step 1 **reproduced** (GAP 35.97 ≈ paper 36.1), step 2 (paintings-only) and
step 3 (synthetic retrieval) done, step 4 (training **with** synthetic) in progress. The **synthetic
dataset** — 24,760 Blender gallery renders of 4,898→4,952 Met paintings ×5 views — is at
`/mnt/storage_6/project_data/pl0896-03/visart-dataset/` (folder→Met-id via each `metadata.json`; a
known camera-framing bug makes the `right upper` view near-useless). Reusable helpers live in
[`scripts/`](scripts/) (`eval_fullgrid.py` = the correct full-K-grid eval, `eval_paintings.py`,
`extract_synthetic.py` + `eval_synthetic_retrieval.py`, `count_paintings.py`, `build_finetune_data.py`);
SLURM jobs live in [`slurm/`](slurm/).

## HPC environment (PCSS Eagle) — how to run

This repo lives on the **PCSS Eagle** cluster (`eagle.man.poznan.pl`) under grant **`pl0896-03`**, in
the **backed-up `project_data`** area (`/mnt/storage_6/project_data/pl0896-03/met`). Jobs run under
**SLURM**. Docs: [Getting Started](https://help.pcss.plcloud.pl/portal/hpc/2%20Getting%20Started/) ·
[Job Management](https://help.pcss.plcloud.pl/portal/hpc/4%20Job%20Management%20and%20Scheduling/) ·
[Submitting jobs](https://help.pcss.plcloud.pl/portal/hpc/submit/) ·
[Data Management](https://help.pcss.plcloud.pl/portal/hpc/5%20Data%20Management%20and%20Transfer/#data-storage-policies).

**Python — ALWAYS a repo-local `.venv`, never the system/bare Python.** Proven build recipe (pip
downloads are fine on a login node; GPU-smoke-test on a `tesla` node afterwards):

```bash
python3 -m venv --without-pip .venv     # py3.9 base (faiss-gpu has no 3.13 wheels); ensurepip is broken here
curl -sS https://bootstrap.pypa.io/pip/3.9/get-pip.py | .venv/bin/python -    # bootstrap pip
.venv/bin/pip install torch torchvision faiss-gpu-cu12 numpy pillow           # -> torch 2.8.0+cu128
```

Every `python -m code.examples.*` command runs via `.venv/bin/python` (no system Python; no conda
unless asked). **faiss runs on CPU here** — the prebuilt `faiss-gpu` wheel lacks H100/sm_90 kernels,
so the GPU-index lines in `knn_classifier.py`/`train_utils.py` are commented out (identical exact-IP
results). The full proven recipe + results are in [`EXPERIMENTS.md`](EXPERIMENTS.md).

**Never compute on login nodes** — they are for editing, `git`, and `sbatch`/`srun` only.

**SLURM essentials** (always pass the grant with `--account=pl0896-03`):

| Action | Command |
|---|---|
| Interactive GPU shell | `srun --account=pl0896-03 -p <gpu_partition> --gpus-per-node=1 --time=1:00:00 --pty bash` |
| Submit a batch job | `sbatch slurm/train.slurm` |
| My queue | `squeue -u $USER` |
| Cancel | `scancel <jobid>` |
| Post-run efficiency | `seff <jobid>` |
| GPU usage (on the node) | `nvidia-smi`, `nvtop` |

GPU jobs for grant `pl0896-03` (QOS `normal,tesla`) go to **`--partition=tesla`**; choose the GPU by
**GRES type** — `--gres=gpu:h100:1` (H100, fast/plentiful) or `--gres=gpu:tesla:1` (V100) — **not**
`--constraint`. CPU-only jobs → `--partition=standard`. (`proxima` is not in this grant's QOS.)

The committed **[`slurm/train.slurm`](slurm/train.slurm)** reproduces the paper's best single model (R18-SWSL
Con-Syn+Real-closest, target **GAP 36.1**):

```bash
#!/bin/bash
#SBATCH --account=pl0896-03
#SBATCH --partition=tesla
#SBATCH --gres=gpu:h100:1           # GPU by GRES type, NOT --constraint
#SBATCH --cpus-per-task=16          # 8 dataloader workers + CPU-faiss threads
#SBATCH --mem=48G
#SBATCH --time=3-00:00:00           # checkpoints every epoch; resume with --resume
#SBATCH --output=logs/%x-%j.out

cd "$SLURM_SUBMIT_DIR"; mkdir -p logs data/models
export TORCH_HOME="$SLURM_SUBMIT_DIR/data/torch_home"   # cached SWSL weights -> offline
.venv/bin/python -m code.examples.train_contrastive ./data/models/r18SWSL_con-syn+real-closest \
    --net r18_sw-sup --pretrained --pairs_type new_pos+new_neg --emb_proj --pca \
    --seed 0 --info_dir ./data/ground_truth --im_root ./data/ --gpuid 0
```

**`--net r18_sw-sup` is required** for the SWSL model (GAP 36.1) — the default `resnet18` trains the
ImageNet model (GAP 32.5). `--gpuid 0` is correct: SLURM exposes the one granted GPU as device 0 (the
code sets `CUDA_VISIBLE_DEVICES` from `--gpuid`). Submit with `sbatch slurm/train.slurm`. **Eval needs the
full K-grid** — via [`scripts/eval_fullgrid.py`](scripts/eval_fullgrid.py) (the README's `knn_eval --autotune`
under-tunes — it only sweeps τ at K=1) — see [`EXPERIMENTS.md`](EXPERIMENTS.md).

**Storage policy** (quota-managed; see [Data Management]):

- **`$HOME` — 1 GB only**, a "bag" of symlinks to grant dirs. Never put data, the venv, or
  checkpoints here.
- **`project_data`** (this repo): shared, **backed up**, guaranteed by the grant, kept 6 months after
  it ends. Good for code, the `.venv`, and final checkpoints. Deletions linger in
  `.recyclebininternal` ~7 days and still count toward quota.
- **`scratch`** (`$HOME/grant_$SLURM_JOB_ACCOUNT/scratch/$USER/$SLURM_JOB_ID`): large, **not** backed
  up — stage the dataset and write job I/O here.
- **`archive`**: slow, very large; for inactive/processed results.
- **Node-local NVMe** on proxima (`/mnt/local`, request `--constraint=local_ssd --tmp=<size>`): fastest
  for the Met images (heavy random small-file reads); auto-wiped at job end, so copy data in at start.

**Data transfer:** `rsync`/`scp` from a login/interactive node, or
`rclone copy <src> <dst> --progress --multi-thread-streams=8` (≤8 streams) for large parallel copies.

## Running code

> **Run every command below inside the repo-local `.venv`, on a compute node via SLURM — never the
> system Python and never on a login node (see [HPC environment](#hpc-environment-pcss-eagle--how-to-run)).**

All scripts are run as **Python modules from the repository root** (the directory containing `code/`), because every file uses absolute imports rooted at `code.` (e.g. `from code.utils.train_utils import *`). Running a script by path (`python code/examples/knn_eval.py`) will fail with import errors.

```bash
# Train embedding model with contrastive loss (writes checkpoints + per-epoch descriptor pkls to EXPORT_DIR)
python -m code.examples.train_contrastive ./data/models/<run_name> --seed 0 --pretrained \
    --pairs_type new_pos+new_neg --emb_proj --pca --info_dir ./data/ground_truth --im_root ./data/ --gpuid 0

# Extract descriptors with a backbone (writes descriptors.pkl into EXPORT_DIR/<exp_name>/)
python -m code.examples.extract_descriptors ./data/descriptors --net r18_contr_loss_gem_fc_swsl \
    --netpath ./data/models/<checkpoint> --ms --info_dir ./data/ground_truth --im_root ./data/ --gpuid 0

# Evaluate pre-extracted descriptors with the kNN classifier (reads descriptors.pkl from EXPORT_DIR)
python -m code.examples.knn_eval ./data/descriptors/<exp_name>/ --autotune --info_dir ./data/ground_truth/

# Every script supports -h for the full option list
python -m code.examples.<script> -h
```

**Prerequisites** (no `requirements.txt` exists): Python 3, NumPy, PyTorch, torchvision, `faiss-gpu`, PIL. A **CUDA GPU is mandatory** — training, descriptor extraction, and the faiss kNN index all call `.cuda()` unconditionally, and each script sets `CUDA_VISIBLE_DEVICES` from `--gpuid`.

**Data is already downloaded** (full layout, schema, and the `images/` path gotcha are in the
[Dataset](#dataset-local-copy-on-eagle) section below). It lives at
`/mnt/storage_6/project_data/pl0896-03/met-dataset` and is wired into the repo via git-ignored
`data/` symlinks, so the `--info_dir ./data/ground_truth --im_root ./data/` flags in the commands
above work as written: `--info_dir` must contain the GT JSONs, and `--im_root` is the image root from
which the loaders read `<im_root>/images/<path>`.

## Dataset (local copy on Eagle)

The Met dataset is **already downloaded** to `/mnt/storage_6/project_data/pl0896-03/met-dataset`
(also reachable as `~/pl0896-03/project_data/met-dataset`; on backed-up `project_data`, ≈32 GB).
It is **not** committed to the repo. Layout:

```
met-dataset/
├── MET/<class_id>/<n>.jpg   # 224,408 class folders; 397,121 exhibit (training) images, 1–10 per class, named 0.jpg…
├── test_met/<hash>.jpg      #  1,132 Met query photos (visitor photos of exhibits)
├── test_other/<hash>.jpg    # 11,520 other-artwork distractor queries
├── test_noart/<hash>.jpg    #  8,832 non-artwork distractor queries
├── MET_database.json        # training GT      — 397,121 entries
├── mini_MET_database.json   # mini training GT —  38,307 entries / 33,501 classes (use with --mini)
├── testset.json             # test query GT    —  19,319 = 1,003 met + 10,352 other + 7,964 noart
└── valset.json              # val query GT     —   2,165 =   129 met +  1,168 other +   868 noart
```

The three `test_*/` folders physically hold **both** val and test images; the val/test split is
defined solely by which JSON lists each file. All images are `.jpg` (≤500×500).

**Ground-truth JSON schema** (each file is a JSON array of objects):
- `MET_database.json` / `mini_MET_database.json`: `{"id": <class_id int>, "path": "MET/<id>/<n>.jpg"}`.
- `testset.json` / `valset.json`: `{"path": "test_*/<hash>.jpg", ...}`. Met queries carry
  `"MET_id": <class_id>` (+ `photographer`, `url`); **distractors have no `MET_id`** (they carry a
  Wikimedia `category` + `url`) and are mapped to label **`-1`** by `MET_queries`.

**⚠️ Path gotcha + wiring.** The JSON `path` values have **no `images/` segment**, but the loaders
build `<im_root>/images/<path>` (`code/utils/datasets.py`). Two repo-local symlinks (already created
under git-ignored `data/`) bridge this so the documented commands run unchanged:

```
data/images       -> /mnt/storage_6/project_data/pl0896-03/met-dataset   # → data/images/MET/34/0.jpg ✓
data/ground_truth -> /mnt/storage_6/project_data/pl0896-03/met-dataset   # → data/ground_truth/MET_database.json ✓
```

Hence `--im_root ./data/ --info_dir ./data/ground_truth`. If the symlinks ever go missing, recreate:
`ln -sfn /mnt/storage_6/project_data/pl0896-03/met-dataset data/images` (likewise `data/ground_truth`).

**Performance:** training/extraction does heavy random small-file reads over ~397k JPEGs. Reading
straight from `project_data` works, but for real runs stage the dataset onto **node-local NVMe**
(`/mnt/local`, `--constraint=local_ssd --tmp=<size>`) or `scratch` and repoint the symlinks there —
see the HPC storage notes above.

## Architecture

The three example scripts are stages of one pipeline; descriptors are the interchange format, passed between stages as pickled NumPy arrays (`descriptors.pkl` with keys `train_descriptors` / `test_descriptors` / `val_descriptors`).

```
images + GT JSON ──> Embedder (CNN backbone → GeM pool → L2-norm → [optional FC projector])
                       │
                       └─> descriptors ──> PCA-whitening ──> faiss kNN classifier ──> GAP / accuracy
```

**Embedder** (`code/networks/backbone.py`) is the core descriptor extractor: a fully-convolutional ResNet trunk → `GeM` generalized-mean pooling → L2 normalization, optionally followed by an FC `projector` that is L2-normalized again. The projector can be **initialized from PCA-whitening statistics** (`init_projector`), so PCA whitening is used in two distinct places: as train-time FC initialization (`--pca`) and as eval-time descriptor post-processing in `knn_eval.py`. `OUTPUT_DIM` maps architecture name → descriptor dimension. Multi-scale extraction (`extract_ms`, scales `[1, 1/√2, 1/2]`) is selected by `--ms`.

**siamese_network** (`code/networks/SiameseNet.py`) is a thin training wrapper that runs `Embedder` on two augmented views and returns both descriptors for `ContrastiveLoss` (`code/utils/losses.py`, margin-based). Only `.backbone` is used at inference time.

**Datasets** (`code/utils/datasets.py`):
- `MET_database` — training-set images (label = exhibit `id`).
- `MET_queries` — val/test queries; missing `MET_id` becomes label **`-1`, the distractor label** (query depicts no known exhibit).
- `MET_pairs_dataset` — yields `(img1, img2), label` pairs for contrastive training (label 1 = same class, 0 = different).

**Online pair mining is the heart of training.** `MET_pairs_dataset.create_epoch_pairs` is called once per epoch and rebuilds all pairs from the *current* model's descriptors using faiss (`mine_negatives` = hard negatives among nearest neighbors of a different class; `mine_positive` = closest same-class sample). `--pairs_type` selects the strategy: `sim_siam_pos`, `sim_siam_pos+new_neg`, `pos+new_neg`, `new_pos+new_neg`. The **first** epoch mines from ImageNet (or `--init_descr`) descriptors; **subsequent** epochs reuse the descriptors returned by `validate()` (`code/utils/train_utils.py`), which runs a full kNN eval each epoch and feeds its train descriptors back into mining — a tight train↔validate coupling.

**Classifier & metrics:**
- `KNN_Classifier` (`code/classifiers/knn_classifier.py`) — faiss-GPU inner-product search over L2-normalized descriptors; per-class scores are softmax-weighted by temperature `t`, yielding a prediction + confidence. `tune_KNN` grid-searches `K`/`t` on the val set (used by `--autotune`).
- `evaluate` (`code/utils/utils.py`) reports three numbers: **GAP** (Global Average Precision, the primary metric, ranks all queries by confidence and accounts for distractors), **GAP without distractors**, and accuracy (distractors excluded). Distractors are identified by label `-1`.

## Conventions & gotchas

- **Wildcard imports** (`from X import *`) are used throughout; new public helpers in `utils/` become available transitively.
- Training **freezes all BatchNorm layers** to ImageNet running statistics (`set_batchnorm_eval`) because effective batch size per anchor is tiny.
- `--vbsizemul` implements **virtual (gradient-accumulation) batching**; the *effective* batch size is `bsize * vbsizemul`, which is what gets baked into the checkpoint filename.
- Several argparse options (`--bsize`, `--epochs`, `--backbone_lr`) have **no `type=`**, so they arrive as strings and are cast inline (`int(args.bsize)`, `float(args.backbone_lr)`); preserve the casts when editing.
- Checkpoint and descriptor filenames are auto-generated from hyperparameters (long `method:_..._epoch:_N` strings); the consuming script must be pointed at the matching directory.
- `--net` accepts two families in `extract_descriptors.py`: ImageNet/self-supervised backbones built from scratch (e.g. `r18INgem`, `r50_swav_gem`, `r18_sw-sup_gem`) and **checkpoint-loadable** variants requiring `--netpath` (`r18_contr_loss_gem`, `r18_contr_loss_gem_fc`, `r18_contr_loss_gem_fc_swsl`).
