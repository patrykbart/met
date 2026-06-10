# The Met benchmark, extended — synthetic gallery photos + a geometric re-rank

Research fork of [nikosips/met](https://github.com/nikosips/met), the official code of
**"The Met Dataset: Instance-level Recognition for Artworks"** (NeurIPS 2021), being extended for a
**VISART** (Vision for Art workshop) submission. The benchmark: given a visitor's photo, identify
which of **~224k museum-exhibit classes** it shows — or correctly reject it as not in the
collection. Its core difficulty is the **distribution shift** between clean studio training photos
and real visitor photos, plus a long tail (60.8% of classes have a single training image).

We keep the task, metrics, and evaluation protocol **unchanged** — every number below is directly
comparable to the original paper — and add two contributions on top:

1. **A synthetic dataset of visitor-style gallery photos.** 24,760 Blender renders of 4,952 Met
   paintings hung in a virtual gallery. Adding it to training **beats the paper's best single model
   with no method changes**.
2. **A new retrieval method** (in progress): a frozen **DINOv3** backbone with a **geometric
   re-rank** that fixes its weak spot — distractor rejection.

![GAP progression: paper 36.1 → +synthetic 38.15 → DINOv3+re-rank 53.07](experiments/dinov3-backbone/figures/progression.png)

## Results so far

Full benchmark: 397k-image database, 19,319 test queries (1,003 real + 18,316 distractors),
multi-scale descriptors, kNN classifier with K and τ tuned on the validation set over the full
grid. **GAP / GAP⁻ / ACC are defined once in [`experiments/README.md`](experiments/README.md)**;
GAP (open-set, distractors included) is the headline metric.

| Model | GAP | GAP⁻ | ACC | Write-up |
|---|--:|--:|--:|---|
| Paper's best single model (R18-SWSL Con-Syn+Real-closest) | 36.1 | 52.4 | 55.0 | [`reference/`](reference/README.md) |
| Our from-scratch reproduction of it | 35.97 | 52.14 | 54.64 | [`training-with-synthetic`](experiments/training-with-synthetic/README.md) |
| **+ our synthetic data** (same recipe, clean A/B) | **38.15** | 55.49 | 58.23 | [`training-with-synthetic`](experiments/training-with-synthetic/README.md) |
| DINOv3 ViT-L, frozen, zero-shot kNN | 48.16 | 72.14 | 77.07 | [`dinov3-backbone`](experiments/dinov3-backbone/README.md) |
| **DINOv3 ViT-L + our geometric re-rank** | **53.07** | 74.69 | 77.07 | [`dinov3-backbone`](experiments/dinov3-backbone/README.md) |

DINOv3's zero-shot strength on Met is the DINOv3 paper's own result; our delta is the re-rank on
top — **+4.9 GAP and +2.6 GAP⁻ at unchanged accuracy**, i.e. pure distractor rejection.

Current status, all experiments, and every number live in **[`EXPERIMENTS.md`](EXPERIMENTS.md)**
(the running lab notebook) and **[`experiments/`](experiments/README.md)** (per-experiment
write-ups).

## Repository map

| Path | What |
|---|---|
| [`code/`](code/) | The upstream pipeline — contrastive training → descriptor extraction → kNN eval — lightly patched (CPU faiss on H100, torch 2.8 checkpoint loading) |
| [`scripts/`](scripts/) | This fork's tooling: [`eval_fullgrid.py`](scripts/eval_fullgrid.py) (the canonical full-K×τ-grid eval), [`build_finetune_data.py`](scripts/build_finetune_data.py) (wires synthetic data into training), painting-subset evals, the DINOv3 + re-rank pipeline |
| [`slurm/`](slurm/) | Batch jobs for the PCSS Eagle cluster — [`train.slurm`](slurm/train.slurm) reproduces the paper's best model |
| [`experiments/`](experiments/README.md) | Per-experiment write-ups; its README defines the task and the metrics once |
| [`EXPERIMENTS.md`](EXPERIMENTS.md) | Running lab notebook: status snapshot, exact commands, job ids, all results |
| [`reference/`](reference/README.md) | LaTeX source of the original paper + a summary of the result tables to beat |
| [`CLAUDE.md`](CLAUDE.md) | Working notes: HPC environment, storage policy, dataset layout and its gotchas |
| `data/` | Git-ignored symlinks to the datasets plus generated models/descriptors |

## Running the code

Everything runs as **Python modules from the repo root** (`python -m code.examples.…`, absolute
imports) inside a repo-local `.venv` — on this cluster, always via SLURM. The proven environment
recipe (Python 3.9 venv, torch 2.8 cu128, faiss — and why faiss runs on CPU here) is in
[`EXPERIMENTS.md`](EXPERIMENTS.md) → Environment and [`CLAUDE.md`](CLAUDE.md).

```bash
# 1) Train the paper's best single model (R18-SWSL, con-syn+real-closest pairs)
sbatch slurm/train.slurm
# …which runs:
.venv/bin/python -m code.examples.train_contrastive ./data/models/<run_name> \
    --net r18_sw-sup --pretrained --pairs_type new_pos+new_neg --emb_proj --pca \
    --seed 0 --info_dir ./data/ground_truth --im_root ./data/ --gpuid 0

# 2) Extract multi-scale descriptors from a checkpoint
.venv/bin/python -m code.examples.extract_descriptors ./data/descriptors \
    --net r18_contr_loss_gem_fc_swsl --netpath ./data/models/<checkpoint> --ms \
    --info_dir ./data/ground_truth --im_root ./data/ --gpuid 0

# 3) Evaluate (GAP / GAP⁻ / ACC) with the full K×τ grid
.venv/bin/python scripts/eval_fullgrid.py ./data/descriptors/<exp_name> ./data/ground_truth 512
```

> ⚠️ Evaluate with [`scripts/eval_fullgrid.py`](scripts/eval_fullgrid.py), not the upstream
> `knn_eval --autotune`: autotune sweeps τ at K=1 only and under-reports any model whose best K > 1
> (details in [`EXPERIMENTS.md`](EXPERIMENTS.md)).

The Met dataset itself comes from the [official webpage](http://cmp.felk.cvut.cz/met/) and is wired
in via the git-ignored `data/images` / `data/ground_truth` symlinks — [`CLAUDE.md`](CLAUDE.md) →
Dataset documents the layout and the `images/` path gotcha.

## The synthetic dataset

**24,760 images = 4,952 Met paintings × 5 gallery viewpoints**, rendered with Blender/Cycles: each
painting hung as a framed, glass-covered canvas with a placard, with randomized lighting, floor
material, and camera pose. Folders map to Met class ids via their `metadata.json`, and the renders
cover **all 148 painting test queries** of the benchmark.
[`scripts/build_finetune_data.py`](scripts/build_finetune_data.py) builds the augmented training
manifests (`data/gt_aug`, `data/gt_synth`) consumed by the training runs.

On the cluster it lives at `/mnt/storage_6/project_data/pl0896-03/visart-dataset/` (~8.8 GB, not
yet public). ⚠️ Known issue: one of the five camera rigs (`right upper`) frames paintings poorly
([`EXPERIMENTS.md`](EXPERIMENTS.md) → EXP-3); a rig fix + regeneration is planned.

## Upstream & citation

All credit for the dataset and the original code to the Met-dataset authors:
**[official webpage](http://cmp.felk.cvut.cz/met/)** (images, ground truth,
[pretrained models](http://cmp.felk.cvut.cz/met/#models),
[pre-extracted descriptors](http://cmp.felk.cvut.cz/met/#descriptors)) ·
**[upstream repo](https://github.com/nikosips/met)** · MIT [license](LICENSE).

```bibtex
@inproceedings{ypsilantis2021met,
  title     = {The Met Dataset: Instance-level Recognition for Artworks},
  author    = {Ypsilantis, Nikolaos-Antonios and Garcia, Noa and Han, Guangxing and
               Ibrahimi, Sarah and van Noord, Nanne and Tolias, Giorgos},
  booktitle = {NeurIPS Datasets and Benchmarks Track},
  year      = {2021}
}
```
