# The Met benchmark, extended with synthetic gallery photos

Research fork of [nikosips/met](https://github.com/nikosips/met), the official code of
**"The Met Dataset: Instance-level Recognition for Artworks"** (NeurIPS 2021), being extended for a
**VISART** (Vision for Art workshop) submission. The benchmark: given a visitor's photo, identify
which of **~224k museum-exhibit classes** it shows — or correctly reject it as not in the
collection. Its core difficulty is the **distribution shift** between clean studio training photos
and real visitor photos, plus a long tail (60.8% of classes have a single training image).

**Our contribution is a synthetic dataset of visitor-style gallery photos** — 24,760 Blender
renders of 4,952 Met paintings hung in a virtual gallery — and the finding that simply **adding it
to the training set beats the paper's best single model**, with the task, metrics, training recipe,
and evaluation protocol left completely **unchanged**. The renders attack both difficulties
directly: they look like visitor photos (perspective, frame, glass, gallery lighting), and they
give single-image classes extra views.

## Results so far

Full benchmark: 397k-image database, 19,319 test queries (1,003 real + 18,316 distractors),
multi-scale descriptors, kNN classifier with K and τ tuned on the validation set over the full
grid. **GAP / GAP⁻ / ACC are defined once in [`experiments-v2/README.md`](experiments-v2/README.md)**;
GAP (open-set, distractors included) is the headline metric. The two right columns score only the
**148 painting test queries** — the classes the synthetic data covers — on the same full database.

| Model | GAP | GAP⁻ | ACC | Paint GAP⁻ (148 q) | Paint ACC (148 q) |
|---|--:|--:|--:|--:|--:|
| Paper's best single model (R18-SWSL Con-Syn+Real-closest) | 36.1 | 52.4 | 55.0 | — | — |
| Our from-scratch reproduction of it | 35.97 | 52.14 | 54.64 | 67.86 | 69.59 |
| **+ our synthetic data** (identical recipe, clean A/B) | **38.15** | **55.49** | **58.23** | **70.41** | **72.30** |

The only change between the last two rows is adding the renders to the training set — no new
method, no extra real data. Main write-up:
[`experiments-v2/training-with-synthetic/`](experiments-v2/training-with-synthetic/README.md).
Supporting experiments:

- [`renders-as-queries`](experiments-v2/renders-as-queries/README.md) — can the baseline model (trained
  on real data only) recognize the renders at all? Yes for well-framed views — and this exposed the
  camera-rig framing bug below.
- [`real-vs-synthetic-mix`](experiments-v2/real-vs-synthetic-mix/README.md) — how the real↔synthetic
  training mix and the amount of synthetic data move painting recognition.
- [`phone-photo-augmentation`](experiments-v2/phone-photo-augmentation/README.md) — simulated
  phone-camera artifacts (JPEG / blur / noise) as training augmentation: a clean negative result.

Current status, all experiments, and every number live in **[`EXPERIMENTS.md`](EXPERIMENTS.md)**
(the running lab notebook) and **[`experiments-v2/`](experiments-v2/README.md)** (per-experiment
write-ups).

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

## Repository map

| Path | What |
|---|---|
| [`code/`](code/) | The upstream pipeline — contrastive training → descriptor extraction → kNN eval — lightly patched (CPU faiss on H100, torch 2.8 checkpoint loading) |
| [`scripts/`](scripts/) | This fork's tooling: [`eval_fullgrid.py`](scripts/eval_fullgrid.py) (the canonical full-K×τ-grid eval), [`build_finetune_data.py`](scripts/build_finetune_data.py) (wires synthetic data into training), painting-subset and synthetic-retrieval evals |
| [`slurm/`](slurm/) | Batch jobs for the PCSS Eagle cluster — [`train.slurm`](slurm/train.slurm) reproduces the paper's best model |
| [`experiments-v2/`](experiments-v2/README.md) | Per-experiment write-ups (the canonical v2 set); its README defines the task, metrics, and protocol once |
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

# 1b) The same recipe WITH the synthetic data (after scripts/build_finetune_data.py)
#     --info_dir data/gt_aug --im_root data/aug   → see slurm/train_synth.slurm

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
