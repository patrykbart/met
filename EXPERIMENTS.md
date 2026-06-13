# Experiments log — Met / VISART fork

Running lab notebook: what we've done, the exact settings, results, and how to continue.
Goal: beat the paper's best single model (**R18-SWSL Con-Syn+Real-closest, GAP 36.1**) by adding a
synthetic gallery phone-photo dataset (+ a new method). Plan & targets-to-beat in `reference/README.md`.

_Last updated: 2026-06-13 (EXP-14 done incl. `noframe` leave-one-out — randomization **inverts** on the full benchmark, frozen-room synth-only ties the all-real model at fGAP 36.09; EXP-13 from-scratch run still training)._

## Status snapshot

| Step | What | State |
|---|---|---|
| Eval pipeline | Validate our eval reproduces the paper | ✅ GAP 36.10 / 52.41 / 55.03 on the authors' descriptors |
| 1 | Reproduce best model from scratch (full benchmark) | ✅ **GAP 35.97** / 52.14 / 54.64 (= paper 36.1) |
| 2 | Same model, test only on paintings | ✅ GAP⁻ 67.86 / ACC 69.59 (148 paintings, `Classification=="Paintings"`) |
| 3 | Same model, retrieve synthetic gallery images | ✅ done — exposes a **camera-rig framing bug** (EXP-3) |
| 4 | Train/fine-tune **with synthetic data**, eval on real paintings | ✅ clean **from-scratch +synth = GAP 38.15** (+2.18 over step 1, +2.71 paint ACC), **beats paper 36.1** — synthetic data helps on its own |
| 5 | New method | 🟡 **DINOv3 + geometric re-rank** (EXP-6): ViT-L+gate **GAP 53.07** (+4.9 over DINOv3 ZS, both GAP/GAP⁻ up); cross-domain mining (additional exp) running (7332307) |
| 6 | **Dataset v2** (GN-randomized scene, arc cameras) — rerun EXP-3/7/8/4 | 🟡 EXP-10/11/12 ✅ (v2 ≥ v1 as training data everywhere; synthall **beats the all-real 397k model** on GAP⁻/ACC); EXP-13: FT-synth **GAP 38.99**, FT-combined 38.66 ✅, from-scratch training (7372507) |
| 7 | **Dataset ablation** — which v2 ingredients drive the gain (randomization ladder / resolution / viewpoints) | ✅ EXP-14: randomization **hurts** the open-set benchmark (frozen room best, **fGAP 36.09** ≈ all-real 35.97/paper 36.1); viewpoints are the key ingredient (3≈5 angles, frontal-only collapses); frame variety harmful (leave-one-out `noframe` recovers +0.53 fGAP but still +1.2 below frozen room); 1024² no benchmark gain |

## Headline results
All eval'd identically: multi-scale descriptors, **original 397k studio DB**, real test queries, full K×τ grid.

| Model | Full GAP | GAP⁻ | ACC | Paint GAP⁻ (148) | Paint ACC (148) |
|---|--:|--:|--:|--:|--:|
| Paper R18-SWSL-SRC | 36.1 | 52.4 | 55.0 | — | — |
| Authors' descriptors | 36.10 | 52.41 | 55.03 | — | — |
| **Step 1 — ours, no synthetic** | 35.97 | 52.14 | 54.64 | 67.86 | 69.59 |
| Synth-only FT *(confounded)* | **38.61** | 55.59 | 57.83 | **72.42** | **73.65** |
| Combined FT *(confounded)* | 37.38 | 53.99 | 56.33 | 70.54 | 72.30 |
| **From-scratch +synth** *(clean A/B)* | **38.15** | **55.49** | **58.23** | **70.41** | **72.30** |

## Environment (proven, on PCSS Eagle)

- **Cluster/SLURM:** account `pl0896-03`, QOS `normal,tesla`. GPU jobs → `--partition=tesla --gres=gpu:h100:1` (GRES type, **not** `--constraint`). CPU-only → `--partition=standard`.
- **venv (`.venv/`, git-ignored):** built on **Python 3.9** (faiss-gpu wheels don't cover 3.13); `venv` ensurepip is broken here, so:
  ```bash
  python3 -m venv --without-pip .venv
  curl -sS https://bootstrap.pypa.io/pip/3.9/get-pip.py | .venv/bin/python -
  .venv/bin/pip install torch torchvision faiss-gpu-cu12 numpy pillow   # torch 2.8.0+cu128, faiss-gpu-cu12 1.12
  ```
- **faiss runs on CPU.** The prebuilt faiss-gpu wheel has no H100/sm_90 kernels (CUDA err 209), so the GPU-move lines in `knn_classifier.py` (`fit`) and `train_utils.py` (`mine_negatives`) are commented out → CPU `IndexFlatIP` (identical exact-IP results).
- **TORCH_HOME=`data/torch_home`** caches the SWSL hub repo + weights → compute nodes need no internet.
- **torch 2.8 quirk:** loading our checkpoints needs `torch.load(..., weights_only=False)` (they store a numpy scalar) — handled in `extract_descriptors.py` and the `--init_weights` path.

## Data

- **Met dataset** (downloaded): `/mnt/storage_6/project_data/pl0896-03/met-dataset` (397,121 train / 224,408 classes; test 19,319 = 1,003 Met + 10,352 other-art + 7,964 non-art; val 2,165). Wired via git-ignored symlinks `data/images`, `data/ground_truth`. Full layout/schema in `CLAUDE.md` → "Dataset".
- **Painting test set (committed):** `Classification == "Paintings"` (exact) = **4,898** dataset classes and **148** of the 1,003 Met test queries (`data/gt_paint/testset.json`) — the single painting definition used project-wide. The exact field excludes painted *objects* (snuffboxes, miniatures) while keeping Asian-format scrolls/fans. **Val has only 1 painting query** → can't tune k,τ on val (reuse the full-set K=7, τ=50). `scripts/count_paintings.py`.

### Synthetic dataset (the VISART contribution)
- `/mnt/storage_6/project_data/pl0896-03/visart-dataset/` (~8.8 GB). **24,760 images = 4,952 MET paintings × 5 gallery viewpoints** (`front`, `left/right × upper/bottom`), 512² RGBA PNG. Blender/Cycles renders (framed canvas + glass + placard, randomized lighting/floor/camera). Generated by the `visart2026` pipeline.
- **Folder `<idx>` → Met class id** via each folder's `metadata.json` source path (also the manifest `visart2026/data/paintings_train_images_unique.json`). e.g. folder 0 → MET 35155.
- **Coverage:** the synthetic renders cover **all 148 committed painting test queries** (`Classification=="Paintings"`). (`scripts/count_paintings.py` + `data/synth_gen/` lists.)
- ⚠️ **Camera-rig framing bug** (found in EXP-3): the 5 cameras frame paintings very unevenly — `left upper`/`front` well-framed, `right upper` grazing/edge-on (painting barely visible), `*bottom` foreshortened. Fix in the rig + regenerate for clean per-angle results.
- **Training wiring:** `scripts/build_finetune_data.py` creates `data/aug/images/{MET,test_*,SYNTH}` (symlinks) + augmented manifests `data/gt_aug` (studio+synthetic) and `data/gt_synth` (synthetic only). Run with `--im_root data/aug --info_dir data/gt_aug|gt_synth`.

### Synthetic dataset **v2** (regenerated, June 2026)
- `/mnt/storage_6/project_data/pl0896-03/visart-dataset-v2/` (~9 GB). **24,490 images = 4,898 paintings × 5 views** — exactly the committed `Classification=="Paintings"` classes (folders `0..4897`; v1's 54 non-painting extras dropped; folder *i* = the same painting as v1, verified 4,898/4,898). Generated by `visart2026` PR #2 (GN-randomized scene), render jobs 7366197/7366202; the dataset's own `README.md` is authoritative.
- **New camera rig:** 5 cameras on a horizontal eye-height **arc**, named by angle `30/60/90/120/150` (90 ≈ frontal), ± pose jitter — replaces v1's named rig and its `right upper` framing bug. Files `<idx>/0_rgb_<angle>.png` (prefix = frame 0).
- **More randomization** (largely Geometry Nodes): floor material (5, recorded) + wall/roof mapping, frame molding/color, painting size, glass p=0.25, placard position/visibility, light shape/spread, camera jitter (recorded). GN picks (frame color, glass, light, size) are **not** in `metadata.json` — v2's analogue of v1's unrecorded lighting.
- **v2 wiring (suffix `_v2`):** all synthetic-data scripts/jobs take dataset-root + suffix params (defaults = v1) — `data/{aug,gt_aug,gt_synth}_v2`, `data/gt_paint_mix_*_v2`, `data/descriptors/synthetic_v2`, `data/synth_dino_v2`, models `r18SWSL_*_v2`. Reruns: **EXP-10..13** ↔ v1's EXP-3/7/8/4; write-ups in [`experiments-v2/`](experiments-v2/README.md).

## Experiment log

### EXP-0 — eval-pipeline validation (authors' descriptors) ✅
- `.venv/bin/python scripts/eval_fullgrid.py data/authors/descriptors data/ground_truth 512`
- **GAP 36.10 / GAP⁻ 52.41 / ACC 55.03** (best **K=10, τ=50**) = paper (36.1 / 52.4 / 55.0) ✓.
- **Lesson:** `knn_eval.py --autotune` defaults `--k 1` (tunes only τ) → degenerate τ=500, **GAP ≈ 23**. Must sweep the full K grid → `scripts/eval_fullgrid.py`. ACC is unaffected by the bug.

### EXP-1 — step 1, from-scratch reproduction ✅
- Train: `sbatch slurm/train.slurm` → job **7313742** (H100, 10 epochs, 21h36m). R18-SWSL Con-Syn+Real-closest, `--net r18_sw-sup --pretrained --pairs_type new_pos+new_neg --emb_proj --pca`, paper defaults (lr 1e-7, sched 6×0.1, margin 1.8, 64 pairs/batch). **`--net r18_sw-sup` is required** (default `resnet18` → ImageNet model, GAP 32.5).
- Extract+eval: `slurm/extract_eval.slurm` (job 7318393) → best **K=7, τ=50**.
- **Result: GAP 35.97 / GAP⁻ 52.14 / ACC 54.64** vs paper 36.1 / 52.4 / 55.0 (authors' 36.10/52.41/55.03). Reproduced within training variance. ✅

### EXP-2 — step 2, paintings-only test ✅
Step-1 model, reuse **K=7, τ=50**; painting set = the committed **148** queries (`Classification=="Paintings"`, `data/gt_paint/testset.json`). `scripts/eval_paintings.py`.

| queries | GAP (+all 18,316 distractors) | GAP⁻ (no distr) | ACC |
|---|--:|--:|--:|
| full test (1,003 Met) | 35.97 | 52.14 | 54.64 |
| **paintings (148, `Classification=="Paintings"`)** | **39.50** | **67.86** | **69.59** |

**Takeaway:** paintings recognized markedly better than the average Met query (ACC 69% vs 55%) — the tractable, high-value subset. This is the paintings baseline steps 3–4 must beat.

### EXP-3 — step 3, synthetic gallery images as queries ✅ (with caveat)
Step-1 model; the 24,760 synthetic renders as queries (correct = source Met class), scored the paper's way — **GAP / GAP⁻ / ACC** + **recall@k**, per camera angle — against **two databases**: **(A)** the full 397k studio DB and **(B)** a paintings-only DB (12,403 photos / 4,898 `Classification=="Paintings"` classes). GAP borrows the 18,316 real distractors (open-set); GAP⁻ excludes them (= closed-world GAP); R@1 == ACC. `slurm/synth_eval.slurm` (`scripts/extract_synthetic.py` → `eval_synthetic_retrieval.py` recall@k / `eval_synthetic_gap.py` Table A GAP / `eval_painting_db.py` Table B).

**(A) full 397k studio DB:**

| angle | N | GAP | GAP⁻ | ACC | R@1 | R@5 | R@10 |
|---|--:|--:|--:|--:|--:|--:|--:|
| ALL angles | 24,760 | 31.42 | 33.69 | 36.93 | 36.93 | 43.97 | 46.81 |
| left upper | 4,952 | 65.49 | 74.72 | 75.85 | 75.85 | 83.10 | 85.20 |
| front | 4,952 | 54.54 | 62.55 | 64.84 | 64.84 | 74.64 | 78.68 |
| right bottom | 4,952 | 9.58 | 16.23 | 21.95 | 21.95 | 31.10 | 34.77 |
| left bottom | 4,952 | 9.57 | 15.43 | 20.62 | 20.62 | 28.55 | 31.91 |
| right upper | 4,952 | 0.02 | 0.09 | 1.41 | 1.41 | 2.48 | 3.49 |

**(B) paintings-only DB** (12,403 / 4,898 cls; synthetic queries = the 24,490 in-DB renders): uniformly easier — all-angles ACC 36.93→**45.15**, front 64.84→**78.99**; the real painting queries rise to GAP⁻ 71.48 / ACC 72.30. Full per-angle Table B in the write-up. **Not** comparable to (A) or the paper's 36.1.

⚠️ **Camera-framing artifact, not a clean domain result.** Verified by viewing renders: `right upper` is edge-on/grazing (painting a barely-visible sliver → 1.41%); `left upper`/`front` are well-framed (65–76% R@1). So the per-angle spread tracks framing, and "all angles" 36.93 is dragged down by the broken views. **Cross-check:** the well-framed synthetic **front R@1 64.84** approaches **step-2 real-photo ACC 69.59** — a well-framed render is about as recognizable as a real photo. **Action:** fix the `right upper` (and `*bottom`) camera poses + regenerate, then re-run for a clean per-angle measurement.

**Standalone write-up: [`experiments/renders-as-queries/`](experiments/renders-as-queries/README.md)** — recall@k re-ran as job 7342800 (`slurm/synth_eval.slurm`, ~7 min H100), GAP/GAP⁻/ACC as jobs 7342973 (full DB) / 7342987 (paint DB); numbers reproduced exactly. Numbers trace to `data/descriptors/synthetic/{retrieval_summary,gap_summary,painting_db_summary}.json`.

### EXP-4 — step 4, train/fine-tune WITH synthetic data ✅
Question: does adding the synthetic gallery data to training improve recognition of **real painting photos** (step-2 set) without hurting the rest? Eval DB stays the original studio set → directly comparable.

| Run | Recipe | Data | Job(s) | Status |
|---|---|---|---|---|
| Baseline | from-SWSL, 10 ep | studio only | (step 1) | ✅ GAP 35.97 |
| Synth-only FT | fine-tune ep10, 5 ep @1e-7 | synthetic only | train 7330026 / eval 7330036 | ✅ |
| Combined FT | fine-tune ep10, 5 ep @1e-7 | studio + synthetic | train 7330025 / eval 7332888 | ✅ GAP 37.38 |
| **From-scratch +synth** | from-SWSL, 10 ep | studio + synthetic | train 7330059 / eval 7342026 | ✅ **GAP 38.15** — clean A/B vs step 1 |

Fine-tunes load epoch-10 weights via **`--init_weights`** (added to `train_contrastive.py`) with a fresh optimizer @ **LR 1e-7** — a literal `--resume` runs at the decayed **1e-8 ≈ frozen** (no-op). `slurm/finetune.slurm <combined|synth>`; from-scratch is `slurm/train_synth.slurm`. Eval: `slurm/extract_eval_ft.slurm <variant>`.

**Synth-only FT result** (original studio DB + real queries; best K=15, τ=50):

| | Full GAP | GAP⁻ | ACC | Paint GAP⁻ (148) | Paint ACC (148) |
|---|--:|--:|--:|--:|--:|
| Baseline (step 1) | 35.97 | 52.14 | 54.64 | 67.86 | 69.59 |
| **Synth-only FT** | **38.61** | **55.59** | **57.83** | **72.42** | **73.65** |
| Δ | +2.64 | +3.45 | +3.19 | +4.56 | +4.06 |

**Combined FT result** (studio + synthetic together; best K=5, τ=100):

| | Full GAP | GAP⁻ | ACC | Paint GAP⁻ (148) | Paint ACC (148) |
|---|--:|--:|--:|--:|--:|
| Baseline (step 1) | 35.97 | 52.14 | 54.64 | 67.86 | 69.59 |
| **Combined FT** | 37.38 | 53.99 | 56.33 | 70.54 | 72.30 |
| Δ | +1.41 | +1.85 | +1.69 | +2.68 | +2.71 |

Combined FT gains *less* than synth-only (+1.4 vs +2.6 full
GAP) — plausibly because mixing 397k studio + 25k synthetic dilutes the synthetic signal per epoch.
Both FTs share the same LR-rewarm + extra-epochs confound, so neither is the headline.

**From-scratch +synth result — the clean A/B** (identical 10-epoch recipe to step 1, only +synthetic — no LR rewarm, no extra epochs; best K=5, τ=50):

| | Full GAP | GAP⁻ | ACC | Paint GAP⁻ (148) | Paint ACC (148) |
|---|--:|--:|--:|--:|--:|
| Baseline (step 1) | 35.97 | 52.14 | 54.64 | 67.86 | 69.59 |
| **From-scratch +synth** | **38.15** | **55.49** | **58.23** | **70.41** | **72.30** |
| Δ | **+2.18** | **+3.35** | **+3.59** | **+2.55** | **+2.71** |

**Finding (✅ confound resolved):** synth-only fine-tuning improved *everything*, including **non-paintings** (+2.6 full GAP) — surprising; the prediction was forgetting. Because non-paintings also rose, it's not pure forgetting — the diverse renders taught **transferable lighting/viewpoint/glass invariance** that helps real photos broadly. The two FTs re-warmed LR (1e-8→1e-7) + trained 5 extra epochs, so part of their gain could have been extra training, not synthetic. **The clean A/B settles it:** the from-scratch +synth run (identical 10-epoch step-1 recipe, only +synthetic) still gains **+2.18 full GAP / +3.59 ACC / +2.71 paint ACC** over step 1, landing at **GAP 38.15** — within 0.5 of the confounded synth-only FT (38.61). So the lift is **real and attributable to the synthetic data itself**, not to extra training, and it clears the paper's best single model (36.1). This is the headline result for contribution 1.

### EXP-5 — step 5, new method 🟡
**Additional experiment — cross-domain pair mining.** New `--pairs_type cross_domain_pos+new_neg`
(`code/utils/datasets.py`): each anchor's positive is the *closest same-class sample in the OTHER
domain* (studio↔synthetic) via `mine_positive` on the cross-domain subset; classes with no
cross-domain partner fall back to standard `new_pos`, singletons self-pair. Directly optimizes the
studio→gallery-photo bridge. Fires for the **4,952** painting classes that have both a studio and a
synthetic image (verified on `data/gt_aug`). Mining the *closest* render also sidesteps the broken
grazing-view renders (they sit far from the studio image). Recipe = `slurm/train_synth.slurm` (from-SWSL,
10 ep, combined manifest) with only the pairs_type changed → **clean A/B vs scratch+synth** (job
7330059). Job `slurm/train_crossdomain.slurm` → **7332307 (queued**, waiting on a GPU). Mining logic
smoke-tested (tiny mixed manifest, asserts cross/fallback/singleton/negatives). Eval TBD (epoch-10
ckpt → `data/models/r18SWSL_crossdomain/`; needs the same eval tweak as scratch+synth, not the
`slurm/extract_eval_ft.slurm` epoch-5 globber).

**Main method:** see EXP-6 (pivoted to DINOv3 + geometric re-rank).

### EXP-6 — DINOv3 backbone + geometric re-rank (the new-method thread) 🟡
Pivoted step 5 to a foundation backbone after finding DINOv3 already ~doubles the R18 GAP. All
eval'd in OUR pipeline (canonical kNN-softmax GAP). DINOv3 features **reused** from the sibling
`art-research` repo (Met split verified **byte-identical**, md5). **DINOv3 ZS on Met is the DINOv3
paper's own claim** (Met GAP = DINOv2 40.0 **+10.8** ≈ 50.8) — NOT our contribution; our delta is the
geometric re-rank on top.

**Standalone write-up: [`experiments/dinov3-backbone/`](experiments/dinov3-backbone/README.md)** (split out of `training-with-synthetic` so the R18 + synthetic-data story stays separate; EXP-6 keeps its number).

**Step 2 — DINOv3 reproduced in our pipeline** (`scripts/build_dinov3_pkl.py` + `eval_fullgrid.py`,
`slurm/eval_dinov3.slurm` job 7332349): assemble raw DINOv3 CLS feats (aspect512, 4096-d) → our PCAw
4096→512 + faiss kNN + GAP. Reproduces the paper → bridge faithful; earlier k=1 "0.68" was the
degeneracy artifact (our full grid picks K=5).

| backbone (frozen, zero-shot) | GAP | GAP⁻ | ACC |
|---|--:|--:|--:|
| R18-SWSL (our step-1) | 35.97 | 52.14 | 54.64 |
| DINOv3-ViT-L | 48.16 | 72.14 | 77.07 |
| DINOv3-7B (best K=5, τ=20) | 52.11 | 75.46 | 81.95 |

**Step 3 — geometric re-rank for GAP** (C2 PatchMatch = mutual-NN + RANSAC on DINOv3 ViT-L patches,
over the strong CLS top-50; `scripts/patchmatch_rerank.py` job 7332350 → PM scores saved to
`data/rerank/pm_scores_*.npz`). Goal: reject distractors (DINOv3's weak spot — GAP⁻ 72 vs GAP 48 =
24-pt gap). PM separates true (maxPM mean **34**) from distractors (**21**).
- *Take 1 — additive into pre-softmax sim:* **NULL** (tuner picks λ=0). The softmax saturates (conf≈1
  for true AND distractor), washing the signal out.
- *Take 2 — fuse into the CONFIDENCE, ACC frozen* (`scripts/rerank_confidence_fusion.py`,
  `slurm/rerank_fusion.slurm`): **works.**

| DINOv3-ViT-L (Met protocol) | GAP | GAP⁻ | ACC |
|---|--:|--:|--:|
| baseline (CLS kNN) | 48.16 | 72.14 | 77.07 |
| **+ geom gate** `conf·(1+w·maxPM/g0)`, w=0.25 | **53.07** | **74.69** | 77.07 |
| + geom RRF, w=0.25 | 56.15 | 66.40 | 77.07 |

**Gate improves BOTH GAP (+4.9) and GAP⁻ (+2.55), ACC unchanged** — clean distractor rejection, no
tradeoff, and ViT-L+gate (53.07) already edges DINOv3-7B ZS (52.11). RRF's +8 GAP is a val-overfit
mirage (K=1, GAP⁻ −5.7). **Next:** apply gate to the **7B** stage-1 (baseline 52.11) for the headline
— needs ViT-L patches for the ~30% of the 7B top-50 candidate union not yet covered (~49k imgs, a
ViT-L patch-extraction job in the art-research env), then re-match + gate (expect ~57 if +5 transfers).
**Caveats:** ViT-L not yet 7B; val has only ~129 Met queries (noisy K/τ/w tuning); reuses
art-research DINOv3 feats+patches; planar-homography geometry is iffy for 3D (non-painting) exhibits.

### EXP-7 — DINOv3 embedding structure of the synthetic dataset ✅
What organizes the **frozen DINOv3 ViT-L** (aspect512 CLS, L2-norm/cosine) embedding space of the
24,760 synthetic renders — camera angle vs procedural hyperparameters — and how does it sit relative
to the real domains? Real side **reuses art-research's identical ViT-L Met features** (same model +
aspect512 preprocessing): studio sources `MET/<id>/0.jpg` (4,952, paired) + real painting test queries
(**148**, committed `Classification=="Paintings"` — the project-wide def). Pipeline: `scripts/extract_synth_dino.py` (GPU job **7333958**, 24,760×1024
in **2 min @275 img/s**, batched since renders are all 512²) → `scripts/assemble_real_dino.py` +
`scripts/analyze_synth_dino.py` (CPU `slurm/analysis_synth_dino.slurm`, job 7333977). Factors parsed by
`scripts/synth_meta.py`. Artifacts: `data/synth_dino/analysis/{summary.json, 6 PNGs}`.
**Standalone write-up with all figures: [`experiments/dinov3-embedding-analysis/`](experiments/dinov3-embedding-analysis/README.md).**

**1) Content dominates; angle is the strong secondary axis; procedural nuisances are weakly encoded.**

| factor (synthetic only) | lin-probe | kNN | silhouette | chance |
|---|--:|--:|--:|--:|
| camera angle (5) | **0.990** | 0.72 | 0.02 | 0.20 |
| floor material (5) | 0.59 | 0.50 | ~0 | 0.21 |
| placard-x quartile | 0.55 | 0.54 | ~0 | 0.25 |
| canvas aspect quartile *(content-correlated)* | 0.73 | 0.67 | ~0 | 0.25 |

kNN(k=10) neighbour composition — **de-confounded** (floor is randomized **once per painting**, so all
5 views share it → same-painting⇒same-floor; each painting has 1 view/angle → same-painting⇒diff-angle):
**same-painting 0.247** (chance 0.0002, ~1500×); among *different*-painting neighbours **same-angle
0.68** (3.4× chance) vs **same-floor 0.28** (1.4×). Read: DINOv3 keys on **painting identity** (tight
per-artwork clusters in t-SNE; angle near-perfectly *linearly* separable but **not** isolated blobs,
silhouette≈0); **angle** is the dominant cross-painting organizer; **floor/placard barely encoded**.

**2) Clean, large domain gap — and the renders are "too clean".** Linear separability: studio/synth/query
3-way **0.995**; studio↔synth 0.992, studio↔query 0.973, synth↔query 0.973 — synthetic is its **own**
region. Centroid cosine dist: **studio↔query 0.22** (the real gap); synth **front is closest to studio
(0.15)** — *closer than the real queries are* — and **no** synthetic view lands closer to
the real-query centroid than studio already is (front↔query 0.23 ≈ studio↔query 0.22). So in frozen-
DINOv3 space the renders add **viewpoint/glass/lighting variation** but do **not** reproduce the real
phone-photo shift → EXP-4's synthetic gain is more plausibly **augmentation/invariance** than domain-matching.

**3) Framing bug corroborated independently** (DINOv3 + cosine vs EXP-3's R18 retrieval): `right upper`
renders sit far from their **own** studio source (cos **0.44** vs front **0.84**) and far in centroid
space (**0.62** from studio); per-view cos-to-source bottoms out exactly where EXP-3 R@1 does.

**Caveats:** frozen DINOv3 ViT-L (not the FT-R18 of EXP-1/4, nor 7B); L2-normed CLS cosine (not the
eval's train-fit PCAw); light randomization not recoverable from `metadata.json`; mid-view cos-to-source
doesn't perfectly track R@1 (retrievability = *discriminability*, not raw similarity; EXP-3 used R18).

### EXP-8 — real↔synthetic training mix for paintings (closed-world) ✅
Train the painting recognizer on a fixed **12,403-image** budget, varying the **real:synthetic** blend
100:0→0:100 (6 runs, identical step-1 recipe, only the data changes); eval always on the **real** painting
test set, two ways — a **closed painting world** (search only the 12,403 painting photos) and the **full
397k benchmark**. `scripts/build_paintings_mix_data.py` → `slurm/paint_train.slurm`/`slurm/paint_eval.slurm`
(closed, K/τ via 2-fold CV on the 148 since val=1) + `slurm/eval_full.slurm`/`slurm/eval_paint_cls.slurm`
(full; painting slice at fixed K=7/τ=50).

**Painting recognition** — the 148 real painting photos, vs both DBs:

| mix (real:synth) | GAP⁻ (paint DB) | ACC (paint DB) | GAP⁻ (full DB) |
|---|--:|--:|--:|
| 100:0 (all real) | 67.18 | 70.27 | 61.83 |
| 80:20 | 70.56 | 72.97 | 66.09 |
| 60:40 | 70.65 | 72.30 | 67.22 |
| 40:60 | 71.37 | 72.97 | 67.92 |
| 20:80 | 71.24 | 72.30 | 69.62 |
| **0:100 (all synth)** | **72.47** | **73.65** | **70.04** |
| *ref: all-real model (397k)* | *71.62* | *72.30* | *67.86* |

**Full Met benchmark** — all 1,003 Met queries vs the full 397k DB:

| mix (real:synth) | GAP | GAP⁻ | ACC |
|---|--:|--:|--:|
| 100:0 (all real) | 28.83 | 49.08 | 52.14 |
| 80:20 | 30.23 | 50.03 | 52.84 |
| 60:40 | 31.15 | 50.60 | 53.34 |
| 40:60 | 30.38 | 50.74 | 53.54 |
| 20:80 | 30.85 | 50.92 | 53.64 |
| **0:100 (all synth)** | **31.32** | **51.47** | **54.04** |
| *ref: all-real model (397k)* | *35.97* | *52.14* | *54.64* |

(paint DB = the 12,403 painting photos, no distractors so GAP = GAP⁻; full DB = all 397,121, GAP incl. the 18,316 distractors.)

**Synth-only data scaling** (0% real, lifting the 12,403 cap): paint-DB GAP⁻ 72.47→73.73→74.39→**75.09**
at 1×/1.25×/1.5×/all-24,490 renders (no plateau ⇒ data-limited); the full-DB painting GAP⁻
plateaus (70.04→70.81→70.93→70.90).

**Finding:** synthetic gallery renders are **better training material than real studio photos** for this
test — **synth-only (0% real) is the best of the six**, beating the all-real baseline everywhere and the
**full-data 397k model on paintings** (paint-DB 72.47 vs 71.62; full-DB 70.04 vs 67.86) with ~32×
less data and no real painting photo. The win is painting-specific: painting-only training can't reject
the 18k distractors, so full **GAP** stays below the full-data model (31.32 vs 35.97). **Caveats:**
closed-world numbers are **not** comparable to the paper's GAP 36.1 (12k vs 397k DB); 148 test / val=1
(≤~2-pt diffs are noise — trust the all-real→all-synth jump + monotone trend).

**Standalone write-up: [`experiments/real-vs-synthetic-mix/`](experiments/real-vs-synthetic-mix/README.md).**

### EXP-9 — phone-photo augmentation on the synth-only recognizer ✅ (negative result)
Iterates on EXP-8's best synth-only point (all **24,490** renders, 0% real → closed paint-DB GAP⁻
**75.09**). EXP-7 said the renders are "too clean" → inject **phone-capture artifacts as train-time
augmentation** (each at p=0.5, atop the paper's crop+jitter; never applied at eval): **jpeg** (re-encode
q 30–90), **blur** (Gaussian σ 0.1–2 or 5×5 motion), **sensor** (downscale 0.3–0.7× + Gaussian noise
σ 0.01–0.06), **phoneall** (all three). 4 independent runs, otherwise byte-identical recipe/data/seed
(`--aug` flag in `train_contrastive.py` → `build_train_transform` in `code/utils/augmentations.py`;
`base` arm verified == the original transform). Train jobs 7346709/11/13/15 (~1–1.3 h each), closed
evals 7346710/12/14/16.

| aug arm (closed paint DB, 148q, 2-fold CV) | GAP⁻ | ACC | Δ GAP⁻ |
|---|--:|--:|--:|
| **base (no phone aug)** | **75.09** | **76.35** | — |
| +jpeg | 74.87 | 76.35 | −0.22 |
| +sensor (noise+res) | 73.37 | 75.68 | −1.72 |
| +blur | 72.22 | 73.65 | −2.87 |
| +phoneall | 71.26 | 72.97 | −3.83 |

**Finding: no arm beats the baseline; damage is monotone in augmentation aggressiveness.** JPEG = tie
(identical ACC), all-three-stacked = worst (−3.8, beyond the ±2 noise floor). Read: instance-level
painting recognition rides on fine brushwork/texture detail; blur/noise/downscale erase it on *both*
views of every contrastive pair, while the useful invariances (viewpoint/glass/lighting) were already
supplied by the renders (EXP-7) + the base recipe. The remaining studio→phone gap is evidently **not**
pixel-degradation-shaped. Headroom is in more/better renders (EXP-8 scaling + camera-rig fix), not
heavier augmentation. **Full-benchmark confirmation, all arms** (397k DB; jobs 7356779 jpeg,
7358858–60 blur/sensor/phoneall) upholds it on both query sets — full GAP: base 32.68, +jpeg 33.53,
+blur 32.80, +sensor 32.26, +all 30.84; paint-slice GAP⁻ (148q, K=7/τ=50): base **70.90** vs
68.63/67.47/66.89/67.59 — every arm 2.3–4.0 below base on paintings; jpeg's +0.85 GAP (the only
positive delta) is distractor-side, not painting-side. **Caveats:** single seed, 148 photos (fold
halves spread up to 6 pts — sensor 70.41/76.33), one strength schedule (p=0.5, mild–moderate ranges).

**Standalone write-up: [`experiments/phone-photo-augmentation/`](experiments/phone-photo-augmentation/README.md).**

### EXP-10 — renders-as-queries, dataset v2 ✅ (rerun of EXP-3)
v2 renders (24,490) as queries against the step-1 model — full 397k DB + paintings-only DB, identical
protocol/jsons to EXP-3 (`slurm/synth_eval.slurm` job 7372439 + CPU re-scores 7372440; scripts pointed
at v2 via `SYNTH_DS`/`SYNTH_OUT`). **Per-view, full DB (GAP/GAP⁻/ACC):** 90° **64.97/73.21/74.52**;
60°≈120° ~20/28/33; 30°≈150° ~0.1/0.4/2–4; ALL 23.62/25.72/29.36 (v1 ALL: 31.42/33.69/36.93). Paint
DB: 90° **76.06/84.18/84.75** (beats v1's best view 70.97/80.12/80.93 *and* the real 148 queries'
72.30 ACC); real-148 row bit-identical to v1 (sanity ✓). **Finding:** v1's *accidental* one-camera
collapse is replaced by a *by-design*, symmetric arc fall-off — frontal better than any v1 view,
±30° better than v1's bottoms (+12 ACC), ±60° grazing ≈ v1's broken view. All-view averages are
**not** a v1↔v2 quality metric (different viewpoint distributions); per-view + training-side results
are. Write-up: [`experiments-v2/renders-as-queries/`](experiments-v2/renders-as-queries/README.md).

### EXP-11 — DINOv3 embedding analysis, dataset v2 ✅ (rerun of EXP-7)
Frozen ViT-L CLS over the v2 renders + the same studio/real reference clouds (extract 7372542,
analysis 7372543 → `data/synth_dino_v2/analysis/summary.json`; `analyze_synth_dino.py --r1-json` now
overlays EXP-10's per-view R@1). **Identity still dominates** (same-painting kNN 0.26, ~1,600×
chance); **angle weakened as intended** by pose jitter (xpaint same-angle 0.68→0.49; probe 0.99→0.94,
kNN 0.72→0.50); floors/placards ≈ chance; **camera height is encoded** (probe 0.80 — new factor);
GN picks unrecorded/untestable. **Domain gap:** still ~99% separable, but where v1 had *every* view
studio-side, v2's frontal **ties the studio↔real distance** (0.222 vs 0.224) and oblique views lean
real-ward. Per-view cos→studio 0.84/0.69/0.49 reproduces EXP-10's ordering (two-method check, as in
v1). Write-up: [`experiments-v2/dinov3-embedding-analysis/`](experiments-v2/dinov3-embedding-analysis/README.md).

### EXP-12 — real↔synthetic mix, dataset v2 ✅ (rerun of EXP-8)
Same blends/budget/seeds/recipe; only the renders change (manifests `data/gt_paint_mix_*_v2` job
7372438; trainings 7372513–34, ~24–60 min each; closed evals `paint_eval`, full evals `eval_full`
incl. the PAINT148 slice — no separate cls batch needed). 100:0 + references reused from v1.
**Closed GAP⁻:** 67.18 → 71.10/71.97/72.27/72.77/**73.47** (v1 0:100: 72.47) — synth-only still best,
each blend ≥ v1. **Full benchmark:** v2 ≥ v1 on every metric at every blend, margin grows with synth
fraction — 0:100 GAP **32.75** (v1 31.32). **Scaling (synth-only):** v1's plateau is **gone** — full
GAP⁻ 52.37→**53.78**, ACC 54.94→**56.63** at all 24,490 renders, **beating the all-real 397k model**
(52.14/54.64) on both; GAP 34.38 still < 35.97 (distractors). Closed scaling tops at 74.38 (v1 75.09;
within the ±2 noise floor). Oracle audit ≤0.33 everywhere ✓. **Finding:** every EXP-8 conclusion
holds with v2, slightly stronger — and per image the v2 renders are worth *more* as training data,
despite the hard ±60° views. Write-up: [`experiments-v2/real-vs-synthetic-mix/`](experiments-v2/real-vs-synthetic-mix/README.md).

### EXP-13 — training with synthetic, dataset v2 🟡 (rerun of EXP-4; from-scratch still training)
Full-benchmark training with v2 renders added (manifests `data/gt_aug_v2`/`gt_synth_v2`; eval =
`eval_full.slurm`, same protocol as EXP-1/4). Done: **FT-synth GAP 38.99** / GAP⁻ 55.15 / ACC 57.23 /
paint GAP⁻ 72.77 (job 7372509+7372510; v1: 38.61/55.59/57.83/72.42) and **FT-combined GAP 38.66** /
53.73 / 55.53 / 70.02 (7372511+7372512; v1: 37.38/53.99/56.33/70.54) — both beat the paper (36.1) and
their v1 counterparts on GAP. Pending: the **clean from-scratch A/B** (10 epochs studio+v2, job
7372507, eval 7372508; v1: GAP 38.15). Write-up after it lands:
[`experiments-v2/training-with-synthetic/`](experiments-v2/training-with-synthetic/README.md).

### EXP-14 — dataset ablation: what makes the renders work? ✅
Which ingredients of the v2 dataset produce the training gain? Nine rows along three axes —
**procedural-randomization ladder** (`visart-dataset-v2-abl0-none` frozen room → `abl1` +textures →
`abl2` +light → `abl3` +glass → `abl4` +frame → default v2 = +camera jitter; 24,490 renders each),
**render resolution** (`visart-dataset-v2-1024`, default config @1024²), and **viewpoint count**
(default v2 filtered to arc angles {90} = 4,898 / {60,90,120} = 14,694 / all five). Every row gets
**EXP-12's strongest treatment, byte-identical**: synth-only "all renders" manifest (seed-1 shuffle
→ same slot order as `synthall_v2` for full-dataset rows), step-1 recipe (`paint_train.slurm`),
eval closed paint world (`paint_eval.slurm`) + full 397k benchmark incl. PAINT148 (`eval_full.slurm`).
Default-v2 row = EXP-12's `synthall_v2` (closed GAP⁻ 74.38 / full GAP 34.38), **reused**. Manifests:
`scripts/build_ablation_data.py` (job 7397360) → `data/gt_paint_synth_<tag>` + `data/aug_<tag>`
(angle rows reuse `data/aug_v2`); models `r18SWSL_paint_synth_<tag>`. Trains 7397361/64/67/70/73/76
(1024, 8h limit for the 4× decode)/79/82 + paired closed/full evals (…62/63 etc., afterok-chained,
queued behind EXP-13). Angle rows are volume-confounded — read against EXP-12's scaling arm.
**Results (all 24 jobs ✅): the ladder INVERTS on the full benchmark** — fGAP falls 36.09 → 35.56 →
35.13 → 35.29 → 34.35 → 34.38 as randomization is added, so the **frozen-room** (abl0) synth-only
model is the best training material and **ties the all-real 397k model** (fGAP 36.09 vs ours 35.97 /
paper 36.1; fGAP⁻ 54.28 / fACC 56.73 also top) — EXP-12's "painting-only can't reject distractors"
gap was the price of randomization, not of synthetic data. Closed world is flat across the ladder
(73.78–75.32 ≈ noise): the randomization cost is distractor-side. **Frame variety is the one harmful
family** (abl3→abl4: −0.94 fGAP / −2.7 cGAP⁻ / −2.9 pGAP⁻ — frames are an identity cue); glass is
the best closed-world rung (75.32). **Viewpoints dominate:** {60,90,120} ≈ all-5 at 60% data (fGAP
−0.7), frontal-only collapses to the all-real baseline (cGAP⁻ 68.55 / fGAP 28.64). **1024²:** best
closed row (75.41/76.35) but ≈ default on the full benchmark (34.48) — not worth ~3× render cost.
Caveats: single seed/run per rung; ±2 noise on the 148-q slices (the inversion rests on the
monotone 6-point fGAP trend over 1,003 queries). **Leave-one-out `noframe`** (`visart-dataset-v2-noframe`
= default config + `--bake-frames`, render array 7398961 + merge 7398962, chained prep/train/evals
7398963–66): **fGAP 34.91 / fGAP⁻ 54.06 / fACC 56.53 / pGAP⁻ 71.37 / closed 73.91** — dropping just
the frame from the full default recovers +0.53 fGAP (confirming frame hurts from the ladder's other
end), but stays +1.18 below the frozen room (abl0 36.09): frame is one harmful family among several,
not the whole penalty. **The single best training material remains the frozen room, not
default-minus-frame.** Suggested follow-up: rerun EXP-13's full-benchmark trainings (FT-synth 38.99)
with the **abl0 frozen-room** renders. Write-up:
[`experiments-v2/dataset-ablation/`](experiments-v2/dataset-ablation/README.md).

## How to evaluate any model (the reusable recipe)
```bash
# GPU job: extract MS descriptors (original studio DB + real queries) then full-grid + paintings eval
sbatch --job-name=met-fteval-<name> slurm/extract_eval_ft.slurm <combined|synth>   # for fine-tuned ckpts
# or manually for any checkpoint:
.venv/bin/python -m code.examples.extract_descriptors data/descriptors_<name> \
    --net r18_contr_loss_gem_fc_swsl --netpath <ckpt> --ms --info_dir data/ground_truth --im_root data/ --gpuid 0
.venv/bin/python scripts/eval_fullgrid.py data/descriptors_<name>/r18_contr_loss_gem_fc_swsl_ms data/ground_truth 512
.venv/bin/python scripts/eval_paintings.py data/descriptors_<name>/r18_contr_loss_gem_fc_swsl_ms
```

## Gotchas & decisions
- **Eval must tune the full K grid** (`scripts/eval_fullgrid.py`), not the README's `--autotune` (K=1, → GAP ≈ 23).
- `--net r18_sw-sup` for training, `r18_contr_loss_gem_fc_swsl` (+`--netpath`) for extraction.
- faiss on **CPU** (H100 sm_90 gap); identical exact-IP results.
- **Checkpoint load needs `weights_only=False`** (torch 2.8).
- **Fine-tune LR:** load weights via `--init_weights` @ 1e-7; literal `--resume` = decayed 1e-8 = no-op.
- **Eval DB = original 397k studio images** for every run (so all models are comparable; synthetic is added to *training* only, never the DB).
- **Synthetic camera-rig bug (v1):** `right upper` grazing view (R@1 1.4%). **Resolved by the v2 regeneration** (arc rig — EXP-10): no accidental view, but the ±60° arc ends are *by-design* near-zero as queries; per-angle claims should quote 90° + the fall-off, and all-view averages are not comparable across v1/v2.
- GPU = `--gres=gpu:h100:1` on `tesla`. Authors' artifacts under `data/authors/` (separate from our `data/models/`).

## Repo artifacts

**Tracked (in git):**
- SLURM jobs: `slurm/train.slurm` (step 1), `slurm/extract_eval.slurm` (eval step 1), `slurm/finetune.slurm` (fine-tune combined/synth), `slurm/train_synth.slurm` (from-scratch +synth), `slurm/extract_eval_ft.slurm` (eval fine-tuned), `slurm/synth_eval.slurm` (step 3).
- `scripts/eval_fullgrid.py` — full-K-grid eval (every model). `scripts/eval_paintings.py` — paintings-subset eval.
- `scripts/extract_synthetic.py` + `scripts/eval_synthetic_retrieval.py` (recall@k) + `scripts/eval_synthetic_gap.py` + `scripts/eval_painting_db.py` — EXP-3 synthetic retrieval (recall@k + GAP/GAP⁻/ACC; full DB + paintings-only DB).
- `scripts/count_paintings.py` — painting counts (Met Open Access join). `scripts/build_finetune_data.py` — augmented manifests + image-root.
- `scripts/smoke_gpu.py` — GPU + CPU-faiss env smoke test.
- **EXP-6** (DINOv3 backbone + geometric re-rank): `scripts/build_dinov3_pkl.py`, `scripts/extract_dino_ckpt.py`, `scripts/patchmatch_rerank.py`, `scripts/rerank_confidence_fusion.py` (`slurm/eval_dinov3.slurm`, `slurm/rerank_fusion.slurm`, `slurm/ftdino.slurm`, `slurm/eval_dino_ft.slurm`). Run in `.venv-dino`.
- **EXP-7** (DINOv3 embedding-structure analysis): `scripts/synth_meta.py` (per-folder procedural factors), `scripts/extract_synth_dino.py` (`slurm/extract_synth_dino.slurm`), `scripts/assemble_real_dino.py` + `scripts/analyze_synth_dino.py` (`slurm/analysis_synth_dino.slurm`). Run in `.venv-dino` (+ scikit-learn/matplotlib).
- **EXP-8** (real↔synth mixing): `scripts/build_paintings_mix_data.py`, `scripts/eval_paintings_closed.py`, `scripts/eval_paintings_cls.py`, `scripts/plot_mixing_report.py` (`slurm/paint_train.slurm`, `slurm/paint_eval.slurm`, `slurm/eval_full.slurm`, `slurm/eval_paint_cls.slurm`, `slurm/eval_paint148.slurm`).
- **EXP-9** (phone-photo augmentation): phone-artifact transforms + `build_train_transform`/`ARMS` in `code/utils/augmentations.py`, `--aug` flag in `train_contrastive.py`, 4th positional `AUG` in `slurm/paint_train.slurm`, `scripts/plot_phone_aug.py` (montage + results figs).
- `code/`: faiss→CPU patch, `extract_descriptors.py` weights_only fix, `train_contrastive.py` `--init_weights`.
- `reference/README.md` — paper targets + method↔`pairs_type` mapping.
- **EXP-10..13 (dataset v2):** the same scripts/jobs with dataset-root/suffix params (`--syn/--suffix`, `SYNTH_DS`/`SYNTH_OUT`, positional slurm args; defaults = v1) + `scripts/plot_mixing_report.py --v2`; write-ups in `experiments-v2/`.
- **EXP-14** (dataset ablation): `scripts/build_ablation_data.py` (fixed 8-row ladder → manifests + im_roots; reuses `paint_train`/`paint_eval`/`eval_full` slurm jobs).

**Local only (git-ignored `data/`):**
- `data/images`, `data/ground_truth` — dataset symlinks. `data/aug/images`, `data/gt_aug`, `data/gt_synth` — augmented training wiring.
- `data/models/*` — checkpoints (`r18SWSL_con-syn+real-closest` step-1, `r18SWSL_ft_synth`, `r18SWSL_ft_combined`, `r18SWSL_scratch_synth`).
- `data/descriptors*/` — extracted descriptors per model. `data/authors/` — authors' checkpoint + descriptors.
- `data/synth_dino/` — EXP-7: synthetic + real DINOv3 ViT-L feats, records, `analysis/` (summary.json + figures).
- **v2 (suffix `_v2`):** `data/{aug,gt_aug,gt_synth}_v2`, `data/gt_paint_{mix_*,synth*}_v2` manifests; `data/descriptors/synthetic_v2` (EXP-10 jsons), `data/synth_dino_v2` (EXP-11), `data/descriptors{,_full}_*_v2` + `data/models/r18SWSL_{paint_*,ft_*,scratch_synth}_v2` (EXP-12/13).
- **EXP-14:** `data/gt_paint_synth_<tag>` manifests + `data/aug_<tag>` im_roots (tags `abl0..abl4,1024,ang3,ang1`), `data/descriptors{,_full}_synth_<tag>`, `data/models/r18SWSL_paint_synth_<tag>`.
- `data/torch_home/` — cached SWSL weights. `data/MetObjects.csv` (~303 MB). `data/synth_gen/` — painting-class lists.
