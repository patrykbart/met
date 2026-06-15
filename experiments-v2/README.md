# Experiments — synthetic dataset **v2**

The canonical experiment set for the VISART submission, on **v2 of the synthetic gallery dataset**
(`/mnt/storage_6/project_data/pl0896-03/visart-dataset-v2`). Four are reruns of the original
synthetic-data experiments (renders-as-queries, real↔synthetic mix, DINOv3 embedding analysis,
training-with-synthetic), two are new v2-only studies (dataset ablation, DINOv3 fine-tuning), and one
(phone-photo-augmentation) is a v1-data study kept here so the set is complete. Same task, recipe,
seeds, and evaluation throughout — **only the synthetic dataset changes** — so every v1↔v2 difference
is attributable to the regenerated renders; v1 reference numbers are quoted inline where useful. The
metrics and shared evaluation protocol are defined once below.

> *The original v1 per-experiment write-ups have been retired from the repo (archived on the
> `archive/experiments-v1` branch); these docs are self-contained.*

## Metrics & evaluation protocol

All scores are **0–100, higher is better**. A non-parametric **kNN classifier** over L2-normalized,
PCA-whitened global descriptors gives each query a predicted class **and a confidence**; K and τ
(softmax temperature) are tuned on the validation set over the full K×τ grid.

- **ACC** — of the **real (non-distractor) queries**, the fraction whose **top-1** prediction is
  correct. Ignores confidence and distractors.
- **GAP⁻** — Global Average Precision over the **real queries only**: rank by confidence, average
  precision over the correct ones. Rewards the correct answers also being the confident ones.
- **GAP** (headline, open-set) — GAP over **all 19,319 queries** (1,003 real + 18,316 distractors):
  rank everything by confidence; distractors always count as wrong. The realistic metric, and the
  lowest of the three. **GAP⁻ − GAP = distractor-rejection quality.**
- **R@k** — is the correct class among the query's **k** nearest database items. With the τ=50 vote,
  R@1 == ACC.

**Databases.** Default = all **397,121** studio photos / 224,408 classes (the full benchmark). Some
experiments also use the **paintings-only** DB (**12,403** photos / **4,898** `Classification=="Paintings"`
classes) — easier, and with no distractors there **GAP == GAP⁻** (the "closed painting world"). Closed
numbers are **not** comparable to the full-DB numbers or the paper's GAP 36.1.

The paper's best single model scores **GAP 36.1 / GAP⁻ 52.4 / ACC 55.0** — the baseline to beat.
Painting experiments use the committed `Classification=="Paintings"` set (4,898 classes / 148 test queries).

## What changed in dataset v2

v2 (24,490 images) regenerates the renders with the **Geometry-Nodes-randomized scene**
(`visart2026` PR #2, June 2026; the dataset's own `README.md` is the authoritative description):

- **Exactly the committed painting set.** 4,898 source paintings = the project-wide
  `Classification=="Paintings"` classes, folders `0..4897`, ×5 views = **24,490** renders
  (v1: 4,952 paintings / 24,760 — its 54 non-painting extras are dropped). Folder *i* shows the
  **same painting** in both versions (per-index source agreement verified 4,898/4,898), and all
  148 painting test queries' classes are covered.
- **New camera rig.** Five cameras on a horizontal **arc at eye height**, named by viewing angle —
  `30°/60°/90°/120°/150°` (90° ≈ frontal) — with per-scene **pose jitter** (±0.5 Y / ±1.0 Z
  translation, ±5° rotation). This replaces v1's `front` / `left|right upper|bottom` rig, whose
  `right upper` view was a known grazing/edge-on **framing bug** (v1 EXP-3/EXP-7).
- **More randomization, largely via Geometry Nodes:** wall/floor/roof texture mapping + floor
  material, frame molding variant + frame color/roughness/metallic, painting size on the wall,
  glass sheet with probability 0.25, placard color/visibility/position, area-light shape and spread.
  Of these, the per-folder `metadata.json` records floor material, placard position, canvas aspect,
  and camera poses; the GN-driven picks (frame color, glass presence, light shape, painting size)
  are **not** recorded — the v2 analogue of v1's unrecorded lighting.
- Same output format otherwise: 512×512 RGBA PNG, `metadata.json` provenance
  (`<index>/0_rgb_<angle>.png`; the file prefix is the frame number, always `0`).

## The experiments

| Experiment | Question (v2 edition) | Status / headline |
|---|---|---|
| [`renders-as-queries/`](renders-as-queries/README.md) | Can the real-data-only Met model recognize the v2 renders — and did the new camera rig fix v1's per-view collapse? | ✅ frontal 90° ACC **74.5** full-DB / **84.8** paint-DB (≥ any v1 view); ±60° arc ends near-zero *by design* |
| [`dinov3-embedding-analysis/`](dinov3-embedding-analysis/README.md) | How does a frozen DINOv3 organize the v2 renders (angle / floor / new jitter factors), and did the domain gap to real photos move? | ✅ identity still ~1,600× chance; angle softened by jitter; frontal now **ties** the studio↔real distance |
| [`real-vs-synthetic-mix/`](real-vs-synthetic-mix/README.md) | Same 12,403-image real:synth blends + synth-only scaling, with v2 renders: does "synthetic-only wins" hold, and by how much? | ✅ v2 ≥ v1 everywhere on the full benchmark; synth-only scaling no longer plateaus — all-renders model **beats the all-real 397k model** on GAP⁻/ACC |
| [`training-with-synthetic/`](training-with-synthetic/README.md) | Does adding the v2 renders to full-benchmark training still beat the paper's 36.1 (clean A/B + the two FT variants)? | ✅ clean from-scratch A/B **GAP 38.48** (+2.51, **beats paper 36.1**); FT-synth **38.99**, FT-combined 38.66 — every arm ≥ v1 |
| [`dataset-ablation/`](dataset-ablation/README.md) | *(new, v2-only)* Which dataset ingredients drive the gain — the procedural-randomization ladder (frozen room → +tex → +light → +glass → +frame → +cam jitter), 1024² rendering, viewpoint count, + a frame leave-one-out? Each variant gets EXP-12's synth-only all-renders treatment. | ✅ randomization **inverts** on the full benchmark — frozen-room synth-only **fGAP 36.09** ties the all-real 397k model; viewpoints are the key ingredient; frame variety harmful on the full benchmark (closed-paint effect within noise); 1024² no benchmark gain |
| [`dinov3-finetune/`](dinov3-finetune/README.md) | *(new, v2-only)* Does the synthetic data help a **strong foundation backbone**? LoRA-fine-tune DINOv3 ViT-L on real / budget-synth / all-synth paintings vs frozen zero-shot, in both DINOv3 readouts (CLS / patch-mean), tested with EXP-12's two protocols. | ✅ synthetic > real as FT data **everywhere**; frozen **CLS** best overall (full GAP 51.34, reproduces EXP-6); on the weak **patch-mean** readout synth FT **helps** (+6.6 GAP⁻) while real hurts — "FT beats frozen" depends on baseline strength |
| [`phone-photo-augmentation/`](phone-photo-augmentation/README.md) | *(v1 data)* Do simulated phone-camera artifacts (JPEG / blur / noise) as training augmentation improve the synth-only recognizer? | ✅ negative result — mild JPEG ties, anything stronger hurts (kept for completeness; no v2 rerun) |

Identical-by-construction comparisons: the blend/scaling subsets reuse the **same shuffle seeds and
sizes** as v1 (12,403 budget; 15,504 / 18,604 / 24,490 scaling), the recognizer recipe and seeds are
unchanged, the all-real points (step-1 baseline, 100:0 blend) are **reused from v1** since they contain
no synthetic data, and every evaluation runs on the same real queries/databases as v1.

Raw log, job ids, and exact commands: [`../EXPERIMENTS.md`](../EXPERIMENTS.md) (EXP-10…EXP-15).
