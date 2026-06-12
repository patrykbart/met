# Experiments — synthetic dataset **v2**

Reproductions of the four synthetic-data experiments on **v2 of the synthetic gallery dataset**
(`/mnt/storage_6/project_data/pl0896-03/visart-dataset-v2`). Same task, metrics, protocol, models,
and budgets as the originals in [`../experiments/`](../experiments/README.md) — **only the synthetic
dataset changes** — so every v1↔v2 difference is attributable to the regenerated renders. The metric
definitions (GAP / GAP⁻ / ACC / R@k) and the shared evaluation protocol are defined once in the
[v1 experiments README](../experiments/README.md) and are not repeated here.

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
| [`training-with-synthetic/`](training-with-synthetic/README.md) | Does adding the v2 renders to full-benchmark training still beat the paper's 36.1 (clean A/B + the two FT variants)? | 🟡 FT-synth **GAP 38.99**, FT-combined **38.66** (both > v1 + paper); from-scratch A/B still training (job 7372507) |

Identical-by-construction comparisons: the blend/scaling subsets reuse the **same shuffle seeds and
sizes** as v1 (12,403 budget; 15,504 / 18,604 / 24,490 scaling), the recognizer recipe and seeds are
unchanged, the all-real points (step-1 baseline, 100:0 blend) are **reused from v1** since they contain
no synthetic data, and every evaluation runs on the same real queries/databases as v1.

Raw log, job ids, and exact commands: [`../EXPERIMENTS.md`](../EXPERIMENTS.md) (EXP-10…EXP-13).
