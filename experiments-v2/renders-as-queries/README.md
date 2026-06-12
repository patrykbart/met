# Can the Met model recognize the v2 gallery renders?

*v2 rerun of [`experiments/renders-as-queries/`](../../experiments/renders-as-queries/README.md):
the recognition model trained **only on the original Met data** (zero synthetic exposure), with the
**24,490 v2 renders as queries**, against the **full 397k Met benchmark** and the **paintings-only**
database — scored exactly like v1 (GAP / GAP⁻ / ACC + recall@k, per camera view). Protocol, model
(step-1 reproduction, GAP 35.97), databases, and distractors are **identical to v1; only the renders
change.** (Lab notebook: [`EXPERIMENTS.md` → EXP-10](../../EXPERIMENTS.md); v1 = EXP-3.)*

The v2 camera rig replaces v1's named poses (`front`, `left/right upper/bottom`) with **five cameras
on a horizontal arc at eye height** — `30°/60°/90°/120°/150°`, 90° ≈ frontal — plus per-scene pose
jitter ([what changed in v2](../README.md)). v1's `right upper` view was an *accidental* grazing
shot (the camera-rig bug); the arc's ±60° endpoints are *deliberate* extreme viewpoints.

## TL;DR

- **The frontal view got better, the extremes got harder.** At 90° the v2 renders match v1's best
  view on the full DB (ACC 74.5 vs 75.9) and **beat every v1 view on the paintings DB (84.8 vs
  80.9)** — while the ±60° views collapse to ACC 2–6, like v1's broken camera.
- **The per-view spread is now symmetric and by-design.** 60° ≈ 120° (ACC ~33 twice) and
  30° ≈ 150° (~2–4) — a clean monotone fall-off with viewing angle, not a one-camera accident.
  The DINOv3 study ([EXP-11](../dinov3-embedding-analysis/README.md)) confirms the same ordering
  from embeddings alone.
- **All-view averages drop accordingly** (full-DB ACC 36.9 → 29.4): two of five arc views are
  near-unanswerable as *queries*. Per-view, every well-framed v2 view ≥ its closest v1 counterpart.
- **Same model, same DBs, real-query rows unchanged** — the 148 real painting queries score
  identically to v1 (GAP⁻ 71.48 / ACC 72.30 on the paint DB), pinning the comparison.

## Table A — against the full Met benchmark database

Database = all **397,121** studio photos / 224,408 classes; GAP includes the 18,316 real test
distractors. Real-photo baselines first, then the v2 renders per view (best→worst), then the v1
all-angles row for reference.

| query (DB = full Met, 397k) | N | GAP | GAP⁻ | ACC | R@1 | R@5 | R@10 |
|---|--:|--:|--:|--:|--:|--:|--:|
| *real — paper R18-SWSL Con-Syn+Real-closest* | 1,003 | 36.1 | 52.4 | 55.0 | — | — | — |
| *real — our step-1 reproduction* | 1,003 | 35.97 | 52.14 | 54.64 | — | — | — |
| *real — our paintings only* | 148 | 39.50 | 67.86 | 69.59 | — | — | — |
| **synthetic v2 — ALL angles** | 24,490 | 23.62 | 25.72 | 29.36 | 29.36 | 35.79 | 38.67 |
| synthetic v2 — **90°** (frontal) | 4,898 | 64.97 | 73.21 | 74.52 | 74.52 | 81.77 | 84.79 |
| synthetic v2 — 60° | 4,898 | 19.88 | 28.39 | 33.48 | 33.48 | 43.14 | 47.71 |
| synthetic v2 — 120° | 4,898 | 20.09 | 28.18 | 33.34 | 33.34 | 43.69 | 47.82 |
| synthetic v2 — 150° | 4,898 | 0.14 | 0.59 | 3.61 | 3.61 | 6.51 | 8.23 |
| synthetic v2 — 30° | 4,898 | 0.03 | 0.16 | 1.86 | 1.86 | 3.84 | 4.82 |
| *v1 — ALL angles (24,760)* | *24,760* | *31.42* | *33.69* | *36.93* | *36.93* | *43.97* | *46.81* |

**v1 ↔ v2 per-view, full DB (ACC):** v1's two good views were `left upper` **75.9** and `front`
**64.8**; v2's single frontal 90° lands at **74.5** — one view now does the work of v1's best. v1's
foreshortened `*bottom` pair scored ~21; v2's ±30° pair scores ~33 (**+12 on the comparable
mid-views**). v1's broken `right upper` was 1.4; v2's deliberate ±60° pair is 1.9 / 3.6.

## Table B — against the paintings-only database

Database = **12,403** photos / **4,898** `Classification=="Paintings"` classes. Unlike v1 (which
dropped 270 renders of non-painting classes), **every v2 render's class is in this DB** — v2 *is*
the painting set. Same 18,316 distractors for GAP; GAP⁻ = the closed-world painting GAP.

| query (DB = paintings only, 12,403) | N | GAP | GAP⁻ | ACC | R@1 | R@5 | R@10 |
|---|--:|--:|--:|--:|--:|--:|--:|
| *real — our paintings only* | 148 | 45.69 | 71.48 | 72.30 | 72.30 | 79.73 | 83.78 |
| **synthetic v2 — ALL angles** | 24,490 | 31.49 | 34.66 | 37.77 | 37.77 | 46.03 | 49.62 |
| synthetic v2 — **90°** (frontal) | 4,898 | 76.06 | 84.18 | 84.75 | 84.75 | 90.61 | 92.94 |
| synthetic v2 — 60° | 4,898 | 29.20 | 43.05 | 47.31 | 47.31 | 60.66 | 66.03 |
| synthetic v2 — 120° | 4,898 | 29.52 | 42.57 | 46.67 | 46.67 | 60.49 | 65.64 |
| synthetic v2 — 150° | 4,898 | 0.41 | 2.11 | 6.25 | 6.25 | 11.05 | 13.84 |
| synthetic v2 — 30° | 4,898 | 0.08 | 0.64 | 3.86 | 3.86 | 7.33 | 9.68 |
| *v1 — ALL angles (in-DB 24,490)* | *24,490* | *39.14* | *42.19* | *45.15* | *45.15* | *53.61* | *56.98* |

Here the v2 frontal renders don't just beat the v1 views (best: `left upper` ACC 80.9) — they beat
the **real painting queries** (84.8 vs 72.3) by a wider margin than v1's best did. The real-query
row is bit-identical to v1 (same model, same DB): the change is all in the renders.

## The rig: from accidental bug to by-design difficulty

v1's per-view spread was an **accident** — one camera (`right upper`) grazed the canvas edge-on.
v2's spread is **structural**: recognizability falls monotonically with arc angle, symmetrically on
both sides of frontal (60°≈120°, 30°≈150°). The
[DINOv3 embedding study](../dinov3-embedding-analysis/README.md) reproduces the same ordering with
a different model and no retrieval: per-view cosine to the paired studio source runs
0.84 (90°) → 0.69 (60°/120°) → 0.49 (30°/150°).

Two readings, both true:

- **As queries** (this experiment), the ±60° views are nearly unanswerable — a model trained on
  frontal studio photos cannot match a painting it sees at a grazing angle, and real visitors rarely
  shoot from there. Per-view claims should quote 90° (and note the fall-off), not the all-view mean.
- **As training data**, hard views are not necessarily a defect — they supply exactly the viewpoint
  variation studio photos lack. Whether they help or hurt is an *empirical* question answered by
  [EXP-12](../real-vs-synthetic-mix/README.md) (they help: v2 beats v1 as training data everywhere
  on the full benchmark) and [EXP-13](../training-with-synthetic/README.md).

## Caveats

- Same as v1: GAP borrows the real test distractors (open-set) while GAP⁻ is the closed-world
  treatment; ACC == R@1 by construction (τ=50 vote follows the nearest neighbour); "correct" =
  source class; step-1 model only.
- **All-angle averages are not comparable v1↔v2 as "dataset quality"** — the rigs sample different
  viewpoint distributions (v1: 4 usable + 1 broken; v2: 3 usable + 2 extreme). Per-view rows and
  the training-side experiments are the meaningful comparison.
- The 90°-view GAP (65.0) is still below the real-query GAP (39.5 → *not* comparable directly:
  different query counts and confidence distributions against the same distractor pool; rank-based
  GAP depends on both).

## How to reproduce

```bash
# 1) GPU: extract MS descriptors for the 24,490 v2 renders with the step-1 model, then recall@k.
sbatch slurm/synth_eval.slurm /mnt/storage_6/project_data/pl0896-03/visart-dataset-v2 \
    data/descriptors/synthetic_v2                            # job 7372439 (~6 min H100)
# 2) CPU re-scores (GAP tables A+B; reuses the descriptors).
SYNTH_OUT=data/descriptors/synthetic_v2 .venv/bin/python scripts/eval_synthetic_gap.py    # job 7372440
SYNTH_OUT=data/descriptors/synthetic_v2 .venv/bin/python scripts/eval_painting_db.py      #   (same job)
```

Outputs (git-ignored): `data/descriptors/synthetic_v2/{synth_descriptors.pkl, retrieval_summary.json,
gap_summary.json, painting_db_summary.json}`. Same scripts as v1, pointed at v2 via
`SYNTH_DS`/`SYNTH_OUT`/positional args.
