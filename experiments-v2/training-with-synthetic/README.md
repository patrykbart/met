# Beating the Met benchmark with synthetic data — dataset v2

*The main contribution, on **v2** of the synthetic gallery dataset: retrain the original Met
recognizer with the v2 renders added and ask whether that **alone** — same recipe, no new method —
beats the paper. It does, more strongly than v1. Metric definitions (GAP / GAP⁻ / ACC) are in the
[experiments-v2 README](../README.md). Lab notebook: [`EXPERIMENTS.md` → EXP-13](../../EXPERIMENTS.md)
(v1 = EXP-4). Only the synthetic dataset changes vs v1; recipe, seeds, and evaluation are identical.*

## What we did

Reproduce the paper's best single model from scratch (the **baseline**, GAP 35.97), then retrain the
**identical** recipe with the v2 gallery renders added to the training set — the synthetic images are
used *only* for training, never enter the test database — and evaluate both on the original 397k-image
benchmark. Three ways of adding the data:

- **From-scratch + synth (clean A/B)** — 10 epochs over studio + v2 renders, identical to the baseline
  except for the added data. The doubt-free comparison (same epochs, same everything).
- **FT-synth / FT-combined** — fine-tune the epoch-10 baseline for 5 more epochs on the v2 renders
  (synth-only / studio+synth). Larger gains but *confounded* (they train longer than the baseline).

## TL;DR

- **Adding v2 synthetic data lifts full-benchmark GAP from 35.97 → 38.48** with the identical recipe
  (clean A/B, **+2.51**) — **beats the paper's best single model (36.1)** with no new method and no
  extra real data.
- **Every metric improves** (clean A/B): GAP⁻ 52.14 → 55.04, ACC 54.64 → 57.63, paint GAP⁻ 67.86 →
  71.47, paint ACC 69.59 → 72.97.
- **The fine-tune variants go higher still** — **FT-synth GAP 38.99** (best overall), FT-combined 38.66
  — but are confounded by longer training; the clean A/B is the headline.
- **v2 ≥ v1 on the headline GAP for every arm** (from-scratch 38.48 vs 38.15, FT-synth 38.99 vs 38.61,
  FT-combined 38.66 vs 37.38) — the regenerated renders are at least as good as v1 as training data.

## Results — full Met benchmark (397k DB, 1,003 real + 18,316 distractor queries)

| training data | GAP | GAP⁻ | ACC | paint GAP⁻ (148) | paint ACC (148) |
|---|--:|--:|--:|--:|--:|
| Paper R18-SWSL Con-Syn+Real-closest | 36.1 | 52.4 | 55.0 | — | — |
| Baseline — studio only (step 1) | 35.97 | 52.14 | 54.64 | 67.86 | 69.59 |
| **From-scratch + v2 synth** *(clean A/B)* | **38.48** | **55.04** | **57.63** | **71.47** | **72.97** |
| FT-synth (v2) | **38.99** | 55.15 | 57.23 | 72.77 | — |
| FT-combined (v2) | 38.66 | 53.73 | 55.53 | 70.02 | — |

*Baseline and paper rows contain no synthetic data. The clean A/B differs from the baseline only in the
added renders, so its +2.51 GAP is attributable to the synthetic data alone. v1 GAP for reference:
from-scratch 38.15, FT-synth 38.61, FT-combined 37.38 (every v2 arm ≥ v1).*

## Findings

1. **Synthetic data helps on its own — and beats the paper.** The clean A/B (+2.51 GAP, no recipe
   change) settles that the gain is the data, not training tricks; the result clears the paper's 36.1.
2. **Non-painting queries improve too** (overall GAP⁻ / ACC rise, not just the painting slice) — the
   renders teach broadly useful lighting / viewpoint / glass invariance, not merely "more painting data".
3. **v2 ≥ v1 as training data.** Every arm's headline GAP matches or beats its v1 counterpart, with the
   GN-randomized scene and arc cameras (the v2 dataset change); consistent with
   [`real-vs-synthetic-mix`](../real-vs-synthetic-mix/README.md) finding v2 renders worth more per image.
4. **Fine-tuning > from-scratch on GAP** (38.99 vs 38.48) but is confounded by extra epochs; we report
   the clean A/B as the defensible headline and the FT numbers as an upper-ish bound.

## Caveats

- The **FT variants train longer** than the baseline (15 vs 10 effective epochs), so their larger gains
  are not a clean A/B — only the from-scratch run is. Paint ACC wasn't logged for the FT variants.
- All numbers use the project's standard eval (multi-scale R18-SWSL descriptors, 397k studio DB, full
  K×τ grid on val); the painting slice is the committed 148 `Classification=="Paintings"` queries.
- This is the v2 dataset (24,490 renders, arc cameras, GN-randomized scene); see the
  [experiments-v2 README](../README.md) for what changed from v1, and
  [`dataset-ablation`](../dataset-ablation/README.md) for which ingredients drive the gain.

## Reproduce

```bash
# from-scratch + v2 synth (clean A/B), identical recipe to step 1 with the data swapped
sbatch slurm/train_synth.slurm _v2                       # -> data/models/r18SWSL_scratch_synth_v2 (job 7372507)
sbatch slurm/eval_full.slurm data/models/r18SWSL_scratch_synth_v2 10 scratchsynth-v2   # eval (7372508)
# fine-tune variants (5 more epochs from the epoch-10 baseline, lr 1e-7)
sbatch slurm/finetune.slurm synth    _v2                 # FT-synth    (7372509 -> eval 7372510)
sbatch slurm/finetune.slurm combined _v2                 # FT-combined (7372511 -> eval 7372512)
```

Recipe, seeds, and evaluation are identical to step 1; only `--info_dir`/`--im_root` point at the v2
manifests (`data/gt_aug_v2` / `data/gt_synth_v2`, `data/aug_v2`). Full commands + job ids:
[`EXPERIMENTS.md` → EXP-13](../../EXPERIMENTS.md).
