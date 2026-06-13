# Dataset ablation — what makes the renders work?

*New v2-only experiment (no v1 counterpart). [`real-vs-synthetic-mix`](../real-vs-synthetic-mix/README.md)
(EXP-12) showed the strongest synthetic-only recognizer is the **“all renders” run** — every one of
the 24,490 v2 renders as training data (1.97× the 12,403 budget), zero real photos. This experiment
repeats **exactly that treatment** over controlled variants of the dataset to isolate *which
ingredients* produce the gain, along three axes: **procedural environment randomization** (an
ablation ladder), **render resolution** (512² vs 1024²), and **viewpoint count** (5 vs 3 vs 1 arc
angles). Recipe, seeds, manifest construction, and both evaluations are identical across rows —
only the renders change. Metric definitions and protocol: [v1 experiments README](../../experiments/README.md);
lab notebook: [`EXPERIMENTS.md` → EXP-14](../../EXPERIMENTS.md).*

## TL;DR

- **On the real open-set benchmark, *less* randomization is better.** Full-benchmark GAP falls
  almost monotonically as the ladder adds randomization (36.09 frozen room → 34.38 fully
  randomized); the **frozen-room, synthetic-only model ties the all-real 397k-image model**
  (fGAP 36.09 vs 35.97 ours / 36.1 paper) and leads fGAP⁻/fACC — from 24,490 renders of 4,898
  painting classes and zero real photos.
- **Viewpoints are the load-bearing ingredient.** 3 arc angles ≈ all 5 nearly everywhere at 60% of
  the data; frontal-only collapses to barely above the all-real baseline.
- **Frame variety is the one clearly harmful family** (worst rung on both worlds), confirmed by a
  leave-one-out render: dropping just the frame from the full default recovers +0.53 fGAP — but
  still lands +1.2 below the frozen room, so it's one harmful ingredient among several, not the
  whole story. **1024² rendering** buys the best closed-world score but nothing on the full
  benchmark — not worth 4× the render cost.

## Design — one treatment, nine rows

Per row: training manifest = **all painting-class renders of that variant** (synth-only; angle rows
filter views first), shuffled @seed 1 exactly as EXP-12's pool — for the full-dataset rows this
yields the **same slot order** as `synthall_v2`, re-rendered. Training = the step-1 recipe
(`slurm/paint_train.slurm`, 10 epochs, seed 0). Evaluation on **real photos only**:
**(a)** closed painting world — 148 painting queries vs the 12,403-photo paint DB, K/τ by 2-fold CV
(`slurm/paint_eval.slurm`); **(b)** full 397k Met benchmark, K/τ tuned on val, + the 148-painting
slice at fixed K=7/τ=50 (`slurm/eval_full.slurm`). The default-v2 row is EXP-12's `synthall_v2`
run, **reused verbatim** (identical treatment, same dataset).

Manifests + image roots: `scripts/build_ablation_data.py` →
`data/gt_paint_synth_<tag>` + `data/aug_<tag>` (angle rows reuse `data/aug_v2`).

### Axis 1 — the procedural-randomization ladder (24,490 renders each, 512²)

Each rung re-renders the full dataset with one more randomization family enabled (cumulative;
each dataset's own `README.md` is authoritative). In **every** rung the painting itself, its scale
on the wall, and the placard still vary; folder *i* = the same source painting everywhere.

| rung | tag | dataset | adds |
|---|---|---|---|
| 0 | `abl0` | `visart-dataset-v2-abl0-none` | nothing — one frozen room for all images |
| 1 | `abl1` | `…-abl1-tex` | wall/floor/roof textures + floor material |
| 2 | `abl2` | `…-abl2-tex-light` | light shape/spread variety |
| 3 | `abl3` | `…-abl3-tex-light-glass` | glass sheet at p=0.25 |
| 4 | `abl4` | `…-abl4-tex-light-glass-frame` | frame molding/color variety |
| 5 | — | `visart-dataset-v2` (default) | camera-pose jitter *(= EXP-12 `synthall_v2`, reused)* |
| LOO | `noframe` | `…-v2-noframe` | **leave-one-out**: the default config with only frame variety frozen (`--bake-frames`) — isolates the frame effect inside the full randomization context |

### Axis 2 — render resolution

`visart-dataset-v2-1024` (tag `1024`): the default configuration re-rendered at **1024²** (only
flag change; scenes re-drawn). The train transform still crops-and-resizes to 500², and eval images
are untouched real photos — so this measures how much painting detail survives the render→resample
pipeline, not a bigger network input.

### Axis 3 — viewpoint count (subsets of default v2)

| tag | arc angles kept | renders |
|---|---|--:|
| `ang1` | {90°} (≈ frontal) | 4,898 |
| `ang3` | {60°, 90°, 120°} | 14,694 |
| — | all five (default) | 24,490 |

⚠️ These rows change data **volume** as well as viewpoint coverage. Read them against EXP-12's
synth-only scaling arm (random all-angle subsets: 12,403 → 24,490 renders moved closed GAP⁻
73.47 → 74.38, full GAP 32.75 → 34.38) — if `ang3` at 14,694 lands near the all-angle run, angle
coverage matters little; if it lands at/below the like-sized random subset, the ±60° views pull
their weight.

## Results

*(complete, 2026-06-12, all ten rows incl. the leave-one-out `noframe`. The default-v2 row is
EXP-12's `synthall_v2` run, reused.)*

One training per row, identified by its factor combination (the ladder is cumulative; every row
still varies the painting, its wall scale, and the placard).

**Factors** (✓ = randomization active in the renders):

- `tex` — wall/floor/roof textures + floor material variety
- `light` — light shape/spread variety
- `glass` — glass sheet at p=0.25
- `frame` — frame molding/color variety
- `jitter` — camera-pose jitter
- `res` — render resolution (px)
- `angles` — arc views per painting: 5 = all, 3 = {60°, 90°, 120°}, 1 = {90°} (renders = 4,898 × angles)

**Metrics** (the prefix says which evaluation):

- `cGAP⁻`, `cACC` — **c**losed painting world: 148 real painting queries vs the 12,403-photo paint DB, K/τ by 2-fold CV
- `fGAP`, `fGAP⁻`, `fACC` — **f**ull 397k benchmark: 1,003 Met queries + 18,316 distractors, K/τ tuned on val
- `pGAP⁻` — the 148-**p**ainting slice of the full benchmark, fixed K=7/τ=50

**Special rows:** *italics* = reused results (the all-✓ 512/5 row is EXP-12's `synthall_v2`);
`†` = the all-real reference — the same recipe on the 12,403 real painting photos, no renders
at all (EXP-8).

| tex | light | glass | frame | jitter | res | angles | cGAP⁻ | cACC | fGAP | fGAP⁻ | fACC | pGAP⁻ |
|:-:|:-:|:-:|:-:|:-:|--:|--:|--:|--:|--:|--:|--:|--:|
| — | — | — | — | — | 512 | 5 | 73.78 | 75.00 | 36.09 | 54.28 | 56.73 | 70.92 |
| ✓ | — | — | — | — | 512 | 5 | 73.94 | 75.00 | 35.56 | 53.29 | 55.73 | 71.84 |
| ✓ | ✓ | — | — | — | 512 | 5 | 74.42 | 75.68 | 35.13 | 53.47 | 55.93 | 71.29 |
| ✓ | ✓ | ✓ | — | — | 512 | 5 | 75.32 | 76.35 | 35.29 | 53.62 | 56.03 | 71.24 |
| ✓ | ✓ | ✓ | ✓ | — | 512 | 5 | 72.61 | 73.65 | 34.35 | 52.10 | 54.74 | 68.32 |
| ✓ | ✓ | ✓ | ✓ | ✓ | 512 | 5 | *74.38* | *75.68* | *34.38* | *53.78* | *56.63* | *70.98* |
| ✓ | ✓ | ✓ | — | ✓ | 512 | 5 | 73.91 | 75.00 | 34.91 | 54.06 | 56.53 | 71.37 |
| ✓ | ✓ | ✓ | ✓ | ✓ | 1024 | 5 | 75.41 | 76.35 | 34.48 | 52.12 | 54.64 | 68.36 |
| ✓ | ✓ | ✓ | ✓ | ✓ | 512 | 3 | 74.31 | 75.68 | 33.70 | 53.71 | 56.53 | 70.51 |
| ✓ | ✓ | ✓ | ✓ | ✓ | 512 | 1 | 68.55 | 70.27 | 28.64 | 49.97 | 53.04 | 63.84 |
| † | — | — | — | — | — | — | *67.18* | *70.27* | *28.83* | *49.08* | *52.14* | *61.83* |

### Findings

1. **The randomization ladder inverts on the full benchmark.** fGAP: 36.09 → 35.56 → 35.13 →
   35.29 → 34.35 → 34.38 — every randomization family added costs open-set GAP, and the frozen
   room ends up the strongest training material (fGAP⁻ 54.28 / fACC 56.73 also top the table).
   The closed world shows the opposite, much flatter picture (73.78 → 75.32 across the same rungs,
   a spread ≈ the noise floor): without distractors, scene variety barely matters — the cost of
   randomization is **distractor-side**.
2. **A synthetic-painting-only model now ties the full-data real model.** EXP-12's "distractor
   rejection is the price of painting-only training" (default v2: fGAP 34.38 vs 35.97) was
   apparently the price of *randomization*: the frozen-room variant closes the whole gap
   (36.09 ≈ 35.97 ours / 36.1 paper) while keeping the painting-slice gains (pGAP⁻ 70.92 vs 61.83
   all-real).
3. **Frame variety is the one clearly harmful ingredient — confirmed from both ends.** Adding it on
   the ladder (abl3 → abl4) drops every metric (−0.94 fGAP, −2.7 cGAP⁻, −2.9 pGAP⁻); the
   **leave-one-out `noframe`** row (the full default config with *only* frame frozen) independently
   *recovers* +0.53 fGAP over the default (34.91 vs 34.38), +0.28 fGAP⁻, +0.39 pGAP⁻ — two
   independent measurements agreeing the frame costs ~0.5–0.9 fGAP. Consistent with the frame being
   a stable per-painting identity cue that randomization destroys. The glass sheet is the most
   *helpful* rung closed-world (abl3 = 75.32, the best 512² row).
4. **Viewpoint count matters more than any scene factor.** {60°, 90°, 120°} matches all-5 within
   noise on everything but fGAP (33.70 vs 34.38) with 40% less data — the ±60° grazing views
   contribute mainly distractor rejection. Frontal-only loses ~6 closed points (68.55) and 5.7
   fGAP (28.64), landing at the all-real baseline: multi-view is what the renders are *for*.
5. **1024² helps only where it isn't needed**: best closed-world row (75.41/76.35) but full-
   benchmark ≈ default (34.48 vs 34.38, fGAP⁻/fACC slightly lower) at ~3× render time and ~4× disk.

**Removing frame variety helps but does not win.** The `noframe` row was the obvious "best of both
worlds" candidate — keep every useful randomization, drop the one harmful family. It does land
above the fully-randomized default (fGAP 34.91 vs 34.38), but it recovers only ~⅓ of the gap to the
**frozen room** (abl0 = 36.09, still +1.18 ahead): frame is one harmful ingredient among several,
and the other randomization families (textures/light/glass/jitter) carry the rest of the open-set
penalty. Closed-world, `noframe` (73.91) is in the flat pack, not the top (abl3 = 75.32). So the
single best training material here remains the **frozen room**, not "default-minus-frame".

Follow-up this suggests: regenerate the *full-benchmark* training mixes (EXP-13's FT-synth /
from-scratch, currently using default v2) with the **frozen-room** renders — if the +1.7 fGAP
transfers, the headline GAP 38.99 has room above it.

## How to reproduce

```bash
PREP=$(sbatch --parsable --partition=standard --time=1:00:00 --mem=8G --cpus-per-task=2 \
    --job-name=met-abl-data --output=logs/%x-%j.out \
    --wrap '.venv/bin/python scripts/build_ablation_data.py')          # manifests + im_roots
declare -A ROOT=( [abl0]=data/aug_abl0 [abl1]=data/aug_abl1 [abl2]=data/aug_abl2 \
                  [abl3]=data/aug_abl3 [abl4]=data/aug_abl4 [1024]=data/aug_1024 \
                  [ang3]=data/aug_v2 [ang1]=data/aug_v2 [noframe]=data/aug_noframe )
for tag in abl0 abl1 abl2 abl3 abl4 1024 ang3 ang1 noframe; do
  extra=""; [ "$tag" = "1024" ] && extra="--time=8:00:00"              # 4x decode cost in mining
  T=$(sbatch --parsable --kill-on-invalid-dep=yes --dependency=afterok:$PREP $extra \
      --job-name=met-tr-$tag slurm/paint_train.slurm data/gt_paint_synth_$tag ${ROOT[$tag]} paint_synth_$tag)
  sbatch --kill-on-invalid-dep=yes --dependency=afterok:$T --job-name=met-ev-$tag \
      slurm/paint_eval.slurm data/models/r18SWSL_paint_synth_$tag 10 synth_$tag
  sbatch --kill-on-invalid-dep=yes --dependency=afterok:$T --job-name=met-full-$tag \
      slurm/eval_full.slurm  data/models/r18SWSL_paint_synth_$tag 10 synth_$tag
done
```

The `noframe` dataset is rendered first in the `visart2026` repo (default config + `--bake-frames`):
`VISART_SAVE_BASE=…scratch…/visart-dataset-v2-noframe VISART_EXTRA_ARGS="--bake-frames" sbatch
--parsable slurm/render_array.sbatch`, then `slurm/merge_shards.sbatch` into
`…/visart-dataset-v2-noframe` — the prep → train → eval chain above hangs off the merge via
`--dependency=afterok`.

Models land in `data/models/r18SWSL_paint_synth_<tag>`, descriptors in
`data/descriptors{,_full}_synth_<tag>`; the table reads the `>> 2-fold mean` line of each closed
eval and the best-grid + `PAINT148` lines of each full eval. Job ids: [`EXPERIMENTS.md` → EXP-14](../../EXPERIMENTS.md).

## Caveats

- **148 test photos → ±2-point noise floor** (EXP-8/12), and a single seed/run per rung: adjacent
  rungs will often differ by less than noise — trust monotone trends across the ladder and
  large/full-benchmark deltas, not single-rung gaps.
- **Rungs compare distributions, not pixel-paired scenes:** every dataset re-draws its active
  randomization (folder *i* = same painting, different room draw); the 1024² set re-draws scenes too.
- **Angle rows confound volume with coverage** — interpret via the EXP-12 scaling arm (axis-3 note).
- Closed-world numbers are **not comparable to the paper's GAP 36.1**; painting-only models can't
  reject the 18k distractors, so full GAP is structurally penalized (EXP-8/12 finding).
