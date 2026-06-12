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

## Design — one treatment, nine rows

Per row: training manifest = **all painting-class renders of that variant** (synth-only; angle rows
filter views first), shuffled @seed 1 exactly as EXP-12's pool — for the full-dataset rows this
yields the **same slot order** as `synthall_v2`, re-rendered. Training = the step-1 recipe
(`slurm/paint_train.slurm`, 10 epochs, seed 0). Evaluation on **real photos only**:
**(a)** closed painting world — 148 painting queries vs the 12,403-photo paint DB, K/τ by 2-fold CV
(`slurm/paint_eval.slurm`); **(b)** full 397k Met benchmark, K/τ tuned on val, + the 148-painting
slice at fixed K=7/τ=50 (`slurm/eval_full.slurm`). The default-v2 row is EXP-12's `synthall_v2`
run, **reused verbatim** (identical treatment, same dataset).

Manifests + image roots: `scripts/build_ablation_data.py` (job 7397360) →
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

*(pending — jobs queued 2026-06-12, behind the EXP-13 from-scratch run; the default-v2 row is
EXP-12's `synthall_v2` run, reused)*

One training per row. The first five columns mark which randomization families the renders carry
(✓ = active; the ladder is cumulative — in every row the painting, its wall scale, and the placard
still vary). `res` = render resolution; `angles` = arc views kept per painting (renders =
4,898 × angles; 3 = {60°, 90°, 120°}, 1 = {90°}). Metrics: **closed** painting world (148 q vs the
12,403-photo paint DB, 2-fold CV) · **full** 397k benchmark (1,003 q + 18,316 distractors, K/τ
tuned on val) · **paint GAP⁻** = its 148-painting slice (fixed K=7/τ=50).

| row | textures | light | glass | frame | cam jitter | res | angles | closed GAP⁻ | closed ACC | full GAP | full GAP⁻ | full ACC | paint GAP⁻ (148) |
|---|:-:|:-:|:-:|:-:|:-:|--:|--:|--:|--:|--:|--:|--:|--:|
| `abl0` | — | — | — | — | — | 512 | 5 | | | | | | |
| `abl1` | ✓ | — | — | — | — | 512 | 5 | | | | | | |
| `abl2` | ✓ | ✓ | — | — | — | 512 | 5 | | | | | | |
| `abl3` | ✓ | ✓ | ✓ | — | — | 512 | 5 | | | | | | |
| `abl4` | ✓ | ✓ | ✓ | ✓ | — | 512 | 5 | | | | | | |
| default *(EXP-12, reused)* | ✓ | ✓ | ✓ | ✓ | ✓ | 512 | 5 | *74.38* | *75.68* | *34.38* | *53.78* | *56.63* | *70.98* |
| `1024` | ✓ | ✓ | ✓ | ✓ | ✓ | 1024 | 5 | | | | | | |
| `ang3` | ✓ | ✓ | ✓ | ✓ | ✓ | 512 | 3 | | | | | | |
| `ang1` | ✓ | ✓ | ✓ | ✓ | ✓ | 512 | 1 | | | | | | |
| *ref: all-real, 12,403 imgs (EXP-8)* | — | — | — | — | — | — | — | *67.18* | *70.27* | *28.83* | *49.08* | *52.14* | *61.83* |

## Jobs

| row | train | closed eval | full eval |
|---|---|---|---|
| abl0 | 7397361 | 7397362 | 7397363 |
| abl1 | 7397364 | 7397365 | 7397366 |
| abl2 | 7397367 | 7397368 | 7397369 |
| abl3 | 7397370 | 7397371 | 7397372 |
| abl4 | 7397373 | 7397374 | 7397375 |
| 1024 | 7397376 | 7397377 | 7397378 |
| ang3 | 7397379 | 7397380 | 7397381 |
| ang1 | 7397382 | 7397383 | 7397384 |

Manifest prep = 7397360; models land in `data/models/r18SWSL_paint_synth_<tag>`, descriptors in
`data/descriptors{,_full}_synth_<tag>`. Closed numbers = the `>> 2-fold mean` line of each
`logs/met-ev-<tag>-*.out`; full numbers = the best-grid + `PAINT148` lines of `logs/met-full-<tag>-*.out`.

## How to reproduce

```bash
PREP=$(sbatch --parsable --partition=standard --time=1:00:00 --mem=8G --cpus-per-task=2 \
    --job-name=met-abl-data --output=logs/%x-%j.out \
    --wrap '.venv/bin/python scripts/build_ablation_data.py')          # manifests + im_roots
declare -A ROOT=( [abl0]=data/aug_abl0 [abl1]=data/aug_abl1 [abl2]=data/aug_abl2 \
                  [abl3]=data/aug_abl3 [abl4]=data/aug_abl4 [1024]=data/aug_1024 \
                  [ang3]=data/aug_v2 [ang1]=data/aug_v2 )
for tag in abl0 abl1 abl2 abl3 abl4 1024 ang3 ang1; do
  extra=""; [ "$tag" = "1024" ] && extra="--time=8:00:00"              # 4x decode cost in mining
  T=$(sbatch --parsable --kill-on-invalid-dep=yes --dependency=afterok:$PREP $extra \
      --job-name=met-tr-$tag slurm/paint_train.slurm data/gt_paint_synth_$tag ${ROOT[$tag]} paint_synth_$tag)
  sbatch --kill-on-invalid-dep=yes --dependency=afterok:$T --job-name=met-ev-$tag \
      slurm/paint_eval.slurm data/models/r18SWSL_paint_synth_$tag 10 synth_$tag
  sbatch --kill-on-invalid-dep=yes --dependency=afterok:$T --job-name=met-full-$tag \
      slurm/eval_full.slurm  data/models/r18SWSL_paint_synth_$tag 10 synth_$tag
done
```

## Caveats

- **148 test photos → ±2-point noise floor** (EXP-8/12), and a single seed/run per rung: adjacent
  rungs will often differ by less than noise — trust monotone trends across the ladder and
  large/full-benchmark deltas, not single-rung gaps.
- **Rungs compare distributions, not pixel-paired scenes:** every dataset re-draws its active
  randomization (folder *i* = same painting, different room draw); the 1024² set re-draws scenes too.
- **Angle rows confound volume with coverage** — interpret via the EXP-12 scaling arm (axis-3 note).
- Closed-world numbers are **not comparable to the paper's GAP 36.1**; painting-only models can't
  reject the 18k distractors, so full GAP is structurally penalized (EXP-8/12 finding).
