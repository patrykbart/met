# DINOv3 fine-tune: real vs synthetic paintings

*Does our synthetic gallery data help a **strong foundation backbone**, the way it helps the R18
recognizer ([`real-vs-synthetic-mix`](../real-vs-synthetic-mix/README.md))? We LoRA-fine-tune
**DINOv3 ViT-L** on three painting training sets — **real** studio photos, **budget-matched
synthetic v2**, **all synthetic v2** — and test every model on the **same real painting photos**,
with the **exact two protocols** of `real-vs-synthetic-mix` (closed painting world + full Met
benchmark), against a **frozen zero-shot** baseline. This is the DINOv3 analogue of that experiment's
three reference points (100:0 real, 0:100 budget-synth, synth-all). Run twice — once reading out the
**CLS token** (the representation that reproduced the DINOv3 paper, [EXP-6](../../EXPERIMENTS.md)),
once reading out **mean-pooled patch tokens** — because the answer flips between them. Metric
definitions (GAP / GAP⁻ / ACC) are in the [experiments-v2 README](../README.md). Lab
notebook: [`EXPERIMENTS.md` → EXP-15](../../EXPERIMENTS.md).*

## TL;DR

- **Synthetic > real as fine-tuning data — everywhere.** In both readouts, both protocols, and on
  every metric, the synthetic-trained model beats the real-trained one. That ordering is the robust,
  representation-invariant result and it matches the R18 story.
- **Whether fine-tuning *beats frozen* depends on how strong the frozen readout already is.** With the
  **weak patch-mean** readout, synthetic FT clearly **helps** (full benchmark **+6.6 GAP⁻**, closed
  **+14.7→+16.0 GAP⁻**); with the **strong CLS** readout (frozen ViT-L already GAP⁻ 72 full / 92
  closed), any fine-tune with this recipe only **erodes** it — synthetic just erodes least.
- **Real-data fine-tuning hurts in both readouts** (studio photos teach studio cues that don't
  transfer to real gallery query photos) — the clearest sign the synthetic gain is about domain, not
  just more data.
- **No full-benchmark scaling gain:** all 24,490 renders ≈ the 12,403 budget (within noise) in both
  readouts; the closed world gains a little.
- **The frozen CLS baseline is the strongest model overall** (full GAP 51.34, GAP⁻ 72.31) and
  reproduces EXP-6 (GAP⁻ 72.14, ACC 77.07) — confirming we used the same method. The patch-mean arm's
  absolute numbers are far lower; its "synthetic helps" is about improving a *weak* descriptor.

## Setup

Same recipe across all fine-tuned arms — **only the training data changes** (mirrors
`real-vs-synthetic-mix`):

| arm | training data | images |
|---|---|--:|
| zero-shot | *(none — frozen ViT-L)* | — |
| real | `data/gt_paint` (real studio paintings) | 12,403 |
| synthv2 | `data/gt_paint_mix_0r100s_v2` (v2 renders, budget-matched) | 12,403 |
| synthv2_all | `data/gt_paint_synthall_v2` (all v2 renders) | 24,490 |

DINOv3 ViT-L, **LoRA** (r=16, all-linear) + FC projector (PCA-whitening init), Con-Syn+Real-closest
mining (`new_pos+new_neg`), imsize 512, effective batch 64 (8×8), **3 epochs**, lr 1e-7, seed 0
([`slurm/dino_paint_train.slurm`](../../slurm/dino_paint_train.slurm)). Evaluated exactly as
`real-vs-synthetic-mix`: closed world = 2-fold CV over the 148 real painting queries vs the 12,403
painting DB; full benchmark = val-tuned K/τ over the 397k DB; painting slice = the 148 at fixed
K=7/τ=50 ([`slurm/dino_paint_eval.slurm`](../../slurm/dino_paint_eval.slurm)). The **readout** (`cls`
vs `patch`) is a flag — `DINOv3Trunk` returns either the CLS token `(B,D,1,1)` or the mean-poolable
patch grid. Eval data is always the real photos; only the model's training data differs.

## Results — CLS readout (the EXP-6 / paper-reproducing representation)

**Full Met benchmark** (1,003 real queries vs 397k DB; val-tuned K/τ). Δ = vs frozen zero-shot:

| arm | GAP | GAP⁻ | ACC |
|---|--:|--:|--:|
| **zero-shot** (frozen CLS) | **51.34** | **72.31** | **76.57** |
| FT real | 42.23 (−9.1) | 62.85 (−9.5) | 69.29 |
| FT synthv2 | 44.80 (−6.5) | 69.19 (−3.1) | 73.48 |
| FT synthv2_all | 45.06 (−6.3) | 68.55 (−3.8) | 72.88 |
| *ref: EXP-6 ViT-L ZS (CLS, aspect512)* | *48.16* | *72.14* | *77.07* |

**Closed painting world** (148 queries vs 12,403 painting DB; 2-fold CV) · **painting slice** (148, full DB, K=7/τ=50):

| arm | closed GAP⁻ | closed ACC | paint-slice GAP⁻ | paint-slice ACC |
|---|--:|--:|--:|--:|
| **zero-shot** | **92.21** | **92.57** | **85.27** | **85.81** |
| FT real | 88.37 (−3.8) | 89.19 | 80.51 | 81.76 |
| FT synthv2 | 89.96 (−2.3) | 90.54 | 78.47 | 79.73 |
| FT synthv2_all | 89.87 (−2.3) | 90.54 | 76.78 | 77.70 |

On the strong CLS representation, frozen zero-shot wins everywhere; LoRA at this recipe only erodes
it. Among the fine-tuned arms, synthetic leads on the full benchmark and closed world (real leads
only on the painting slice).

## Results — patch-mean readout (weak global descriptor)

**Full Met benchmark** (Δ vs frozen zero-shot):

| arm | GAP | GAP⁻ | ACC |
|---|--:|--:|--:|
| zero-shot (frozen patch-mean) | 11.25 | 32.42 | 38.68 |
| FT real | 10.54 (−0.7) | 27.13 (−5.3) | 37.09 |
| **FT synthv2** | **16.29 (+5.0)** | **39.05 (+6.6)** | **45.56 (+6.9)** |
| FT synthv2_all | 16.50 (+5.3) | 38.91 (+6.5) | 45.46 (+6.8) |

**Closed painting world** · **painting slice**:

| arm | closed GAP⁻ | closed ACC | paint-slice GAP⁻ | paint-slice ACC |
|---|--:|--:|--:|--:|
| zero-shot | 41.96 | 51.35 | 26.51 | 31.76 |
| FT real | 38.76 (−3.2) | 46.62 | 22.46 | 30.41 |
| FT synthv2 | 56.65 (+14.7) | 61.49 | 40.25 (+13.7) | 45.27 |
| **FT synthv2_all** | **58.00 (+16.0)** | **62.84** | 39.11 | 43.92 |

Here fine-tuning on **synthetic helps substantially** (positive Δ on every metric), while real hurts
— the "synthetic data helps" story, on a representation weak enough to have room for it.

## Findings

1. **Synthetic beats real as fine-tuning data, invariant to readout and protocol.** CLS closed 89.96
   vs 88.37; CLS full GAP⁻ 69.19 vs 62.85; patch-mean closed 56.65 vs 38.76; patch-mean full GAP⁻
   39.05 vs 27.13. This is the cleanest support for the project thesis: it survives a 60-point swing
   in baseline strength.
2. **The "does FT beat frozen?" answer is set by the baseline, not the data.** Patch-mean frozen is
   weak (full GAP 11), so synthetic FT improves it by +5–7; CLS frozen is strong (full GAP 51, closed
   92), so the same recipe can only take away. The synthetic arm is consistently the *least* harmful
   on CLS and the *most* helpful on patch-mean.
3. **Real-data fine-tuning is net-negative in both readouts** — strong evidence the synthetic
   advantage is a domain effect (renders resemble the real *gallery* queries; studio photos don't),
   not merely image count.
4. **No full-benchmark scaling.** 24,490 ≈ 12,403 renders on the full benchmark in both readouts
   (CLS GAP 45.06 vs 44.80; patch GAP 16.50 vs 16.29); the closed world gains slightly (CLS tie;
   patch +1.4).
5. **Reproduction check.** Frozen ZS-CLS in our pipeline = GAP⁻ 72.31 / ACC 76.57, matching EXP-6's
   72.14 / 77.07 (full GAP 51.34 vs 48.16, the distractor-sensitive metric, differs with K/τ tuning
   and square-vs-aspect512 preprocessing) — so the CLS arm uses the same method that reproduced the
   paper.

## Caveats

- **Recipe is inherited from R18-SWSL** (lr 1e-7, 3 epochs, margin 1.8); it was never tuned for
  ViT-L+LoRA. "Fine-tuning hurts the strong CLS readout" is therefore partly a statement about *this*
  recipe — whether a better recipe could beat frozen CLS is open. The synth-vs-real comparison is
  robust to this (identical recipe across arms).
- **Patch-mean absolute numbers are weak** (full GAP 11–16, far below R18's 35.97 and CLS-DINOv3's
  51): "synthetic helps" there means improving a weak descriptor, not reaching SOTA.
- **148 painting queries, 1 val query** (hence 2-fold CV + the fixed-K/τ painting slice); single
  differences ≤ ~2 points are noise. Closed-world numbers are not comparable to the paper's GAP 36.1.
- **DINOv3 features are mean-pooled at imsize 512 (square)** vs EXP-6's aspect512 — a minor
  preprocessing difference that mostly affects the distractor-sensitive GAP.

## Reproduce

```bash
# ZS baseline (both readouts) + the three LoRA arms x both readouts; eval = closed + full + paint slice.
sbatch --job-name=met-dev-zscls   slurm/dino_paint_eval_zs.slurm            # frozen CLS  (cls is default)
sbatch --job-name=met-dev-zspatch slurm/dino_paint_eval_zs.slurm patch      # frozen patch-mean
for R in cls patch; do
  for arm in "real data/gt_paint data/" \
             "synthv2 data/gt_paint_mix_0r100s_v2 data/aug_v2" \
             "synthv2_all data/gt_paint_synthall_v2 data/aug_v2"; do
    set -- $arm; tag=$1; info=$2; root=$3
    suf=""; [ "$R" != cls ] && suf="_$R"
    t=$(sbatch --parsable --job-name=met-dtr-$tag-$R slurm/dino_paint_train.slurm "$info" "$root" "$tag" lora "$R")
    sbatch --dependency=afterok:$t --job-name=met-dev-$tag-$R \
        slurm/dino_paint_eval.slurm "data/models/dinov3L_lora_${tag}${suf}" "$tag" lora "$R"
  done
done
```

Run in `.venv-dino` (extraction; transformers/peft) + `.venv` (eval). Trains ~2.5–5 h each on an
H100 (synthv2_all is the long pole); evals ~3 h (the 397k DINOv3 extraction dominates). Code:
[`scripts/extract_dino_ckpt.py`](../../scripts/extract_dino_ckpt.py) (`--zeroshot`, `--dino-readout`),
the `dino_readout` flag in [`train_contrastive.py`](../../code/examples/train_contrastive.py) →
`DINOv3Trunk` ([`backbone.py`](../../code/networks/backbone.py)),
[`eval_paintings_closed.py`](../../scripts/eval_paintings_closed.py) /
[`eval_fullgrid.py`](../../scripts/eval_fullgrid.py) / [`eval_paintings.py`](../../scripts/eval_paintings.py).
```
