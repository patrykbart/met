# Reference paper — The Met Dataset (NeurIPS 2021)

This directory holds the **LaTeX source of the original Met dataset paper**, the primary
reference for this fork's research. It was downloaded from arXiv (`arxiv.org/src/2202.01747`)
and extracted here.

> Ypsilantis, Garcia, Han, Ibrahimi, van Noord, Tolias.
> **"The Met Dataset: Instance-level Recognition for Artworks."**
> NeurIPS 2021 Datasets & Benchmarks Track. arXiv:2202.01747.
> Dataset/code: http://cmp.felk.cvut.cz/met/ · https://github.com/nikosips/met

The bulky source (figures, sample images — ~33 MB) is **git-ignored**; only this README is
tracked. Re-download with `curl -L arxiv.org/src/2202.01747 -o src.tgz && tar -xzf src.tgz`.

## Source file map

| File | Section |
|------|---------|
| `ms.tex` | Main file — preamble, author list, abstract, `\input`s the rest |
| `intro.tex` | Introduction & motivation |
| `dataset.tex` | Dataset construction, splits, **task & evaluation protocol (GAP / GAP⁻ / ACC)** |
| `baselines.tex` | **Methods**: representation, kNN classifier, training variants, pretrained backbones |
| `exp.tex` | Experiments & all result tables |
| `appendix.tex` | Implementation details (hyper-params), ablations, mini-dataset, OOD-ratio study |
| `datasheet.tex` | Datasheet-for-datasets documentation |
| `abbrev.tex`, `plots.tex` | Macros / plot styling |
| `ms.bbl` | Bibliography (no `bib` source shipped) |
| `fig/`, `suppl-fig/` | Figures and sample images |

## The task in one paragraph

Instance-level recognition (ILR): each of ~224k Met exhibits is its own class. **Training
images** are studio photos of exhibits (~397k images, long-tailed — 60.8% of classes have a
single image). **Query images** are visitor photos of exhibits (a distribution shift from the
studio training set) plus two kinds of **distractors** (other-artwork and non-artwork) that are
out-of-distribution and labeled `-1`. The method of choice is a **non-parametric kNN classifier**
over L2-normalized global descriptors from a fine-tuned backbone; it beats parametric classifiers,
especially on the long tail. See [../AGENTS.md](../AGENTS.md) for how this maps onto the code.

**Metrics:** ACC (Met queries only), GAP⁻ = GAP on Met queries only, GAP = Global Average
Precision (a.k.a. μAP) over *all* queries — rewards giving distractors lower confidence than true
Met queries. GAP is the headline metric.

## Dataset facts (composition, collection, splits)

- **Scale:** 397,121 exhibit (training) images / **224,408 classes**; val 129 Met + 1,168 other-art
  + 868 non-art queries; test 1,003 Met + 10,352 other-art + 7,964 non-art queries. Distractors are
  the `+1` class. Val/test classes are subsets of train classes; **no class overlap** between val
  and test, and Met queries are **grouped by photographer** before splitting (no user leakage).
- **Long tail (exact histogram, training images per class):** 1→136,466 classes, 2→57,297,
  3→12,157, 4→6,836, 5→3,739, 6→2,310, 7→1,325, 8→868, 9→628, 10→2,782 (sums to 224,408).
  → **60.8% of classes have a single training image**; only 1.2% have the max of 10.
  Query side: 689 classes have a single query, falling off fast.
- **Collection:** exhibit images from The Met open-access catalog (≤10 per exhibit, skewed aspect
  ratios excluded, de-duplicated). Met queries from museum visitors — **partly shot by the authors
  on an iPhone 11 Pro Max**, partly crawled from Flickr (39 photographers; a few prolific ones with
  300–500 queries). Distractors from Wikimedia Commons: **art categories → other-art**, **generic
  categories → non-art**. Built Sept 2020 – Sept 2021.
- **Format:** JPEG, max resolution 500×500, aspect ratio preserved.
- **License:** annotations CC BY 4.0; images are public-domain / Creative Commons (no copyright
  owned by the authors). Hosted at CTU Prague.

The **iPhone-11-Pro-Max** detail matters for this fork: the real Met queries are partly genuine
phone photos, so synthetic gallery phone-photos emulate the same capture device/conditions.

## `pairs_type` ⇄ paper-method mapping (key bridge)

The training code's `--pairs_type` flag uses different names than the paper. This mapping is the
single most useful thing to remember when reproducing or extending results:

| Code `--pairs_type` | Paper method | Positive pair | Negative pair |
|---------------------|--------------|---------------|---------------|
| `sim_siam_pos` | SimSiam | augmented anchor | none |
| `sim_siam_pos+new_neg` | **Con-Syn** | augmented anchor (synthetic) | hard negative |
| `pos+new_neg` | **Con-Syn+Real** | random same-class *or* synthetic | hard negative |
| `new_pos+new_neg` | **Con-Syn+Real-closest** | closest same-class *or* synthetic | hard negative |

`new_pos+new_neg` (Con-Syn+Real-closest) is the paper's best method and matches the released
`r18SWSL_con-syn+real-closest` checkpoint.

## Key results — the numbers to beat

**Pretrained backbone + kNN, no Met training** (MS + PCAw→512D; paper Table 2):

| Backbone | GAP | GAP⁻ | ACC |
|----------|----:|----:|----:|
| R18-ImageNet | 15.9 | 37.5 | 42.3 |
| R18-SWSL | 24.7 | 47.0 | 50.9 |
| R50-ImageNet | 22.2 | 41.8 | 46.4 |
| R50-SWSL | 30.4 | 52.9 | 56.3 |

**Trained on the Met + kNN** (paper Table 3):

| Method | GAP | GAP⁻ | ACC |
|--------|----:|----:|----:|
| R18-IN baseline (no Met train) | 15.9 | 37.5 | 42.3 |
| R18-IN SimSiam | 26.8 | 42.3 | 45.6 |
| R18-IN Con-Syn | 30.4 | 46.6 | 49.4 |
| R18-IN Con-Syn+Real | 29.8 | 46.0 | 48.8 |
| R18-IN Con-Syn+Real-closest | 32.5 | 47.5 | 50.0 |
| R18-SWSL baseline (no Met train) | 24.7 | 47.0 | 50.9 |
| **R18-SWSL Con-Syn+Real-closest** | **36.1** | **52.4** | **55.0** |

The bottom row (GAP **36.1**) is the paper's best **single-model** R18 result and the natural
primary baseline for this fork. Parametric DNet classifiers (CE/AF) are reported but consistently
underperform kNN.

**Dimensionality + descriptor concatenation** (appendix `suppl-fig/data.tex`; `★` =
Con-Syn+Real-closest, `+` = concatenate descriptors before PCAw). This is the paper's *strongest*
configuration and is easy to overlook — GAP at varying PCAw dimension `d`:

| d | R18IN | R18SWSL | R18SWSL★ | R18SWSL★+R18IN | R18SWSL★+R18SWSL |
|--:|------:|--------:|---------:|---------------:|-----------------:|
| 128 | 9.8 | 15.9 | 28.3 | 28.5 | 29.5 |
| 256 | 14.3 | 23.3 | 34.1 | 34.8 | 35.5 |
| 512 | 15.9 | 24.7 | **36.1** | 37.5 | 37.6 |
| 1024 | – | – | – | **38.0** | 37.5 |

So the **highest GAP reported in the paper is 38.0** (R18IN+R18SWSL★ concatenated, 1024-D), and the
highest ACC is **59.6** (R18SWSL★+R18SWSL, 1024-D). Takeaways for this fork: (a) report against
**36.1** for single-model comparisons but be aware of the **~38** concatenation ceiling; (b)
concatenating complementary descriptors before PCAw reliably helps; (c) higher `d` keeps helping.

## Implementation details (from the appendix, for reproduction)

- **Optimizer** Adam, weight decay `1e-6`; backbone LR `1e-7`, decayed ×0.1 halfway through.
- **Contrastive training:** 10 epochs, margin 1.8, batch 128 (= 64 pairs); one epoch = every
  training image used once as anchor. Hard negative = random pick among the 10 nearest images of a
  different class, recomputed each epoch from the current backbone.
- **Augmentations:** random crop scale `[0.7, 1.0]` resized to 500×500, color jitter p=0.8,
  grayscale p=0.2. (Matches `code/utils/augmentations.py`.)
- **kNN tuning:** grid search on GAP over k∈{1,2,3,5,7,10,15,20,50}, τ∈{0.01,0.1,1,5,…,100,500}.
- **Best-epoch selection** uses single-scale, no-PCAw kNN for speed.
- **Mini dataset** (appendix): 38,307 images / 33,501 classes — a fast sanity-check subset before
  full-database training (`--mini` flag in the code). R18-IN kNN on it: 27.1 GAP.

## Ablations & findings relevant to method design (appendix)

These are useful when designing the fork's new method — what the authors tried and what happened:

- **Long-tail recipes that did *not* help.** Starting from the DNet+ArcFace reference (36.6 ACC),
  class weighting (35.8), class-balanced sampling (33.4), and classifier retraining à la Kang et al.
  (35.0) all *failed to improve* accuracy. Standard imbalance fixes don't move the needle here — a
  bar the new method should clear or avoid.
- **Local descriptors are a promising untested direction.** HOW local descriptors + ASMK reach
  25.3 GAP / 47.6 GAP⁻ / 50.9 ACC for an R18 — best for that backbone with much higher cost. The
  authors flag local descriptors as promising future work given high inter-class similarity and the
  importance of fine artwork detail.
- **OOD ratio.** Difficulty scales with the fraction of distractors in the test set; a *small* amount
  of distractors in the validation set is enough to tune k,τ well.
- **Confidence normalization is essential.** Fixing k=1 (no soft-max over classes) collapses GAP on
  all queries (2.9 vs 15.9) — handling distractors needs the temperature-normalized confidence.

## Relevance to this fork (VISART work)

The paper's central difficulty is the **distribution shift** between studio exhibit images (train)
and real visitor photos (queries), compounded by the long tail. This fork's planned contributions —
a **synthetic dataset of phone photos of paintings in a gallery**, plus a **new method** — target
exactly that gap. Concrete hooks from a full read of the paper:

- The datasheet explicitly names **domain adaptation** as a supported use "given the domain shift
  between the exhibit and the query images" — direct support for the synthetic-data framing.
- Real Met queries were partly captured on an **iPhone 11 Pro Max**, so synthetic gallery
  phone-photos emulate the genuine query device/conditions, not an arbitrary augmentation.
- The synthetic images most naturally enter the existing pipeline as **extra positives** for a class
  (mining in `MET_pairs_dataset`) and/or extra exhibit/training images — both reuse current code
  paths. Keep GAP/GAP⁻/ACC and the val/test protocol unchanged so numbers stay comparable.
- For the **new method**, note what already failed (standard long-tail recipes, above) and what's
  flagged as promising but untested (local descriptors + ASMK). Descriptor **concatenation** before
  PCAw is a cheap, reliable booster worth combining with any new representation.

When extending the code or writing up results, compare against the tables above — primarily
**R18-SWSL Con-Syn+Real-closest, GAP 36.1** (single model), while being aware of the **~38 GAP**
concatenation ceiling — and reuse this paper's task definition, metrics, and protocol. See the
project notes in `AGENTS.md`.
