"""Build real/synthetic MIXING-RATIO training manifests for the closed painting world.

Sweep train-data composition at FIXED total size = 12,403 images (the real painting set
size, = the 100%-real baseline data/gt_paint). Each ratio is a real:synth split of 12,403:
  80:20 -> 9922:2481 | 60:40 -> 7442:4961 | 40:60 -> 4961:7442 | 20:80 -> 2481:9922 | 0:100 -> 0:12403
Subsamples are NESTED (real shuffled once @seed0, synth once @seed1, then prefixes) so
higher-real mixes are supersets of lower-real ones. val/test stay the REAL closed-world
queries -- evaluation is ALWAYS on real data; synthetic enters TRAINING only. The 100:0
point is the existing data/gt_paint (no manifest needed here).

Train each with:  --im_root data/aug  --info_dir data/gt_paint_mix_<tag>
(real paths MET/.. and synth paths SYNTH/.. both resolve under data/aug/images).
stdlib only; login-node safe.  Run: .venv/bin/python scripts/build_paintings_mix_data.py

For another dataset version, pass --syn and --suffix (defaults reproduce the v1 layout), e.g.:
  --syn .../visart-dataset-v2 --suffix _v2  ->  data/gt_paint_mix_<tag>_v2, data/gt_paint_synth*_v2
(train those with --im_root data/aug_v2, built by build_finetune_data.py --suffix _v2)
"""
import os, re, glob, json, random, argparse
HERE = os.path.dirname(os.path.abspath(__file__)); REPO = os.path.dirname(HERE)
ap = argparse.ArgumentParser()
ap.add_argument("--syn", default="/mnt/storage_6/project_data/pl0896-03/visart-dataset")
ap.add_argument("--suffix", default="", help="appended to every output manifest dir")
args = ap.parse_args()
SYN, SUF = args.syn, args.suffix
GT_PAINT = os.path.join(REPO, "data/gt_paint")
TOTAL = 12403
RATIOS = [(80, 20), (60, 40), (40, 60), (20, 80), (0, 100)]

real_pool = json.load(open(os.path.join(GT_PAINT, "MET_database.json")))   # 12,403 real painting imgs
paint_ids = {int(e["id"]) for e in real_pool}

# synthetic pool: renders for the painting classes only (folder -> Met id via metadata.json)
synth_pool = []
for folder in sorted((f for f in os.listdir(SYN) if os.path.isdir(os.path.join(SYN, f))), key=int):
    mfile = os.path.join(SYN, folder, "metadata.json")
    if not os.path.exists(mfile):
        continue
    m = re.search(r'MET/(\d+)/\d+\.jpg', open(mfile).read())
    if not m or int(m.group(1)) not in paint_ids:
        continue
    mid = int(m.group(1))
    for png in sorted(glob.glob(os.path.join(SYN, folder, "*_rgb_*.png"))):
        synth_pool.append({"id": mid, "path": f"SYNTH/{folder}/{os.path.basename(png)}"})

random.Random(0).shuffle(real_pool)
random.Random(1).shuffle(synth_pool)
print(f"real pool {len(real_pool):,} | synth pool {len(synth_pool):,} (painting classes) | total/run {TOTAL:,}\n")
assert len(synth_pool) >= TOTAL, "need >= TOTAL synth imgs for the 0%-real run"

def link(src, dst):
    if os.path.islink(dst) or os.path.exists(dst):
        os.remove(dst)
    os.symlink(os.path.abspath(src), dst)

for r, s in RATIOS:
    n_real = round(r / 100 * TOTAL); n_synth = TOTAL - n_real
    entries = real_pool[:n_real] + synth_pool[:n_synth]
    tag = f"{r}r{s}s"
    out = os.path.join(REPO, f"data/gt_paint_mix_{tag}{SUF}"); os.makedirs(out, exist_ok=True)
    json.dump(entries, open(os.path.join(out, "MET_database.json"), "w"))
    for j in ("valset.json", "testset.json"):
        link(os.path.join(GT_PAINT, j), os.path.join(out, j))
    n_cls = len({e["id"] for e in entries})
    print(f"  {tag:>8}: real {n_real:>6,} + synth {n_synth:>6,} = {len(entries):,} imgs / {n_cls:,} classes "
          f"-> data/gt_paint_mix_{tag}{SUF}")

# Synth-only DATA-SCALING runs (beyond the fixed 12,403 budget): 0% real, longer prefixes of the
# SAME shuffled synth_pool -> nested supersets of the 0r100s (1x) set. 200% (24,806) exceeds the
# pool, so the top point is capped at all 24,490 renders (~1.97x). Eval stays the same real closed
# world + full benchmark (synthetic in TRAINING only).
SCALE = [("synth125", round(1.25 * TOTAL)), ("synth150", round(1.50 * TOTAL)), ("synthall", len(synth_pool))]
for tag, n in SCALE:
    n = min(n, len(synth_pool))
    entries = synth_pool[:n]                                  # 0% real
    out = os.path.join(REPO, f"data/gt_paint_{tag}{SUF}"); os.makedirs(out, exist_ok=True)
    json.dump(entries, open(os.path.join(out, "MET_database.json"), "w"))
    for j in ("valset.json", "testset.json"):
        link(os.path.join(GT_PAINT, j), os.path.join(out, j))
    n_cls = len({e["id"] for e in entries})
    print(f"  {tag:>9}: synth {n:>6,} imgs (x{n / TOTAL:.2f} budget) / {n_cls:,} classes -> data/gt_paint_{tag}{SUF}")

print(f"\ntrain mix:     sbatch --job-name=met-tr-<tag> slurm/paint_train.slurm data/gt_paint_mix_<tag>{SUF} data/aug{SUF} paint_<tag>{SUF}")
print(f"train scaling: sbatch --job-name=met-tr-<tag> slurm/paint_train.slurm data/gt_paint_<tag>{SUF}     data/aug{SUF} paint_<tag>{SUF}")
