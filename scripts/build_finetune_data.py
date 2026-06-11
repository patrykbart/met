"""Build training manifests + a shared image-root for the synthetic fine-tune runs.

Creates:
  data/aug/images/{MET,test_met,test_other,test_noart,SYNTH}  -> symlinks
  data/gt_aug/MET_database.json    = 397,121 studio + 24,760 synthetic   (COMBINED)
  data/gt_synth/MET_database.json  = 24,760 synthetic only               (SYNTH-ONLY)
  (+ valset/testset/mini symlinks in both gt dirs)

Synthetic path scheme: "SYNTH/<folder>/<file>.png" resolves via data/aug/images/SYNTH -> visart-dataset.

For another dataset version, pass --syn and --suffix (defaults reproduce the v1 layout above), e.g.:
  --syn .../visart-dataset-v2 --suffix _v2   ->  data/aug_v2, data/gt_aug_v2, data/gt_synth_v2
"""
import os, re, glob, json, argparse
HERE = os.path.dirname(os.path.abspath(__file__)); REPO = os.path.dirname(HERE)
MET = "/mnt/storage_6/project_data/pl0896-03/met-dataset"
ap = argparse.ArgumentParser()
ap.add_argument("--syn", default="/mnt/storage_6/project_data/pl0896-03/visart-dataset")
ap.add_argument("--suffix", default="", help="appended to data/aug, data/gt_aug, data/gt_synth")
args = ap.parse_args()
SYN, SUF = args.syn, args.suffix
AUG_IMG = os.path.join(REPO, f"data/aug{SUF}/images"); os.makedirs(AUG_IMG, exist_ok=True)

def link(src, dst):
    if os.path.islink(dst) or os.path.exists(dst):
        if os.path.realpath(dst) == os.path.realpath(src):
            return
        os.remove(dst)
    os.symlink(src, dst)

for name in ("MET", "test_met", "test_other", "test_noart"):
    link(os.path.join(MET, name), os.path.join(AUG_IMG, name))
link(SYN, os.path.join(AUG_IMG, "SYNTH"))

synth = []
for folder in sorted((f for f in os.listdir(SYN) if os.path.isdir(os.path.join(SYN, f))), key=int):
    mfile = os.path.join(SYN, folder, "metadata.json")
    if not os.path.exists(mfile):
        continue
    m = re.search(r'MET/(\d+)/\d+\.jpg', open(mfile).read())
    if not m:
        continue
    mid = int(m.group(1))
    for png in sorted(glob.glob(os.path.join(SYN, folder, "*_rgb_*.png"))):
        synth.append({"id": mid, "path": f"SYNTH/{folder}/{os.path.basename(png)}"})
orig = json.load(open(os.path.join(MET, "MET_database.json")))
print(f"original studio entries: {len(orig)} | synthetic entries: {len(synth)}")

for d, entries in ((f"data/gt_aug{SUF}", orig + synth), (f"data/gt_synth{SUF}", synth)):
    dd = os.path.join(REPO, d); os.makedirs(dd, exist_ok=True)
    json.dump(entries, open(os.path.join(dd, "MET_database.json"), "w"))
    for j in ("valset.json", "testset.json", "mini_MET_database.json"):
        link(os.path.join(MET, j), os.path.join(dd, j))
    print(f"{d}/MET_database.json: {len(entries):,} entries "
          f"({sum(1 for e in entries if e['path'].startswith('SYNTH'))} synthetic)")
