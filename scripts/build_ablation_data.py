"""Build the dataset-ablation training manifests + image roots (EXP-14).

One row per controlled variant of the v2 synthetic dataset, each trained EXACTLY like
EXP-12's best point -- the synth-only "all renders" run (data/gt_paint_synthall_v2):
training data = every painting-class render of that variant (no real images), recipe =
step 1, eval = the real closed painting world + the full 397k benchmark.

Rows (fixed ladder; see experiments-v2/dataset-ablation/README.md):
  abl0..abl4   visart-dataset-v2-abl{0..4}-*  -- procedural-randomization ladder
  1024         visart-dataset-v2-1024         -- default config rendered at 1024^2
  ang3 / ang1  visart-dataset-v2 filtered to arc angles {60,90,120} / {90} -- viewpoint count
  noframe      visart-dataset-v2-noframe      -- leave-one-out: default config + --bake-frames
  (rung 5 -- default v2, all renders -- is EXP-12's synthall_v2 run, reused, not rebuilt here)

Creates per row:  data/aug_<tag>/images/{MET,test_*,SYNTH} symlinks (ang* reuse data/aug_v2)
                  data/gt_paint_synth_<tag>/{MET_database.json, valset.json, testset.json}
Pool built folder->Met-id via metadata.json, painting classes only, shuffled @seed1 (same
as build_paintings_mix_data.py, so the full-dataset rows share EXP-12 synthall's slot order);
val/test stay the REAL closed-world queries -- synthetic enters TRAINING only.
stdlib only.  Run: .venv/bin/python scripts/build_ablation_data.py
"""
import os, re, glob, json, random

HERE = os.path.dirname(os.path.abspath(__file__)); REPO = os.path.dirname(HERE)
MET = "/mnt/storage_6/project_data/pl0896-03/met-dataset"
DS = "/mnt/storage_6/project_data/pl0896-03"
GT_PAINT = os.path.join(REPO, "data/gt_paint")

#        tag     dataset root                                       angles         im_root (None -> data/aug_<tag>)
ROWS = [("abl0", f"{DS}/visart-dataset-v2-abl0-none",                  None,          None),
        ("abl1", f"{DS}/visart-dataset-v2-abl1-tex",                   None,          None),
        ("abl2", f"{DS}/visart-dataset-v2-abl2-tex-light",             None,          None),
        ("abl3", f"{DS}/visart-dataset-v2-abl3-tex-light-glass",       None,          None),
        ("abl4", f"{DS}/visart-dataset-v2-abl4-tex-light-glass-frame", None,          None),
        ("1024", f"{DS}/visart-dataset-v2-1024",                       None,          None),
        ("ang3", f"{DS}/visart-dataset-v2",                            {60, 90, 120}, "data/aug_v2"),
        ("ang1", f"{DS}/visart-dataset-v2",                            {90},          "data/aug_v2"),
        ("noframe", f"{DS}/visart-dataset-v2-noframe",                 None,          None)]

paint_ids = {int(e["id"]) for e in json.load(open(os.path.join(GT_PAINT, "MET_database.json")))}

def link(src, dst):
    if os.path.islink(dst) or os.path.exists(dst):
        if os.path.realpath(dst) == os.path.realpath(src):
            return
        os.remove(dst)
    os.symlink(os.path.abspath(src), dst)

for tag, syn, angles, im_root in ROWS:
    if im_root is None:                      # wire data/aug_<tag>/images (as build_finetune_data.py)
        im_root = f"data/aug_{tag}"
        aug_img = os.path.join(REPO, im_root, "images"); os.makedirs(aug_img, exist_ok=True)
        for name in ("MET", "test_met", "test_other", "test_noart"):
            link(os.path.join(MET, name), os.path.join(aug_img, name))
        link(syn, os.path.join(aug_img, "SYNTH"))
    pool = []
    for folder in sorted((f for f in os.listdir(syn) if os.path.isdir(os.path.join(syn, f))), key=int):
        mfile = os.path.join(syn, folder, "metadata.json")
        if not os.path.exists(mfile):
            continue
        m = re.search(r'MET/(\d+)/\d+\.jpg', open(mfile).read())
        if not m or int(m.group(1)) not in paint_ids:
            continue
        for png in sorted(glob.glob(os.path.join(syn, folder, "*_rgb_*.png"))):
            if angles is None or int(os.path.basename(png).rsplit("_", 1)[1][:-4]) in angles:
                pool.append({"id": int(m.group(1)), "path": f"SYNTH/{folder}/{os.path.basename(png)}"})
    random.Random(1).shuffle(pool)
    n_cls = len({e["id"] for e in pool})
    want = 4898 * (5 if angles is None else len(angles))
    assert (n_cls, len(pool)) == (4898, want), f"{tag}: got {len(pool)} renders / {n_cls} classes, want {want} / 4898"
    out = os.path.join(REPO, f"data/gt_paint_synth_{tag}"); os.makedirs(out, exist_ok=True)
    json.dump(pool, open(os.path.join(out, "MET_database.json"), "w"))
    for j in ("valset.json", "testset.json"):
        link(os.path.join(GT_PAINT, j), os.path.join(out, j))
    print(f"  {tag:>4}: {len(pool):>6,} renders / {n_cls:,} classes  im_root={im_root}  -> data/gt_paint_synth_{tag}")

print("\nsubmit per row (see experiments-v2/dataset-ablation/README.md):")
print("  T=$(sbatch --parsable --job-name=met-tr-<tag> slurm/paint_train.slurm data/gt_paint_synth_<tag> <im_root> paint_synth_<tag>)")
print("  sbatch --dependency=afterok:$T --job-name=met-ev-<tag>   slurm/paint_eval.slurm data/models/r18SWSL_paint_synth_<tag> 10 synth_<tag>")
print("  sbatch --dependency=afterok:$T --job-name=met-full-<tag> slurm/eval_full.slurm  data/models/r18SWSL_paint_synth_<tag> 10 synth_<tag>")
