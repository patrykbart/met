"""Step 3b: retrieval of synthetic gallery images against the Met train DB.
Correct = the source Met class (whose studio image is in the DB). Reports recall@1/5/10
pooled over all 5 camera angles, then per angle. Same PCAw (learned on train) as the eval.
Env override (default = v1): SYNTH_OUT = dir with synth_descriptors.pkl (also gets the summary json).
"""
import os, sys, json, pickle
HERE = os.path.dirname(os.path.abspath(__file__)); REPO = os.path.dirname(HERE)
sys.path.insert(0, REPO)
import numpy as np, faiss
from code.utils.utils import estimate_pca_whiten_with_shrinkage, apply_pca_whiten_and_normalize

SYNDIR = os.path.join(REPO, os.environ.get("SYNTH_OUT", "data/descriptors/synthetic"))
DB  = os.path.join(REPO, "data/descriptors/r18_contr_loss_gem_fc_swsl_ms/descriptors.pkl")
SYN = os.path.join(SYNDIR, "synth_descriptors.pkl")
GT  = os.path.join(REPO, "data/ground_truth"); DIM = 512

train = np.ascontiguousarray(pickle.load(open(DB, "rb"))["train_descriptors"], dtype="float32")
train_labels = np.array([e["id"] for e in json.load(open(GT + "/MET_database.json"))])
s = pickle.load(open(SYN, "rb"))
synd = np.ascontiguousarray(s["descriptors"], dtype="float32"); mids = np.asarray(s["met_ids"]); angles = np.asarray(s["angles"])

m, P = estimate_pca_whiten_with_shrinkage(train, shrinkage=1.0, dimensions=DIM)   # PCAw learned on train, applied to both
train = apply_pca_whiten_and_normalize(train, m, P).astype("float32")
synd  = apply_pca_whiten_and_normalize(synd,  m, P).astype("float32")

index = faiss.IndexFlatIP(DIM); index.add(train)
_, I = index.search(synd, 10)                 # top-10 train neighbours per synthetic query
nbl = train_labels[I]                          # neighbour class labels (N, 10)

def recalls(mask):
    sub, gt = nbl[mask], mids[mask]; n = int(mask.sum())
    if n == 0:
        return 0, 0.0, 0.0, 0.0
    r1 = np.mean(sub[:, 0] == gt) * 100
    r5 = np.mean([gt[i] in sub[i, :5] for i in range(n)]) * 100
    r10 = np.mean([gt[i] in sub[i, :10] for i in range(n)]) * 100
    return n, r1, r5, r10

# query groups: all synthetic, then the committed painting subset.
# Committed painting def = Met Open Access Classification=="Paintings" (the 148 painting test
# queries in data/gt_paint/testset.json, spanning 122 distinct classes). Painting subset = renders
# whose source class is one of those test-query classes.
groups = [("ALL synthetic", np.ones(len(mids), bool))]
gt_paint = os.path.join(REPO, "data/gt_paint/testset.json")
if os.path.exists(gt_paint):
    paint_cls = {int(e["MET_id"]) for e in json.load(open(gt_paint)) if "MET_id" in e}
    groups.append((f'paintings (Classification=="Paintings", {len(paint_cls)} test classes)',
                   np.isin(mids, list(paint_cls))))

def row(mask):
    n, r1, r5, r10 = recalls(mask)
    return {"N": int(n), "R@1": round(float(r1), 2), "R@5": round(float(r5), 2), "R@10": round(float(r10), 2)}

print(f"DB: {len(train_labels):,} train images")
summary = {"db_train_images": int(len(train_labels)), "n_synthetic": int(len(mids)), "groups": {}}
for gname, gmask in groups:
    print(f"\n=== {gname} ===")
    print(f"{'angle':<14}{'N':>8}{'R@1':>8}{'R@5':>8}{'R@10':>8}")
    g = {"all_angles": row(gmask), "per_angle": {}}
    r = g["all_angles"]; print(f"{'ALL angles':<14}{r['N']:>8}{r['R@1']:>8.2f}{r['R@5']:>8.2f}{r['R@10']:>8.2f}")
    for a in sorted(set(angles.tolist())):
        r = row(gmask & (angles == a)); g["per_angle"][a] = r
        print(f"{a:<14}{r['N']:>8}{r['R@1']:>8.2f}{r['R@5']:>8.2f}{r['R@10']:>8.2f}")
    summary["groups"][gname] = g

out = os.path.join(SYNDIR, "retrieval_summary.json")
json.dump(summary, open(out, "w"), indent=2)
print(f"\nsaved {out}")
