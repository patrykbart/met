"""Table B: retrieve against a PAINTING-ONLY database instead of the full 397k Met DB.

DB = Met Open Access Classification=="Paintings" (4,898 classes / 12,403 studio imgs), built by
subsetting the step-1 model's already-extracted train descriptors (NO re-extraction). PCAw is
re-learned on the painting DB; kNN K=7/tau=50. GAP includes the 18,316 real test distractors
(open-set); GAP- excludes them (== the closed-world painting GAP) — so both distractor treatments
show in one table. Scores the synthetic renders (all + per camera view, restricted to source classes
present in the painting DB) and the 148 real painting test queries; reports GAP / GAP- / ACC and
recall@1/5/10. ACC == R@1 by construction (tau=50 vote follows the nearest neighbour).

CPU-only; run via SLURM, not the login node.
Env override (default = v1): SYNTH_OUT = dir with synth_descriptors.pkl (also gets the summary json).
"""
import os, sys, csv, json, pickle
HERE = os.path.dirname(os.path.abspath(__file__)); REPO = os.path.dirname(HERE); sys.path.insert(0, REPO)
import numpy as np, faiss
from code.utils.utils import *                      # gap, gap_non_distr, classif_accuracy, PCAw helpers
from code.classifiers.knn_classifier import KNN_Classifier
K, TAU, DIM = 7, 50.0, 512

INFO = os.path.join(REPO, "data/ground_truth")
SYNDIR = os.path.join(REPO, os.environ.get("SYNTH_OUT", "data/descriptors/synthetic"))
DB   = os.path.join(REPO, "data/descriptors/r18_contr_loss_gem_fc_swsl_ms/descriptors.pkl")
SYN  = os.path.join(SYNDIR, "synth_descriptors.pkl")
CSVF = os.path.join(REPO, "data/MetObjects.csv")

# committed painting class set: Classification == "Paintings"
paint = set()
for r in csv.DictReader(open(CSVF, encoding="utf-8-sig", newline="")):
    oid = (r.get("Object ID") or "").strip()
    if oid and (r.get("Classification") or "").strip() == "Paintings":
        paint.add(int(oid))

# descriptors + labels
d = pickle.load(open(DB, "rb"))
train = np.ascontiguousarray(d["train_descriptors"], dtype="float32")
test  = np.ascontiguousarray(d["test_descriptors"],  dtype="float32")
train_labels_full = np.array([e["id"] for e in json.load(open(os.path.join(INFO, "MET_database.json")))])
test_meta   = json.load(open(os.path.join(INFO, "testset.json")))
test_labels = np.array([int(e["MET_id"]) if "MET_id" in e else -1 for e in test_meta])

# painting-only DB (subset of train) + real painting queries (148) + distractors (18,316)
db_pmask = np.isin(train_labels_full, list(paint))
pdb, pdb_labels = train[db_pmask], train_labels_full[db_pmask]
distr = test[test_labels == -1]; n_distr = len(distr)
paint_paths = {e["path"] for e in json.load(open(os.path.join(REPO, "data/gt_paint/testset.json")))}
rq_mask = np.array([e["path"] in paint_paths for e in test_meta])
realq, realq_lab = test[rq_mask], test_labels[rq_mask]

# synthetic renders whose source class is in the painting DB (every Classification=="Paintings" class
# has renders; the 54 synthetic classes outside this set (4,952 rendered - 4,898) are dropped as unanswerable)
s = pickle.load(open(SYN, "rb"))
synd = np.ascontiguousarray(s["descriptors"], dtype="float32")
mids = np.asarray(s["met_ids"]).astype(np.int64); angles = np.asarray(s["angles"])
in_db = np.isin(mids, list(paint)); synd, mids, angles = synd[in_db], mids[in_db], angles[in_db]
print(f"painting DB: {len(pdb)} imgs / {len(set(pdb_labels.tolist()))} classes | synthetic in-DB: {len(synd)} "
      f"({len(set(mids.tolist()))} classes) | real paint q: {len(realq_lab)} | distractors: {n_distr} | K={K} tau={TAU}", flush=True)

# PCAw learned on the painting DB, applied to DB + all queries
m, P = estimate_pca_whiten_with_shrinkage(pdb, shrinkage=1.0, dimensions=DIM)
pdb   = apply_pca_whiten_and_normalize(pdb,   m, P).astype("float32")
synd  = apply_pca_whiten_and_normalize(synd,  m, P).astype("float32")
realq = apply_pca_whiten_and_normalize(realq, m, P).astype("float32")
distr = apply_pca_whiten_and_normalize(distr, m, P).astype("float32")

# recall@k (faiss IP over the painting DB)
index = faiss.IndexFlatIP(DIM); index.add(pdb)
def recall(q, lab):
    _, I = index.search(q, 10); nbl = pdb_labels[I]; n = len(lab)
    r1  = np.mean(nbl[:, 0] == lab) * 100
    r5  = np.mean([lab[i] in nbl[i, :5]  for i in range(n)]) * 100
    r10 = np.mean([lab[i] in nbl[i, :10] for i in range(n)]) * 100
    return round(float(r1), 2), round(float(r5), 2), round(float(r10), 2)

# GAP/GAP-/ACC (kNN classifier on the painting DB); predict once over [synthetic + realq + distractors]
clf = KNN_Classifier(K=K, t=TAU); clf.fit(pdb, pdb_labels)
allq = np.ascontiguousarray(np.vstack([synd, realq, distr]), dtype="float32")
preds, confs = clf.predict(allq); preds = np.array(preds); confs = np.array(confs)
nS, nQ = len(mids), len(realq_lab)
sp, sc = preds[:nS], confs[:nS]
rp, rc = preds[nS:nS+nQ], confs[nS:nS+nQ]
dp, dc = preds[nS+nQ:], confs[nS+nQ:]
dlab = -np.ones(n_distr, dtype=np.int64)

def score(qp, qc, qlab, qdesc, label):
    g_open = gap(np.concatenate([qp, dp]), np.concatenate([qc, dc]), np.concatenate([qlab, dlab])) * 100
    g_no   = gap_non_distr(qp, qc, qlab) * 100          # == closed-world GAP (no distractors)
    acc    = classif_accuracy(qp, qlab) * 100
    r1, r5, r10 = recall(qdesc, qlab)
    print(f"PDB {label:<24} N={len(qlab):>5} GAP {g_open:6.2f}  GAP- {g_no:6.2f}  ACC {acc:6.2f}  "
          f"R@1 {r1:6.2f}  R@5 {r5:6.2f}  R@10 {r10:6.2f}", flush=True)
    return {"N": int(len(qlab)), "GAP": round(float(g_open), 2), "GAP-": round(float(g_no), 2),
            "ACC": round(float(acc), 2), "R@1": r1, "R@5": r5, "R@10": r10}

out = {"K": K, "tau": TAU, "db_imgs": int(len(pdb_labels)), "db_classes": int(len(set(pdb_labels.tolist()))),
       "n_distractors": int(n_distr),
       "real_paintings_148": score(rp, rc, realq_lab, realq, "real paintings (148)"),
       "synthetic": {"all_views": score(sp, sc, mids, synd, "synthetic ALL views")}}
for a in sorted(set(angles.tolist())):
    am = angles == a
    out["synthetic"][a] = score(sp[am], sc[am], mids[am], synd[am], f"synthetic {a}")
json.dump(out, open(os.path.join(SYNDIR, "painting_db_summary.json"), "w"), indent=2)
print(f"\nsaved {os.path.join(SYNDIR, 'painting_db_summary.json')}", flush=True)
