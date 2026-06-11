"""GAP / GAP- / ACC for the SYNTHETIC renders used as queries.

Mirrors the real painting slice (scripts/eval_paintings_cls.py): kNN classifier K=7/tau=50 over the
full 397k studio DB, GAP scored against all 18,316 real test distractors, GAP-/ACC over the renders
only. Two query groups, each all-views + per camera view:
  - ALL synthetic renders (24,760 = 4,952 painting classes x 5 views)  -> the section-2 table
  - committed painting test subset (Classification=="Paintings", data/gt_paint/testset.json:
    122 classes x 5 views = 610 renders)                               -> the section-3 table
Distractor preds are identical to eval_paintings_cls.py (same descriptors + classifier), so the
synthetic GAP is directly comparable to the real-painting-query GAP. ACC == R@1 by construction
(the per-class max-sim score makes the tau=50 vote follow the single nearest neighbour). Reuses
already-extracted descriptors (no re-extraction).

CPU-only; run via SLURM, not the login node.
Env override (default = v1): SYNTH_OUT = dir with synth_descriptors.pkl (also gets the summary json).
"""
import os, sys, json, pickle
HERE = os.path.dirname(os.path.abspath(__file__)); REPO = os.path.dirname(HERE); sys.path.insert(0, REPO)
import numpy as np
from code.utils.utils import *                      # gap, gap_non_distr, classif_accuracy, PCAw helpers
from code.classifiers.knn_classifier import KNN_Classifier
K, TAU, DIM = 7, 50.0, 512

INFO = os.path.join(REPO, "data/ground_truth")
SYNDIR = os.path.join(REPO, os.environ.get("SYNTH_OUT", "data/descriptors/synthetic"))
DB   = os.path.join(REPO, "data/descriptors/r18_contr_loss_gem_fc_swsl_ms/descriptors.pkl")
SYN  = os.path.join(SYNDIR, "synth_descriptors.pkl")

# DB train + real distractors (label -1) from the step-1 descriptors
d = pickle.load(open(DB, "rb"))
train = np.ascontiguousarray(d["train_descriptors"], dtype="float32")
test  = np.ascontiguousarray(d["test_descriptors"],  dtype="float32")
train_labels = np.array([e["id"] for e in json.load(open(os.path.join(INFO, "MET_database.json")))])
test_labels  = np.array([int(e["MET_id"]) if "MET_id" in e else -1 for e in json.load(open(os.path.join(INFO, "testset.json")))])
distr = test[test_labels == -1]; n_distr = len(distr)

# all synthetic renders + committed painting subset
s = pickle.load(open(SYN, "rb"))
synd = np.ascontiguousarray(s["descriptors"], dtype="float32")
mids = np.asarray(s["met_ids"]).astype(np.int64); angles = np.asarray(s["angles"])
paint_cls = {int(e["MET_id"]) for e in json.load(open(os.path.join(REPO, "data/gt_paint/testset.json"))) if "MET_id" in e}
pmask = np.isin(mids, list(paint_cls))
print(f"synthetic renders: {len(synd)} (all) | painting subset: {int(pmask.sum())} ({len(paint_cls)} classes) | "
      f"real distractors: {n_distr} | K={K} tau={TAU}", flush=True)

# PCAw learned on train, applied to train + queries (same recipe as the eval)
m, P = estimate_pca_whiten_with_shrinkage(train, shrinkage=1.0, dimensions=DIM)
train = apply_pca_whiten_and_normalize(train, m, P).astype("float32")
synd  = apply_pca_whiten_and_normalize(synd,  m, P).astype("float32")
distr = apply_pca_whiten_and_normalize(distr, m, P).astype("float32")

clf = KNN_Classifier(K=K, t=TAU); clf.fit(train, train_labels)

# predict once over [all renders + distractors]; queries are independent, so slice afterwards
Q = np.ascontiguousarray(np.vstack([synd, distr]), dtype="float32")
preds, confs = clf.predict(Q); preds = np.array(preds); confs = np.array(confs)
nR = len(mids)
rp, rc = preds[:nR], confs[:nR]                       # render preds / confs
dp, dc = preds[nR:], confs[nR:]                       # distractor preds / confs
dlab = -np.ones(n_distr, dtype=np.int64)

def score(mask, label):
    p = np.concatenate([rp[mask], dp]); c = np.concatenate([rc[mask], dc])
    lab = np.concatenate([mids[mask], dlab])          # renders labeled by source class, distractors -1
    g_all = gap(p, c, lab) * 100                       # GAP : renders + all distractors
    g_no  = gap_non_distr(rp[mask], rc[mask], mids[mask]) * 100   # GAP- : renders only
    acc   = classif_accuracy(rp[mask], mids[mask]) * 100          # ACC  : renders only (== R@1)
    print(f"SYNGAP {label:<40} N={int(mask.sum()):>5} GAP {g_all:6.2f}  GAP- {g_no:6.2f}  ACC {acc:6.2f}", flush=True)
    return {"N": int(mask.sum()), "GAP": round(float(g_all), 2), "GAP-": round(float(g_no), 2), "ACC": round(float(acc), 2)}

def group(gmask, gname):
    print(f"\n=== {gname} ===", flush=True)
    g = {"all_views": score(gmask, f"{gname}: ALL views")}
    for a in sorted(set(angles[gmask].tolist())):
        g[a] = score(gmask & (angles == a), f"{gname}: {a}")
    return g

out = {"K": K, "tau": TAU, "n_distractors": int(n_distr),
       "all_synthetic": group(np.ones(nR, bool), "ALL synthetic"),
       "paintings": group(pmask, 'paintings (Classification=="Paintings")')}
json.dump(out, open(os.path.join(SYNDIR, "gap_summary.json"), "w"), indent=2)
print(f"\nsaved {os.path.join(SYNDIR, 'gap_summary.json')}", flush=True)
