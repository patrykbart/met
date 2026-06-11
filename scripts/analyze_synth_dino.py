"""Analyze the structure of DINOv3 ViT-L (frozen, aspect512 CLS) embeddings of the
synthetic gallery dataset, and compare to the real Met domains.

Questions:
  (A) Substructure: do synthetic embeddings cluster by CAMERA ANGLE (5) or by
      procedural NUISANCE hyperparameters (floor material, placard position)?
      -> linear-probe + kNN decodability and silhouette per factor (vs chance).
  (B) Domain gap: are studio (catalog) / synthetic (renders) / real painting-query
      (visitor photos) clouds separable? Centroid distances; per-VIEW cosine to the
      paired studio source (vs EXP-3 per-view retrieval R@1).

Inputs (in --dir): synth_dino_dinov3_vitl16_aspect512.npz, synth_dino_records.json,
                   real_dino_vitl16_aspect512.npz
Outputs (in --dir/analysis): summary.json + PNG figures (PCA / t-SNE / UMAP).
All features L2-normalized (cosine geometry) -- model-intrinsic, no train-fit PCAw.
"""
import argparse
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
from sklearn.model_selection import train_test_split

# Per-view retrieval R@1 overlay (R18-SWSL renders-as-queries; different model, but the
# framing ordering should transfer). Default = the v1 EXP-3 numbers; pass --r1-json to
# overlay another run (either a plain {angle: R@1} mapping or a retrieval_summary.json).
EXP3_R1 = {"left upper": 75.85, "front": 64.84, "right bottom": 21.95,
           "left bottom": 20.62, "right upper": 1.41}
SEED = 0


def load_r1(path, angles):
    """angle -> R@1 from --r1-json (plain mapping or eval_synthetic_retrieval summary)."""
    if not path:
        return EXP3_R1 if all(a in EXP3_R1 for a in angles) else {}
    d = json.load(open(path))
    if "groups" in d:  # retrieval_summary.json: use the ALL-synthetic per-angle block
        per = next(iter(d["groups"].values()))["per_angle"]
        return {a: v["R@1"] for a, v in per.items()}
    return d


def l2n(x):
    x = x.astype(np.float32)
    return x / np.clip(np.linalg.norm(x, axis=1, keepdims=True), 1e-8, None)


def faiss_knn_acc(X, y, k=10, seed=SEED):
    """kNN (cosine) decodability via a stratified 50/50 split, majority vote."""
    import faiss
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.5, random_state=seed, stratify=y)
    index = faiss.IndexFlatIP(Xtr.shape[1])
    index.add(np.ascontiguousarray(Xtr))
    _, I = index.search(np.ascontiguousarray(Xte), k)
    votes = ytr[I]  # (Nte, k)
    pred = np.array([np.bincount(row).argmax() for row in votes])
    return float((pred == yte).mean())


def lin_probe_acc(X, y, seed=SEED):
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.5, random_state=seed, stratify=y)
    clf = LogisticRegression(max_iter=2000, C=1.0, n_jobs=-1)
    clf.fit(Xtr, ytr)
    return float(clf.score(Xte, yte))


def quartile_bins(v):
    """Rank-based balanced quartiles (robust to value ties) -> 4 bins 0..3."""
    r = np.argsort(np.argsort(v))
    return (r * 4 // len(v)).astype(int)


def knn_composition(X, met_id, angle_idx, floor_idx, k=10):
    """For each render, what fraction of its k nearest (cosine) neighbours share its
    painting / angle / floor? Directly measures what dominates LOCAL structure."""
    import faiss
    index = faiss.IndexFlatIP(X.shape[1])
    index.add(np.ascontiguousarray(X))
    _, I = index.search(np.ascontiguousarray(X), k + 1)
    I = I[:, 1:]  # drop self (exact search -> col 0 is self)
    n = len(X)
    # NOTE confound: floor is randomized once PER PAINTING, so all 5 views share a floor
    # -> same-painting neighbours are always same-floor (and never same-angle). Condition
    # the angle/floor affinity on DIFFERENT-painting neighbours to de-confound.
    sp = met_id[I] == met_id[:, None]           # same painting
    diff = ~sp                                   # cross-painting neighbours
    ndiff = max(int(diff.sum()), 1)
    comp = dict(
        k=k,
        same_painting=float(sp.mean()),
        same_angle=float((angle_idx[I] == angle_idx[:, None]).mean()),
        same_floor=float((floor_idx[I] == floor_idx[:, None]).mean()),
        # de-confounded: among cross-painting neighbours only
        same_angle_xpaint=float(((angle_idx[I] == angle_idx[:, None]) & diff).sum() / ndiff),
        same_floor_xpaint=float(((floor_idx[I] == floor_idx[:, None]) & diff).sum() / ndiff),
        chance_same_painting=float(4.0 / (n - 1)),     # 4 other views of the same painting
        chance_same_angle=float(1.0 / len(np.unique(angle_idx))),
        chance_same_floor=float(1.0 / len(np.unique(floor_idx))),
    )
    print(f"  kNN(k={k}): same-painting {comp['same_painting']:.3f} (chance {comp['chance_same_painting']:.4f})\n"
          f"    cross-painting neighbours -> same-angle {comp['same_angle_xpaint']:.3f} | "
          f"same-floor {comp['same_floor_xpaint']:.3f}  (chance {comp['chance_same_angle']:.2f} each)",
          flush=True)
    return comp


def decode_factor(X, y, name):
    classes, counts = np.unique(y, return_counts=True)
    chance = float(counts.max() / counts.sum())  # majority-class baseline
    lin = lin_probe_acc(X, y)
    knn = faiss_knn_acc(X, y)
    # silhouette on a cosine sample
    ssize = min(4000, len(y))
    sil = float(silhouette_score(X, y, metric="cosine", sample_size=ssize, random_state=SEED))
    print(f"  [{name}] {len(classes)}-way | chance {chance:.3f} | "
          f"linprobe {lin:.3f} | knn {knn:.3f} | silhouette {sil:.3f}", flush=True)
    return dict(n_classes=int(len(classes)), chance=chance, linprobe=lin, knn=knn, silhouette=sil)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="data/synth_dino")
    ap.add_argument("--model", default="dinov3_vitl16")
    ap.add_argument("--r1-json", default=None,
                    help="per-angle R@1 overlay (plain mapping or retrieval_summary.json); "
                         "default = the v1 EXP-3 numbers when angle names match")
    args = ap.parse_args()
    outdir = os.path.join(args.dir, "analysis")
    os.makedirs(outdir, exist_ok=True)

    s = np.load(os.path.join(args.dir, f"synth_dino_{args.model}_aspect512.npz"))
    rec = json.load(open(os.path.join(args.dir, "synth_dino_records.json")))
    r = np.load(os.path.join(args.dir, "real_dino_vitl16_aspect512.npz"))
    angles = rec["angles"]; floors = rec["floors"]

    synth = l2n(s["features"])
    angle_idx = s["angle_idx"].astype(int)
    floor_idx = s["floor_idx"].astype(int)
    placard_x = s["placard_x"]; aspect = s["aspect"]; smet = s["met_id"]
    studio = l2n(r["studio_feats"]); studio_met = r["studio_met_id"]
    pq = l2n(r["pq_feats"])
    print(f"synth {synth.shape} | studio {studio.shape} | painting-queries {pq.shape}", flush=True)

    summary = {"counts": {"synth": int(len(synth)), "studio": int(len(studio)),
                          "painting_queries": int(len(pq))},
               "angles": angles, "floors": floors}

    # ---------- (A) synthetic substructure: which factor does DINOv3 encode? ----------
    print("\n=== (A) substructure decodability (synthetic only) ===", flush=True)
    # continuous nuisance -> balanced rank quartiles (robust to ties)
    placard_bin = quartile_bins(placard_x)
    aspect_bin = quartile_bins(aspect)
    summary["substructure"] = {
        "camera_angle": decode_factor(synth, angle_idx, "camera_angle"),
        "floor_material": decode_factor(synth, floor_idx, "floor_material"),
        "placard_x_quartile": decode_factor(synth, placard_bin, "placard_x_quartile"),
        "aspect_quartile": decode_factor(synth, aspect_bin, "aspect_quartile"),
    }
    # extra recorded factors (v2): placard row + per-camera pose jitter. Decode only what
    # varies WITHIN angle groups (else it's just the angle factor in disguise).
    for extra in ("placard_z", "cam_y", "cam_z"):
        if extra not in s.files:
            continue
        v = s[extra]
        within = max(float(v[angle_idx == i].std()) for i in range(len(angles)))
        if within < 1e-4:
            continue
        summary["substructure"][f"{extra}_quartile"] = decode_factor(
            synth, quartile_bins(v), f"{extra}_quartile")
    # local structure: what dominates each render's nearest neighbours?
    summary["knn_composition"] = knn_composition(synth, smet, angle_idx, floor_idx, k=10)

    # ---------- (B) domain separability + geometry ----------
    print("\n=== (B) domain separability ===", flush=True)
    dom_X = np.concatenate([studio, synth, pq], 0)
    dom_y = np.concatenate([np.zeros(len(studio)), np.ones(len(synth)), np.full(len(pq), 2)]).astype(int)
    summary["domain_3way"] = decode_factor(dom_X, dom_y, "domain_3way(studio/synth/query)")
    # pairwise (subsample synth to balance)
    rng = np.random.default_rng(SEED)
    ssub = synth[rng.choice(len(synth), min(len(synth), len(studio)), replace=False)]
    for a_name, A, b_name, B in [("studio", studio, "synth", ssub),
                                 ("studio", studio, "query", pq),
                                 ("synth", ssub, "query", pq)]:
        X = np.concatenate([A, B], 0); y = np.r_[np.zeros(len(A)), np.ones(len(B))].astype(int)
        summary.setdefault("domain_pairwise", {})[f"{a_name}_vs_{b_name}"] = dict(
            linprobe=lin_probe_acc(X, y), n=(len(A), len(B)))
        print(f"  {a_name} vs {b_name}: linprobe {summary['domain_pairwise'][f'{a_name}_vs_{b_name}']['linprobe']:.3f}", flush=True)

    # centroids (cosine distance between domain/per-view means, then re-normalized)
    def cen(x):
        c = x.mean(0); return c / np.linalg.norm(c)
    cents = {"studio": cen(studio), "query": cen(pq)}
    for i, a in enumerate(angles):
        cents[f"synth:{a}"] = cen(synth[angle_idx == i])
    names = list(cents); M = np.stack([cents[n] for n in names])
    cosdist = 1 - M @ M.T
    summary["centroid_cosine_distance"] = {"order": names, "matrix": cosdist.round(4).tolist()}

    # per-view cosine to the PAIRED studio source (key domain-gap-per-view number)
    print("\n=== per-view cosine to paired studio source ===", flush=True)
    r1_overlay = load_r1(args.r1_json, angles)
    studio_row = {int(m): i for i, m in enumerate(studio_met)}
    per_view = {}
    for i, a in enumerate(angles):
        m = angle_idx == i
        sims = [float(synth[j] @ studio[studio_row[int(smet[j])]])
                for j in np.where(m)[0] if int(smet[j]) in studio_row]
        per_view[a] = dict(mean_cos_to_studio=float(np.mean(sims)), n=len(sims),
                           retrieval_R1=r1_overlay.get(a))
        print(f"  {a:>12}: cos->studio {per_view[a]['mean_cos_to_studio']:.3f}  (retrieval R@1 {r1_overlay.get(a)})", flush=True)
    summary["per_view_to_studio"] = per_view

    json.dump(summary, open(os.path.join(outdir, "summary.json"), "w"), indent=2)
    print(f"\nwrote {os.path.join(outdir, 'summary.json')}", flush=True)

    # ---------------- figures ----------------
    make_figures(outdir, synth, angle_idx, floor_idx, studio, pq, angles, floors,
                 per_view, summary["knn_composition"])
    print("figures written.", flush=True)


def _proj(Xsample, seed=SEED):
    """PCA-50 -> t-SNE-2D (standard), PCA-2D, and UMAP-2D (cosine, on the L2-normed
    vectors). Returns (pca2d, tsne2d, umap2d)."""
    import umap
    p50 = PCA(n_components=min(50, Xsample.shape[1]), random_state=seed).fit_transform(Xsample)
    tsne = TSNE(n_components=2, init="pca", perplexity=30, random_state=seed,
                max_iter=1000).fit_transform(p50)
    pca2 = PCA(n_components=2, random_state=seed).fit_transform(Xsample)
    umap2 = umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.1, metric="cosine",
                      random_state=seed).fit_transform(Xsample)
    return pca2, tsne, umap2


def make_figures(outdir, synth, angle_idx, floor_idx, studio, pq, angles, floors, per_view, comp):
    rng = np.random.default_rng(SEED)
    # (0) kNN neighbour composition: what dominates local structure? (de-confounded)
    fig, ax = plt.subplots(figsize=(6.5, 4.4))
    vals = [comp["same_painting"], comp["same_angle_xpaint"], comp["same_floor_xpaint"]]
    chance = [comp["chance_same_painting"], comp["chance_same_angle"], comp["chance_same_floor"]]
    labels = ["same painting\n(of any angle)", "same angle\n(diff. painting)", "same floor\n(diff. painting)"]
    xs = range(3)
    ax.bar(xs, vals, color=["#9467bd", "#ff7f0e", "#8c564b"], alpha=.85, label=f"observed (k={comp['k']})")
    ax.plot(xs, chance, "k_", markersize=34, markeredgewidth=2, label="chance")
    for i, (v, c) in enumerate(zip(vals, chance)):
        ax.text(i, v + .02, f"{v:.2f}\n(chance {c:.3f})", ha="center", fontsize=8)
    ax.set_xticks(list(xs)); ax.set_xticklabels(labels)
    ax.set_ylabel("fraction of 10 nearest neighbours"); ax.set_ylim(0, 1.05)
    ax.set_title("What dominates local DINOv3 structure? (synthetic)")
    ax.legend(fontsize=9, loc="upper right"); fig.tight_layout()
    fig.savefig(os.path.join(outdir, "knn_composition.png"), dpi=140); plt.close(fig)
    # balanced subsample of synth for plotting (per angle), + studio sample + all queries
    per_ang = 1200
    sidx = np.concatenate([rng.choice(np.where(angle_idx == i)[0],
                                      min(per_ang, (angle_idx == i).sum()), replace=False)
                           for i in range(len(angles))])
    st_idx = rng.choice(len(studio), min(2000, len(studio)), replace=False)
    Xsyn = synth[sidx]; a_s = angle_idx[sidx]; f_s = floor_idx[sidx]
    Xst = studio[st_idx]
    Xcomb = np.concatenate([Xst, Xsyn, pq], 0)
    dom = np.concatenate([np.zeros(len(Xst)), np.ones(len(Xsyn)), np.full(len(pq), 2)]).astype(int)
    print(f"projecting {len(Xcomb)} points (PCA, t-SNE, UMAP)...", flush=True)
    pca2, tsne, umap2 = _proj(Xcomb)

    # (1) domain, all three projections
    for proj, tag in [(tsne, "tsne"), (pca2, "pca"), (umap2, "umap")]:
        fig, ax = plt.subplots(figsize=(7, 6))
        for d, lab, c, mk, al in [(0, "studio (catalog)", "#1f77b4", "o", .5),
                                  (1, "synthetic (renders)", "#ff7f0e", ".", .35),
                                  (2, "real painting query", "#2ca02c", "*", .9)]:
            mm = dom == d
            ax.scatter(proj[mm, 0], proj[mm, 1], s=(28 if d == 2 else 9), c=c, marker=mk,
                       alpha=al, label=lab, linewidths=0)
        ax.set_title(f"DINOv3 ViT-L embeddings by domain ({tag})")
        ax.legend(markerscale=2, fontsize=9); ax.set_xticks([]); ax.set_yticks([])
        fig.tight_layout(); fig.savefig(os.path.join(outdir, f"proj_domain_{tag}.png"), dpi=140)
        plt.close(fig)

    # (2) synthetic colored by angle, and by floor (t-SNE + UMAP)
    syn_mask = dom == 1
    for proj, ptag in [(tsne, "tsne"), (umap2, "umap")]:
        syn_proj = proj[syn_mask]
        for labels, names, fname, title in [
            (a_s, angles, f"proj_synth_angle_{ptag}.png", "Synthetic embeddings by camera angle"),
            (f_s, floors, f"proj_synth_floor_{ptag}.png", "Synthetic embeddings by floor material")]:
            fig, ax = plt.subplots(figsize=(7, 6))
            cmap = plt.get_cmap("tab10")
            for i, nm in enumerate(names):
                mm = labels == i
                ax.scatter(syn_proj[mm, 0], syn_proj[mm, 1], s=8, color=cmap(i), alpha=.5, label=nm, linewidths=0)
            ax.set_title(f"{title} ({ptag})"); ax.legend(markerscale=2, fontsize=9)
            ax.set_xticks([]); ax.set_yticks([])
            fig.tight_layout(); fig.savefig(os.path.join(outdir, fname), dpi=140); plt.close(fig)

    # (3) per-view cosine-to-studio vs paired retrieval R@1 (when an overlay is available)
    order = sorted(angles, key=lambda a: per_view[a]["mean_cos_to_studio"], reverse=True)
    cos = [per_view[a]["mean_cos_to_studio"] for a in order]
    r1 = [per_view[a]["retrieval_R1"] for a in order]
    fig, ax1 = plt.subplots(figsize=(7, 4.5))
    ax1.bar(range(len(order)), cos, color="#ff7f0e", alpha=.8)
    ax1.set_ylabel("mean cosine to paired studio source", color="#ff7f0e")
    ax1.set_xticks(range(len(order))); ax1.set_xticklabels(order, rotation=20)
    if all(v is not None for v in r1):
        ax2 = ax1.twinx(); ax2.plot(range(len(order)), r1, "o-", color="#1f77b4")
        ax2.set_ylabel("retrieval R@1 (%)", color="#1f77b4")
    ax1.set_title("Per-view: render↔studio similarity vs retrievability")
    fig.tight_layout(); fig.savefig(os.path.join(outdir, "per_view_to_studio.png"), dpi=140); plt.close(fig)


if __name__ == "__main__":
    main()
