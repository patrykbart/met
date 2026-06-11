"""Extract frozen DINOv3 (ViT-L) CLS features over the synthetic gallery renders,
using the IDENTICAL aspect-preserve preprocessing as art-research's Met extraction
(scripts/extract_met_dinov3_aspect.py) so synthetic and real features are directly
comparable. Synthetic renders are all 512x512, so aspect_preserve_resize is a no-op
and we can BATCH (the real path used batch=1 only because Met images vary in size).

Output (--out-dir):
  synth_dino_<model>_aspect<L>.npz : features(N,1024 f16) + aligned label arrays
                                     (met_id, folder, angle_idx, floor_idx, aspect, placard_x)
  synth_dino_records.json          : legends (angles, floors), src_paths, counts

Run with .venv-dino (transformers). Needs HF cache + offline env (see slurm).
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # scripts/
from synth_meta import SYNTH_ROOT, angles_for, parse_metadata

HF_MODELS = {
    "dinov3_vitl16": "facebook/dinov3-vitl16-pretrain-lvd1689m",
    "dinov3_vit7b16": "facebook/dinov3-vit7b16-pretrain-lvd1689m",
}


def aspect_preserve_resize(img, long_side, patch):
    """Identical to art-research: long side -> long_side, both dims floored to mult of patch."""
    W, H = img.size
    if W >= H:
        new_w = long_side
        new_h = max(int(round(H * long_side / W)), patch)
    else:
        new_h = long_side
        new_w = max(int(round(W * long_side / H)), patch)
    new_w = max((new_w // patch) * patch, patch)
    new_h = max((new_h // patch) * patch, patch)
    if (new_w, new_h) != img.size:
        img = img.resize((new_w, new_h), Image.BICUBIC)
    return img


def build_records(synth_root, angles, limit=None):
    """Flat list of per-(folder,angle) records, in deterministic folder->angle order."""
    folders = sorted(
        (d for d in os.listdir(synth_root)
         if d.isdigit() and os.path.isdir(os.path.join(synth_root, d))),
        key=int,
    )
    if limit:
        folders = folders[:limit]
    recs = []
    for fl in folders:
        fdir = os.path.join(synth_root, fl)
        m = parse_metadata(fdir)
        for ang in angles:
            path = os.path.join(fdir, f"0_rgb_{ang}.png")
            cam = m["cams"].get(ang) or {}
            loc = cam.get("loc") or (None, None, None)
            recs.append(dict(
                folder=int(fl), met_id=m["met_id"], angle=ang, floor=m["floor"],
                aspect=m["aspect"], placard_x=m["placard_x"], placard_z=m["placard_z"],
                cam_y=loc[1], cam_z=loc[2], path=path,
            ))
    return recs


class SynthDataset(Dataset):
    def __init__(self, records, long_side, patch):
        self.records = records
        self.long_side = long_side
        self.patch = patch
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, i):
        rec = self.records[i]
        try:
            img = Image.open(rec["path"]).convert("RGB")
        except Exception as e:
            print(f"[warn] failed to open {rec['path']}: {e}", flush=True)
            img = Image.new("RGB", (self.long_side, self.long_side), (128, 128, 128))
        img = aspect_preserve_resize(img, self.long_side, self.patch)
        x = torch.from_numpy(np.asarray(img).copy()).permute(2, 0, 1).float() / 255.0
        x = (x - self.mean) / self.std
        return x, i


@torch.inference_mode()
def run_extract(model, loader, device, dtype, n_total):
    feats = []
    t0 = time.time()
    done = 0
    for x, _ in loader:
        x = x.to(device, dtype=dtype)
        out = model(pixel_values=x)
        cls = out.last_hidden_state[:, 0]  # CLS token (matches art-research)
        feats.append(cls.float().to(torch.float16).cpu().numpy())
        done += x.shape[0]
        if done % (x.shape[0] * 20) < x.shape[0]:
            rate = done / (time.time() - t0)
            eta = (n_total - done) / max(rate, 1e-3) / 60
            print(f"  {done}/{n_total} | {rate:.1f} img/s | ETA {eta:.1f} min", flush=True)
    return np.concatenate(feats, axis=0)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=list(HF_MODELS.keys()), default="dinov3_vitl16")
    p.add_argument("--long-side", type=int, default=512)
    p.add_argument("--patch", type=int, default=16)
    p.add_argument("--dtype", default="bfloat16", choices=["float32", "bfloat16"])
    p.add_argument("--synth-root", default=SYNTH_ROOT)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--limit", type=int, default=None, help="first N folders (smoke)")
    p.add_argument("--dry-run", action="store_true", help="build records + checks, no model/GPU")
    args = p.parse_args()

    angles = angles_for(args.synth_root)
    print(f"building records from {args.synth_root} (angles={angles})", flush=True)
    recs = build_records(args.synth_root, angles, args.limit)
    n = len(recs)
    floors = sorted({r["floor"] for r in recs})
    floor_idx = {f: i for i, f in enumerate(floors)}
    ang_idx = {a: i for i, a in enumerate(angles)}
    n_missing_id = sum(r["met_id"] is None for r in recs)
    n_missing_file = sum(not os.path.isfile(r["path"]) for r in recs)
    print(f"records: {n}  (folders x {len(angles)} angles)", flush=True)
    print(f"floors: {floors}", flush=True)
    print(f"missing met_id: {n_missing_id} | missing files: {n_missing_file}", flush=True)
    assert n_missing_id == 0, "some folders have no recoverable met_id"
    assert n_missing_file == 0, "some expected render files are missing"

    labels = dict(
        met_id=np.array([r["met_id"] for r in recs], dtype=np.int64),
        folder=np.array([r["folder"] for r in recs], dtype=np.int64),
        angle_idx=np.array([ang_idx[r["angle"]] for r in recs], dtype=np.int8),
        floor_idx=np.array([floor_idx[r["floor"]] for r in recs], dtype=np.int8),
        aspect=np.array([r["aspect"] for r in recs], dtype=np.float32),
        placard_x=np.array([r["placard_x"] for r in recs], dtype=np.float32),
    )
    # extra factors only meaningfully randomized in v2 (placard row + per-camera pose jitter)
    for extra in ("placard_z", "cam_y", "cam_z"):
        vals = [r[extra] for r in recs]
        if all(v is not None for v in vals) and len({round(v, 4) for v in vals}) > 1:
            labels[extra] = np.array(vals, dtype=np.float32)

    os.makedirs(args.out_dir, exist_ok=True)
    recs_path = os.path.join(args.out_dir, "synth_dino_records.json")
    with open(recs_path, "w") as f:
        json.dump(dict(angles=angles, floors=floors, n=n,
                       src_paths=[r["path"] for r in recs]), f)
    print(f"wrote {recs_path}", flush=True)

    if args.dry_run:
        print("dry-run: records OK, skipping model.", flush=True)
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = getattr(torch, args.dtype)
    from transformers import AutoModel
    print(f"loading {HF_MODELS[args.model]} (dtype={dtype})...", flush=True)
    model = AutoModel.from_pretrained(HF_MODELS[args.model], dtype=dtype).eval().to(device)

    ds = SynthDataset(recs, args.long_side, args.patch)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, pin_memory=True)
    feats = run_extract(model, loader, device, dtype, n)
    assert feats.shape[0] == n, (feats.shape, n)
    print(f"features: {feats.shape} {feats.dtype}", flush=True)

    out_path = os.path.join(args.out_dir, f"synth_dino_{args.model}_aspect{args.long_side}.npz")
    np.savez(out_path, features=feats, **labels)
    print(f"wrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
