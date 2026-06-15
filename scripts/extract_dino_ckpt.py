"""Extract descriptors from an adapted DINOv3 checkpoint (head or LoRA), over the
ORIGINAL 397k studio DB + real queries, into THIS repo's descriptors.pkl format so
eval_fullgrid.py / eval_paintings.py evaluate it exactly like every other model.

Reuses /met's siamese_network/Embedder + datasets; batched (the inference-resize
transform makes all images 512x512, so we can batch instead of the batch=1 conv path
-> ~10x faster than extract_embeddings over 397k). Run with .venv-dino (transformers).
"""
import argparse
import os
import pickle
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch
from torch.utils.data import DataLoader

from code.networks.SiameseNet import siamese_network
from code.utils.datasets import MET_database, MET_queries
from code.utils.augmentations import augmentation


@torch.no_grad()
def extract(net, ds, name, bs, workers):
    loader = DataLoader(ds, batch_size=bs, shuffle=False, num_workers=workers, pin_memory=True)
    outs = []
    for i, (x, _) in enumerate(loader):
        outs.append(net(x.cuda()).cpu().numpy())
        if i % 200 == 0:
            print(f"  {name}: {i * bs}/{len(ds)}", flush=True)
    return np.concatenate(outs, axis=0).astype("float32")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--net", default="dinov3_vitl16")
    ap.add_argument("--netpath", default=None, help="adapted checkpoint (omit with --zeroshot)")
    ap.add_argument("--lora", action="store_true")
    ap.add_argument("--zeroshot", action="store_true",
                    help="frozen DINOv3, no projector, no checkpoint -- the ZS baseline")
    ap.add_argument("--dino-readout", default="cls", choices=["cls", "patch"],
                    help="token read out: 'cls' (EXP-6 method, GAP 48.16) or 'patch' (mean-pooled grid)")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--info-dir", default="data/ground_truth")
    ap.add_argument("--im-root", default="data/")
    ap.add_argument("--imsize", type=int, default=512)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--num-workers", type=int, default=8)
    args = ap.parse_args()

    if args.zeroshot:
        # frozen DINOv3, no FC projector -> same mean-pooled-patch representation the fine-tuned
        # arms use (minus the trained projector), so it is an apples-to-apples FT-vs-ZS control.
        print(f"zero-shot: frozen DINOv3, no projector, no checkpoint, readout={args.dino_readout}", flush=True)
        model = siamese_network(args.net, emb_proj=False, lora=False, dino_img_size=args.imsize,
                                dino_readout=args.dino_readout)
        net = model.backbone.cuda().eval()
        print(f"built ZS backbone; outputdim={net.meta['outputdim']}", flush=True)
    else:
        if args.netpath is None:
            sys.exit("--netpath is required unless --zeroshot")
        print(f"loading checkpoint: {args.netpath}", flush=True)
        ckpt = torch.load(args.netpath, weights_only=False, map_location="cuda")
        sd = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt

        model = siamese_network(args.net, emb_proj=True, lora=args.lora, dino_img_size=args.imsize,
                                dino_readout=args.dino_readout)
        # projector bias is saved as (1,D) from PCA init; reshape target to match before load
        _bk = "backbone.projector.bias"
        if _bk in sd and model.state_dict()[_bk].shape != sd[_bk].shape:
            model.backbone.projector.bias.data = model.backbone.projector.bias.data.reshape(sd[_bk].shape)
        model.load_state_dict(sd)
        net = model.backbone.cuda().eval()
        print(f"loaded; outputdim={net.meta['outputdim']}  lora={args.lora}", flush=True)

    tf = augmentation("augment_inference_resize", args.imsize)
    splits = {
        "train_descriptors": MET_database(root=args.info_dir, transform=tf, im_root=args.im_root),
        "test_descriptors":  MET_queries(root=args.info_dir, test=True,  transform=tf, im_root=args.im_root),
        "val_descriptors":   MET_queries(root=args.info_dir, test=False, transform=tf, im_root=args.im_root),
    }
    out = {}
    for key, ds in splits.items():
        print(f"extracting {key} ({len(ds)} images)...", flush=True)
        out[key] = extract(net, ds, key, args.batch_size, args.num_workers)
        print(f"  -> {out[key].shape}", flush=True)

    os.makedirs(args.out_dir, exist_ok=True)
    p = os.path.join(args.out_dir, "descriptors.pkl")
    with open(p, "wb") as f:
        pickle.dump(out, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"wrote {p}", flush=True)


if __name__ == "__main__":
    main()
