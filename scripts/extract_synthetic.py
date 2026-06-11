"""Step 3a: extract multi-scale descriptors for the synthetic gallery images using
our step-1 model (epoch 10). Saves descriptors + source Met id + camera angle.
Synthetic images are uniform 512x512 so we batch (unlike the variable-size Met images).
Set LIMIT=N for a quick CPU pre-flight. Output: data/descriptors/synthetic/synth_descriptors.pkl
Env overrides (defaults = v1): SYNTH_DS = dataset root, SYNTH_OUT = output dir, e.g. for v2:
  SYNTH_DS=.../visart-dataset-v2 SYNTH_OUT=data/descriptors/synthetic_v2
"""
import os, sys, re, glob, pickle
HERE = os.path.dirname(os.path.abspath(__file__)); REPO = os.path.dirname(HERE)
os.environ.setdefault("TORCH_HOME", os.path.join(REPO, "data", "torch_home"))
sys.path.insert(0, REPO)
import numpy as np, torch, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets.folder import default_loader
from code.networks.SiameseNet import siamese_network
from code.utils.augmentations import augmentation

DS  = os.environ.get("SYNTH_DS", "/mnt/storage_6/project_data/pl0896-03/visart-dataset")
CKPT = os.path.join(REPO, "data/models/r18SWSL_con-syn+real-closest/"
       "method:_contrastive_net:_r18_sw-sup_bckbn_lr:1e-07_b_size:_64_epochs:_10_wdecay:_1e-06_"
       "margin:_1.8_schedstep:_6_schedgamma:_0.1_imsize:_500_pairs_type:_new_pos+new_neg_pca_emb_proj_pretrained_seed:_0_epoch:_10")
OUT = os.path.join(REPO, os.environ.get("SYNTH_OUT", "data/descriptors/synthetic")); os.makedirs(OUT, exist_ok=True)
LIMIT = int(os.environ.get("LIMIT", "0"))

# (path, source Met id, camera angle) for every synthetic image
items = []
for folder in sorted(os.listdir(DS)):
    fdir = os.path.join(DS, folder); mfile = os.path.join(fdir, "metadata.json")
    if not os.path.isdir(fdir) or not os.path.exists(mfile):
        continue
    m = re.search(r'MET/(\d+)/\d+\.jpg', open(mfile).read())
    if not m:
        continue
    mid = int(m.group(1))
    for png in sorted(glob.glob(os.path.join(fdir, "*_rgb_*.png"))):
        angle = os.path.basename(png).split("_rgb_")[1].rsplit(".", 1)[0]
        items.append((png, mid, angle))
if LIMIT:
    items = items[:LIMIT]
print(f"synthetic query images: {len(items)}", flush=True)

tf = augmentation("augment_inference")
class SynDS(Dataset):
    def __init__(self, it): self.it = it
    def __len__(self): return len(self.it)
    def __getitem__(self, i): return tf(default_loader(self.it[i][0])), i   # default_loader -> RGB (drops alpha)
loader = DataLoader(SynDS(items), batch_size=64, shuffle=False, num_workers=8, pin_memory=True)

dev = "cuda" if torch.cuda.is_available() else "cpu"
model = siamese_network("r18_sw-sup", pooling="gem", pretrained=False, emb_proj=True)
model.backbone.projector.bias.data = model.backbone.projector.bias.data.unsqueeze(0)
model.load_state_dict(torch.load(CKPT, weights_only=False, map_location=dev)["state_dict"])
net = model.backbone.to(dev).eval()
dim = net.meta["outputdim"]; scales = [1.0, 1/np.sqrt(2), 0.5]
print(f"device={dev} dim={dim} scales={scales}", flush=True)

descr = np.zeros((len(items), dim), dtype="float32")
with torch.no_grad():
    for imgs, idx in loader:
        imgs = imgs.to(dev, non_blocking=True)
        v = torch.zeros(imgs.size(0), dim, device=dev)
        for sc in scales:                                   # multi-scale: normalize(sum_s net(x_s)) == repo extract_ms (msp=1)
            x = imgs if sc == 1.0 else F.interpolate(imgs, scale_factor=sc, mode="bilinear", align_corners=False)
            v = v + net(x)
        descr[idx.numpy()] = F.normalize(v, dim=1).cpu().numpy()
met_ids = np.array([it[1] for it in items]); angles = np.array([it[2] for it in items])
pickle.dump({"descriptors": descr, "met_ids": met_ids, "angles": angles},
            open(os.path.join(OUT, "synth_descriptors.pkl"), "wb"), protocol=4)
print(f"saved {OUT}/synth_descriptors.pkl  descr={descr.shape}  angles={sorted(set(angles.tolist()))}", flush=True)
