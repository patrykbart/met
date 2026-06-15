"""Export the paper teaser for MET 437372 (Raphael, Madonna and Child Enthroned with Saints).

Three-panel story:  Met studio training image  ->  our synthetic gallery render (v2, 30 deg,
cropped to the painting)  ->  real visitor query photo. The render's 30 deg oblique leans the
same way as the visitor photo, illustrating that our synthetic data mimics real visitor shots.

Self-contained (re-derives the crop from source). Writes to committed figures/teaser/:
  teaser_437372.pdf            composite, VECTOR captions + full-res embedded images (drop-in)
  teaser_437372.png            raster preview
  panel_{studio,render,query}.{png,pdf}   individual full-res panels for LaTeX \\subfigure
Run: .venv/bin/python scripts/export_teaser.py
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

REPO  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MET   = "/mnt/storage_6/project_data/pl0896-03/met-dataset"
V2FOL = "/mnt/storage_6/project_data/pl0896-03/visart-dataset-v2/1897"
OUT   = os.path.join(REPO, "figures/teaser"); os.makedirs(OUT, exist_ok=True)

STUDIO = f"{MET}/MET/437372/0.jpg"
QUERY  = f"{MET}/test_met/8f3a69ef8d7b.jpg"
R30    = f"{V2FOL}/0_rgb_30.png"


def square_crop(path):
    """Square crop centered on the painting (located via its blue pigment + dark frame)."""
    im = Image.open(path).convert("RGB")
    a = np.asarray(im).astype(np.float32)
    r, g, b = a[..., 0], a[..., 1], a[..., 2]
    val = a.max(2) / 255.0
    mask = ((b - np.maximum(r, g)) > 10) | (val < 0.10)
    ys, xs = np.where(mask)
    x0, x1 = np.percentile(xs, [1, 99]); y0, y1 = np.percentile(ys, [1, 99])
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    W0, H0 = im.size
    side = min((y1 - y0) * 1.12, W0, H0)
    sx0 = int(min(max(0, cx - side/2), W0 - side)); sy0 = int(min(max(0, cy - side/2), H0 - side))
    return im.crop((sx0, sy0, int(sx0 + side), int(sy0 + side)))


# --- panels (common height; preserve aspect) ------------------------------------------
H = 512
panels = [("studio", Image.open(STUDIO).convert("RGB"), "Met training image"),
          ("render", square_crop(R30),                  "Our synthetic render"),
          ("query",  Image.open(QUERY).convert("RGB"),  "Real visitor query")]

def fit(im):
    return im.resize((max(1, int(im.width * H / im.height)), H), Image.LANCZOS)

imgs = [fit(im) for _, im, _ in panels]

# save individual panels (full native res) as PNG + PDF for LaTeX \subfigure
for (name, im, _), _ in zip(panels, imgs):
    im.save(f"{OUT}/panel_{name}.png")
    im.convert("RGB").save(f"{OUT}/panel_{name}.pdf", "PDF", resolution=300.0)

# --- images-only composite (PIL: pixel-perfect layout, white gaps) --------------------
M, G = 22, 30                                  # outer margin, inter-panel gap
Wc = 2*M + sum(p.width for p in imgs) + 2*G
Hc = 2*M + H
comp = Image.new("RGB", (Wc, Hc), "white")
centers, x = [], M
for p in imgs:
    comp.paste(p, (x, M))
    centers.append((x + p.width/2) / Wc)       # panel x-center as fraction of composite width
    x += p.width + G

# --- matplotlib: composite as raster + vector captions underneath ---------------------
cap_frac = 0.135                               # caption band height as a fraction of the page
fig_w = 8.0
fig_h = fig_w * (Hc / Wc) / (1 - cap_frac)
fig = plt.figure(figsize=(fig_w, fig_h))
ax = fig.add_axes([0, cap_frac, 1, 1 - cap_frac]); ax.imshow(comp); ax.axis("off")
for cx, (_, _, cap) in zip(centers, panels):
    fig.text(min(max(cx, 0.12), 0.88), cap_frac * 0.50, cap, ha="center", va="center",
             fontsize=12.5, fontweight="bold", color=(0.10, 0.10, 0.14))
fig.savefig(f"{OUT}/teaser_437372.pdf")
fig.savefig(f"{OUT}/teaser_437372.png", dpi=200)
plt.close(fig)
print("wrote:", *sorted(os.listdir(OUT)), sep="\n  ")
print("->", OUT)
