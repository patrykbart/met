"""Export the paper teaser (caption-free 3-panel strip; captions added later in LaTeX):
  studio training image | our synthetic gallery render (FULL-FRAME 1024^2, no zoom) | visitor query.

Current hero (edit CONFIG to change):
  MET 435894 -- Giuseppe Bartolomeo Chiari, "Bathsheba at Her Bath", arc angle 60.

Resolution notes (the Met dataset ships images downsized to <=500px):
  studio : Met Open Access high-res original (public domain), committed downscaled to
           figures/teaser/src_studio_<id>.jpg. Falls back to the 500px dataset file if absent.
  render : v2 gallery at 1024^2 (visart-dataset-v2-1024), FULL frame -- no crop/zoom.
  query  : dataset visitor photo, intrinsically <=500px (no higher-res source); mild upscale only.

Writes to committed figures/teaser/:
  teaser_<id>.pdf / .png                      caption-free 3-panel composite (drop-in)
  panel_{studio,render,query}_<id>.{png,pdf}  individual panels for a LaTeX \\subfigure layout
Run: .venv/bin/python scripts/export_teaser.py
"""
import os, glob
from PIL import Image

REPO    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MET     = "/mnt/storage_6/project_data/pl0896-03/met-dataset"
V2_1024 = "/mnt/storage_6/project_data/pl0896-03/visart-dataset-v2-1024"
OUT     = os.path.join(REPO, "figures/teaser"); os.makedirs(OUT, exist_ok=True)

# ---- CONFIG (the chosen hero) --------------------------------------------------------
MET_ID = "435894"
FOLDER = "886"          # v2-1024 render folder for this painting
ANGLE  = "60"
QUERY  = f"{MET}/test_met/4b484589e4ef.jpg"
H      = 768            # composite common height
# --------------------------------------------------------------------------------------

STUDIO_HIRES = f"{OUT}/src_studio_{MET_ID}.jpg"
STUDIO_DS    = f"{MET}/MET/{MET_ID}/0.jpg"
RENDER       = glob.glob(f"{V2_1024}/{FOLDER}/*_rgb_{ANGLE}.png")[0]

# --- panels at full native resolution (render = FULL frame, no crop) ------------------
studio = Image.open(STUDIO_HIRES if os.path.exists(STUDIO_HIRES) else STUDIO_DS).convert("RGB")
render = Image.open(RENDER).convert("RGB")
query  = Image.open(QUERY).convert("RGB")
native = [("studio", studio), ("render", render), ("query", query)]
print("native panel sizes:", {n: "%dx%d" % im.size for n, im in native})

for name, im in native:
    im.save(f"{OUT}/panel_{name}_{MET_ID}.png")
    im.save(f"{OUT}/panel_{name}_{MET_ID}.pdf", "PDF", resolution=300.0)

# --- caption-free composite (common height; white gaps + thin margin) -----------------
def fit(im):
    return im.resize((max(1, round(im.width * H / im.height)), H), Image.LANCZOS)

imgs = [fit(im) for _, im in native]
M, G = 20, 42
Wc = 2*M + sum(p.width for p in imgs) + 2*G
comp = Image.new("RGB", (Wc, 2*M + H), "white")
x = M
for p in imgs:
    comp.paste(p, (x, M)); x += p.width + G
comp.save(f"{OUT}/teaser_{MET_ID}.png")
comp.save(f"{OUT}/teaser_{MET_ID}.pdf", "PDF", resolution=300.0)
print("composite: %dx%d" % comp.size, "->", f"{OUT}/teaser_{MET_ID}.pdf")
