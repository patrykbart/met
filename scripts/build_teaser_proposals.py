"""Teaser proposals: first 10 painting candidates, each as a full triplet using the FULL-FRAME
1024^2 gallery render (arc angle 30, NO zoom/crop) as the middle panel. Studio + query shown at
dataset resolution (selection only; the chosen one gets a high-res studio in the final export).

Out: figures/teaser/proposals/proposal_1024_30deg.jpg  (+ manifest.json)
Run: .venv/bin/python scripts/build_teaser_proposals.py
"""
import os, re, json, glob
from PIL import Image, ImageDraw, ImageFont

REPO   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MET    = "/mnt/storage_6/project_data/pl0896-03/met-dataset"
V2_1024 = "/mnt/storage_6/project_data/pl0896-03/visart-dataset-v2-1024"
OUT    = os.path.join(REPO, "figures/teaser/proposals"); os.makedirs(OUT, exist_ok=True)
ANGLE  = "30"

cand = json.load(open(os.path.join(REPO, "data/teaser/candidates.json")))[:10]
want = {c["met_id"] for c in cand}

# met_id -> 1024 render folder (independent set; scan metadata)
fol_of = {}
for f in os.listdir(V2_1024):
    d = os.path.join(V2_1024, f); m = os.path.join(d, "metadata.json")
    if not os.path.isdir(d) or not os.path.exists(m):
        continue
    mm = re.search(r'MET/(\d+)/\d+\.jpg', open(m).read())
    if mm and mm.group(1) in want:
        fol_of[mm.group(1)] = f

rows = []
for c in cand:
    fol = fol_of[c["met_id"]]
    render = glob.glob(os.path.join(V2_1024, fol, f"*_rgb_{ANGLE}.png"))[0]
    rows.append({**c, "render1024": render})
json.dump(rows, open(os.path.join(OUT, "manifest.json"), "w"), indent=1)

# --- montage -------------------------------------------------------------------------
TH, GUT, PAD, HEAD = 248, 320, 14, 60
COLS = ["Met training (studio)", "Our render — full 1024² frame, 30°", "Visitor photo (query)"]
try:
    font  = ImageFont.truetype("/usr/share/fonts/dejavu/DejaVuSans.ttf", 16)
    fontb = ImageFont.truetype("/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf", 18)
    fonth = ImageFont.truetype("/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf", 22)
except OSError:
    font = fontb = fonth = ImageFont.load_default()

def thumb(path):
    im = Image.open(path).convert("RGB"); im.thumbnail((TH, TH), Image.LANCZOS)
    c = Image.new("RGB", (TH, TH), "white"); c.paste(im, ((TH-im.width)//2, (TH-im.height)//2)); return c

def wrap(d, t, fnt, maxw):
    words, lines, cur = t.split(), [], ""
    for w in words:
        s = (cur+" "+w).strip()
        if d.textlength(s, font=fnt) <= maxw: cur = s
        else:
            if cur: lines.append(cur)
            cur = w
    if cur: lines.append(cur)
    return lines

W = GUT + 3*TH + 4*PAD
H = HEAD + len(rows)*(TH+PAD) + PAD
canvas = Image.new("RGB", (W, H), "white"); d = ImageDraw.Draw(canvas)
d.text((PAD, 12), "Teaser proposals — full-frame 1024² renders (angle 30°, no zoom). Pick a MET id.",
       font=fonth, fill="black")
for ci, t in enumerate(COLS):
    d.text((GUT+PAD+ci*(TH+PAD), HEAD-22), t, font=fontb, fill=(20, 20, 90))
for ri, c in enumerate(rows):
    y = HEAD + ri*(TH+PAD); ty = y+6
    d.text((PAD, ty), f"MET {c['met_id']}", font=fontb, fill=(150, 0, 0)); ty += 24
    for ln in wrap(d, c["title"] or "(untitled)", font, GUT-2*PAD)[:3]:
        d.text((PAD, ty), ln, font=font, fill="black"); ty += 19
    if c["artist"]:
        for ln in wrap(d, c["artist"], font, GUT-2*PAD)[:2]:
            d.text((PAD, ty), ln, font=font, fill=(90, 90, 90)); ty += 19
    for ci, key in enumerate(("train", "render1024", "photo")):
        canvas.paste(thumb(c[key]), (GUT+PAD+ci*(TH+PAD), y))
canvas.save(os.path.join(OUT, "proposal_1024_30deg.jpg"), quality=90)
print("saved", os.path.join(OUT, "proposal_1024_30deg.jpg"), canvas.size)
