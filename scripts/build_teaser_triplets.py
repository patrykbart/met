"""Teaser picker: assemble [Met training | our v2 render | visitor test photo] triplets
for every PAINTING test query that also has a synthetic render, so we can choose the hero
painting for the paper teaser. stdlib + PIL only; login-node safe (light image I/O, no GPU).

Painting def = committed project definition (Met Open Access Classification == "Paintings").
Render shown here = v2 arc angle 90 (frontal, clearest identity); the rendering *degree*
(abl0..abl4 / full / noframe / 1024) is chosen separately, later.

Run:  .venv/bin/python scripts/build_teaser_triplets.py
Out:  data/teaser/pageNN.jpg  +  data/teaser/candidates.json (manifest for the degree step)
"""
import os, re, csv, json, glob
from collections import defaultdict
from PIL import Image, ImageDraw, ImageFont

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MET  = "/mnt/storage_6/project_data/pl0896-03/met-dataset"
V2   = "/mnt/storage_6/project_data/pl0896-03/visart-dataset-v2"
CSVF = os.path.join(REPO, "data/MetObjects.csv")
TEST = os.path.join(REPO, "data/ground_truth/testset.json")
OUT  = os.path.join(REPO, "data/teaser"); os.makedirs(OUT, exist_ok=True)
ANGLE = "90"   # frontal render for the picker

# --- 1. Met test queries that depict a known exhibit ----------------------------------
test = json.load(open(TEST))
met_q = [e for e in test if "MET_id" in e]
want_ids = {str(e["MET_id"]) for e in met_q}

# --- 2. join Met Open Access metadata: classification + title/artist/department -------
info = {}
with open(CSVF, encoding="utf-8-sig", newline="") as f:
    for row in csv.DictReader(f):
        oid = (row.get("Object ID") or "").strip()
        if oid in want_ids:
            info[oid] = {"cls":    (row.get("Classification") or "").strip(),
                         "title":  (row.get("Title") or "").strip(),
                         "artist": (row.get("Artist Display Name") or "").strip(),
                         "dept":   (row.get("Department") or "").strip()}

# --- 3. v2 synthetic folder -> Met id --------------------------------------------------
folder_of = {}
for folder in sorted((f for f in os.listdir(V2) if os.path.isdir(os.path.join(V2, f))), key=lambda x: int(x) if x.isdigit() else 1<<30):
    mfile = os.path.join(V2, folder, "metadata.json")
    if not os.path.exists(mfile):
        continue
    m = re.search(r'MET/(\d+)/\d+\.jpg', open(mfile).read())
    if m:
        folder_of[m.group(1)] = folder

# --- 4. build painting triplets (one row per class; all three sources must exist) -----
photos_per = defaultdict(list)
for e in met_q:
    photos_per[str(e["MET_id"])].append(e["path"])

cands, missing_render = [], 0
seen = set()
for e in met_q:
    mid = str(e["MET_id"])
    if mid in seen:
        continue
    meta = info.get(mid)
    if not meta or meta["cls"] != "Paintings":
        continue
    seen.add(mid)
    # training image (canonical first frame, fall back to any)
    train = os.path.join(MET, "MET", mid, "0.jpg")
    if not os.path.exists(train):
        g = sorted(glob.glob(os.path.join(MET, "MET", mid, "*.jpg")))
        train = g[0] if g else None
    n_train = len(glob.glob(os.path.join(MET, "MET", mid, "*.jpg")))
    # synthetic render at ANGLE (fall back to any angle)
    render = None
    fol = folder_of.get(mid)
    if fol:
        cand = os.path.join(V2, fol, f"{fol}_rgb_{ANGLE}.png")
        if not os.path.exists(cand):
            g = sorted(glob.glob(os.path.join(V2, fol, "*_rgb_*.png")))
            cand = g[0] if g else None
        render = cand
    photo = os.path.join(MET, e["path"])
    if not (train and render and os.path.exists(photo)):
        if not render:
            missing_render += 1
        continue
    cands.append({"met_id": mid, "train": train, "render": render, "photo": photo,
                  "n_train": n_train, "n_photos": len(photos_per[mid]), **meta})

# --- 5. order: well-known painting depts first, then richer classes --------------------
PRIO = {"European Paintings": 0, "The American Wing": 1, "Modern and Contemporary Art": 2,
        "Asian Art": 3, "Robert Lehman Collection": 4, "Islamic Art": 5}
cands.sort(key=lambda c: (PRIO.get(c["dept"], 9), c["dept"], -c["n_train"], int(c["met_id"])))

print(f"painting test queries with a full triplet: {len(cands)} classes "
      f"(missing render: {missing_render})")
json.dump(cands, open(os.path.join(OUT, "candidates.json"), "w"), indent=1)

# --- 6. montage pages ------------------------------------------------------------------
T, GUT, PAD, HEAD = 300, 360, 12, 64          # thumb, label gutter, padding, header
PER = 12
COLS = ["Met training (studio)", "Our render — v2 gallery", "Visitor photo (test query)"]
try:
    font  = ImageFont.truetype("/usr/share/fonts/dejavu/DejaVuSans.ttf", 17)
    fontb = ImageFont.truetype("/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf", 19)
    fonth = ImageFont.truetype("/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf", 22)
except OSError:
    font = fontb = fonth = ImageFont.load_default()

def thumb(path):
    im = Image.open(path).convert("RGB")
    im.thumbnail((T, T), Image.LANCZOS)
    c = Image.new("RGB", (T, T), "white")
    c.paste(im, ((T - im.width)//2, (T - im.height)//2))
    return c

def wrap(d, text, fnt, maxw):
    words, lines, cur = text.split(), [], ""
    for w in words:
        t = (cur + " " + w).strip()
        if d.textlength(t, font=fnt) <= maxw:
            cur = t
        else:
            if cur: lines.append(cur)
            cur = w
    if cur: lines.append(cur)
    return lines

pages = [cands[i:i+PER] for i in range(0, len(cands), PER)]
npages = len(pages)
for pi, page in enumerate(pages, 1):
    W = GUT + 3*T + 4*PAD
    H = HEAD + len(page)*(T + PAD) + PAD
    canvas = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(canvas)
    d.text((PAD, 10), f"Met painting teaser candidates — page {pi}/{npages}  "
           f"(render = v2 arc-angle {ANGLE})", font=fonth, fill="black")
    for ci, title in enumerate(COLS):
        x = GUT + PAD + ci*(T + PAD)
        d.text((x, HEAD - 24), title, font=fontb, fill=(20, 20, 90))
    for ri, c in enumerate(page):
        y = HEAD + ri*(T + PAD)
        # gutter label
        ty = y + 6
        d.text((PAD, ty), f"MET {c['met_id']}", font=fontb, fill=(150, 0, 0)); ty += 26
        for ln in wrap(d, c["title"] or "(untitled)", font, GUT - 2*PAD)[:3]:
            d.text((PAD, ty), ln, font=font, fill="black"); ty += 21
        if c["artist"]:
            for ln in wrap(d, c["artist"], font, GUT - 2*PAD)[:2]:
                d.text((PAD, ty), ln, font=font, fill=(80, 80, 80)); ty += 21
        ty += 4
        for ln in wrap(d, c["dept"], font, GUT - 2*PAD)[:2]:
            d.text((PAD, ty), ln, font=font, fill=(0, 90, 0)); ty += 20
        d.text((PAD, ty), f"{c['n_train']} train img · {c['n_photos']} visitor photo(s)",
               font=font, fill=(120, 120, 120))
        # thumbs
        for ci, key in enumerate(("train", "render", "photo")):
            x = GUT + PAD + ci*(T + PAD)
            canvas.paste(thumb(c[key]), (x, y))
    canvas.save(os.path.join(OUT, f"page{pi:02d}.jpg"), quality=90)
    print(f"saved page {pi:02d}: {len(page)} triplets")
print(f"\nTotal: {len(cands)} candidate paintings across {npages} pages -> {OUT}")
