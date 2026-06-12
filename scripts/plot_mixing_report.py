"""Figures for the real/synthetic mixing report (experiments/real-vs-synthetic-mix/README.md).

Parses the eval logs and writes two clean, plain-titled figures:
  fig_paintings.png      -- painting recognition (GAP- on the 148 real painting photos) vs
                            synthetic %, in BOTH test settings (closed painting world + full
                            397k benchmark). Needs the cls re-score log (met-paint-cls-*).
  fig_full_benchmark.png -- whole Met benchmark GAP / GAP- / ACC vs synthetic %.

Each figure is written only when its inputs are present, so this can run before the cls
re-score finishes (it will just skip fig_paintings and print a note).

Run: .venv-dino/bin/python scripts/plot_mixing_report.py [--v2]
--v2 plots the dataset-v2 reproduction (EXP-12): reads the met-*-v2 logs (painting slice from
their PAINT148 lines; the shared all-real 0% point + references stay the v1 logs/values) and
writes into experiments-v2/real-vs-synthetic-mix/figures.
"""
import os, re, glob, argparse
os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl-" + os.environ.get("USER", "x"))
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

ap = argparse.ArgumentParser(); ap.add_argument("--v2", action="store_true")
V2 = ap.parse_args().v2
HERE = os.path.dirname(os.path.abspath(__file__)); REPO = os.path.dirname(HERE)
LOGS = os.path.join(REPO, "logs")
DOCS = os.path.join(REPO, ("experiments-v2" if V2 else "experiments") + "/real-vs-synthetic-mix/figures")
os.makedirs(DOCS, exist_ok=True)
PCTS = [0, 20, 40, 60, 80, 100]
MIXTAG = {20: "80r20s", 40: "60r40s", 60: "40r60s", 80: "20r80s", 100: "0r100s"}
BLUE, RED, GREEN = "#1f77b4", "#d62728", "#2ca02c"

def last(globpat):
    fs = sorted(glob.glob(os.path.join(LOGS, globpat))); return fs[-1] if fs else None
def find1(text, pat):
    m = re.search(pat, text, re.M); return float(m.group(1)) if m else None
def paint148(f):
    return find1(open(f).read(), r"PAINT148\s+GAP\(\+all distr\)\s+[\d.]+\s+GAP-\s+([\d.]+)") if f else None

# closed-world painting GAP- (2-fold-CV mean), by synthetic %. The 0% (all-real) point and the
# step-1 control are version-independent -> always the original v1 jobs. [0-9]* keeps the v1
# globs from matching the -v2- logs.
CLOSED = {0: "met-paint-eval-7336122.out"}
for p in MIXTAG:
    CLOSED[p] = f"met-ev-{MIXTAG[p]}-v2-*.out" if V2 else f"met-ev-{MIXTAG[p]}-[0-9]*.out"
def closed_gap(g):
    f = last(g); return find1(open(f).read(), r"2-fold mean:\s*GAP-\s*([\d.]+)") if f else None
closed = [closed_gap(CLOSED[p]) for p in PCTS]
cf = last("met-paint-eval-7336269.out"); closed_ref = (find1(open(cf).read(), r"2-fold mean:\s*GAP-\s*([\d.]+)") if cf else None)

# full benchmark GAP / GAP- / ACC, by synthetic % (0% = the shared v1 all-real model)
def full_log(pct):
    if pct == 0: return last("met-full-0s-[0-9]*.out")
    return last(f"met-full-{MIXTAG[pct]}-v2-*.out" if V2 else f"met-full-{pct}s-[0-9]*.out")
def full_metrics(pct):
    f = full_log(pct)
    if not f: return (None, None, None)
    t = open(f).read(); return (find1(t, r"^GAP\s+([\d.]+)"), find1(t, r"^GAP-\s+([\d.]+)"), find1(t, r"^ACC\s+([\d.]+)"))
fm = [full_metrics(p) for p in PCTS]
fgap, fgnd, facc = [x[0] for x in fm], [x[1] for x in fm], [x[2] for x in fm]
FULL_REF = dict(gap=35.97, gnd=52.14, acc=54.64)  # step-1 all-real-data model (EXP-1)

# full-benchmark painting GAP- (148q committed def): v1 from the cls re-score batch log,
# v2 from each full-eval log's PAINT148 line (same K=7/tau=50 protocol).
clsf = last("met-paint-cls-*.out"); full_paint, full_paint_ref = {}, None
if clsf:
    for m in re.finditer(r"CLSPAINT synth=(\w+) GAP [\d.]+ GAP- ([\d.]+)", open(clsf).read()):
        if m.group(1) == "ref": full_paint_ref = float(m.group(2))
        elif m.group(1).isdigit(): full_paint[int(m.group(1))] = float(m.group(2))
        # else: scaling keys (synth125/150/all) parsed separately for Fig 3
if V2:
    # PAINT148 lines from the v2 full-eval logs; the shared 0% (all-real) point keeps its
    # CLSPAINT value (the v1-era met-full-0s log predates the PAINT148 step in eval_full).
    for p in PCTS:
        v = paint148(full_log(p))
        if v is not None:
            full_paint[p] = v
full = [full_paint.get(p) for p in PCTS]

def topaxis(ax):
    s = ax.secondary_xaxis("top", functions=(lambda x: 100 - x, lambda x: 100 - x))
    s.set_xlabel("real % of training data"); s.set_xticks(PCTS)

# Fig 1 — painting recognition (headline): closed + full, same 148 real photos
if all(v is not None for v in closed) and all(v is not None for v in full):
    fig, ax = plt.subplots(figsize=(7.6, 4.9))
    ax.plot(PCTS, closed, "o-", color=BLUE, lw=2.5, ms=8, label="closed painting world (search 12.4k photos)")
    ax.plot(PCTS, full,   "s-", color=RED,  lw=2.5, ms=7, label="full Met benchmark (search 397k photos)")
    if closed_ref: ax.axhline(closed_ref, color=BLUE, ls=":", lw=1.3, alpha=.6); ax.text(1, closed_ref + .25, f"all-real-data model {closed_ref:.1f}", fontsize=7, color=BLUE)
    if full_paint_ref: ax.axhline(full_paint_ref, color=RED, ls=":", lw=1.3, alpha=.6); ax.text(1, full_paint_ref + .25, f"all-real-data model {full_paint_ref:.1f}", fontsize=7, color=RED)
    ax.set_xlabel("synthetic % of training data   (real % = 100 - synthetic %)")
    ax.set_ylabel("painting recognition  —  GAP- on 148 real photos (%)")
    ax.set_title("More synthetic training data -> better real-painting recognition")
    ax.set_xticks(PCTS); ax.grid(alpha=.3); ax.legend(loc="lower right", framealpha=.9); topaxis(ax)
    fig.tight_layout(); fig.savefig(os.path.join(DOCS, "fig_paintings.png"), dpi=150); print("wrote fig_paintings.png")
else:
    print("fig_paintings.png SKIPPED — cls re-score (met-paint-cls-*.out) not ready yet")

# Fig 2 — whole Met benchmark
if all(v is not None for v in fgap):
    fig, ax = plt.subplots(figsize=(7.6, 4.9))
    ax.plot(PCTS, facc, "^-", color=GREEN, lw=2, ms=7, label="ACC (top-1 on real Met queries)")
    ax.plot(PCTS, fgnd, "s-", color=BLUE,  lw=2, ms=7, label="GAP- (no distractors)")
    ax.plot(PCTS, fgap, "o-", color=RED,   lw=2, ms=7, label="GAP (with distractors)")
    for k, c in [("acc", GREEN), ("gnd", BLUE), ("gap", RED)]:
        ax.axhline(FULL_REF[k], color=c, ls=":", lw=1, alpha=.4)
    ax.text(99, FULL_REF["acc"] + .25, "dotted = all-real-data model (all 397k photos)", fontsize=6.5, ha="right", color="#555")
    ax.set_xlabel("synthetic % of training data   (real % = 100 - synthetic %)")
    ax.set_ylabel("whole Met benchmark score (%)")
    ax.set_title("Whole-benchmark scores vs training mix\n(painting-trained models; 397k DB + 19,319 queries incl. distractors)")
    ax.set_xticks(PCTS); ax.grid(alpha=.3); ax.legend(loc="center right", framealpha=.9); topaxis(ax)
    fig.tight_layout(); fig.savefig(os.path.join(DOCS, "fig_full_benchmark.png"), dpi=150); print("wrote fig_full_benchmark.png")
else:
    print("fig_full_benchmark.png SKIPPED — full-benchmark logs missing")
print("closed:", closed, "| full-paint:", full, "| full GAP/GAP-/ACC:", fgap, fgnd, facc)

# Fig 3 — synth-only DATA SCALING (0% real; more synthetic images, beyond the 12,403 budget)
def closed_for(tag):
    tag = "0r100s" if tag == "1x" else tag
    f = last(f"met-ev-{tag}-v2-*.out" if V2 else f"met-ev-{tag}-[0-9]*.out")
    return find1(open(f).read(), r"2-fold mean:\s*GAP-\s*([\d.]+)") if f else None
def fullpaint_for(key):
    if V2:
        tag = "0r100s" if key == "100" else key
        return paint148(last(f"met-full-{tag}-v2-*.out"))
    if not clsf: return None
    m = re.search(rf"CLSPAINT synth={key} GAP [\d.]+ GAP- ([\d.]+)", open(clsf).read())
    return float(m.group(1)) if m else None
SC_IMGS   = [12403, 15504, 18604, 24490]
SC_CLOSED = [closed_for(t) for t in ["1x", "synth125", "synth150", "synthall"]]
SC_FULLP  = [fullpaint_for(k) for k in ["100", "synth125", "synth150", "synthall"]]
base_closed = closed[0]; base_fullp = full_paint.get(0)   # all-real baselines (0 synthetic)
if all(v is not None for v in SC_CLOSED) and all(v is not None for v in SC_FULLP):
    fig, ax = plt.subplots(figsize=(7.8, 4.9))
    ax.plot(SC_IMGS, SC_CLOSED, "o-", color=BLUE, lw=2.5, ms=8, label="closed painting world (search 12.4k photos)")
    ax.plot(SC_IMGS, SC_FULLP,  "s-", color=RED,  lw=2.5, ms=7, label="full Met benchmark (search 397k photos)")
    if base_closed: ax.axhline(base_closed, color=BLUE, ls=":", lw=1.2, alpha=.55); ax.text(12600, base_closed + .25, f"all-real baseline {base_closed:.1f}", fontsize=7, color=BLUE)
    if base_fullp:  ax.axhline(base_fullp,  color=RED,  ls=":", lw=1.2, alpha=.55); ax.text(12600, base_fullp + .25,  f"all-real baseline {base_fullp:.1f}", fontsize=7, color=RED)
    ax.set_xlabel("number of synthetic training images (0% real)")
    ax.set_ylabel("painting recognition  —  GAP- on 148 real photos (%)")
    ax.set_title("Does MORE synthetic data keep helping?  (synth-only, beyond the 12,403 budget)")
    ax.set_xticks(SC_IMGS); ax.set_xticklabels([f"{n:,}\n({n/12403:.2f}x)" for n in SC_IMGS])
    ax.grid(alpha=.3); ax.legend(loc="center right", framealpha=.9)
    fig.tight_layout(); fig.savefig(os.path.join(DOCS, "fig_synth_scaling.png"), dpi=150)
    print("wrote fig_synth_scaling.png | scaling closed:", SC_CLOSED, "| scaling full-paint:", SC_FULLP)
else:
    print("fig_synth_scaling.png SKIPPED — scaling logs not all present")
