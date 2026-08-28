"""figs3.py -- Paper 3 figures.  Each is checked for overlapping/clipped labels by
constrained_layout + explicit margins; LOOK AT THEM before shipping."""
import _path  # noqa: F401  (adds code/ and code/vendor/ to sys.path)

import json, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ablate2 as A, robust as RB
from run_feature2 import FAMS

R = "/mnt/user-data/outputs/p3/results"
OUT = "/mnt/user-data/outputs/p3/figs"; os.makedirs(OUT, exist_ok=True)
rob = json.load(open(f"{R}/robust.json"))
lea = json.load(open(f"{R}/learners.json"))
robk = json.load(open(f"{R}/robust_k.json"))
plt.rcParams.update({"font.size": 9, "axes.grid": True, "grid.alpha": .3,
                     "figure.dpi": 160, "savefig.bbox": "tight"})

# ---------------------------------------------------------------- fig 1: gain vs cost
COST = {"v2_48": 1.16, "v2_24grey": 0.25, "v2_16grey": 0.11}
ORIG = {"orig_48": 1.12, "orig_16": 0.11}
fig, ax = plt.subplots(figsize=(5.2, 3.6), constrained_layout=True)
def band(d, key, c, lab, mk):
    x = [d[k] for k in d]; 
    ys = [np.array(rob[f"{k}|{key}"]["C22"]) for k in d]
    m = [np.median(y) for y in ys]
    lo = [np.percentile(y, 25) for y in ys]; hi = [np.percentile(y, 75) for y in ys]
    o = np.argsort(x); x = np.array(x)[o]; m = np.array(m)[o]
    lo = np.array(lo)[o]; hi = np.array(hi)[o]
    ax.fill_between(x, lo, hi, color=c, alpha=.15)
    ax.plot(x, m, mk + "-", color=c, label=lab, ms=6)
band(COST, "frac+mode2", "#1f77b4", "frac + mode-2 char (4-D), this work", "o")
band(ORIG, "frac+mode2", "#d62728", "frac + mode-2 char, original extractor", "s")
fr = np.array(rob["shared|frac"]["C22"]); px = np.array(rob["shared|pixels"]["C22"])
ax.axhline(np.median(fr), color="k", ls="--", lw=1)
ax.text(.085, np.median(fr) + .015, "solid fraction alone (free)", ha="left", fontsize=8)
ax.axhline(np.median(px), color="gray", ls=":", lw=1)
ax.text(.085, np.median(px) - .050, "2304 raw pixels", ha="left", color="gray", fontsize=8)
ax.set_xscale("log"); ax.set_xlim(.08, 1.5)
ax.set_xlabel("descriptor cost / cost of one homogenisation")
ax.set_ylabel(r"held-out $C_{22}$  $R^2$  (median, IQR)")
ax.set_title("Transfer to unseen topologies vs. descriptor cost", fontsize=10)
ax.legend(loc="upper left", fontsize=8, framealpha=.95, bbox_to_anchor=(0.0, 0.99))
ax.set_ylim(-.13, .60)
fig.savefig(f"{OUT}/fig1_gain_vs_cost.png"); plt.close(fig)

# ---------------------------------------------------------------- fig 2: per-family
a = lea["v2_24grey|frac|RF"]["perfam"]; b = lea["v2_24grey|frac+mode2|RF"]["perfam"]
am = lea["v2_24grey|frac|MLP"]["perfam"]; bm = lea["v2_24grey|frac+mode2|MLP"]["perfam"]
d = {f: b[f] - a[f] for f in a}; dm = {f: bm[f] - am[f] for f in am}
order = sorted(d, key=lambda f: d[f])
y = np.arange(len(order))
fig, ax = plt.subplots(figsize=(5.6, 5.0), constrained_layout=True)
ax.barh(y - .2, [np.clip(d[f], -3, 3) for f in order], .4, color="#1f77b4", label="random forest")
ax.barh(y + .2, [np.clip(dm[f], -3, 3) for f in order], .4, color="#ff7f0e", label="MLP")
for i, f in enumerate(order):
    for v, off in ((d[f], -.2), (dm[f], .2)):
        if abs(v) > 3:
            ax.text(3.05 if v > 0 else -3.05, i + off, f"{v:+.0f}", va="center",
                    ha="left" if v > 0 else "right", fontsize=6.5)
ax.axvline(0, color="k", lw=.8)
ax.set_yticks(y); ax.set_yticklabels(order, fontsize=7.5)
ax.set_xlim(-4.6, 4.6)
ax.set_xlabel(r"change in held-out $C_{22}$ $R^2$ when mode-2 character is added"
              "\n(clipped to $\\pm$3; true value labelled)")
ax.set_title("Which held-out families the feature helps", fontsize=10)
ax.legend(fontsize=8, loc="upper left", framealpha=.95)
fig.savefig(f"{OUT}/fig2_per_family.png"); plt.close(fig)

# ---------------------------------------------------------------- fig 3: learners + K
fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.3), constrained_layout=True)
ax = axes[0]
LN = ["RF", "GBR", "MLP", "KRRt", "KRR"]
LAB = {"RF":"RF","GBR":"GBR","MLP":"MLP","KRRt":"kernel\nridge","KRR":"kernel ridge\n(mistuned)"}
f0 = [np.median(lea[f"v2_24grey|frac|{l}"]["C22"]) for l in LN]
f1 = [np.median(lea[f"v2_24grey|frac+mode2|{l}"]["C22"]) for l in LN]
x = np.arange(len(LN))
ax.bar(x - .2, f0, .4, color="#bbbbbb", label="solid fraction")
ax.bar(x + .2, f1, .4, color="#1f77b4", label="+ mode-2 character")
for i, v in enumerate(f1):
    if v < -.1: ax.text(i + .2, .03, "unstable\nextrapolation", ha="center", fontsize=6.5, color="#d62728")
ax.axhline(0, color="k", lw=.8); ax.set_xticks(x); ax.set_xticklabels([LAB[l] for l in LN], fontsize=8)
ax.set_ylabel(r"held-out $C_{22}$ $R^2$ (median)"); ax.set_ylim(-.35, .55)
ax.set_title("Surrogate class (24$\\times$24 feature)", fontsize=9.5)
ax.legend(fontsize=7.5, loc="upper left")
ax = axes[1]
Ks = [1, 2, 4]
def med(k, s, arm):
    key = f"K{k}|{'shared' if arm=='frac' else s}|{arm}" if k < 4 else f"{'shared' if arm=='frac' else s}|{arm}"
    src = robk if k < 4 else rob
    v = np.array(src[key]["C22"]); return np.median(v), np.percentile(v, 25), np.percentile(v, 75)
for s, c, lab in [("v2_24grey", "#1f77b4", "frac + mode-2 (24$\\times$24)"), (None, "k", "solid fraction")]:
    arm = "frac" if s is None else "frac+mode2"
    m = np.array([med(k, s or "v2_24grey", arm) for k in Ks])
    ax.plot(Ks, m[:, 0], "o-", color=c, label=lab)
    ax.fill_between(Ks, m[:, 1], m[:, 2], color=c, alpha=.13)
ax.axhline(0, color="gray", lw=.8, ls=":")
ax.set_xticks(Ks); ax.set_xlabel("number of training families $K$")
ax.set_ylabel(r"held-out $C_{22}$ $R^2$")
ax.set_title("Training-set size", fontsize=9.5); ax.legend(fontsize=7.5, loc="upper left")
fig.savefig(f"{OUT}/fig3_robustness.png"); plt.close(fig)

# ---------------------------------------------------------------- fig 4: mechanism
sel = A.selection(); C = A.load(RB.SETS["v2_24grey"], sel)
fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.3), constrained_layout=True)
ax = axes[0]
rat, rot, names = [], [], []
for f in FAMS:
    y = C[f]["y"]; rat.append(np.median(y[:, 0] / y[:, 2]))
    rot.append(np.median(C[f]["D"][:, 6])); names.append(f)
rat = np.array(rat); rot = np.array(rot)
iso = np.abs(rat - 1) < 1e-6
ax.scatter(rat[~iso], rot[~iso], s=22, color="#1f77b4", label="anisotropic")
ax.scatter(rat[iso], rot[iso], s=40, color="#d62728", marker="^",
           label=r"$C_{11}/C_{22}=1.000$ exactly")
for f, xx, yy in zip(names, rat, rot):
    if f in ("cross", "star_hole", "honeycomb", "layered", "circle", "rand_lattice"):
        ax.annotate(f, (xx, yy), textcoords="offset points",
                    xytext=(-5, 5) if xx > 2 else (5, 4), fontsize=6.5,
                    ha="right" if xx > 2 else "left")
ax.set_xscale("log"); ax.set_xlim(0.15, 14); ax.set_xticks([0.2,0.5,1,2,5,10])
ax.set_xticklabels(["0.2","0.5","1","2","5","10"])
ax.set_xlabel(r"$C_{11}/C_{22}$  (family median)")
ax.set_ylabel("rotation fraction of mode 2")
ax.set_title("Stiffness says nothing; the mode still differs", fontsize=9.5)
ax.legend(fontsize=7, loc="lower left")
ax = axes[1]
for f, c in [("cross", "#d62728"), ("star_hole", "#ff7f0e"), ("circle", "#1f77b4"), ("square", "#2ca02c")]:
    ax.hist(C[f]["D"][:, 6], bins=25, histtype="step", lw=1.4, color=c, label=f, density=True)
ax.set_xlabel("rotation fraction of mode 2"); ax.set_ylabel("density")
ax.set_title(r"Four families with $C_{11}/C_{22}=1.000$", fontsize=9.5)
ax.legend(fontsize=7.5)
fig.savefig(f"{OUT}/fig4_mechanism.png"); plt.close(fig)
print("wrote", os.listdir(OUT))
