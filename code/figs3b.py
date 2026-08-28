"""figs3b.py -- the figures the acceptance checklist demands and figs3.py lacked:
  fig0  pipeline schematic (what is computed from what)
  fig5  (a) property-space coverage of the 22-family pool, (b) in-distribution parity,
        (c) cross-family parity for pixels vs the 4-number feature
"""
import _path  # noqa: F401  (adds code/ and code/vendor/ to sys.path)

import os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ablate2 as A, robust as RB
from run_feature2 import FAMS
import subcell as SB

OUT = "/mnt/user-data/outputs/p3/figs"
plt.rcParams.update({"font.size": 9, "figure.dpi": 160, "savefig.bbox": "tight"})

# ------------------------------------------------------------------ fig 0: pipeline
fig, ax = plt.subplots(figsize=(7.4, 3.0))
ax.set_xlim(0, 100); ax.set_ylim(-4, 42); ax.axis("off")
def box(x, y, w, h, t, fc, ec="#333333", fs=8):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.6",
                                fc=fc, ec=ec, lw=1.0))
    ax.text(x + w / 2, y + h / 2, t, ha="center", va="center", fontsize=fs)
def arrow(x1, y1, x2, y2, t="", up=True):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                                 mutation_scale=11, lw=1.0, color="#333333"))
    if t: ax.text((x1 + x2) / 2, (y1 + y2) / 2 + (2.2 if up else -3.0), t,
                  ha="center", fontsize=7, color="#555555")
box(1, 16, 17, 10, "unit cell\n48$\\times$48 pixels", "#f2f2f2")
arrow(18.5, 27, 30, 33)
arrow(18.5, 15, 30, 9)
box(30, 29, 24, 10, "periodic FE operator $K$\n$+$ stress averaging", "#dce9f5")
box(30, 3, 24, 10, "same $K$, $+$ mass $M$\n$K\\phi=\\omega^2M\\phi$", "#fbe3d6")
arrow(54.5, 34, 66, 34)
arrow(54.5, 8, 66, 8)
box(66, 29, 30, 10, "$\\mathbf{C}^H$  (target)\n2 rigid modes retained", "#dce9f5")
box(66, 3, 30, 10, "mode-2 character\n$(c_{rot},c_{dil},c_{shr})$  (feature)", "#fbe3d6")
ax.text(81, 22.5, "everything else discarded by the projection", ha="center",
        fontsize=8, color="#b03030", style="italic")
ax.annotate("", xy=(70, 20), xytext=(92, 20),
            arrowprops=dict(arrowstyle="<->", color="#b03030", lw=1.0))
ax.text(24.5, 33.0, "full 48$\\times$48", fontsize=7, color="#555555", ha="center")
ax.text(15.0, 1.2, "coarsened to 24$\\times$24 first;\ncost 0.25$\\times$ one homogenisation",
        fontsize=7, color="#555555", ha="center")
ax.text(50, 40.5, "Sub-cell modes measure what stress-averaging throws away",
        ha="center", fontsize=9.5)
fig.savefig(f"{OUT}/fig0_pipeline.png"); plt.close(fig)

# ------------------------------------------------------------------ data for fig 5
sel = A.selection(); C = A.load(RB.SETS["v2_24grey"], sel)
rng = np.random.default_rng(11)
# Pick the subset whose (feature - pixels) gain is CLOSEST TO THE MEDIAN of all 60,
# so the parity panel is representative rather than favourable.  Stated in the caption.
import json
_R = json.load(open("/mnt/user-data/outputs/p3/results/robust.json"))
_px = np.array(_R["shared|pixels"]["C22"]).reshape(-1, 2).mean(1)
_ft = np.array(_R["v2_24grey|frac+mode2"]["C22"]).reshape(-1, 2).mean(1)
_g = _ft - _px
_i = int(np.argmin(np.abs(_g - np.median(_g))))
train = RB.subsets()[_i]
print(f"median-gain subset #{_i}: {train}  gain {_g[_i]:+.3f} (median {np.median(_g):+.3f})")
test = [f for f in FAMS if f not in train]

def feats(fams, arm):
    return np.vstack([RB.feats(C, f, arm) for f in fams])
def targ(fams):
    return np.concatenate([C[f]["y"][:, 2] for f in fams])

def fit_pred(arm):
    m = RandomForestRegressor(n_estimators=60, random_state=0, max_features=0.3,
                              min_samples_leaf=2, n_jobs=1)
    Xtr = feats(train, arm); ytr = targ(train)
    n = len(ytr); idx = rng.permutation(n); cut = int(.75 * n)
    m.fit(Xtr[idx[:cut]], ytr[idx[:cut]])
    idp = m.predict(Xtr[idx[cut:]]); idy = ytr[idx[cut:]]
    m2 = RandomForestRegressor(n_estimators=60, random_state=0, max_features=0.3,
                               min_samples_leaf=2, n_jobs=1).fit(Xtr, ytr)
    return (idy, idp), (targ(test), m2.predict(feats(test, arm)))

id_px, oo_px = fit_pred("pixels")
id_ft, oo_ft = fit_pred("frac+mode2")

fig, axes = plt.subplots(1, 3, figsize=(9.6, 3.3), constrained_layout=True)
# (a) coverage
ax = axes[0]
for f in FAMS:
    y = C[f]["y"]
    ax.scatter(y[:, 0], y[:, 2], s=2, alpha=.35,
               color="#d62728" if f in train else "#7f9fbf")
ax.plot([], [], "o", color="#d62728", ms=5, label="training families (median-gain draw)")
ax.plot([], [], "o", color="#7f9fbf", ms=5, label="18 held-out families")
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlabel(r"$C_{11}/E_s$"); ax.set_ylabel(r"$C_{22}/E_s$")
ax.set_title("(a) property-space coverage", fontsize=9.5)
ax.legend(fontsize=7, loc="upper left"); ax.grid(alpha=.25)
# (b),(c) parity
def parity(ax, d, title, c):
    y, p = d
    ax.scatter(y, p, s=3, alpha=.3, color=c)
    lo = min(y.min(), p.min()); hi = max(y.max(), p.max())
    ax.plot([lo, hi], [lo, hi], "k--", lw=.9)
    ax.set_xlabel(r"FE homogenisation $C_{22}/E_s$")
    ax.set_ylabel(r"surrogate prediction")
    ax.set_title(title, fontsize=9.5); ax.grid(alpha=.25)
    ax.text(.04, .93, f"$R^2$ = {r2_score(y, p):+.3f}", transform=ax.transAxes, fontsize=8.5)
parity(axes[1], id_px, "(b) in-distribution, 2304 pixels", "#7f7f7f")
axes[1].text(.04, .84, "held-out cells,\ntraining families", transform=axes[1].transAxes,
             fontsize=7, color="#555555")
parity(axes[2], oo_ft, "(c) 18 unseen families", "#1f77b4")
axes[2].text(.04, .75, "4 numbers", transform=axes[2].transAxes, fontsize=8.5, color="#1f77b4")
axes[2].scatter(*oo_px, s=3, alpha=.25, color="#d62728")
axes[2].text(.04, .84, f"pixels $R^2$ = {r2_score(*oo_px):+.3f}", transform=axes[2].transAxes,
             fontsize=8.5, color="#d62728")
fig.savefig(f"{OUT}/fig5_coverage_parity.png"); plt.close(fig)
print("wrote fig0, fig5")
