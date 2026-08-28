"""make_figs.py -- figures for the training-family-selection paper."""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

import selcore as SC
import selrules as SR
import run_sweep as RS

OUT = "/mnt/user-data/outputs"
B = json.load(open(f"{OUT}/results_big.json"))
POOL = B["pool"]; AN = B["aniso"]

HOLE = {"circle", "square", "cross", "star_hole", "ellipse", "rect_hole", "slot_pair",
        "tri_hole", "hex_hole", "diamond_hole", "star6_hole", "two_holes", "cross_aniso"}

# ---------------------------------------------------------------- Fig 1: the pool
fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
fr = {f: SC.frac(f).mean() for f in POOL}
for f in POOL:
    hole = f in HOLE
    ax[0].scatter(fr[f], AN[f], s=70, marker="o" if hole else "^",
                  c="#2b6cb0" if hole else "#c05621", edgecolor="k", lw=.5, zorder=3)
    ax[0].annotate(f, (fr[f], AN[f]), fontsize=5.5, xytext=(3, 3),
                   textcoords="offset points")
ax[0].set_xlabel("mean solid fraction (histogram-matched)")
ax[0].set_ylabel(r"anisotropy  median $|C_{11}-C_{22}|/(C_{11}+C_{22})$")
ax[0].set_title("22-family pool: anisotropy is decoupled from topology")
ax[0].scatter([], [], marker="o", c="#2b6cb0", label="perforation (hole)")
ax[0].scatter([], [], marker="^", c="#c05621", label="strut network")
ax[0].legend(fontsize=8); ax[0].grid(alpha=.3)

r = np.array([np.median(SC.yall(f)[:, 0] / SC.yall(f)[:, 3]) for f in POOL])
o = np.argsort(r)
ax[1].barh(range(len(POOL)), np.log10(r[o]),
           color=["#c05621" if POOL[i] not in HOLE else "#2b6cb0" for i in o])
ax[1].set_yticks(range(len(POOL))); ax[1].set_yticklabels([POOL[i] for i in o], fontsize=6)
ax[1].axvline(0, color="k", lw=.8)
ax[1].set_xlabel(r"$\log_{10}$ median $C_{11}/C_{22}$")
ax[1].set_title("both signs of anisotropy are represented")
ax[1].grid(alpha=.3, axis="x")
plt.tight_layout(); plt.savefig(f"{OUT}/fig1_pool.png", dpi=160); plt.close()

# ------------------------------------------- Fig 2: how much does selection matter
fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))
for ax_, t in zip(axes, ["C11", "C22", "C33"]):
    for K, col in [(2, "#2b6cb0"), (4, "#c05621")]:
        a = np.array([x["wc"] for x in B["subsets"][f"{t}|{K}"]])
        ax_.hist(-a, bins=np.logspace(np.log10(max(1e-2, -a.max())), np.log10(-a.min()), 30),
                 alpha=.6, color=col, label=f"K={K}  (n={len(a)})")
    ax_.set_xscale("log")
    ax_.set_xlabel(r"$-R^2$ on the worst held-out family  (log)")
    ax_.set_ylabel("number of training-family subsets")
    ax_.set_title(t); ax_.legend(fontsize=8); ax_.grid(alpha=.3)
plt.tight_layout(); plt.savefig(f"{OUT}/fig2_spread.png", dpi=160); plt.close()

# ------------------------------------------- Fig 3: criterion correlation heatmap
crit = ["mmd", "wass", "kcenter_geo", "resp_cover", "resp_span",
        "aniso_sum", "aniso_max", "aniso_cover", "aniso_range"]
jobs = ["C11|2", "C11|4", "C22|2", "C22|4", "C33|2", "C33|4"]
M = np.array([[B["corr"][j][c]["rho"] for j in jobs] for c in crit])
fig, ax = plt.subplots(figsize=(7.2, 4.4))
im = ax.imshow(M, cmap="RdBu_r", norm=TwoSlopeNorm(vmin=-0.6, vcenter=0, vmax=0.6))
ax.set_xticks(range(len(jobs))); ax.set_xticklabels(jobs, fontsize=8)
ax.set_yticks(range(len(crit))); ax.set_yticklabels(crit, fontsize=8)
for i in range(len(crit)):
    for j in range(len(jobs)):
        ax.text(j, i, f"{M[i,j]:+.2f}", ha="center", va="center", fontsize=7.5)
ax.set_title(r"Spearman $\rho$: cheap criterion vs achieved worst-case $R^2$"
             "\n(positive = criterion predicts good transfer)", fontsize=9)
plt.colorbar(im, shrink=.8); plt.tight_layout()
plt.savefig(f"{OUT}/fig3_criteria.png", dpi=160); plt.close()

# ------------------------------------------- Fig 4: the sign reversal
imgs = {f: SC.images(f) for f in POOL}
fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))
for ax_, t in zip(axes, ["C11", "C22", "C33"]):
    ctx = SR.Ctx(POOL, t, imgs)
    cf = RS._cost_fn("mmd", ctx)
    recs = B["subsets"][f"{t}|2"]
    x = [cf(tuple(r["set"])) for r in recs]
    y = [r["wc"] for r in recs]
    ax_.scatter(x, y, s=16, alpha=.7, color="#2b6cb0")
    ax_.set_yscale("symlog", linthresh=1)
    ax_.set_xlabel("MMD cost (worst unseen family to training union)")
    ax_.set_ylabel(r"worst-case held-out $R^2$")
    ax_.set_title(f"{t}, K=2   " + r"$\rho$ = " + f"{B['corr'][t+'|2']['mmd']['rho']:+.2f}")
    ax_.grid(alpha=.3)
plt.tight_layout(); plt.savefig(f"{OUT}/fig4_mmd.png", dpi=160); plt.close()
print("wrote fig1_pool.png fig2_spread.png fig3_criteria.png fig4_mmd.png")
