"""
analyze.py -- turn results_sweep.json into the tables and figures for the paper.
"""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

IN  = "/mnt/user-data/outputs/results_sweep.json"
OUT = "/mnt/user-data/outputs"

R = json.load(open(IN))
POOL, TARGETS, KS = R["pool"], R["targets"], R["ks"]
RULES = ["maxdiv", "kcenter_geo", "mmd", "wass",
         "aniso_top", "aniso_cover", "resp_cover", "resp_span", "hardest"]
PRIOR = {"maxdiv", "kcenter_geo", "mmd", "wass"}
SHORT = {"circle": "ci", "square": "sq", "cross": "cr", "star_hole": "st",
         "ellipse": "el", "rect_hole": "re", "slot_pair": "sl", "kagome": "ka",
         "honeycomb": "hc", "rand_lattice": "rl"}
s = lambda fs: "+".join(SHORT[f] for f in fs)

lines = []
def P(x=""):
    print(x); lines.append(x)


# ---------------------------------------------------------------- main table
P("=" * 100)
P("TRAINING-FAMILY SELECTION -- worst-case held-out R^2 (mean over 3 RF seeds)")
P("de-confounded 10-family pool, solid-fraction histogram matched, percolating cells only")
P("=" * 100)

for t in TARGETS:
    P(f"\n### TARGET {t}")
    hdr = f"{'K':>2} | {'oracle':>26} | {'random':>7} {'worst':>7} | " + \
          " ".join(f"{r[:11]:>11}" for r in RULES)
    P(hdr); P("-" * len(hdr))
    for K in KS:
        row = R["table"][f"{t}|{K}"]
        o = row["oracle"]
        cells = []
        for r in RULES:
            v = row[r]
            cells.append(f"{v['wc']:+7.2f}/{v['rank']:<3d}" if v["wc"] is not None else "     --    ")
        P(f"{K:>2} | {s(o['set']):>14} {o['wc']:+7.2f}   | "
          f"{row['random_mean']['wc']:+7.2f} {row['worst']['wc']:+7.2f} | " + " ".join(cells))
    P("  (each rule cell = worst-case R^2 / rank of that subset among all C(10,K) subsets)")

# ---------------------------------------------------------------- regret
P("\n" + "=" * 100)
P("REGRET vs ORACLE  (oracle worst-case R^2  minus  rule's worst-case R^2; 0 = optimal)")
P("and PERCENTILE of the rule's pick among all subsets (100 = oracle)")
P("=" * 100)
hdr = f"{'rule':<13}" + "".join(f"{t+' K'+str(K):>11}" for t in TARGETS for K in KS)
P(hdr); P("-" * len(hdr))
reg_summary = {}
for r in RULES + ["random_mean"]:
    cells, regs, pcts = [], [], []
    for t in TARGETS:
        for K in KS:
            row = R["table"][f"{t}|{K}"]
            n = len(R["subsets"][t][str(K)])
            v = row[r]
            if v["wc"] is None:
                cells.append("     --   "); continue
            reg = row["oracle"]["wc"] - v["wc"]
            regs.append(reg)
            if v["rank"] is not None:
                pcts.append(100 * (n - v["rank"]) / max(n - 1, 1))
            cells.append(f"{reg:>10.2f}")
    P(f"{r:<13}" + "".join(cells))
    reg_summary[r] = (float(np.mean(regs)) if regs else np.nan,
                      float(np.median(pcts)) if pcts else np.nan)

P("\n" + f"{'rule':<15}{'mean regret':>13}{'median pct':>12}   (lower regret / higher pct = better)")
P("-" * 56)
for r, (mr, mp) in sorted(reg_summary.items(), key=lambda kv: kv[1][0]):
    tag = "  [prior art]" if r in PRIOR else ("  [baseline]" if r == "random_mean" else "")
    P(f"{r:<15}{mr:>13.2f}{mp:>12.1f}{tag}")

# ---------------------------------------------------------------- correlations
P("\n" + "=" * 100)
P("SPEARMAN rho between each CHEAP criterion and the achieved worst-case R^2,")
P("computed across ALL C(10,K) subsets.  rho > 0 means: lower cost -> better transfer.")
P("A rule can hit the oracle by luck; a high rho means the criterion actually tracks transfer.")
P("=" * 100)
keys = sorted({k for v in R["corr"].values() for k in v})
hdr = f"{'criterion':<14}" + "".join(f"{t+' K'+str(K):>10}" for t in TARGETS for K in [1, 2, 3, 4])
P(hdr); P("-" * len(hdr))
for k in keys:
    cells = []
    vals = []
    for t in TARGETS:
        for K in [1, 2, 3, 4]:
            c = R["corr"].get(f"{t}|{K}", {}).get(k)
            if c is None or not np.isfinite(c["rho"]):
                cells.append("      --  ")
            else:
                star = "*" if c["p"] < 0.05 else " "
                cells.append(f"{c['rho']:>+9.2f}{star}")
                vals.append(c["rho"])
    P(f"{k:<14}" + "".join(cells) + f"   | mean {np.mean(vals):+.2f}" if vals else "")

# ---------------------------------------------------------------- figures
fig, axes = plt.subplots(1, len(TARGETS), figsize=(5 * len(TARGETS), 4), sharey=False)
if len(TARGETS) == 1: axes = [axes]
show = ["oracle", "aniso_cover", "aniso_top", "resp_cover", "maxdiv", "mmd", "wass"]
for ax, t in zip(axes, TARGETS):
    for r in show:
        ys = []
        for K in KS:
            row = R["table"][f"{t}|{K}"]
            ys.append(row[r]["wc"] if row[r]["wc"] is not None else np.nan)
        st = dict(lw=2.4, marker="o") if r == "oracle" else dict(lw=1.3, marker="s", ms=4)
        if r in PRIOR: st["ls"] = "--"
        ax.plot(KS, ys, label=r, **st)
    ax.plot(KS, [R["table"][f"{t}|{K}"]["random_mean"]["wc"] for K in KS],
            color="grey", ls=":", label="random (mean)")
    ax.set_title(f"target {t}"); ax.set_xlabel("K training families")
    ax.set_ylabel("worst-case held-out $R^2$"); ax.grid(alpha=.3)
axes[0].legend(fontsize=7)
plt.tight_layout(); plt.savefig(f"{OUT}/fig_kcurve.png", dpi=150)

# scatter: anisotropy-coverage cost vs achieved wc, K=2, C11
fig, axes = plt.subplots(1, 3, figsize=(13, 4))
for ax, t in zip(axes, TARGETS):
    recs = R["subsets"][t]["2"]
    an = R["aniso"]
    x = [max(min(abs(an[u] - an[c]) for c in r["set"])
             for u in POOL if u not in r["set"]) for r in recs]
    y = [r["wc"] for r in recs]
    ax.scatter(x, y, s=22, alpha=.75)
    ax.set_xlabel("anisotropy-coverage cost (worst uncovered gap)")
    ax.set_ylabel("worst-case held-out $R^2$"); ax.set_title(f"{t}, K=2")
    ax.grid(alpha=.3)
plt.tight_layout(); plt.savefig(f"{OUT}/fig_aniso_scatter.png", dpi=150)

open(f"{OUT}/results_table.txt", "w").write("\n".join(lines) + "\n")
print(f"\nwrote {OUT}/results_table.txt, fig_kcurve.png, fig_aniso_scatter.png")
