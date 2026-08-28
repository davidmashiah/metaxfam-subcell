"""summarize.py -- paper-ready table from robust.json (120 paired evaluations per arm)."""
import _path  # noqa: F401  (adds code/ and code/vendor/ to sys.path)

import json, numpy as np
R = json.load(open("/mnt/user-data/outputs/p3/results/robust.json"))
COST = {"orig_48": 1.12, "orig_16": 0.11, "v2_48": 1.16, "v2_24grey": 0.25, "v2_16grey": 0.11, "v2_16binary": 0.11}
def g(k): return np.array(R[k]["C22"]) if k in R else None
def g11(k): return np.array(R[k]["C11"]) if k in R else None
def fmt(a): return f"{np.median(a):+.3f} [{np.percentile(a,25):+.3f},{np.percentile(a,75):+.3f}]"
pix = g("shared|pixels"); frac = g("shared|frac"); pixfrac = g("shared|pix+frac")
print("Pooled held-out C22 R^2, K=4 training families, 3 subset draws x 20 subsets x 2 RF seeds = 120 paired evaluations.")
print("median [IQR]; 'gain' = paired median difference; 'win' = fraction of the 120 evaluations where the arm beats its control.\n")
print(f"{'control: frac only':30s} {fmt(frac)}")
if pix is not None: print(f"{'control: pixels only':30s} {fmt(pix)}")
if pixfrac is not None: print(f"{'control: pixels+frac':30s} {fmt(pixfrac)}   gain vs pixels {np.median(pixfrac-pix):+.3f}  win {np.mean(pixfrac>pix):.2f}")
print()
print(f"{'set':12s}{'cost':>6s} | {'mode2 (3-D)':>26s} {'gain/frac':>10s} {'win':>5s} | {'frac+mode2 (4-D)':>26s} {'gain/frac':>10s} {'win':>5s} | {'chars(18)':>10s} {'all24':>8s}")
for s in COST:
    m = g(f"{s}|mode2"); fm = g(f"{s}|frac+mode2"); ch = g(f"{s}|chars"); a = g(f"{s}|all24")
    if m is None: continue
    print(f"{s:12s}{COST[s]:6.2f} | {fmt(m):>26s} {np.median(m-frac):+10.3f} {np.mean(m>frac):5.2f} | "
          f"{fmt(fm):>26s} {np.median(fm-frac):+10.3f} {np.mean(fm>frac):5.2f} | {np.median(ch):+10.3f} {np.median(a):+8.3f}")
print("\nPixel-surrogate arms (feature APPENDED to 2304 raw pixels):")
print(f"{'set':12s} | {'pix+mode2':>26s} {'gain/pix':>9s} {'win':>5s} {'gain/pix+frac':>14s} | {'pix+mode2+frac':>26s} {'gain/pix':>9s} {'win':>5s} {'gain/pix+frac':>14s} {'win':>5s}")
for s in COST:
    pm = g(f"{s}|pix+mode2"); pmf = g(f"{s}|pix+mode2+frac")
    if pm is None: continue
    print(f"{s:12s} | {fmt(pm):>26s} {np.median(pm-pix):+9.3f} {np.mean(pm>pix):5.2f} {np.median(pm-pixfrac):+14.3f} | "
          f"{fmt(pmf):>26s} {np.median(pmf-pix):+9.3f} {np.mean(pmf>pix):5.2f} {np.median(pmf-pixfrac):+14.3f} {np.mean(pmf>pixfrac):5.2f}")
print("\nC11 (the component that did NOT collapse in Paper 1) -- mode2 alone, median:")
for s in COST:
    m = g11(f"{s}|mode2")
    if m is not None: print(f"  {s:12s} {np.median(m):+.3f}   (frac only {np.median(g11('shared|frac')):+.3f})")
