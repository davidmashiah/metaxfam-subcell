"""robust.py -- the ablation, but with 3 subset draws (rng 7,8,9; 20 subsets each, K=4)
x 2 RF seeds = 120 paired evaluations per arm.  Same 5,500 matched cells as ablate2.
Reports median and IQR of pooled held-out C22 / C11 R^2 and paired gains.
Checkpointed per (set, arm) in ../results/robust.json."""
import _path  # noqa: F401  (adds code/ and code/vendor/ to sys.path)

import json, os, sys, time
import numpy as np
from sklearn.metrics import r2_score
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ablate2 as A
from run_feature2 import FAMS, rf

OUT = "/mnt/user-data/outputs/p3/results/robust.json"
SETS = {"orig_48": "/mnt/user-data/outputs/descriptors_48",
        "orig_16": "/mnt/user-data/outputs/descriptors_16",
        "v2_48": "/mnt/user-data/outputs/descriptors_v2_48binary",
        "v2_24grey": "/mnt/user-data/outputs/descriptors_v2_24grey",
        "v2_16grey": "/mnt/user-data/outputs/descriptors_v2_16grey",
        "v2_16binary": "/mnt/user-data/outputs/descriptors_v2_16binary"}
PIXSETS = ["orig_48", "v2_24grey", "v2_16grey"]
DESC_ARMS = ["mode2", "frac", "frac+mode2", "chars", "all24", "frac+all24"]
PIX_ARMS = ["pixels", "pix+frac", "pix+mode2", "pix+mode2+frac"]
SEEDS = (0, 1)

def subsets():
    out = []
    for r in (7, 8, 9):
        rng = np.random.default_rng(r)
        out += [tuple(sorted(rng.choice(FAMS, 4, replace=False))) for _ in range(20)]
    return out

def feats(C, f, arm):
    p = []
    if arm.startswith("pix"): p.append(C[f]["X"])
    if "frac" in arm: p.append(C[f]["frac"].astype(np.float32))
    if "mode2" in arm: p.append(C[f]["D"][:, [6, 7, 8]].astype(np.float32))
    if arm == "chars": p.append(C[f]["D"][:, 6:24].astype(np.float32))
    if "all24" in arm: p.append(C[f]["D"].astype(np.float32))
    return np.hstack(p)

def run(C, arm, subs):
    r22, r11 = [], []
    for sub in subs:
        tr = list(sub); te = [f for f in FAMS if f not in sub]
        X = np.vstack([feats(C, f, arm) for f in tr]); Xt = np.vstack([feats(C, f, arm) for f in te])
        Y = np.vstack([C[f]["y"] for f in tr]); Yt = np.vstack([C[f]["y"] for f in te])
        for s in SEEDS:
            p = rf(s).fit(X, Y).predict(Xt)
            r22.append(float(r2_score(Yt[:, 2], p[:, 2]))); r11.append(float(r2_score(Yt[:, 0], p[:, 0])))
    return r22, r11

def main():
    sel = A.selection(); subs = subsets()
    R = json.load(open(OUT)) if os.path.exists(OUT) else {}
    for name, d in SETS.items():
        C = A.load(d, sel)
        arms = DESC_ARMS + (PIX_ARMS if name in PIXSETS else [])
        for arm in arms:
            key = f"{name}|{arm}"
            if arm in ("frac", "pixels", "pix+frac"):        # descriptor-independent: share
                key = f"shared|{arm}"
            if key in R: continue
            t0 = time.time(); r22, r11 = run(C, arm, subs)
            R[key] = {"C22": r22, "C11": r11}
            print(f"  {key:28s} C22 median {np.median(r22):+.3f}  IQR [{np.percentile(r22,25):+.3f},{np.percentile(r22,75):+.3f}]"
                  f"  C11 median {np.median(r11):+.3f}   {time.time()-t0:5.0f}s", flush=True)
            json.dump(R, open(OUT + ".tmp", "w")); os.replace(OUT + ".tmp", OUT)
    print("DONE", flush=True)

if __name__ == "__main__":
    main()
