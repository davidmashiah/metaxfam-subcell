"""
predictor.py -- does sub-cell mode character predict TRANSFERABILITY?

Paper 2's headline negative: across nine criteria (k-centre geometric coverage, MMD,
Wasserstein, anisotropy coverage/max/sum/range, response coverage/span) no cheap
criterion robustly predicts which training families transfer well, and MMD's apparent
strength did not survive a change of surrogate class.

This script asks whether the mode-2 character -- which §4 shows carries the transferable
content -- also predicts transfer error when used purely as a SELECTION criterion, i.e.
without being given to the model at all.  That is a strictly harder test than §4: the
surrogate here sees only raw pixels, exactly as in Paper 2.

Design.  For each of 60 K=4 subsets we compute
    wc  = worst-case held-out R^2 over the 18 unseen families   (pixels -> C22)
    mn  = mean held-out R^2
and a family of candidate criteria, each a scalar function of the training subset only:
    md_cover  : mode-character coverage -- min over held-out families of the distance
                from that family to its nearest training family, in mode-2 character
                space (small = the training set covers the pool).  Negated so that
                larger is better, as for the other criteria.
    md_spread : spread of the training subset in mode-character space (max pairwise)
    md_rot_rng: range of the rotational fraction across the training subset
    frac_rng  : range of solid fraction (density control)
    aniso_rng : range of |C11-C22|/(C11+C22) (Paper 2's best physical criterion)
    mmd_pix   : MMD between the training union and the pool in raw pixel space
                (Paper 2's prior-art criterion, recomputed here)
Reported as Spearman rho against worst-case and mean transfer, with a permutation
p-value, and repeated for a SECOND surrogate class (extra-trees) because Paper 2's
lesson is that criterion rankings can be class-dependent.
"""
import _path  # noqa: F401  (adds code/ and code/vendor/ to sys.path)

import json, os, sys, time
import numpy as np
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.metrics import r2_score
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ablate2 as A, robust as RB
from run_feature2 import FAMS

OUT = "/mnt/user-data/outputs/p3/results/predictor.json"
DESC = "/mnt/user-data/outputs/descriptors_v2_24grey"

sel = A.selection()
C = A.load(DESC, sel)
XR = {f: C[f]["X"] for f in FAMS}
Y22 = {f: C[f]["y"][:, 2] for f in FAMS}
MD = {f: C[f]["D"][:, [6, 7, 8]] for f in FAMS}          # mode-2 character, per cell
MDm = {f: MD[f].mean(0) for f in FAMS}                    # family centroid
FRAC = {f: C[f]["frac"].mean() for f in FAMS}
ANI = {f: float(np.median(np.abs(C[f]["y"][:, 0] - C[f]["y"][:, 2])
                          / (C[f]["y"][:, 0] + C[f]["y"][:, 2]))) for f in FAMS}


def mmd_pixels(train):
    """Linear-kernel MMD between the training union and the whole pool, pixel space."""
    a = np.vstack([XR[f] for f in train]).mean(0)
    b = np.vstack([XR[f] for f in FAMS]).mean(0)
    return float(np.linalg.norm(a - b))


def criteria(train):
    tr = list(train); te = [f for f in FAMS if f not in tr]
    d_near = [min(np.linalg.norm(MDm[t] - MDm[s]) for s in tr) for t in te]
    pair = [np.linalg.norm(MDm[a] - MDm[b]) for i, a in enumerate(tr) for b in tr[i + 1:]]
    rot = [MDm[f][0] for f in tr]
    return {
        "md_cover":   -max(d_near),                 # smaller worst-gap is better
        "md_spread":  max(pair),
        "md_rot_rng": max(rot) - min(rot),
        "frac_rng":   max(FRAC[f] for f in tr) - min(FRAC[f] for f in tr),
        "aniso_rng":  max(ANI[f] for f in tr) - min(ANI[f] for f in tr),
        "mmd_pix":    -mmd_pixels(tr),
    }


def transfer(train, learner, seed=0):
    tr = list(train); te = [f for f in FAMS if f not in tr]
    X = np.vstack([XR[f] for f in tr]); y = np.concatenate([Y22[f] for f in tr])
    M = (RandomForestRegressor if learner == "rf" else ExtraTreesRegressor)(
        n_estimators=60, random_state=seed, max_features=0.3, min_samples_leaf=2, n_jobs=1)
    M.fit(X, y)
    per = {f: float(r2_score(Y22[f], M.predict(XR[f]))) for f in te}
    return min(per.values()), float(np.mean(list(per.values())))


def perm_p(x, y, rho, n=5000, seed=0):
    rng = np.random.default_rng(seed)
    x = np.asarray(x); y = np.asarray(y)
    cnt = sum(abs(spearmanr(x, rng.permutation(y)).statistic) >= abs(rho) for _ in range(n))
    return (cnt + 1) / (n + 1)


def main():
    R = json.load(open(OUT)) if os.path.exists(OUT) else {}
    subs = RB.subsets()
    if "crit" not in R:
        R["crit"] = [criteria(s) for s in subs]
        json.dump(R, open(OUT, "w"))
    for learner in ("rf", "et"):
        k = f"transfer_{learner}"
        if k not in R:
            t0 = time.time()
            vals = [transfer(s, learner) for s in subs]
            R[k] = {"wc": [v[0] for v in vals], "mean": [v[1] for v in vals]}
            print(f"  transfer/{learner}: {len(subs)} subsets in {(time.time()-t0)/60:.1f} min", flush=True)
            json.dump(R, open(OUT + ".tmp", "w")); os.replace(OUT + ".tmp", OUT)

    names = list(R["crit"][0])
    print(f"\n{'criterion':12s}" + "".join(f"{f'{l}/{t}':>16s}" for l in ("rf", "et") for t in ("wc", "mean")))
    summary = {}
    for nm in names:
        x = [c[nm] for c in R["crit"]]
        row = ""
        summary[nm] = {}
        for l in ("rf", "et"):
            for t in ("wc", "mean"):
                y = R[f"transfer_{l}"][t]
                rho = spearmanr(x, y).statistic
                p = perm_p(x, y, rho)
                summary[nm][f"{l}_{t}"] = {"rho": float(rho), "p": float(p)}
                row += f"{rho:+8.2f} (p{p:5.3f})"
        print(f"{nm:12s}{row}", flush=True)
    R["summary"] = summary
    json.dump(R, open(OUT, "w"))
    print("\nDONE")


if __name__ == "__main__":
    main()
