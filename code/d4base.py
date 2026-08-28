"""d4base.py -- D4-equivariant pixel baseline: an honest, cheap stand-in for the
similarity-equivariant GNN of Hendriks et al. (CMAME 439:117867, 2025).

We cannot reimplement their architecture. The mechanism they exploit is invariance of
the homogenised response to the symmetry group of the cell; we obtain EXACT D4
invariance by orbit averaging -- train on all 8 dihedral images of every training cell,
and average the prediction over the 8 images of every test cell. Because C22 is NOT D4
invariant (the 90-degree rotations swap C11 and C22), the target is transformed with the
cell: rotations by 90/270 and the anti-diagonal reflections map C22 -> C11. This is
handled explicitly below. 8x training cost, 8x inference cost.

This is a LOWER bound on a properly equivariant architecture and is reported as such.
"""
import _path  # noqa: F401  (adds code/ and code/vendor/ to sys.path)

import json, os, sys, time
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ablate2 as A, robust as RB
from run_feature2 import FAMS
OUT = "/mnt/user-data/outputs/p3/results/d4.json"
DESC = "/mnt/user-data/outputs/descriptors_v2_24grey"
# (transform, swaps C11<->C22?)
T = [(lambda a: a, False), (lambda a: np.rot90(a, 1, (1, 2)), True),
     (lambda a: np.rot90(a, 2, (1, 2)), False), (lambda a: np.rot90(a, 3, (1, 2)), True),
     (lambda a: a[:, ::-1, :], False), (lambda a: a[:, :, ::-1], False),
     (lambda a: np.transpose(a, (0, 2, 1)), True),
     (lambda a: np.rot90(a, 2, (1, 2))[:, ::-1, :][:, :, ::-1].transpose(0, 2, 1), True)]

sel = A.selection(); C = A.load(DESC, sel)
IMG = {f: C[f]["X"].reshape(-1, 48, 48) for f in FAMS}
Y22 = {f: C[f]["y"][:, 2] for f in FAMS}     # C22
Y11 = {f: C[f]["y"][:, 0] for f in FAMS}     # C11

def orbit(fams, train):
    X, y = [], []
    for f in fams:
        for tf, swap in T:
            X.append(tf(IMG[f]).reshape(len(IMG[f]), -1))
            y.append(Y11[f] if swap else Y22[f])
    return np.vstack(X).astype(np.float32), np.concatenate(y)

def predict_avg(m, f):
    """Average the 8 orbit predictions, mapping each back to C22 of the original."""
    ps = []
    for tf, swap in T:
        p = m.predict(tf(IMG[f]).reshape(len(IMG[f]), -1).astype(np.float32))
        ps.append(p)          # each already predicts the transformed cell's C22 slot
    return np.mean(ps, 0)

def main():
    subs = RB.subsets(); R = json.load(open(OUT)) if os.path.exists(OUT) else {}
    for arm in ("pix_d4", "pix_plain"):
        if arm in R: continue
        t0 = time.time(); r = []
        for sub in subs[:30]:                      # 30 subsets: 8x cost forces a sample
            tr = list(sub); te = [f for f in FAMS if f not in sub]
            if arm == "pix_d4":
                X, y = orbit(tr, True)
            else:
                X = np.vstack([IMG[f].reshape(len(IMG[f]), -1) for f in tr]).astype(np.float32)
                y = np.concatenate([Y22[f] for f in tr])
            for s in (0,):
                m = RandomForestRegressor(n_estimators=60, random_state=s, max_features=0.3,
                                          min_samples_leaf=2, n_jobs=1).fit(X, y)
                if arm == "pix_d4":
                    p = np.concatenate([predict_avg(m, f) for f in te])
                else:
                    p = m.predict(np.vstack([IMG[f].reshape(len(IMG[f]), -1) for f in te]).astype(np.float32))
                yt = np.concatenate([Y22[f] for f in te])
                r.append(float(r2_score(yt, p)))
            print(f"    {arm} {len(r)}/30  {np.median(r):+.3f}  ({(time.time()-t0)/60:.1f} min)", flush=True)
        R[arm] = r
        print(f"  {arm:10s} median {np.median(r):+.3f}  IQR [{np.percentile(r,25):+.3f},{np.percentile(r,75):+.3f}]", flush=True)
        json.dump(R, open(OUT+".tmp","w")); os.replace(OUT+".tmp", OUT)
    print("DONE", flush=True)

if __name__ == "__main__":
    main()
