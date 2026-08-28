"""ablate.py -- WHICH descriptors carry the oracle gain?

The oracle arm showed true descriptors -> C^H transferring at R2=+0.33 where
pixels -> C^H sits at 0.00.  If that gain rests on one or two physically simple
quantities, a cheap purpose-built feature may exist.  If it needs the whole
spectrum, it does not.
"""
import _path  # noqa: F401  (adds code/ and code/vendor/ to sys.path)

import json, sys, numpy as np
sys.path.insert(0,'.')
from sklearn.metrics import r2_score
import run_feature2 as RF
from run_feature2 import load, FAMS, rf

C = load()   # NOTE: 16x16 by default; set DESCRES=48 for full
GROUPS = {
    "all 24":            list(range(24)),
    "w2 only":           [0],
    "w2,w3":             [0,1],
    "all 6 freqs":       list(range(6)),
    "freq ratios only":  None,          # handled specially
    "characters only":   list(range(6,24)),
    "mode2 char only":   [6,7,8],
}
rng = np.random.default_rng(7)
subs = [tuple(sorted(rng.choice(FAMS,4,replace=False))) for _ in range(20)]

def build(f, cols):
    D = C[f]["D"]
    if cols is None:                     # ratios w3/w2, w4/w2 ... (log diffs)
        return np.column_stack([D[:,j]-D[:,0] for j in range(1,6)])
    return D[:, cols]

print(f"{'feature set':20s}{'C22 median':>12s}{'C11 median':>12s}{'dim':>6s}")
for name, cols in GROUPS.items():
    sc22, sc11 = [], []
    for sub in subs:
        tr=list(sub); te=[f for f in FAMS if f not in sub]
        A=np.vstack([build(f,cols) for f in tr]); B=np.vstack([build(f,cols) for f in te])
        Ytr=np.vstack([C[f]["y"] for f in tr]); Yte=np.vstack([C[f]["y"] for f in te])
        p=rf(0).fit(A,Ytr).predict(B)
        sc11.append(r2_score(Yte[:,0],p[:,0])); sc22.append(r2_score(Yte[:,2],p[:,2]))
    print(f"{name:20s}{np.median(sc22):>+12.3f}{np.median(sc11):>+12.3f}"
          f"{(len(cols) if cols else 5):>6d}")
