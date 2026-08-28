"""learners.py -- does the frac+mode2 gain survive a change of surrogate class?
Same 60 K=4 subsets (rng 7,8,9), same 5,500 cells, pooled held-out C22 R^2.
Also: per-held-out-family breakdown (RF, v2_24grey) of frac+mode2 vs frac."""
import _path  # noqa: F401  (adds code/ and code/vendor/ to sys.path)

import json, os, sys, numpy as np
from sklearn.metrics import r2_score
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ablate2 as A, robust as RB
from run_feature2 import FAMS
OUT = "/mnt/user-data/outputs/p3/results/learners.json"
sel = A.selection(); subs = RB.subsets()
LEARN = {
 "RF":   lambda: RandomForestRegressor(n_estimators=60, random_state=0, max_features=0.3, min_samples_leaf=2, n_jobs=1),
 "GBR":  lambda: GradientBoostingRegressor(n_estimators=200, max_depth=3, learning_rate=0.05, random_state=0),
 "KRR":  lambda: make_pipeline(StandardScaler(), KernelRidge(alpha=0.01, kernel="rbf", gamma=0.5)),
 "MLP":  lambda: make_pipeline(StandardScaler(), MLPRegressor(hidden_layer_sizes=(32,32), max_iter=2000, random_state=0, alpha=1e-3)),
}
R = json.load(open(OUT)) if os.path.exists(OUT) else {}
for setname in ("v2_24grey", "v2_48"):
    C = A.load(RB.SETS[setname], sel)
    for arm in ("frac", "frac+mode2"):
        for ln, mk in LEARN.items():
            key = f"{setname}|{arm}|{ln}"
            if key in R: continue
            r22, perfam = [], {f: [] for f in FAMS}
            for sub in subs:
                tr = list(sub); te = [f for f in FAMS if f not in sub]
                X = np.vstack([RB.feats(C, f, arm) for f in tr]); Xt = np.vstack([RB.feats(C, f, arm) for f in te])
                y = np.concatenate([C[f]["y"][:, 2] for f in tr]); yt = np.concatenate([C[f]["y"][:, 2] for f in te])
                m = mk().fit(X, y); p = m.predict(Xt)
                r22.append(float(r2_score(yt, p)))
                off = 0
                for f in te:
                    n = len(C[f]["y"]); perfam[f].append(float(r2_score(yt[off:off+n], p[off:off+n]))); off += n
            R[key] = {"C22": r22, "perfam": {f: float(np.median(v)) for f, v in perfam.items()}}
            print(f"  {key:28s} C22 median {np.median(r22):+.3f}  IQR [{np.percentile(r22,25):+.3f},{np.percentile(r22,75):+.3f}]", flush=True)
            json.dump(R, open(OUT+".tmp","w")); os.replace(OUT+".tmp", OUT)
print("DONE", flush=True)
