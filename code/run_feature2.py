"""
run_feature2.py -- sub-cell modes as a CHEAP INPUT FEATURE.

The descriptor is only a method if it can be computed for much less than the
homogenisation it helps replace.  A full 48x48 eigensolve costs the same as the
homogenisation itself (0.96x), so is useless as a shortcut.  A 16x16 eigensolve
costs a tenth as much.  This script asks whether the cross-family gain survives
that degradation.

Arms (identical model, protocol, seeds):
  pixels     image                       -> C^H
  pix_frac   image + solid fraction      -> C^H   (cheap-scalar control)
  pix_desc   image + coarse descriptors  -> C^H
  desc       coarse descriptors alone    -> C^H
"""
import _path  # noqa: F401  (adds code/ and code/vendor/ to sys.path)

import json, os, sys, time
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import datapath

FAMS = ["circle","square","cross","star_hole","star6_hole","tri_hole","hex_hole",
        "diamond_hole","ellipse","rect_hole","slot_pair","two_holes","cross_aniso",
        "kagome","square_lattice","diag_lattice","rand_lattice","honeycomb",
        "rect_lattice","chevron","reentrant","layered"]
KEEP = [0, 1, 3, 5]; KN = ["C11", "C12", "C22", "C33"]
N_PER = 250
RES = int(os.environ.get("DESCRES", "16"))
DESC = f"/mnt/user-data/outputs/descriptors_{RES}"
OUT = f"/mnt/user-data/outputs/results_feature{RES}.json"

def rf(seed, n=60):
    return RandomForestRegressor(n_estimators=n, random_state=seed, n_jobs=1,
                                 max_features=0.3, min_samples_leaf=2)

_C = None
def load():
    global _C
    if _C is not None: return _C
    rng = np.random.default_rng(0); raw = {}; fr = {}
    for f in FAMS:
        d = datapath.load(f"clean48_{f}")
        om = np.load(f"{DESC}/omega_{f}.npy"); ch = np.load(f"{DESC}/char_{f}.npy")
        raw[f] = (d["X"], d["y"], d["frac"], om, ch); fr[f] = d["frac"]
    edges = np.linspace(0.45, 0.75, 11)
    binid = {f: np.clip(np.digitize(v, edges)-1, 0, 9) for f, v in fr.items()}
    counts = np.array([[np.sum(binid[f] == b) for b in range(10)] for f in FAMS])
    share = counts.min(axis=0).astype(float)
    if share.sum() == 0: share = counts.mean(axis=0).astype(float)
    quota = np.floor(share/share.sum()*N_PER).astype(int)
    out = {}
    for f in FAMS:
        X, y, fc, om, ch = raw[f]
        good = np.isfinite(om).all(1) & np.isfinite(ch).all(axis=(1,2)) & (om > 1e-9).all(1)
        idx = []
        for b in range(10):
            pool = np.where((binid[f] == b) & good)[0]
            k = min(int(quota[b]), len(pool))
            if k: idx.extend(rng.choice(pool, k, replace=False))
        idx = np.array(idx, dtype=int)
        if len(idx) < N_PER:
            rest = np.setdiff1d(np.where(good)[0], idx)
            if len(rest):
                idx = np.concatenate([idx, rng.choice(rest, min(N_PER-len(idx), len(rest)), replace=False)])
        idx = np.sort(idx[:N_PER])
        D = np.hstack([np.log(np.clip(om[idx], 1e-6, None)), ch[idx].reshape(len(idx), -1)])
        out[f] = dict(X=X[idx].reshape(len(idx), -1).astype(np.float32),
                      y=y[idx][:, KEEP], D=D.astype(np.float64), frac=fc[idx])
    _C = out; return out

def run(seed, K, nsub, rng):
    C = load()
    subs = [(f,) for f in FAMS] if K == 1 else \
           [tuple(sorted(rng.choice(FAMS, K, replace=False))) for _ in range(nsub)]
    out = []
    for sub in subs:
        tr = list(sub); te = [f for f in FAMS if f not in sub]
        Xtr = np.vstack([C[f]["X"] for f in tr]); Xte = np.vstack([C[f]["X"] for f in te])
        Dtr = np.vstack([C[f]["D"] for f in tr]); Dte = np.vstack([C[f]["D"] for f in te])
        Ftr = np.vstack([C[f]["frac"][:,None] for f in tr]); Fte = np.vstack([C[f]["frac"][:,None] for f in te])
        Ytr = np.vstack([C[f]["y"] for f in tr]); Yte = np.vstack([C[f]["y"] for f in te])
        rec = {"train": tr}
        for nm, A, B in [("pixels", Xtr, Xte),
                         ("pix_frac", np.hstack([Xtr,Ftr]), np.hstack([Xte,Fte])),
                         ("pix_desc", np.hstack([Xtr,Dtr]), np.hstack([Xte,Dte])),
                         ("desc", Dtr, Dte)]:
            p = rf(seed).fit(A, Ytr).predict(B)
            rec[nm] = [float(r2_score(Yte[:,j], p[:,j])) for j in range(len(KEEP))]
        out.append(rec)
        print("      %-28s C22 pix %+7.2f  +desc %+7.2f  desc %+7.2f  +frac %+7.2f"
              % ("+".join(tr)[:28], rec["pixels"][2], rec["pix_desc"][2],
                 rec["desc"][2], rec["pix_frac"][2]), flush=True)
    return out

if __name__ == "__main__":
    R = json.load(open(OUT)) if os.path.exists(OUT) else {}
    print(f"descriptor resolution {RES}x{RES}\n", flush=True)
    for K, n in [(1,22),(2,30),(4,30)]:
        for s in (0,1):
            k = "K%d|s%d" % (K,s)
            if k in R: print("  %s cached" % k, flush=True); continue
            t = time.time(); print("  K=%d seed=%d" % (K,s), flush=True)
            R[k] = run(s, K, n, np.random.default_rng(100*K+s))
            json.dump(R, open(OUT+".tmp","w")); os.replace(OUT+".tmp", OUT)
            print("  %s done %.1f min" % (k,(time.time()-t)/60), flush=True)
    print("DONE")
