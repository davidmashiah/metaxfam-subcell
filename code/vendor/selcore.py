"""
selcore.py -- efficient core for the training-FAMILY selection study.

Design:
  * load every family ONCE, flatten ONCE, cache to disk.
  * SOLID-FRACTION HISTOGRAM MATCHING.  Every family is subsampled so all ten
    families share (as closely as the data allow) the SAME solid-fraction
    histogram on [0.45,0.75].  Band-matching alone -- what the EML paper did --
    is not enough here, because several families are non-uniform inside the
    band; without histogram matching a "cross-family" error is partly a density
    extrapolation.
  * project onto a SHARED, LABEL-FREE PCA basis fitted on the pooled unlabelled
    images of all ten families.  The basis is identical for every candidate
    training subset, so differences in held-out R^2 come only from WHICH
    FAMILIES are in the training union.  No target values are used to build it.
  * evaluate a subset with one RF fit + cheap per-family predictions.
"""
import os, time
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

import datapath

FAMS10 = ["circle", "square", "cross", "star_hole",
          "ellipse", "rect_hole", "slot_pair",
          "kagome", "honeycomb", "rand_lattice",
          "tri_hole", "hex_hole", "diamond_hole", "star6_hole",
          "two_holes", "cross_aniso",
          "square_lattice", "diag_lattice",
          "rect_lattice", "chevron", "reentrant", "layered"]                       # anisotropic strut

TARGETS = {"C11": 0, "C12": 1, "C22": 3, "C33": 5}

PREFIX    = os.environ.get("SELPREFIX", "clean48_")
N_PER_FAM = int(os.environ.get("SELN", "300"))
N_PCA     = 48
NBINS     = 10
_HERE     = os.path.dirname(os.path.abspath(__file__))
CACHE     = os.path.join(_HERE, f"_selcache_{PREFIX}{N_PER_FAM}.npz")


def _match_indices(fracs, n_want, rng):
    """Indices per family so that all families share one solid-fraction histogram."""
    edges = np.linspace(0.45, 0.75, NBINS + 1)
    binid = {f: np.clip(np.digitize(v, edges) - 1, 0, NBINS - 1) for f, v in fracs.items()}
    counts = np.array([[np.sum(binid[f] == b) for b in range(NBINS)] for f in fracs])
    share = counts.min(axis=0).astype(float)
    if share.sum() == 0:
        share = counts.mean(axis=0).astype(float)
    quota = np.floor(share / share.sum() * n_want).astype(int)
    out = {}
    for f in fracs:
        idx = []
        for b in range(NBINS):
            pool = np.where(binid[f] == b)[0]
            k = min(int(quota[b]), len(pool))
            if k:
                idx.extend(rng.choice(pool, k, replace=False))
        idx = np.array(idx, dtype=int)
        if len(idx) < n_want:
            rest = np.setdiff1d(np.arange(len(binid[f])), idx)
            if len(rest):
                idx = np.concatenate([idx, rng.choice(rest, min(n_want - len(idx), len(rest)),
                                                      replace=False)])
        out[f] = np.sort(idx[:n_want])
    return out


def build_cache(verbose=True):
    if os.path.exists(CACHE):
        d = np.load(CACHE, allow_pickle=True)
        return {k: d[k] for k in d.files}
    rng = np.random.default_rng(0)
    raw = {f: (lambda d: (d["X"], d["y"], d["frac"]))(datapath.load(f"{PREFIX}{f}"))
           for f in FAMS10}
    sel = _match_indices({f: raw[f][2] for f in FAMS10}, N_PER_FAM, rng)

    Xs, ys, fr, sizes = [], [], [], []
    for f in FAMS10:
        X, y, fc = raw[f]; i = sel[f]
        Xs.append(X[i].reshape(len(i), -1).astype(np.float32))
        ys.append(y[i]); fr.append(fc[i]); sizes.append(len(i))
        if verbose:
            a = np.abs(y[i, 0] - y[i, 3]) / (y[i, 0] + y[i, 3])
            print(f"  {f:14s} N={len(i):4d}  frac {fc[i].mean():.3f}"
                  f" [{fc[i].min():.3f},{fc[i].max():.3f}]"
                  f"  aniso(med) {np.median(a):.4f}  C11 mean {y[i,0].mean():.4f}")

    Xall = np.vstack(Xs); mu = Xall.mean(0)
    from sklearn.decomposition import PCA
    t0 = time.time()
    pca = PCA(n_components=N_PCA, svd_solver="randomized", random_state=0).fit(Xall - mu)
    if verbose:
        print(f"  PCA({N_PCA}) {time.time()-t0:.1f}s  "
              f"explained var {pca.explained_variance_ratio_.sum():.4f}")
    Z = pca.transform(Xall - mu).astype(np.float32)
    out = {"sizes": np.array(sizes), "Z": Z, "y": np.vstack(ys),
           "frac": np.concatenate(fr), "Xraw": Xall.astype(np.uint8)}
    np.savez_compressed(CACHE, **out)
    return out


_C = _SLICE = None
def _ensure():
    global _C, _SLICE
    if _C is None:
        _C = build_cache(verbose=False)
        off = np.concatenate([[0], np.cumsum(_C["sizes"])])
        _SLICE = {f: slice(int(off[i]), int(off[i + 1])) for i, f in enumerate(FAMS10)}
    return _C, _SLICE

def feats(fam): C, S = _ensure(); return C["Z"][S[fam]]
def yall(fam):  C, S = _ensure(); return C["y"][S[fam]]
def frac(fam):  C, S = _ensure(); return C["frac"][S[fam]]
def images(fam):
    C, S = _ensure()
    n = int(round(np.sqrt(C["Xraw"].shape[1])))
    return C["Xraw"][S[fam]].reshape(-1, n, n)
def targ(fam, target="C11"): return yall(fam)[:, TARGETS[target]]


_EVAL = {}
def evaluate(train_fams, target="C11", seed=0, pool=FAMS10, n_estimators=60):
    key = (tuple(sorted(train_fams)), target, seed, n_estimators)
    if key in _EVAL:
        return _EVAL[key]
    Xtr = np.vstack([feats(f) for f in train_fams])
    ytr = np.concatenate([targ(f, target) for f in train_fams])
    m = RandomForestRegressor(n_estimators=n_estimators, random_state=seed,
                              n_jobs=1, max_features=0.3, min_samples_leaf=2)
    m.fit(Xtr, ytr)
    out = {f: float(r2_score(targ(f, target), m.predict(feats(f))))
           for f in pool if f not in train_fams}
    _EVAL[key] = out
    return out


def wc_mean(train_fams, target="C11", seeds=(0, 1, 2), **kw):
    wcs, mns = [], []
    for s in seeds:
        r = evaluate(train_fams, target, s, **kw)
        wcs.append(min(r.values())); mns.append(float(np.mean(list(r.values()))))
    return float(np.mean(wcs)), float(np.mean(mns))


if __name__ == "__main__":
    print(f"Building cache  prefix={PREFIX}  N={N_PER_FAM}")
    build_cache()
    t0 = time.time(); r = evaluate(("circle", "honeycomb"), "C11", 0)
    print(f"\none evaluate(): {time.time()-t0:.2f}s")
    for k, v in sorted(r.items(), key=lambda kv: kv[1]):
        print(f"   {k:14s} R2 = {v:+8.3f}")
