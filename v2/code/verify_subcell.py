"""
verify_subcell.py -- do the sub-cell paper's conclusions survive on duplicate-free data?

WHY.  The sub-cell paper (under review, AIP Advances ADV26-AR-04295) was computed on
MetaXFam22, which is now known to be 61.6% duplicated at 48x48 and 92.3% duplicated at
24x24 -- and the mode-2 descriptors were computed at 24x24. Its headline comparisons are
cross-family, so duplication should not favour one representation over another, but that is
an expectation, not a verified fact. This script verifies it.

THE THREE CLAIMS UNDER TEST, in the paper's own numbers (measured on the duplicated data):

  CLAIM 1  fraction + mode-2 character transfers far better than raw pixels
           reported: +0.356 vs -0.008 on C22, and +0.082 for solid fraction alone

  CLAIM 2  it is the ELASTIC spectrum, not any spectrum
           reported: mode-2 character +0.356 beats shape-DNA (geometric Laplacian
           spectrum, the published prior art) at +0.106, on 96% of paired evaluations

  CLAIM 3  the ordering is robust to surrogate class
           reported: every learner with a sane inductive bias improves

Each is recomputed here on MetaXFam-D (18 families x 172 cells, ZERO duplicates), with
descriptors recomputed from scratch at 24x24 on the clean cells. The question is not
whether the absolute numbers move -- they will, because the clean families are harder --
but whether the ORDERING and the SIGN of the effects survive. That is what the paper
claims.
"""
import json, os, sys, time
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import RidgeCV
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import r2_score
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import eigsh

sys.path.insert(0, "/mnt/user-data/outputs/p3/code")
import subcell as SB

DATA = "/mnt/user-data/outputs/metaxfam_d"
DESC = "/mnt/user-data/outputs/p4/descriptors_clean24"
OUT = "/mnt/user-data/outputs/p4/results/verify_subcell.json"
os.makedirs(DESC, exist_ok=True); os.makedirs(os.path.dirname(OUT), exist_ok=True)

FAMS = sorted({f.replace("clean48_", "").replace("__X.npy", "")
               for f in os.listdir(DATA) if f.endswith("__X.npy")})
COL22 = 3      # C22 in raw Voigt order


def load(f):
    X = np.load(f"{DATA}/clean48_{f}__X.npy").astype(float)
    y = np.load(f"{DATA}/clean48_{f}__y.npy")
    fr = np.load(f"{DATA}/clean48_{f}__frac.npy")
    return X, y, fr


# ---------------------------------------------------------------- descriptors
def compute_descriptors():
    for f in FAMS:
        fo = f"{DESC}/{f}__m2.npy"
        if os.path.exists(fo):
            continue
        X, _, _ = load(f)
        t0 = time.time()
        M = np.zeros((len(X), 3))
        for i, x in enumerate(X):
            d = SB.descriptors(SB.coarsen(x, 24, "grey"))
            M[i] = d["char"][0][[2, 0, 1]]        # (rot, dil, shr)
        np.save(fo, M)
        print(f"    {f:22s} {len(X)} cells  {(time.time()-t0)/len(X)*1000:5.1f} ms/cell",
              flush=True)


# shape-DNA: geometric Laplacian spectrum of the solid phase (the prior-art control)
def shape_dna(solid48, n_eig=6):
    a = (SB.coarsen(solid48, 24, "grey") >= .5)
    n = a.shape[0]; m = a.ravel()
    N = int(m.sum())
    if N < n_eig + 2:
        return np.full(n_eig, np.nan)
    idx = -np.ones(n * n, int); idx[m] = np.arange(N)
    g = np.arange(n * n).reshape(n, n)
    rows, cols, vals = [], [], []
    h2 = (1.0 / n) ** 2
    for sh, ax in ((1, 0), (-1, 0), (1, 1), (-1, 1)):
        nb = np.roll(g, sh, axis=ax).ravel()
        both = m & m[nb]
        rows.append(idx[both]); cols.append(idx[nb[both]])
        vals.append(np.full(both.sum(), -1.0 / h2))
    deg = np.zeros(N)
    for r, v in zip(rows, vals):
        np.add.at(deg, r, -v)
    rows.append(np.arange(N)); cols.append(np.arange(N)); vals.append(deg)
    L = coo_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                   shape=(N, N)).tocsc()
    try:
        lam = eigsh(L, k=n_eig + 1, sigma=-1e-6, which="LM",
                    v0=np.linspace(.3, 1., N), tol=1e-8, return_eigenvectors=False)
    except Exception:
        return np.full(n_eig, np.nan)
    return np.sort(np.clip(lam, 0, None))[1:n_eig + 1]


def compute_sdna():
    for f in FAMS:
        fo = f"{DESC}/{f}__sdna.npy"
        if os.path.exists(fo):
            continue
        X, _, _ = load(f)
        S = np.array([shape_dna(x) for x in X])
        np.save(fo, S)
        print(f"    shape-DNA {f:18s} nan {int(np.isnan(S).any(1).sum())}", flush=True)


def main():
    print("Recomputing sub-cell descriptors at 24x24 on the CLEAN benchmark")
    compute_descriptors()
    print("Recomputing shape-DNA (prior-art control)")
    compute_sdna()

    X = {f: load(f)[0].reshape(-1, 48 * 48).astype(np.float32) for f in FAMS}
    Y = {f: load(f)[1][:, COL22] for f in FAMS}
    FR = {f: load(f)[2][:, None].astype(np.float32) for f in FAMS}
    M2 = {f: np.load(f"{DESC}/{f}__m2.npy").astype(np.float32) for f in FAMS}
    SD = {}
    for f in FAMS:
        s = np.load(f"{DESC}/{f}__sdna.npy")
        s = np.log(np.clip(s, 1e-9, None))
        bad = ~np.isfinite(s).all(1)
        if bad.any():
            s[bad] = np.nanmedian(s[~bad], 0)
        SD[f] = s.astype(np.float32)

    ARMS = {
        "pixels (2304)":        lambda f: X[f],
        "solid fraction (1)":   lambda f: FR[f],
        "frac + shape-DNA (7)": lambda f: np.hstack([FR[f], SD[f]]),
        "mode-2 char (3)":      lambda f: M2[f],
        "frac + mode-2 (4)":    lambda f: np.hstack([FR[f], M2[f]]),
    }
    LEARNERS = {
        "RF":    lambda: RandomForestRegressor(n_estimators=60, random_state=0,
                                               max_features=0.3, min_samples_leaf=2, n_jobs=1),
        "ridge": lambda: make_pipeline(StandardScaler(with_mean=False),
                                       RidgeCV(alphas=np.logspace(-2, 4, 13))),
        "MLP":   lambda: make_pipeline(StandardScaler(with_mean=False),
                                       MLPRegressor(hidden_layer_sizes=(64, 64),
                                                    max_iter=800, random_state=0,
                                                    early_stopping=True)),
    }

    rng = np.random.default_rng(7)
    subs = []
    while len(subs) < 20:
        c = tuple(sorted(rng.choice(FAMS, 4, replace=False)))
        if c not in subs:
            subs.append(c)

    R = json.load(open(OUT)) if os.path.exists(OUT) else {}
    print("\n" + "=" * 76)
    print("C22 transfer to unseen families, MetaXFam-D (zero duplicates), median of 20")
    print("=" * 76)
    print(f"{'arm':24s}" + "".join(f"{l:>12s}" for l in LEARNERS))
    for arm, fn in ARMS.items():
        row = f"{arm:24s}"
        for ln, mk in LEARNERS.items():
            key = f"{arm}|{ln}"
            if key not in R:
                v = []
                for s in subs:
                    tr = list(s); te = [f for f in FAMS if f not in s]
                    Xtr = np.vstack([fn(f) for f in tr])
                    ytr = np.concatenate([Y[f] for f in tr])
                    Xte = np.vstack([fn(f) for f in te])
                    yte = np.concatenate([Y[f] for f in te])
                    v.append(float(r2_score(yte, mk().fit(Xtr, ytr).predict(Xte))))
                R[key] = v
                json.dump(R, open(OUT + ".tmp", "w")); os.replace(OUT + ".tmp", OUT)
            row += f"{np.median(R[key]):+12.3f}"
        print(row, flush=True)

    print("\n" + "=" * 76)
    print("PAPER'S CLAIMS, re-checked on clean data (RF, paired over the same 20 subsets)")
    print("=" * 76)
    a = np.array(R["frac + mode-2 (4)|RF"]); p = np.array(R["pixels (2304)|RF"])
    fr = np.array(R["solid fraction (1)|RF"]); sd = np.array(R["frac + shape-DNA (7)|RF"])
    print(f"  CLAIM 1  frac+mode-2 vs pixels      gain {np.median(a-p):+.3f}"
          f"   wins {np.mean(a>p):.0%}   (paper: +0.364, 100%)")
    print(f"           frac+mode-2 vs frac alone  gain {np.median(a-fr):+.3f}"
          f"   wins {np.mean(a>fr):.0%}   (paper: +0.274, 97%)")
    print(f"  CLAIM 2  frac+mode-2 vs shape-DNA   gain {np.median(a-sd):+.3f}"
          f"   wins {np.mean(a>sd):.0%}   (paper: +0.250, 96%)")
    ok3 = all(np.median(R[f"frac + mode-2 (4)|{l}"]) > np.median(R[f"solid fraction (1)|{l}"])
              for l in LEARNERS)
    print(f"  CLAIM 3  improves for every learner: {'YES' if ok3 else 'NO'}")
    R["_verdict"] = dict(c1_vs_pixels=float(np.median(a - p)), c1_win=float(np.mean(a > p)),
                         c1_vs_frac=float(np.median(a - fr)), c2_vs_sdna=float(np.median(a - sd)),
                         c2_win=float(np.mean(a > sd)), c3_all_learners=bool(ok3))
    json.dump(R, open(OUT, "w"))
    print("\nDONE")


if __name__ == "__main__":
    main()
