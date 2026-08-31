"""
wasserstein_test.py -- does the Liu et al. (2026) result reproduce on MetaXFam-D?

THE CLAIM UNDER TEST.  Liu, Baishnab, Pokuri, Ganapathysubramanian & Wodo,
"Constructing generalizable microstructure-property maps across diverse microstructure
classes", MRS Communications 16:449-458 (2026), doi 10.1557/s43579-025-00873-z (open
access).  They extrapolate models trained on one microstructure type (spinodal) to another
(dendritic) using three featurisations -- two-point correlation functions, graph-based
descriptors, and deep network embeddings -- and report that "the Wasserstein distance is an
excellent metric that correlates well with generalizability, serving as a model-agnostic
yet data-aware signature of generalizability."

WHY THIS MATTERS HERE.  The training-family-selection study on this project reached the
opposite conclusion: no cheap criterion, distributional ones included, robustly predicts
which training families transfer, and a criterion that looked strong failed once the
surrogate class changed.  Both cannot be straightforwardly true, so the disagreement has to
be settled rather than ignored.

WHAT IS DIFFERENT ABOUT THIS TEST, AND WHY THAT IS FAIR TO THEM.
  * Their setting: a few microstructure classes (spinodal, dendritic), continuous
    two-phase morphologies.  Ours: 18 parametric metamaterial families.  A failure to
    reproduce here is NOT a refutation of their result in their setting; it bounds its
    generality.  That distinction is preserved in how the result is reported.
  * We test at the level THEY test: distance from the training pool to ONE held-out class,
    against how well the model does on that class.  Per-family, not per-subset.
  * We use their featurisation where we have it (two-point correlation functions) plus
    others, because the claim is that the metric is featurisation-robust.
  * We test three surrogate classes, because "model-agnostic" is an explicit part of the
    claim and is exactly where the earlier criterion broke.

Data: MetaXFam-D (18 families x 172 distinct cells, zero duplicates).
Metric: sliced Wasserstein distance (mean 1-D Wasserstein over random projections), which
is the standard tractable estimator for multivariate distributions.
"""
import json, os, sys, time
import numpy as np
from scipy.stats import wasserstein_distance, spearmanr
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import RidgeCV
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.decomposition import PCA
from sklearn.metrics import r2_score

DATA = "/mnt/user-data/outputs/metaxfam_d"
OUT = "/mnt/user-data/outputs/p4/results/wasserstein.json"
os.makedirs(os.path.dirname(OUT), exist_ok=True)

FAMS = sorted({f.replace("clean48_", "").replace("__X.npy", "")
               for f in os.listdir(DATA) if f.endswith("__X.npy")})
X = {f: np.load(f"{DATA}/clean48_{f}__X.npy").reshape(-1, 48 * 48).astype(np.float32)
     for f in FAMS}
Y = {f: np.load(f"{DATA}/clean48_{f}__y.npy") for f in FAMS}
COL = {"C11": 0, "C22": 3}          # raw Voigt columns


# ------------------------------------------------------------------ featurisations
def two_point(img48, nbins=16):
    """Radially averaged periodic two-point correlation -- the featurisation Liu et al.
    use as their primary descriptor."""
    n = 48
    a = img48.reshape(-1, n, n)
    F = np.fft.rfft2(a - a.mean(axis=(1, 2), keepdims=True))
    C = np.fft.irfft2(F * np.conj(F), s=(n, n)) / (n * n)
    yy, xx = np.mgrid[0:n, 0:n]
    r = np.hypot(np.minimum(xx, n - xx), np.minimum(yy, n - yy))
    edges = np.linspace(0, n / 2, nbins + 1)
    b = np.clip(np.digitize(r.ravel(), edges) - 1, 0, nbins - 1)
    out = np.zeros((len(a), nbins))
    Cf = C.reshape(len(a), -1)
    for k in range(nbins):
        m = b == k
        if m.any():
            out[:, k] = Cf[:, m].mean(axis=1)
    return out

_TP = {f: two_point(X[f]) for f in FAMS}
_pca = PCA(n_components=16, random_state=0).fit(np.vstack([X[f] for f in FAMS]))
_PC = {f: _pca.transform(X[f]) for f in FAMS}

FEATS = {
    "two-point corr (16)": lambda f: _TP[f],
    "pixel PCA (16)":      lambda f: _PC[f],
    "raw pixels (2304)":   lambda f: X[f],
}


def sliced_wasserstein(A, B, n_proj=200, seed=0):
    """Mean 1-D Wasserstein distance over random projections."""
    rng = np.random.default_rng(seed)
    A = np.nan_to_num(A); B = np.nan_to_num(B)
    mu, sd = A.mean(0), A.std(0) + 1e-9
    A = (A - mu) / sd; B = (B - mu) / sd
    d = A.shape[1]
    V = rng.normal(size=(d, n_proj)); V /= np.linalg.norm(V, axis=0, keepdims=True)
    PA, PB = A @ V, B @ V
    return float(np.mean([wasserstein_distance(PA[:, i], PB[:, i]) for i in range(n_proj)]))


LEARNERS = {
    "RF":    lambda: RandomForestRegressor(n_estimators=60, random_state=0,
                                           max_features=0.3, min_samples_leaf=2, n_jobs=1),
    "ridge": lambda: make_pipeline(StandardScaler(with_mean=False),
                                   RidgeCV(alphas=np.logspace(-2, 4, 13))),
    "MLP":   lambda: make_pipeline(StandardScaler(with_mean=False),
                                   MLPRegressor(hidden_layer_sizes=(64, 64), max_iter=800,
                                                random_state=0, early_stopping=True)),
}


def main():
    rng = np.random.default_rng(7)
    subs = []
    while len(subs) < 15:
        c = tuple(sorted(rng.choice(FAMS, 4, replace=False)))
        if c not in subs:
            subs.append(c)

    R = json.load(open(OUT)) if os.path.exists(OUT) else {}
    for fname, ffn in FEATS.items():
        for tname, col in COL.items():
            for ln, mk in LEARNERS.items():
                key = f"{fname}|{tname}|{ln}"
                if key in R:
                    continue
                t0 = time.time()
                W, r2 = [], []
                for sub in subs:
                    tr = list(sub); te = [f for f in FAMS if f not in sub]
                    Xtr = np.vstack([X[f] for f in tr])
                    ytr = np.concatenate([Y[f][:, col] for f in tr])
                    m = mk().fit(Xtr, ytr)
                    Ftr = np.vstack([ffn(f) for f in tr])
                    for f in te:                       # per held-out family, as they do
                        W.append(sliced_wasserstein(Ftr, ffn(f)))
                        r2.append(float(r2_score(Y[f][:, col], m.predict(X[f]))))
                W = np.array(W); r2 = np.array(r2)
                rho = float(spearmanr(W, r2).statistic)
                R[key] = {"W": W.tolist(), "r2": r2.tolist(), "rho": rho, "n": len(W)}
                print(f"  {fname:20s} {tname} {ln:5s}  rho(W, R2) = {rho:+.3f}"
                      f"   n={len(W)}   ({time.time()-t0:.0f}s)", flush=True)
                json.dump(R, open(OUT + ".tmp", "w")); os.replace(OUT + ".tmp", OUT)

    print("\n" + "=" * 74)
    print("Liu et al. predict a STRONG NEGATIVE rho: larger Wasserstein distance from the")
    print("training pool  =>  worse generalisation to that held-out class.")
    print("=" * 74)
    print(f"{'featurisation':22s}{'target':7s}" + "".join(f"{l:>10s}" for l in LEARNERS))
    for fname in FEATS:
        for tname in COL:
            row = f"{fname:22s}{tname:7s}"
            for ln in LEARNERS:
                row += f"{R[f'{fname}|{tname}|{ln}']['rho']:+10.3f}"
            print(row)
    allrho = [R[k]["rho"] for k in R if isinstance(R[k], dict) and "rho" in R[k]]
    print(f"\n  median rho over all {len(allrho)} settings: {np.median(allrho):+.3f}")
    print(f"  settings with rho < -0.5 (claim supported): "
          f"{sum(r < -0.5 for r in allrho)}/{len(allrho)}")
    print(f"  settings with |rho| < 0.2 (no relationship): "
          f"{sum(abs(r) < 0.2 for r in allrho)}/{len(allrho)}")
    json.dump(R, open(OUT, "w"))
    print("\nDONE")


if __name__ == "__main__":
    main()
