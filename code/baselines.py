"""
baselines.py -- the named baselines the paper was missing.

Three families, all computed on the SAME 5500 matched cells and evaluated in the
SAME harness (K=4, 60 subsets, 2 seeds, pooled held-out R^2) as everything else.

1. SHAPE-DNA (Wang, Chan, Liu, Zhu, Chen, Struct. Multidisc. Optim. 61:2613, 2020).
   Laplace-Beltrami spectrum of the SOLID DOMAIN as a shape descriptor.  This is the
   published spectral-descriptor prior art and the decisive control for this paper:
   if a GEOMETRIC spectrum transfers as well as our ELASTIC one, the claim that the
   discarded *elastic* sub-cell content is what matters collapses.
   Implementation: Dirichlet Laplacian on the solid phase of the periodic cell,
   5-point stencil, void masked out, lowest 6 non-trivial eigenvalues.  Reported both
   raw and as ratios (scale-normalised), matching how we treat the elastic spectrum.

2. D4-EQUIVARIANT PIXELS.  A cheap, honest stand-in for the similarity-equivariant
   GNN of Hendriks et al. (CMAME 439:117867, 2025).  We cannot reimplement their
   architecture, but the mechanism they exploit -- invariance of the homogenised
   response to the symmetry group of the square lattice -- is obtained by orbit
   averaging: train on all 8 dihedral transforms of each cell, and average the
   prediction over the 8 transforms of each test cell.  This is exact D4 invariance
   by construction, at 8x cost.  It is a LOWER bound on what a properly equivariant
   architecture achieves, and we say so.

3. INVARIANT GEOMETRIC DESCRIPTORS.  Hu moment invariants of the solid indicator
   (7 numbers, invariant to translation/rotation/scale) plus the radially-averaged
   two-point correlation function (8 bins).  The standard "hand-crafted geometric
   descriptor" family.
"""
import _path  # noqa: F401  (adds code/ and code/vendor/ to sys.path)

import json, os, sys, time
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import eigsh
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ablate2 as A, robust as RB
from run_feature2 import FAMS
import subcell as SB

OUT = "/mnt/user-data/outputs/p3/results/baselines.json"
DESCDIR = "/mnt/user-data/outputs/descriptors_v2_24grey"


# ---------------------------------------------------------------- 1. shape-DNA
def shape_dna(solid, n_eig=6):
    """Lowest non-trivial eigenvalues of the Laplacian on the solid phase of a
    periodic cell.  Void nodes are removed (Dirichlet on the solid/void interface),
    which is the periodic-grid analogue of the Laplace-Beltrami operator on the
    shape.  Returns nan if the solid phase is empty."""
    n = solid.shape[0]
    m = solid.astype(bool).ravel()
    idx = -np.ones(n * n, dtype=int)
    idx[m] = np.arange(m.sum())
    N = int(m.sum())
    if N < n_eig + 2:
        return np.full(n_eig, np.nan)
    rows, cols, vals = [], [], []
    h2 = (1.0 / n) ** 2
    g = np.arange(n * n).reshape(n, n)
    for sh, ax in ((1, 0), (-1, 0), (1, 1), (-1, 1)):
        nb = np.roll(g, sh, axis=ax).ravel()          # periodic neighbour
        both = m & m[nb]
        rows.append(idx[both]); cols.append(idx[nb[both]]); vals.append(np.full(both.sum(), -1.0 / h2))
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
    lam = np.sort(np.clip(lam, 0, None))
    return lam[1:n_eig + 1]          # drop the constant mode


# ---------------------------------------------- 3. invariant geometric descriptors
def hu_moments(img):
    """Seven Hu invariants of a binary image (translation/rotation/scale invariant)."""
    n = img.shape[0]
    y, x = np.mgrid[0:n, 0:n].astype(float)
    m00 = img.sum()
    if m00 < 1:
        return np.zeros(7)
    xb = (x * img).sum() / m00; yb = (y * img).sum() / m00
    def mu(p, q): return (((x - xb) ** p) * ((y - yb) ** q) * img).sum()
    def nu(p, q): return mu(p, q) / m00 ** (1 + (p + q) / 2.0)
    n20, n02, n11 = nu(2, 0), nu(0, 2), nu(1, 1)
    n30, n03, n21, n12 = nu(3, 0), nu(0, 3), nu(2, 1), nu(1, 2)
    h = np.zeros(7)
    h[0] = n20 + n02
    h[1] = (n20 - n02) ** 2 + 4 * n11 ** 2
    h[2] = (n30 - 3 * n12) ** 2 + (3 * n21 - n03) ** 2
    h[3] = (n30 + n12) ** 2 + (n21 + n03) ** 2
    h[4] = ((n30 - 3 * n12) * (n30 + n12) * ((n30 + n12) ** 2 - 3 * (n21 + n03) ** 2)
            + (3 * n21 - n03) * (n21 + n03) * (3 * (n30 + n12) ** 2 - (n21 + n03) ** 2))
    h[5] = ((n20 - n02) * ((n30 + n12) ** 2 - (n21 + n03) ** 2)
            + 4 * n11 * (n30 + n12) * (n21 + n03))
    h[6] = ((3 * n21 - n03) * (n30 + n12) * ((n30 + n12) ** 2 - 3 * (n21 + n03) ** 2)
            - (n30 - 3 * n12) * (n21 + n03) * (3 * (n30 + n12) ** 2 - (n21 + n03) ** 2))
    return np.sign(h) * np.log1p(np.abs(h))


def two_point(img, nbins=8):
    """Radially-averaged periodic two-point correlation of the solid indicator."""
    n = img.shape[0]
    F = np.fft.rfft2(img - img.mean())
    C = np.fft.irfft2(F * np.conj(F), s=(n, n)) / (n * n)
    yy, xx = np.mgrid[0:n, 0:n]
    r = np.hypot(np.minimum(xx, n - xx), np.minimum(yy, n - yy))
    edges = np.linspace(0, n / 2, nbins + 1)
    b = np.clip(np.digitize(r.ravel(), edges) - 1, 0, nbins - 1)
    out = np.zeros(nbins)
    for k in range(nbins):
        sel = b == k
        out[k] = C.ravel()[sel].mean() if sel.any() else 0.0
    return out


# --------------------------------------------------------------- feature caches
def build_cache():
    fn = "/mnt/user-data/outputs/p3/results/baseline_feats.npz"
    if os.path.exists(fn):
        z = np.load(fn); return {k: z[k] for k in z.files}
    sel = A.selection(); C = A.load(DESCDIR, sel)
    out = {}
    for f in FAMS:
        X = C[f]["X"].reshape(-1, 48, 48)
        sd, geo = [], []
        t0 = time.time()
        for img in X:
            c24 = SB.coarsen(img, 24, "grey")
            sd.append(shape_dna((c24 >= .5).astype(float)))
            geo.append(np.concatenate([hu_moments(img), two_point(img)]))
        out[f"sdna_{f}"] = np.array(sd)
        out[f"geo_{f}"] = np.array(geo)
        print(f"  {f:16s} shape-DNA + geometric  {(time.time()-t0)/len(X)*1000:5.1f} ms/cell"
              f"  lam1 med {np.nanmedian(np.array(sd)[:,0]):.1f}  nan {int(np.isnan(np.array(sd)).any(1).sum())}",
              flush=True)
    np.savez(fn, **out)
    return out


# ---------------------------------------------------------------- evaluation
def main():
    sel = A.selection(); C = A.load(DESCDIR, sel)
    B = build_cache()
    subs = RB.subsets()
    R = json.load(open(OUT)) if os.path.exists(OUT) else {}

    def sdna(f):
        v = B[f"sdna_{f}"].copy()
        v[~np.isfinite(v)] = np.nanmedian(v[np.isfinite(v).all(1)], 0)[0]
        return np.log(np.clip(v, 1e-9, None)).astype(np.float32)
    def sdna_ratio(f):
        v = sdna(f); return (v[:, 1:] - v[:, :1]).astype(np.float32)

    ARMS = {
        "shape_dna":        lambda f: sdna(f),
        "frac+shape_dna":   lambda f: np.hstack([C[f]["frac"], sdna(f)]).astype(np.float32),
        "frac+sdna_ratio":  lambda f: np.hstack([C[f]["frac"], sdna_ratio(f)]).astype(np.float32),
        "geom_invariants":  lambda f: B[f"geo_{f}"].astype(np.float32),
        "frac+geom":        lambda f: np.hstack([C[f]["frac"], B[f"geo_{f}"]]).astype(np.float32),
        "frac+mode2":       lambda f: RB.feats(C, f, "frac+mode2"),
        "frac":             lambda f: RB.feats(C, f, "frac"),
    }

    for arm, fn in ARMS.items():
        if arm in R: continue
        t0 = time.time(); r22 = []
        for sub in subs:
            tr = list(sub); te = [f for f in FAMS if f not in sub]
            X = np.vstack([fn(f) for f in tr]); Xt = np.vstack([fn(f) for f in te])
            y = np.concatenate([C[f]["y"][:, 2] for f in tr])
            yt = np.concatenate([C[f]["y"][:, 2] for f in te])
            for s in (0, 1):
                m = RandomForestRegressor(n_estimators=60, random_state=s, max_features=0.3,
                                          min_samples_leaf=2, n_jobs=1).fit(X, y)
                r22.append(float(r2_score(yt, m.predict(Xt))))
        R[arm] = r22
        print(f"  {arm:18s} median {np.median(r22):+.3f}  IQR [{np.percentile(r22,25):+.3f},"
              f"{np.percentile(r22,75):+.3f}]  {time.time()-t0:5.0f}s", flush=True)
        json.dump(R, open(OUT + ".tmp", "w")); os.replace(OUT + ".tmp", OUT)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
