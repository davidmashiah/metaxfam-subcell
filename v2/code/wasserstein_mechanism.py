"""
wasserstein_mechanism.py -- WHY does Wasserstein predict transfer, and how much is density?

The reproduction found that the Liu et al. criterion carries real but weak signal on
MetaXFam-D: all 18 settings negative, median Spearman rho = -0.484, but their own
featurisation (two-point correlations) is the weakest at -0.31, and only 6/18 settings
reach rho < -0.5.

The obvious explanation for the gap between their strong result and this weak one is that
distributional distance between two microstructure classes is dominated by the crudest
difference between them. In their setting (spinodal vs dendritic) that is morphology. In a
metamaterial pool it is usually SOLID FRACTION -- which is also, by itself, a strong
predictor of stiffness. If so, Wasserstein is not a "data-aware signature of
generalizability"; it is largely a density detector, and it should lose most of its power
once density is partialled out.

THREE TESTS.

  T1  PARTIAL CORRELATION.  Spearman rho between Wasserstein distance and transfer R^2,
      controlling for the difference in solid fraction between the training pool and the
      held-out family. If the partial rho collapses toward zero, the criterion was density.

  T2  DENSITY ALONE.  How well does the solid-fraction gap ALONE predict transfer? If it
      matches or beats Wasserstein, the expensive distributional metric buys nothing over
      a single free number -- the same finding the training-family study reported for other
      criteria.

  T3  DOES DE-CONFOUNDING CAUSE THE WEAKNESS?  Build two pools from the same families:
      one solid-fraction MATCHED across families (as MetaXFam-D is), one deliberately
      UNMATCHED (each family drawn from a different part of the density band). If
      Wasserstein looks strong on the unmatched pool and weak on the matched one, that is
      direct evidence that the criterion's apparent power comes from a confound, and it
      explains the disagreement with Liu et al. without either result being wrong.

T3 is the one that turns a disagreement into an explanation.
"""
import json, os, sys, time
import numpy as np
from scipy.stats import wasserstein_distance, spearmanr, rankdata
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

DATA = "/mnt/user-data/outputs/metaxfam_d"
POOL = "/mnt/user-data/outputs/data_distinct"
OUT = "/mnt/user-data/outputs/p4/results/wasserstein_mech.json"
os.makedirs(os.path.dirname(OUT), exist_ok=True)

FAMS = sorted({f.replace("clean48_", "").replace("__X.npy", "")
               for f in os.listdir(DATA) if f.endswith("__X.npy")})
COL = 3      # C22


def load(dirpath, f):
    X = np.load(f"{dirpath}/clean48_{f}__X.npy").reshape(-1, 48 * 48).astype(np.float32)
    y = np.load(f"{dirpath}/clean48_{f}__y.npy")
    fr = np.load(f"{dirpath}/clean48_{f}__frac.npy")
    return X, y, fr


def sliced_w(A, B, n_proj=150, seed=0):
    rng = np.random.default_rng(seed)
    mu, sd = A.mean(0), A.std(0) + 1e-9
    A = (A - mu) / sd; B = (B - mu) / sd
    V = rng.normal(size=(A.shape[1], n_proj)); V /= np.linalg.norm(V, axis=0, keepdims=True)
    PA, PB = A @ V, B @ V
    return float(np.mean([wasserstein_distance(PA[:, i], PB[:, i]) for i in range(n_proj)]))


def partial_spearman(x, y, z):
    """Spearman correlation of x and y controlling for z, via ranks + residualisation."""
    rx, ry, rz = rankdata(x), rankdata(y), rankdata(z)
    def resid(a, b):
        b1 = np.c_[np.ones_like(b), b]
        return a - b1 @ np.linalg.lstsq(b1, a, rcond=None)[0]
    return float(spearmanr(resid(rx, rz), resid(ry, rz)).statistic)


def mk():
    return RandomForestRegressor(n_estimators=60, random_state=0, max_features=0.3,
                                 min_samples_leaf=2, n_jobs=1)


def run_pool(D, fams, subs, tag):
    """D: {family: (X, y, frac)}.  Returns per-held-out-family records."""
    W, dfrac, r2 = [], [], []
    for sub in subs:
        tr = [f for f in sub]; te = [f for f in fams if f not in sub]
        Xtr = np.vstack([D[f][0] for f in tr])
        ytr = np.concatenate([D[f][1][:, COL] for f in tr])
        frtr = np.concatenate([D[f][2] for f in tr])
        m = mk().fit(Xtr, ytr)
        for f in te:
            W.append(sliced_w(Xtr, D[f][0]))
            dfrac.append(abs(np.median(D[f][2]) - np.median(frtr)))
            r2.append(float(r2_score(D[f][1][:, COL], m.predict(D[f][0]))))
    W, dfrac, r2 = map(np.array, (W, dfrac, r2))
    out = dict(
        rho_W=float(spearmanr(W, r2).statistic),
        rho_frac=float(spearmanr(dfrac, r2).statistic),
        rho_W_given_frac=partial_spearman(W, r2, dfrac),
        rho_W_frac=float(spearmanr(W, dfrac).statistic),
        n=len(W))
    print(f"\n  [{tag}]  n={out['n']}")
    print(f"    T2  solid-fraction gap alone      rho = {out['rho_frac']:+.3f}")
    print(f"        Wasserstein alone             rho = {out['rho_W']:+.3f}")
    print(f"    T1  Wasserstein | density         rho = {out['rho_W_given_frac']:+.3f}"
          f"   <- collapses toward 0 if it was density")
    print(f"        (Wasserstein vs density gap   rho = {out['rho_W_frac']:+.3f})", flush=True)
    return out


def main():
    R = json.load(open(OUT)) if os.path.exists(OUT) else {}
    rng = np.random.default_rng(7)
    subs = []
    while len(subs) < 15:
        c = tuple(sorted(rng.choice(FAMS, 4, replace=False)))
        if c not in subs:
            subs.append(c)

    print("=" * 74)
    print("T1/T2 on MetaXFam-D (solid-fraction matched across families)")
    print("=" * 74)
    if "matched" not in R:
        D = {f: load(DATA, f) for f in FAMS}
        R["matched"] = run_pool(D, FAMS, subs, "MATCHED / de-confounded")
        json.dump(R, open(OUT, "w"))

    # ---- T3: deliberately UNMATCHED pool from the same families -----------------
    print("\n" + "=" * 74)
    print("T3  same families, but density deliberately NOT matched")
    print("=" * 74)
    if "unmatched" not in R:
        rng2 = np.random.default_rng(3)
        Du, fams_u = {}, []
        for i, f in enumerate(FAMS):
            X, y, fr = load(POOL, f)
            if len(X) < 150:
                continue
            # push each family toward a different part of the density band
            target = 0.47 + 0.26 * ((i * 7) % len(FAMS)) / (len(FAMS) - 1)
            order = np.argsort(np.abs(fr - target))[:150]
            Du[f] = (X[order], y[order], fr[order])
            fams_u.append(f)
        med = {f: float(np.median(Du[f][2])) for f in fams_u}
        print(f"  {len(fams_u)} families; family median solid fraction now spans "
              f"{min(med.values()):.3f}-{max(med.values()):.3f}")
        print("  (MetaXFam-D spans 0.579-0.695 by design)")
        subs_u = []
        r3 = np.random.default_rng(7)
        while len(subs_u) < 15:
            c = tuple(sorted(r3.choice(fams_u, 4, replace=False)))
            if c not in subs_u:
                subs_u.append(c)
        R["unmatched"] = run_pool(Du, fams_u, subs_u, "UNMATCHED / confounded")
        R["unmatched"]["frac_span"] = [min(med.values()), max(med.values())]
        json.dump(R, open(OUT, "w"))

    print("\n" + "=" * 74)
    print("VERDICT")
    print("=" * 74)
    m, u = R["matched"], R["unmatched"]
    print(f"  Wasserstein rho:  unmatched {u['rho_W']:+.3f}   ->  matched {m['rho_W']:+.3f}")
    print(f"  after removing density:      {u['rho_W_given_frac']:+.3f}   ->  "
          f"{m['rho_W_given_frac']:+.3f}")
    print(f"  density gap alone:           {u['rho_frac']:+.3f}   ->  {m['rho_frac']:+.3f}")
    print("\n  If the criterion is largely a density detector, the unmatched pool shows a")
    print("  strong rho that collapses both when density is partialled out and when the")
    print("  pool is de-confounded. That reconciles the two published conclusions without")
    print("  either being wrong.")
    print("\nDONE")


if __name__ == "__main__":
    main()
