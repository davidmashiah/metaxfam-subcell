"""
benchmark_v2.py -- assemble MetaXFam-D (distinct) and re-measure the headline result.

Regeneration with a distinct-rasterisation filter is complete for all 22 families.  The
achievable distinct count per family is now known and is the honest size of the benchmark:
families that "had 600 cells" often had a few dozen distinct geometries padded with copies.

This script:
  1. Reports the final per-family distinct counts and the total.
  2. Selects families that clear a usability threshold, and sets a per-family cap so every
     family contributes equally (no family dominates the pooled arms).
  3. Verifies ZERO duplicates within and across the assembled set -- the property the
     original dataset lacked.
  4. Re-runs the three-protocol comparison (random-within / random-pooled / family-disjoint)
     at matched training-set size, on four stiffness components and two learners.
  5. Writes the assembled benchmark to /mnt/user-data/outputs/metaxfam_d/ so it is a
     citable artefact independent of the container.

The comparison against the duplicated dataset is the paper's methods contribution; the
family-disjoint gap measured here is the paper's headline claim, now leak-free.
"""
import json, os, shutil, sys
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import r2_score

SRC = "/mnt/user-data/outputs/data_distinct"
DST = "/mnt/user-data/outputs/metaxfam_d"
OUT = "/mnt/user-data/outputs/p4/results/benchmark_v2.json"
os.makedirs(DST, exist_ok=True); os.makedirs(os.path.dirname(OUT), exist_ok=True)

MIN_CELLS = 150                     # a family must supply at least this many distinct cells
# families superseded by a re-parameterised version: keep only the rich one, to avoid
# two near-identical families (the original is a strict subset of the rich family's span)
SUPERSEDED = {"circle", "square", "cross", "diamond_hole", "square_lattice", "diag_lattice"}
COL = {"C11": 0, "C12": 1, "C22": 3, "C33": 5}      # raw y columns

def load(f):
    X = np.load(f"{SRC}/clean48_{f}__X.npy")
    y = np.load(f"{SRC}/clean48_{f}__y.npy")
    fr = np.load(f"{SRC}/clean48_{f}__frac.npy")
    return X, y, fr

def main():
    fams_all = sorted({os.path.basename(p).replace("clean48_", "").replace("__X.npy", "")
                       for p in os.listdir(SRC) if p.endswith("__X.npy")})
    print("=" * 74)
    print("MetaXFam-D : distinct-rasterisation benchmark")
    print("=" * 74)
    counts = {}
    for f in fams_all:
        counts[f] = len(load(f)[0])
    keep = [f for f in fams_all if counts[f] >= MIN_CELLS and f not in SUPERSEDED]
    drop = [f for f in fams_all if f not in keep]
    for f in sorted(fams_all, key=lambda z: -counts[z]):
        mark = "keep" if f in keep else "DROP"
        reason = ("superseded" if f in SUPERSEDED else
                  "" if counts[f] >= MIN_CELLS else "too few distinct")
        print(f"  {f:20s}{counts[f]:5d}   {mark:4s} {reason}")
    cap = min(counts[f] for f in keep)
    print(f"\n  {len(keep)} families kept, {len(drop)} dropped (< {MIN_CELLS} distinct)")
    print(f"  dropped: {drop}")
    print(f"  per-family cap = {cap}  ->  benchmark size {len(keep) * cap} cells")

    # ---- assemble, cap, and verify distinctness -------------------------------------
    rng = np.random.default_rng(0)
    D = {}
    for f in keep:
        X, y, fr = load(f)
        i = np.sort(rng.permutation(len(X))[:cap])
        D[f] = (X[i], y[i], fr[i])
        np.save(f"{DST}/clean48_{f}__X.npy", X[i])
        np.save(f"{DST}/clean48_{f}__y.npy", y[i])
        np.save(f"{DST}/clean48_{f}__frac.npy", fr[i])
    Xall = np.vstack([D[f][0] for f in keep]).reshape(len(keep) * cap, -1)
    n_distinct = len(np.unique(Xall, axis=0))
    print(f"\n  VERIFY: {len(Xall)} cells -> {n_distinct} distinct "
          f"({'PASS: zero duplicates' if n_distinct == len(Xall) else 'FAIL'})")

    # solid-fraction balance across families (the original de-confounding intent)
    fr_med = {f: float(np.median(D[f][2])) for f in keep}
    print(f"  solid fraction, family medians: {min(fr_med.values()):.3f}-{max(fr_med.values()):.3f}")

    R = {"counts": counts, "keep": keep, "drop": drop, "cap": int(cap),
         "n_distinct": int(n_distinct), "frac_median": fr_med}

    # ---- headline experiment ---------------------------------------------------------
    LEARNERS = {
        "ridge": lambda: make_pipeline(StandardScaler(with_mean=False),
                                       RidgeCV(alphas=np.logspace(-2, 4, 13))),
        "RF":    lambda: RandomForestRegressor(n_estimators=60, random_state=0,
                                               max_features=0.3, min_samples_leaf=2, n_jobs=1),
    }
    Xf = {f: D[f][0].reshape(cap, -1).astype(np.float32) for f in keep}
    Yf = {f: D[f][1] for f in keep}

    def stack(fs, col, idx=None):
        xs, ys = [], []
        for f in fs:
            i = slice(None) if idx is None else idx[f]
            xs.append(Xf[f][i]); ys.append(Yf[f][i][:, col])
        return np.vstack(xs), np.concatenate(ys)

    K = 4
    r2 = np.random.default_rng(7)
    subs = []
    while len(subs) < 20:
        c = tuple(sorted(r2.choice(keep, K, replace=False)))
        if c not in subs:
            subs.append(c)

    print("\n" + "=" * 74)
    print("HEADLINE: what a random split hides (median over 20 subsets, K=4)")
    print("=" * 74)
    print(f"{'target':8s}{'learner':8s}{'within':>10s}{'pooled':>10s}{'disjoint':>11s}{'gap':>9s}")
    for tname, col in COL.items():
        for ln, mk in LEARNERS.items():
            w, p, d = [], [], []
            for sub in subs:
                tr_f = list(sub); te_f = [f for f in keep if f not in sub]
                n_train = K * cap
                Xtr, ytr = stack(tr_f, col); Xte, yte = stack(te_f, col)
                d.append(r2_score(yte, mk().fit(Xtr, ytr).predict(Xte)))
                per = int(np.ceil(n_train / len(keep)))
                itr, ite = {}, {}
                for f in keep:
                    pm = rng.permutation(cap); itr[f] = pm[:per]; ite[f] = pm[per:]
                Xp, yp = stack(keep, col, itr); Xp_te, yp_te = stack(keep, col, ite)
                k = rng.permutation(len(yp))[:n_train]
                p.append(r2_score(yp_te, mk().fit(Xp[k], yp[k]).predict(Xp_te)))
                iw, iwt = {}, {}
                for f in tr_f:
                    pm = rng.permutation(cap); c0 = int(.75 * cap)
                    iw[f] = pm[:c0]; iwt[f] = pm[c0:]
                Xw, yw = stack(tr_f, col, iw); Xw_te, yw_te = stack(tr_f, col, iwt)
                w.append(r2_score(yw_te, mk().fit(Xw, yw).predict(Xw_te)))
            s = dict(within=float(np.median(w)), pooled=float(np.median(p)),
                     disjoint=float(np.median(d)))
            R[f"{tname}|{ln}"] = s
            print(f"{tname:8s}{ln:8s}{s['within']:+10.3f}{s['pooled']:+10.3f}"
                  f"{s['disjoint']:+11.3f}{s['within']-s['disjoint']:+9.3f}", flush=True)
            json.dump(R, open(OUT + ".tmp", "w")); os.replace(OUT + ".tmp", OUT)
    json.dump(R, open(OUT, "w"))
    print(f"\nbenchmark written to {DST}")
    print("DONE")

if __name__ == "__main__":
    main()
