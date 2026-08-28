"""
run_big.py -- the definitive study on the expanded 22-family pool, RAW PIXELS.

Two changes from run_sweep.py, both forced by what the 10-family study showed:

  1. RAW 48x48 PIXELS, not the 48-component PCA basis.  The criterion-vs-transfer
     correlations turned out to be partly basis-dependent (physics criteria fell
     from ~+0.45 to ~+0.28 and MMD rose from ~+0.06 to ~+0.20 when the PCA was
     removed), so the PCA pipeline cannot be the one the conclusions rest on.

  2. 22 families instead of 10.  With 10 families the correlations were estimated
     from 45 (K=2) or 120 (K=3) subsets and no criterion was significant on more
     than 4 of 6 target/K jobs -- the effect sat near the noise floor.  22
     families give 231 subsets at K=2 and a densely sampled anisotropy axis
     (rect_lattice, layered, cross_aniso are continuously tunable; chevron and
     reentrant supply the previously-missing C22 > C11 direction).

For K >= 3 the subset space is too large to enumerate, so subsets are SAMPLED
uniformly at random.  For estimating a rank correlation this is not a
compromise -- a random sample of subsets is an unbiased sample of the
population the correlation is defined over.

Checkpointed per (target, K).  Safe to kill and resume.
"""
import itertools, json, os, time
import numpy as np
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

import selcore as SC
import selrules as SR
import run_sweep as RS

TARGETS  = ["C11", "C22", "C33"]
JOBS     = [(t, K) for K in (2, 4) for t in TARGETS]
SEEDS    = (0, 1)
N_SAMPLE = 200
OUT      = "/mnt/user-data/outputs/results_big.json"

POOL = SC.FAMS10
C, S = SC._ensure()
XR = {f: C["Xraw"][S[f]].astype(np.float32) for f in POOL}


def evaluate_raw(train, target, seed):
    Xtr = np.vstack([XR[f] for f in train])
    ytr = np.concatenate([SC.targ(f, target) for f in train])
    m = RandomForestRegressor(n_estimators=50, random_state=seed, n_jobs=1,
                              max_features=0.3, min_samples_leaf=2).fit(Xtr, ytr)
    return {f: float(r2_score(SC.targ(f, target), m.predict(XR[f])))
            for f in POOL if f not in train}


def subsets_for(K, rng):
    all_n = len(list(itertools.combinations(range(len(POOL)), K))) if K <= 2 else None
    if K <= 2:
        return list(itertools.combinations(POOL, K))
    seen, out = set(), []
    while len(out) < N_SAMPLE:
        c = tuple(sorted(rng.choice(len(POOL), K, replace=False)))
        if c in seen:
            continue
        seen.add(c)
        out.append(tuple(POOL[i] for i in c))
    return out


def main():
    res = json.load(open(OUT)) if os.path.exists(OUT) else {"pool": POOL, "subsets": {}, "corr": {}}
    imgs = {f: SC.images(f) for f in POOL}
    ctx = {t: SR.Ctx(POOL, t, imgs) for t in TARGETS}
    res["aniso"] = ctx["C11"].aniso

    for target, K in JOBS:
        key = f"{target}|{K}"
        if key not in res["subsets"]:
            rng = np.random.default_rng(100 + K)
            combos = subsets_for(K, rng)
            recs, t0 = [], time.time()
            for c in combos:
                wcs, mns = [], []
                for s in SEEDS:
                    r = evaluate_raw(c, target, s)
                    wcs.append(min(r.values())); mns.append(float(np.mean(list(r.values()))))
                recs.append({"set": list(c), "wc": float(np.mean(wcs)),
                             "mean": float(np.mean(mns))})
            res["subsets"][key] = recs
            a = np.array([r["wc"] for r in recs])
            print(f"  [{target} K={K}] {len(combos)} subsets in {(time.time()-t0)/60:.1f} min"
                  f" | oracle {a.max():+.2f} median {np.median(a):+.2f} worst {a.min():+.2f}",
                  flush=True)
            json.dump(res, open(OUT + ".tmp", "w")); os.replace(OUT + ".tmp", OUT)

        if key not in res["corr"]:
            recs = res["subsets"][key]
            a = np.array([r["wc"] for r in recs])
            cc, t0 = {}, time.time()
            for name in ["kcenter_geo", "mmd", "wass", "aniso_cover", "resp_cover", "resp_span"]:
                cf = RS._cost_fn(name, ctx[target])
                rr, pp = spearmanr([cf(tuple(r["set"])) for r in recs], -a)
                cc[name] = {"rho": float(rr), "p": float(pp)}
            for nm, fn in [("aniso_max", lambda c: -max(ctx[target].aniso[f] for f in c)),
                           ("aniso_sum", lambda c: -sum(ctx[target].aniso[f] for f in c)),
                           ("aniso_range", lambda c: -(max(ctx[target].aniso[f] for f in c)
                                                       - min(ctx[target].aniso[f] for f in c)))]:
                rr, pp = spearmanr([fn(tuple(r["set"])) for r in recs], -a)
                cc[nm] = {"rho": float(rr), "p": float(pp)}
            res["corr"][key] = cc
            print(f"      correlations in {(time.time()-t0)/60:.1f} min: "
                  + "  ".join(f"{n}={v['rho']:+.2f}" for n, v in cc.items()), flush=True)
            json.dump(res, open(OUT + ".tmp", "w")); os.replace(OUT + ".tmp", OUT)

    print("\nDONE")


if __name__ == "__main__":
    main()
