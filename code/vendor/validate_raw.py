"""
validate_raw.py -- is the 48-component PCA basis load-bearing?

The whole selection study runs on RF surrogates fitted in a shared 48-dim PCA
basis.  If the conclusions are an artefact of that compression, they are worth
nothing.  This script re-runs every K=2 and K=3 subset on RAW 48x48 pixels
(2304 features, no PCA) for all three targets and asks:

  (1) does the RANKING of subsets by worst-case held-out R^2 agree between the
      PCA pipeline and the raw-pixel pipeline?  (Spearman)
  (2) do the criterion-vs-transfer correlations -- the headline result --
      survive on raw pixels?

Checkpointed: safe to kill and resume.
"""
import itertools, json, os, time
import numpy as np
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

import selcore as SC
import selrules as SR
import run_sweep as RS

POOL    = SC.FAMS10
TARGETS = ["C11", "C22", "C33"]
KS      = [2, 3]
SEEDS   = (0, 1, 2)
OUT     = "/mnt/user-data/outputs/results_raw.json"

C, S = SC._ensure()
XR = {f: C["Xraw"][S[f]].astype(np.float32) for f in POOL}


def evaluate_raw(train, target, seed):
    Xtr = np.vstack([XR[f] for f in train])
    ytr = np.concatenate([SC.targ(f, target) for f in train])
    m = RandomForestRegressor(n_estimators=60, random_state=seed, n_jobs=1,
                              max_features=0.3, min_samples_leaf=2).fit(Xtr, ytr)
    return {f: float(r2_score(SC.targ(f, target), m.predict(XR[f])))
            for f in POOL if f not in train}


def main():
    res = json.load(open(OUT)) if os.path.exists(OUT) else {"subsets": {}, "corr": {}}
    imgs = {f: SC.images(f) for f in POOL}
    ctx = {t: SR.Ctx(POOL, t, imgs) for t in TARGETS}
    PCA_R = json.load(open("/mnt/user-data/outputs/results_sweep.json"))

    for t in TARGETS:
        res["subsets"].setdefault(t, {})
        for K in KS:
            key = f"{t}|{K}"
            if str(K) not in res["subsets"][t]:
                combos = list(itertools.combinations(POOL, K))
                recs, t0 = [], time.time()
                for c in combos:
                    wcs, mns = [], []
                    for s in SEEDS:
                        r = evaluate_raw(c, t, s)
                        wcs.append(min(r.values())); mns.append(float(np.mean(list(r.values()))))
                    recs.append({"set": list(c), "wc": float(np.mean(wcs)),
                                 "mean": float(np.mean(mns))})
                res["subsets"][t][str(K)] = recs
                print(f"  [{t} K={K}] {len(combos)} subsets raw-pixel in "
                      f"{(time.time()-t0)/60:.1f} min", flush=True)
                json.dump(res, open(OUT + ".tmp", "w")); os.replace(OUT + ".tmp", OUT)

            # ---- agreement with the PCA pipeline
            raw = res["subsets"][t][str(K)]
            pca = PCA_R["subsets"][t][str(K)]
            pmap = {tuple(r["set"]): r["wc"] for r in pca}
            a = np.array([r["wc"] for r in raw])
            b = np.array([pmap[tuple(r["set"])] for r in raw])
            rho, p = spearmanr(a, b)
            # ---- criterion correlations on RAW targets
            cc = {}
            for name in ["kcenter_geo", "mmd", "wass", "aniso_cover", "resp_cover", "resp_span"]:
                cf = RS._cost_fn(name, ctx[t])
                v = [cf(tuple(r["set"])) for r in raw]
                rr, pp = spearmanr(v, -a)
                cc[name] = {"rho": float(rr), "p": float(pp)}
            for nm, fn in [("aniso_max", lambda c: -max(ctx[t].aniso[f] for f in c)),
                           ("aniso_sum", lambda c: -sum(ctx[t].aniso[f] for f in c))]:
                rr, pp = spearmanr([fn(tuple(r["set"])) for r in raw], -a)
                cc[nm] = {"rho": float(rr), "p": float(pp)}
            res["corr"][key] = {"pca_vs_raw_rho": float(rho), "pca_vs_raw_p": float(p),
                                "criteria": cc,
                                "raw_oracle": float(a.max()), "raw_median": float(np.median(a)),
                                "raw_worst": float(a.min()),
                                "pca_oracle": float(b.max())}
            json.dump(res, open(OUT + ".tmp", "w")); os.replace(OUT + ".tmp", OUT)
            print(f"      PCA-vs-raw subset-ranking Spearman = {rho:+.3f} (p={p:.1e})", flush=True)

    print("\n" + "=" * 78)
    print("PCA vs RAW-PIXEL agreement and criterion correlations")
    print("=" * 78)
    print(f"{'job':<10}{'rank agree':>11}{'raw oracle':>12}{'pca oracle':>12}{'raw median':>12}")
    for k, v in res["corr"].items():
        print(f"{k:<10}{v['pca_vs_raw_rho']:>+11.3f}{v['raw_oracle']:>+12.2f}"
              f"{v['pca_oracle']:>+12.2f}{v['raw_median']:>+12.2f}")
    names = sorted({n for v in res["corr"].values() for n in v["criteria"]})
    print(f"\n{'criterion':<14}" + "".join(f"{k:>11}" for k in res["corr"]))
    for n in names:
        cells = []
        for k, v in res["corr"].items():
            c = v["criteria"][n]
            cells.append(f"{c['rho']:>+10.2f}" + ("*" if c["p"] < 0.05 else " "))
        print(f"{n:<14}" + "".join(cells))


if __name__ == "__main__":
    main()
