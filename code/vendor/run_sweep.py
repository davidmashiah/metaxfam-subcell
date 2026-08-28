"""
run_sweep.py -- the full training-family selection study.

For K = 1..5 over a pool of 10 topological families, for targets C11/C22/C33:
  * EXHAUSTIVELY evaluate every K-subset (worst-case and mean held-out R^2,
    averaged over 3 RF seeds)  -> the ORACLE, and the full distribution
  * evaluate every cheap SELECTION RULE's pick against that oracle
  * measure the Spearman rank correlation between each rule's cheap cost and
    the achieved worst-case R^2 across ALL subsets -- i.e. does the criterion
    actually track transferability, not just get lucky at the argmin?

Results are written to results_sweep.json (+ a printed table).
"""
import itertools, json, os, sys, time
import numpy as np
from scipy.stats import spearmanr

import selcore as SC
import selrules as SR

POOL    = SC.FAMS10
TARGETS = ["C11", "C22", "C33"]
KS      = [1, 2, 3, 4, 5]
SEEDS   = (0, 1, 2)
OUT     = "/mnt/user-data/outputs/results_sweep.json"


def main():
    t_start = time.time()
    print("Building cache ...", flush=True)
    SC.build_cache(verbose=True)

    imgs = {f: SC.images(f) for f in POOL}
    ctx = {t: SR.Ctx(POOL, t, imgs) for t in TARGETS}

    print("\nPilot-FE family statistics (20 cells/family):")
    c0 = ctx["C11"]
    for f in POOL:
        print(f"  {f:14s} aniso={c0.aniso[f]:.4f}  "
              + "  ".join(f"{t}: mu={ctx[t].rmu[f]:.4f} sd={ctx[t].rsd[f]:.4f}"
                          for t in TARGETS))

    if os.path.exists(OUT):
        results = json.load(open(OUT))
        print(f"resuming from {OUT}: {len(results['table'])} jobs already done",
              flush=True)
    else:
        results = {"pool": POOL, "targets": TARGETS, "ks": KS,
                   "aniso": c0.aniso, "subsets": {}, "table": {}, "corr": {}}

    for target in TARGETS:
        results["subsets"].setdefault(target, {})
        for K in KS:
            if f"{target}|{K}" in results["table"]:
                print(f"  [{target} K={K}] cached", flush=True)
                continue
            combos = list(itertools.combinations(POOL, K))
            recs = []
            t0 = time.time()
            for c in combos:
                wc, mn = SC.wc_mean(c, target, SEEDS)
                recs.append({"set": list(c), "wc": wc, "mean": mn})
            results["subsets"][target][K] = recs
            wcs = np.array([r["wc"] for r in recs])
            print(f"  [{target} K={K}] {len(combos):4d} subsets in "
                  f"{(time.time()-t0)/60:.1f} min | "
                  f"oracle wc={wcs.max():+.3f} median={np.median(wcs):+.3f} "
                  f"worst={wcs.min():+.3f}", flush=True)

            # ---- rules
            row = {}
            order = np.argsort(-wcs)
            rank_of = {tuple(recs[i]["set"]): int(np.where(order == i)[0][0]) + 1
                       for i in range(len(recs))}
            row["oracle"] = {"set": recs[int(np.argmax(wcs))]["set"],
                             "wc": float(wcs.max()),
                             "mean": recs[int(np.argmax(wcs))]["mean"], "rank": 1}
            row["random_mean"] = {"set": None, "wc": float(wcs.mean()),
                                  "mean": float(np.mean([r["mean"] for r in recs])),
                                  "rank": None}
            row["worst"] = {"set": recs[int(np.argmin(wcs))]["set"],
                            "wc": float(wcs.min()), "mean": recs[int(np.argmin(wcs))]["mean"],
                            "rank": len(recs)}
            for name, fn in SR.RULES.items():
                pick = tuple(sorted(fn(ctx[target], K)))
                j = combos.index(pick) if pick in combos else None
                if j is None:
                    row[name] = {"set": list(pick), "wc": None, "mean": None, "rank": None}
                else:
                    row[name] = {"set": list(pick), "wc": recs[j]["wc"],
                                 "mean": recs[j]["mean"],
                                 "rank": rank_of[tuple(recs[j]["set"])]}
            results["table"][f"{target}|{K}"] = row

            # ---- rank correlation of each cheap cost with achieved wc R^2
            if 1 <= K <= 4 and len(combos) > 5:
                costs = {}
                for name in ["kcenter_geo", "mmd", "wass", "aniso_cover",
                             "resp_cover", "resp_span"]:
                    cf = _cost_fn(name, ctx[target])
                    costs[name] = [cf(c) for c in combos]
                # scalar summaries that are not k-center costs
                costs["aniso_sum"] = [-sum(ctx[target].aniso[f] for f in c) for c in combos]
                costs["aniso_max"] = [-max(ctx[target].aniso[f] for f in c) for c in combos]
                costs["aniso_range"] = [-(max(ctx[target].aniso[f] for f in c)
                                          - min(ctx[target].aniso[f] for f in c))
                                        for c in combos]
                costs["resp_sd_sum"] = [-sum(ctx[target].rsd[f] for f in c) for c in combos]
                cc = {}
                for name, v in costs.items():
                    rho, p = spearmanr(v, -wcs)   # cost up  <->  wc R2 down
                    cc[name] = {"rho": float(rho), "p": float(p)}
                results["corr"][f"{target}|{K}"] = cc

            os.makedirs(os.path.dirname(OUT), exist_ok=True)
            with open(OUT + ".tmp", "w") as fh:
                json.dump(results, fh)
            os.replace(OUT + ".tmp", OUT)
            print(f"      checkpointed ({len(results['table'])} jobs)", flush=True)

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as fh:
        json.dump(results, fh)
    print(f"\nwrote {OUT}   total {(time.time()-t_start)/60:.1f} min")


def _cost_fn(name, ctx):
    P = ctx.pool
    if name == "kcenter_geo":
        return lambda c: max(min(ctx.gdist(u, x) for x in c) for u in P if u not in c)
    if name == "mmd":
        return lambda c: max(ctx.mmd2(u, c) for u in P if u not in c)
    if name == "wass":
        return lambda c: max(ctx.sw(u, c) for u in P if u not in c)
    if name == "aniso_cover":
        return lambda c: max(min(abs(ctx.aniso[u] - ctx.aniso[x]) for x in c)
                             for u in P if u not in c)
    if name == "resp_cover":
        V = np.array([[ctx.rmu[f], ctx.rsd[f]] for f in P])
        mu, sd = V.mean(0), V.std(0) + 1e-12
        v = {f: (np.array([ctx.rmu[f], ctx.rsd[f]]) - mu) / sd for f in P}
        return lambda c: max(min(float(np.linalg.norm(v[u] - v[x])) for x in c)
                             for u in P if u not in c)
    if name == "resp_span":
        def f_(c):
            lo = min(ctx.rmu[x] - ctx.rsd[x] for x in c)
            hi = max(ctx.rmu[x] + ctx.rsd[x] for x in c)
            return max(max(lo - ctx.rmu[u], ctx.rmu[u] - hi, 0.) for u in P if u not in c)
        return f_
    raise KeyError(name)


if __name__ == "__main__":
    main()
