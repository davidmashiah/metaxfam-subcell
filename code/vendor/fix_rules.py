"""
fix_rules.py -- recompute the rule->subset lookup in results_sweep.json.

Bug: rules returned alphabetically sorted tuples, but itertools.combinations
returns tuples in POOL order, so `pick in combos` failed for K>=2 and the table
showed '--'.  The expensive part (every subset's held-out R^2) is already cached
in results_sweep.json; only the rule argmins need recomputing.
"""
import itertools, json
import numpy as np
import selcore as SC
import selrules as SR

F = "/mnt/user-data/outputs/results_sweep.json"
R = json.load(open(F))
POOL, TARGETS, KS = R["pool"], R["targets"], R["ks"]

canon = lambda c: tuple(f for f in POOL if f in set(c))

imgs = {f: SC.images(f) for f in POOL}
ctx = {t: SR.Ctx(POOL, t, imgs) for t in TARGETS}

for t in TARGETS:
    for K in KS:
        recs = R["subsets"][t][str(K)]
        wcs = np.array([r["wc"] for r in recs])
        idx = {canon(r["set"]): i for i, r in enumerate(recs)}
        order = np.argsort(-wcs)
        rank = {i: int(np.where(order == i)[0][0]) + 1 for i in range(len(recs))}
        row = R["table"][f"{t}|{K}"]
        for name, fn in SR.RULES.items():
            pick = canon(fn(ctx[t], K))
            i = idx.get(pick)
            row[name] = ({"set": list(pick), "wc": recs[i]["wc"],
                          "mean": recs[i]["mean"], "rank": rank[i]}
                         if i is not None else
                         {"set": list(pick), "wc": None, "mean": None, "rank": None})
        print(f"  {t} K={K}: " + "  ".join(
            f"{n}={row[n]['rank']}" for n in SR.RULES), flush=True)

json.dump(R, open(F, "w"))
print("\nrewrote", F)
