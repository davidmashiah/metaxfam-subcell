"""
cnn_check.py -- is the criterion result random-forest-specific?

The one reproducible positive signal on the 22-family pool is that input-
distribution distance (MMD) predicts worst-case cross-family transfer for C22
(rho = +0.54 at both K=2 and K=4) but is useless-to-harmful for C11
(rho = -0.12, -0.10), while response-coverage does the reverse.  Before that can
go in a paper it has to survive a surrogate that is not a random forest.

This replicates it with the from-scratch NumPy CNN (mininet.py, backprop
gradient-checked to 2.3e-9), on the SAME subsets used in the RF run, so we also
get the RF-vs-CNN agreement on which subsets transfer well.

Cost forces a sample: N_SUB random K=2 subsets, one seed, 200 cells/family,
40 epochs.  Checkpointed per target.
"""
import json, os, time
import numpy as np
from scipy.stats import spearmanr

import selcore as SC
import selrules as SR
import run_sweep as RS
from mininet import make_cnn, train

TARGETS = ["C11", "C22"]
K       = 2
N_SUB   = 45
N_CELL  = 200
EPOCHS  = 40
OUT     = "/mnt/user-data/outputs/results_cnn.json"

POOL = SC.FAMS10
IMG  = {f: SC.images(f)[:N_CELL].astype(np.float32)[:, None] for f in POOL}


def r2(y, p):
    y = np.asarray(y).ravel(); p = np.asarray(p).ravel()
    return float(1 - ((y - p) ** 2).sum() / ((y - y.mean()) ** 2).sum())


def evaluate_cnn(train_fams, target, seed=0):
    Xtr = np.concatenate([IMG[f] for f in train_fams])
    ytr = np.concatenate([SC.targ(f, target)[:N_CELL] for f in train_fams])[:, None]
    mu, sd = ytr.mean(), ytr.std() + 1e-9
    net = make_cnn(seed=seed, n_out=1, in_size=48)
    train(net, Xtr, (ytr - mu) / sd, epochs=EPOCHS, bs=32, lr=1e-3, seed=seed)
    out = {}
    for f in POOL:
        if f in train_fams:
            continue
        p = net.forward(IMG[f]) * sd + mu
        out[f] = r2(SC.targ(f, target)[:N_CELL], p)
    return out


def main():
    res = json.load(open(OUT)) if os.path.exists(OUT) else {"subsets": {}, "corr": {}}
    BIG = json.load(open("/mnt/user-data/outputs/results_big.json"))
    imgs = {f: SC.images(f) for f in POOL}
    ctx = {t: SR.Ctx(POOL, t, imgs) for t in TARGETS}

    rng = np.random.default_rng(7)
    all_sets = [tuple(r["set"]) for r in BIG["subsets"]["C11|2"]]
    pick = [all_sets[i] for i in rng.choice(len(all_sets), N_SUB, replace=False)]

    for t in TARGETS:
        if t not in res["subsets"]:
            recs, t0 = [], time.time()
            for i, c in enumerate(pick):
                r = evaluate_cnn(c, t)
                recs.append({"set": list(c), "wc": float(min(r.values())),
                             "mean": float(np.mean(list(r.values())))})
                if i % 10 == 9:
                    print(f"    {t}: {i+1}/{N_SUB} subsets, "
                          f"{(time.time()-t0)/60:.1f} min", flush=True)
            res["subsets"][t] = recs
            json.dump(res, open(OUT + ".tmp", "w")); os.replace(OUT + ".tmp", OUT)
            a = np.array([r["wc"] for r in recs])
            print(f"  [CNN {t} K=2] {N_SUB} subsets in {(time.time()-t0)/60:.1f} min | "
                  f"oracle {a.max():+.2f} median {np.median(a):+.2f} worst {a.min():+.2f}",
                  flush=True)

        if t not in res["corr"]:
            recs = res["subsets"][t]
            a = np.array([r["wc"] for r in recs])
            # agreement with the random forest on the same subsets
            rf = {tuple(r["set"]): r["wc"] for r in BIG["subsets"][f"{t}|2"]}
            b = np.array([rf[tuple(r["set"])] for r in recs])
            rho_rf, p_rf = spearmanr(a, b)
            cc = {}
            for name in ["mmd", "wass", "kcenter_geo", "resp_cover", "resp_span", "aniso_cover"]:
                cf = RS._cost_fn(name, ctx[t])
                rr, pp = spearmanr([cf(tuple(r["set"])) for r in recs], -a)
                cc[name] = {"rho": float(rr), "p": float(pp)}
            for nm, fn in [("aniso_sum", lambda c: -sum(ctx[t].aniso[f] for f in c)),
                           ("aniso_max", lambda c: -max(ctx[t].aniso[f] for f in c))]:
                rr, pp = spearmanr([fn(tuple(r["set"])) for r in recs], -a)
                cc[nm] = {"rho": float(rr), "p": float(pp)}
            res["corr"][t] = {"rf_vs_cnn_rho": float(rho_rf), "rf_vs_cnn_p": float(p_rf),
                              "criteria": cc}
            json.dump(res, open(OUT + ".tmp", "w")); os.replace(OUT + ".tmp", OUT)
            print(f"      RF-vs-CNN subset agreement rho = {rho_rf:+.3f} (p={p_rf:.1e})")
            print("      " + "  ".join(f"{n}={v['rho']:+.2f}"
                                       + ("*" if v["p"] < 0.05 else "")
                                       for n, v in cc.items()), flush=True)
    print("\nDONE")


if __name__ == "__main__":
    main()
