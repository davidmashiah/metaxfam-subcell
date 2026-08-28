"""
cnn_test.py -- the learner-change test that matters most: a CONVOLUTIONAL surrogate
on raw pixels vs the 4-number physical feature on an MLP.

Paper 1's collapse was measured on pixel surrogates including a CNN.  The Paper 3
claim is that a 4-D physically-motivated feature transfers where pixels do not.  A
random forest on flattened pixels is a weak pixel baseline; a CNN is the fair one.

Arms (all predict C22, all on the same K=4 subsets and the same 250 matched cells
per family used everywhere else in Paper 3):
  cnn_pix        NumPy CNN (mininet, gradient-checked to 2.3e-9) on the 48x48 image
  cnn_pix_m2     same CNN, with (frac, mode2 char) concatenated into the dense head
  mlp_frac       MLP on solid fraction alone                  (free control)
  mlp_frac_m2    MLP on frac + mode-2 character at 24x24      (the proposed feature)
Targets are standardised on the training union (as in cnn_check.py) and inverted
before scoring, so R^2 is on the physical scale.

Cost forces a sample: 12 K=4 subsets (the first 12 of the rng-7 draw used by
robust.py), 1 seed, 200 cells/family, 40 epochs.  Checkpointed per arm.
"""
import _path  # noqa: F401  (adds code/ and code/vendor/ to sys.path)

import json, os, sys, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import ablate2 as A, robust as RB
from run_feature2 import FAMS
from mininet import make_cnn, train

OUT = "/mnt/user-data/outputs/p3/results/cnn_test.json"
N_CELL, EPOCHS, N_SUB = 200, 40, 12
DESC = "v2_24grey"

def r2(y, p):
    y = np.asarray(y).ravel(); p = np.asarray(p).ravel()
    return float(1 - ((y - p) ** 2).sum() / ((y - y.mean()) ** 2).sum())

sel = A.selection()
C = A.load(RB.SETS[DESC], sel)
IMG = {f: C[f]["X"][:N_CELL].reshape(-1, 1, 48, 48).astype(np.float32) for f in FAMS}
EXTRA = {f: np.hstack([C[f]["frac"], C[f]["D"][:, [6, 7, 8]]])[:N_CELL].astype(np.float32) for f in FAMS}
Y = {f: C[f]["y"][:N_CELL, 2] for f in FAMS}
subs = RB.subsets()[:N_SUB]

class Concat:
    """Appends a fixed per-sample vector to the flattened conv features.
    forward: [conv_feats | E].  backward: drop the E columns (E is data, not a parameter)."""
    def __init__(self, dim): self.dim = dim; self.E = None
    def forward(self, X): return np.hstack([X, self.E])
    def backward(self, dout): return dout[:, :-self.dim]

def cnn_arm(train_fams, use_extra, seed=0):
    Xtr = np.concatenate([IMG[f] for f in train_fams])
    ytr = np.concatenate([Y[f] for f in train_fams])[:, None]
    mu, sd = ytr.mean(), ytr.std() + 1e-9
    net = make_cnn(seed=seed, n_out=1, in_size=48)
    if not use_extra:
        train(net, Xtr, (ytr - mu) / sd, epochs=EPOCHS, bs=32, lr=1e-3, seed=seed)
        o = {f: r2(Y[f], net.forward(IMG[f]) * sd + mu) for f in FAMS if f not in train_fams}
        te = [f for f in FAMS if f not in train_fams]
        o["_pooled"] = r2(np.concatenate([Y[f] for f in te]),
                          np.concatenate([np.asarray(net.forward(IMG[f]) * sd + mu).ravel() for f in te]))
        return o
    from mininet import Dense, Flatten, Adam
    Etr = np.concatenate([EXTRA[f] for f in train_fams])
    em, es = Etr.mean(0), Etr.std(0) + 1e-9
    i = [j for j, L in enumerate(net.layers) if isinstance(L, Flatten)][0]
    d = net.layers[i + 1]
    rng = np.random.default_rng(seed)
    d.W = np.vstack([d.W, rng.normal(0, np.sqrt(2.0 / d.W.shape[0]), size=(4, d.W.shape[1]))])
    cat = Concat(4)
    net.layers.insert(i + 1, cat)
    opt = Adam(net, lr=1e-3); rng2 = np.random.default_rng(seed)
    yz = (ytr - mu) / sd; N = len(Xtr)
    for ep in range(EPOCHS):
        idx = rng2.permutation(N)
        for s0 in range(0, N, 32):
            b = idx[s0:s0 + 32]
            cat.E = (Etr[b] - em) / es
            pred = net.forward(Xtr[b])
            net.backward(2.0 * (pred - yz[b]) / len(b))
            opt.step()
    out = {}; ys = []; ps = []
    for f in FAMS:
        if f in train_fams: continue
        cat.E = (EXTRA[f] - em) / es
        p = net.forward(IMG[f]) * sd + mu
        out[f] = r2(Y[f], p); ys.append(Y[f]); ps.append(np.asarray(p).ravel())
    out["_pooled"] = r2(np.concatenate(ys), np.concatenate(ps))
    return out

def mlp_arm(train_fams, cols):
    def X(f): return EXTRA[f][:, cols]
    Xtr = np.vstack([X(f) for f in train_fams]); ytr = np.concatenate([Y[f] for f in train_fams])
    m = make_pipeline(StandardScaler(), MLPRegressor(hidden_layer_sizes=(32, 32), max_iter=2000, random_state=0, alpha=1e-3)).fit(Xtr, ytr)
    te = [f for f in FAMS if f not in train_fams]
    o = {f: r2(Y[f], m.predict(X(f))) for f in te}
    o["_pooled"] = r2(np.concatenate([Y[f] for f in te]), np.concatenate([m.predict(X(f)) for f in te]))
    return o

ARMS = {"mlp_frac": lambda s: mlp_arm(s, [0]),
        "mlp_frac_m2": lambda s: mlp_arm(s, [0, 1, 2, 3]),
        "cnn_pix": lambda s: cnn_arm(s, False),
        "cnn_pix_m2": lambda s: cnn_arm(s, True)}

def main():
    R = json.load(open(OUT)) if os.path.exists(OUT) else {}
    for name, fn in ARMS.items():
        if name in R: 
            a = np.array([x["pooled"] for x in R[name]]); print(f"  {name:14s} cached  pooled-R2 median {np.median(a):+.3f}"); continue
        recs, t0 = [], time.time()
        for i, sub in enumerate(subs):
            r = fn(list(sub))
            pooled = r.pop("_pooled")
            recs.append({"set": list(sub), "pooled": pooled,
                         "mean": float(np.mean(list(r.values()))),
                         "wc": float(min(r.values())), "per": r})
            print(f"    {name} {i+1}/{len(subs)}  pooled {recs[-1]['pooled']:+.3f}  ({(time.time()-t0)/60:.1f} min)", flush=True)
        R[name] = recs
        a = np.array([x["pooled"] for x in recs])
        print(f"  {name:14s} pooled-R2 median {np.median(a):+.3f}  IQR [{np.percentile(a,25):+.3f},{np.percentile(a,75):+.3f}]"
              f"  {(time.time()-t0)/60:.1f} min", flush=True)
        json.dump(R, open(OUT + ".tmp", "w")); os.replace(OUT + ".tmp", OUT)
    print("DONE", flush=True)

if __name__ == "__main__":
    main()
