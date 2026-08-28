"""ablate2.py -- the ORIGINAL ablate.py protocol (K=4, 20 subsets rng 7, RF(60), 250
matched cells per family, pooled held-out R^2, median over subsets) with:
  * the cell SELECTION fixed to the one the stored 48x48 descriptors produce
    (so every descriptor set is evaluated on exactly the same 5,500 cells);
  * a swappable descriptor directory (original or rebuilt-extractor);
  * optional pixel arms: pixels, pix+mode2, pix+frac, pix+mode2+frac.
Usage: python ablate2.py <descdir> [--pix] [--tag name]
"""
import _path  # noqa: F401  (adds code/ and code/vendor/ to sys.path)

import json, os, sys, time
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
sys.path.insert(0, "/mnt/user-data/outputs/9_PROJECT_STATE/code")
import datapath
from run_feature2 import FAMS, KEEP, N_PER, rf

DESC48 = "/mnt/user-data/outputs/descriptors_48"
OUT = "/mnt/user-data/outputs/p3/results/ablate2.json"
IDXF = "/mnt/user-data/outputs/p3/results/selection_idx.npz"

def selection():
    """Replicates run_feature2.load()'s matched selection using the stored 48 descriptors."""
    if os.path.exists(IDXF):
        z = np.load(IDXF); return {f: z[f] for f in FAMS}
    rng = np.random.default_rng(0); fr = {}; good = {}
    for f in FAMS:
        d = datapath.load(f"clean48_{f}"); fr[f] = d["frac"]
        om = np.load(f"{DESC48}/omega_{f}.npy"); ch = np.load(f"{DESC48}/char_{f}.npy")
        good[f] = np.isfinite(om).all(1) & np.isfinite(ch).all(axis=(1, 2)) & (om > 1e-9).all(1)
    edges = np.linspace(0.45, 0.75, 11)
    binid = {f: np.clip(np.digitize(v, edges) - 1, 0, 9) for f, v in fr.items()}
    counts = np.array([[np.sum(binid[f] == b) for b in range(10)] for f in FAMS])
    share = counts.min(axis=0).astype(float)
    if share.sum() == 0: share = counts.mean(axis=0).astype(float)
    quota = np.floor(share / share.sum() * N_PER).astype(int)
    out = {}
    for f in FAMS:
        idx = []
        for b in range(10):
            pool = np.where((binid[f] == b) & good[f])[0]
            k = min(int(quota[b]), len(pool))
            if k: idx.extend(rng.choice(pool, k, replace=False))
        idx = np.array(idx, dtype=int)
        if len(idx) < N_PER:
            rest = np.setdiff1d(np.where(good[f])[0], idx)
            if len(rest):
                idx = np.concatenate([idx, rng.choice(rest, min(N_PER - len(idx), len(rest)), replace=False)])
        out[f] = np.sort(idx[:N_PER])
    np.savez(IDXF, **out)
    return out

def load(descdir, sel):
    C = {}
    for f in FAMS:
        d = datapath.load(f"clean48_{f}"); i = sel[f]
        om = np.load(f"{descdir}/omega_{f}.npy")[i]; ch = np.load(f"{descdir}/char_{f}.npy")[i]
        bad = ~(np.isfinite(om).all(1) & np.isfinite(ch).all(axis=(1, 2)))
        if bad.any():   # keep the cell set fixed: impute failures with the family median
            om[bad] = np.nanmedian(om, 0); ch[bad] = np.nanmedian(ch, 0)
        D = np.hstack([np.log(np.clip(om, 1e-6, None)), ch.reshape(len(i), -1)])
        C[f] = dict(X=d["X"][i].reshape(len(i), -1).astype(np.float32), y=d["y"][i][:, KEEP],
                    D=D.astype(np.float64), frac=d["frac"][i][:, None], nbad=int(bad.sum()))
    return C

GROUPS = {"all 24": list(range(24)), "w2 only": [0], "all 6 freqs": list(range(6)),
          "freq ratios only": None, "characters only": list(range(6, 24)), "mode2 char only": [6, 7, 8],
          "mode2 rot+shr": [6, 8]}

def build(C, f, cols):
    D = C[f]["D"]
    if cols is None: return np.column_stack([D[:, j] - D[:, 0] for j in range(1, 6)])
    return D[:, cols]

def main():
    descdir = sys.argv[1]; pix = "--pix" in sys.argv
    tag = sys.argv[sys.argv.index("--tag") + 1] if "--tag" in sys.argv else os.path.basename(descdir.rstrip("/"))
    sel = selection(); C = load(descdir, sel)
    nbad = sum(C[f]["nbad"] for f in FAMS)
    rng = np.random.default_rng(7)
    subs = [tuple(sorted(rng.choice(FAMS, 4, replace=False))) for _ in range(20)]
    res = json.load(open(OUT)) if os.path.exists(OUT) else {}
    res.setdefault(tag, {})["n_imputed"] = nbad
    print(f"== {tag}  (imputed failures: {nbad}/5500)")
    print(f"{'feature set':22s}{'C22 median':>12s}{'C11 median':>12s}{'dim':>6s}")
    arms = [(n, ("desc", c)) for n, c in GROUPS.items()]
    if pix:
        arms += [("pixels", ("pix", [])), ("pix+mode2", ("pix", [6, 7, 8])), ("pix+frac", ("pixf", [])),
                 ("pix+mode2+frac", ("pixf", [6, 7, 8]))]
    for name, (kind, cols) in arms:
        if name in res[tag]: 
            r = res[tag][name]; print(f"{name:22s}{r['C22']:>+12.3f}{r['C11']:>+12.3f}{r['dim']:>6d}  (cached)"); continue
        t0 = time.time(); sc22, sc11 = [], []
        for sub in subs:
            tr = list(sub); te = [f for f in FAMS if f not in sub]
            def feats(f):
                parts = []
                if kind in ("pix", "pixf"): parts.append(C[f]["X"])
                if kind == "pixf": parts.append(C[f]["frac"].astype(np.float32))
                if kind == "desc" or cols: parts.append(build(C, f, cols).astype(np.float32))
                return np.hstack(parts)
            A = np.vstack([feats(f) for f in tr]); B = np.vstack([feats(f) for f in te])
            Ytr = np.vstack([C[f]["y"] for f in tr]); Yte = np.vstack([C[f]["y"] for f in te])
            p = rf(0).fit(A, Ytr).predict(B)
            sc11.append(r2_score(Yte[:, 0], p[:, 0])); sc22.append(r2_score(Yte[:, 2], p[:, 2]))
        dim = A.shape[1]
        res[tag][name] = dict(C22=float(np.median(sc22)), C11=float(np.median(sc11)), dim=int(dim),
                              C22_all=[float(x) for x in sc22], C11_all=[float(x) for x in sc11])
        print(f"{name:22s}{np.median(sc22):>+12.3f}{np.median(sc11):>+12.3f}{dim:>6d}  {time.time()-t0:5.0f}s", flush=True)
        json.dump(res, open(OUT + ".tmp", "w")); os.replace(OUT + ".tmp", OUT)

if __name__ == "__main__":
    main()
