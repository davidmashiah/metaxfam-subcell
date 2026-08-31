"""
dup_audit.py -- how badly is MetaXFam22 duplicated, and why?

The matched 5,500-cell evaluation subset turned out to contain only 2,574 distinct
geometries (64.7% of cells belong to some duplicate group).  Before any result computed on
that subset can be trusted, three things must be established:

 1. HOW MUCH.  Duplication over the FULL 13,200-cell dataset, per family, not just the
    matched subset.
 2. WHY.  The hypothesis is rasterisation collapse: family generators sample continuous
    shape parameters, but at 48x48 many distinct parameter values rasterise to the same
    pixel grid.  If so, duplication should fall sharply at higher resolution and rise at
    lower resolution.  That is a testable prediction, and it also tells us whether the
    defect is in the sampler or is intrinsic to pixelisation.
 3. WHAT IT COSTS.  How much of the reported in-distribution accuracy is memorisation of
    repeated cells: measure the fraction of random-split test cells whose exact geometry
    also appears in the training half.

Nothing here changes any conclusion by itself; it sizes the problem so the rebuild is
principled rather than a guess.
"""
import collections, hashlib, json, os, sys
import numpy as np

sys.path.insert(0, "/mnt/user-data/outputs/p3/code")
os.environ.setdefault("SELPREFIX", "clean48_")
import datapath
from run_feature2 import FAMS

DATA = "/mnt/user-data/outputs/9_PROJECT_STATE/data"
OUT = "/mnt/user-data/outputs/p4/results/dup_audit.json"
os.makedirs(os.path.dirname(OUT), exist_ok=True)

def h(a):
    return hashlib.md5(np.ascontiguousarray(a.astype(np.uint8)).tobytes()).hexdigest()

def coarsen(a, n_out):
    n = a.shape[0]
    if n == n_out:
        return a
    b = n // n_out
    return (a.reshape(n_out, b, n_out, b).mean(axis=(1, 3)) >= 0.5).astype(np.uint8)

def load_family(f):
    """Load direct from the .npy layout in 9_PROJECT_STATE/data (datapath expects .npz)."""
    X = np.load(f"{DATA}/clean48_{f}__X.npy").astype(np.uint8)
    y = np.load(f"{DATA}/clean48_{f}__y.npy")
    fr = np.load(f"{DATA}/clean48_{f}__frac.npy")
    return X, y, fr

def main():
    R = {}
    print("=" * 76)
    print("PER-FAMILY DUPLICATION, FULL DATASET (600 cells per family, 48x48)")
    print("=" * 76)
    print(f"{'family':16s}{'cells':>7s}{'distinct':>10s}{'dup rate':>10s}{'largest group':>15s}")
    allh, allfam = [], []
    per_family = {}
    for i, f in enumerate(FAMS):
        Xf, yf, ff = load_family(f)
        hs = [h(x) for x in Xf]
        c = collections.Counter(hs)
        dup_rate = 1.0 - len(c) / len(hs)
        per_family[f] = dict(n=len(hs), distinct=len(c), dup_rate=float(dup_rate),
                             largest=int(max(c.values())))
        print(f"{f:16s}{len(hs):7d}{len(c):10d}{dup_rate:9.1%}{max(c.values()):15d}")
        allh += hs; allfam += [i] * len(hs)
    R["per_family_48"] = per_family

    c = collections.Counter(allh)
    idx = collections.defaultdict(list)
    for k, hh in enumerate(allh):
        idx[hh].append(k)
    cross = sum(1 for hh, v in c.items() if v > 1 and len({allfam[j] for j in v and idx[hh]}) > 1)
    print(f"\nPOOLED: {len(allh)} cells, {len(c)} distinct, "
          f"overall duplicate rate {1 - len(c)/len(allh):.1%}")
    print(f"        duplicate groups spanning >1 family: {cross}")
    R["pooled_48"] = dict(n=len(allh), distinct=len(c),
                          dup_rate=float(1 - len(c) / len(allh)), cross_family=int(cross))

    # ---------------- why: resolution dependence ----------------
    print("\n" + "=" * 76)
    print("RESOLUTION TEST: is this rasterisation collapse?")
    print("=" * 76)
    sub = FAMS[:8]
    print(f"{'resolution':>12s}{'cells':>9s}{'distinct':>10s}{'dup rate':>10s}")
    res_rows = {}
    for n_out in (12, 16, 24, 48):
        hs = []
        for f in sub:
            Xf, _, _ = load_family(f)
            hs += [h(coarsen(x, n_out)) for x in Xf]
        cc = collections.Counter(hs)
        rate = 1 - len(cc) / len(hs)
        res_rows[n_out] = float(rate)
        print(f"{n_out:>10d}x{n_out:<2d}{len(hs):9d}{len(cc):10d}{rate:9.1%}")
    R["resolution"] = res_rows
    print("\n  Rising duplication as resolution falls => rasterisation collapse:")
    print("  distinct shape parameters landing on the same pixel grid.")

    # ---------------- what it costs: leakage under a random split ----------------
    print("\n" + "=" * 76)
    print("COST: exact-geometry leakage across a random 75/25 split")
    print("=" * 76)
    rng = np.random.default_rng(0)
    hs = np.array(allh)
    perm = rng.permutation(len(hs))
    cut = int(0.75 * len(hs))
    tr, te = set(hs[perm[:cut]]), hs[perm[cut:]]
    leaked = np.mean([x in tr for x in te])
    print(f"  {leaked:.1%} of random-split TEST cells have their exact geometry in TRAIN")
    print("  (this is the mechanism that inflates in-distribution R^2)")
    R["random_split_leakage"] = float(leaked)

    json.dump(R, open(OUT, "w"), indent=1)
    print(f"\nwritten to {OUT}")

if __name__ == "__main__":
    main()
