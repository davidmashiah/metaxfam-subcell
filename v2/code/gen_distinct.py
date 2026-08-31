"""
gen_distinct.py -- regenerate MetaXFam22 with DISTINCT rasterisations only.

WHY.  An audit of the original dataset found it is 61.6% duplicated at 48x48
(13,200 cells -> 5,072 distinct), with several families almost degenerate
(diag_lattice 600 -> 3, square 600 -> 6, square_lattice 600 -> 6, diamond_hole 600 -> 7).
A random 75/25 split leaks the exact geometry of 67.7% of test cells into training, which
inflates every in-distribution score.  The cause is rasterisation collapse: the generators
sample continuous shape parameters, but at 48x48 many parameter values land on the same
pixel grid.  Duplication rises monotonically as resolution falls (48: 72.8%, 24: 92.3%,
16: 96.7%, 12: 98.4% on an 8-family probe), which is the signature of the mechanism.

An independent check on METASET's published 2D subsets (Chan, Ahmed, Wang & Chen,
J. Mech. Des. 2021) found 0% duplication within every subset, so this is a defect of THESE
generators, not of pixelised metamaterial datasets in general.  It is reported as a
methods caution, not as a claim about the field.

THE FIX.  Reject a candidate before homogenising if its exact rasterisation has already
been accepted.  This is cheap (a hash-set lookup) and it also saves the FE solve that a
duplicate would have wasted.

WHAT THIS CANNOT FIX.  A family whose parameterisation simply cannot produce many distinct
48x48 rasterisations will now stall instead of silently emitting copies.  That is the
point: the script reports the achievable distinct count per family so the benchmark's real
size is visible rather than inflated.  Families that cannot reach the target are reported
with what they achieved; the decision to drop or re-parameterise them is then explicit.

Usage:  python gen_distinct.py [nwant] [family ...]
Writes ../data_distinct/clean48_<family>__{X,y,frac}.npy  (checkpointed per family).
"""
import os, sys, time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gen_clean as GC
from families_new import GEN_NEW
from families_rich import GEN_RICH
from homogenize2d import homogenize

N, BAND, MAT = GC.N, GC.BAND, GC.MAT
GEN = dict(GC.GEN); GEN.update(GEN_NEW); GEN.update(GEN_RICH)
OUT = "/mnt/user-data/outputs/data_distinct"
os.makedirs(OUT, exist_ok=True)

ORDER = ["circle", "square", "cross", "star_hole", "star6_hole", "tri_hole", "hex_hole",
         "diamond_hole", "ellipse", "rect_hole", "slot_pair", "two_holes", "cross_aniso",
         "kagome", "square_lattice", "diag_lattice", "rand_lattice", "honeycomb",
         "rect_lattice", "chevron", "reentrant", "layered"]


def generate_distinct(name, nwant=400, seed=0, max_draws=400_000,
                      stall_limit=12_000, time_budget=150.0):
    """Draw until nwant DISTINCT rasterisations are accepted, or until the family stalls
    (stall_limit consecutive draws yielding nothing new) or exhausts its time budget.

    A family that stalls has been enumerated to exhaustion: its parameterisation cannot
    produce more distinct 48x48 rasterisations inside the density band. That is reported,
    not worked around.
    """
    fn = GEN[name]
    rng = np.random.default_rng(1000 + seed)
    seen = set()
    X, Y, F = [], [], []
    t0 = time.time()
    n_draw = n_band = n_perc = n_dup = 0
    since_new = 0
    while (len(X) < nwant and n_draw < max_draws and since_new < stall_limit
           and time.time() - t0 < time_budget):
        n_draw += 1
        since_new += 1
        ph, _ = fn(N, rng)
        solid = (ph == 0)
        f = float(solid.mean())
        if not (BAND[0] <= f <= BAND[1]):
            continue
        n_band += 1
        if not GC.percolates(solid):
            continue
        n_perc += 1
        key = solid.astype(np.uint8).tobytes()        # exact rasterisation identity
        if key in seen:
            n_dup += 1
            continue
        try:
            CH = homogenize(ph, MAT)
        except Exception:
            continue
        if not np.all(np.isfinite(CH)) or min(CH[0, 0], CH[1, 1], CH[2, 2]) <= 1e-3:
            continue
        seen.add(key)
        X.append(solid.astype(np.uint8))
        Y.append([CH[0, 0], CH[0, 1], CH[0, 2], CH[1, 1], CH[1, 2], CH[2, 2]])
        F.append(f)
        since_new = 0
    X = np.array(X, np.uint8); Y = np.array(Y); F = np.array(F)
    assert len(np.unique(X.reshape(len(X), -1), axis=0)) == len(X), "duplicate slipped through"
    if len(X) >= nwant:
        status = "OK"
    elif since_new >= stall_limit:
        status = "EXHAUSTED"
    elif time.time() - t0 >= time_budget:
        status = "TIMEBOX"
    else:
        status = "CAPPED"
    dup_rate = n_dup / max(n_perc, 1)
    print(f"  {name:16s} N={len(X):4d}/{nwant}  {status:9s} {(time.time()-t0)/60:5.2f} min"
          f"  draws {n_draw:6d}  dup-rejected {n_dup:6d} ({dup_rate:5.1%} of percolating)",
          flush=True)
    for k, v in [("X", X), ("y", Y), ("frac", F)]:
        np.save(f"{OUT}/clean48_{name}__{k}.npy", v)
    return len(X), status, float(dup_rate)


if __name__ == "__main__":
    nwant = int(sys.argv[1]) if len(sys.argv) > 1 else 400
    which = sys.argv[2:] or ORDER
    print(f"Regenerating with DISTINCT-rasterisation filter, target {nwant} per family")
    print("frac in [0.45,0.75], percolating in x and y, unique 48x48 rasterisation\n")
    report = {}
    for nm in which:
        if os.path.exists(f"{OUT}/clean48_{nm}__X.npy"):
            n = len(np.load(f"{OUT}/clean48_{nm}__X.npy"))
            print(f"  {nm:16s} cached N={n}", flush=True)
            report[nm] = (n, "cached", None)
            continue
        report[nm] = generate_distinct(nm, nwant=nwant)
    print("\nSUMMARY")
    ok = [k for k, v in report.items() if v[0] >= nwant]
    print(f"  reached target: {len(ok)}/{len(report)}")
    for k, v in sorted(report.items(), key=lambda z: z[1][0]):
        print(f"    {k:16s} {v[0]:4d}  {v[1]}")
