"""
gen_clean.py -- regenerate ALL 10 families under ONE uniform protocol.

Two defects were found in the inherited 10-family pool:

  (1) DENSITY CONFOUND.  gen_ext.py rejection-samples into [0.45,0.75], but the
      parameter ranges of rect_hole / slot_pair / kagome cannot physically reach
      the bottom of that band (a rect hole with w<=0.75, h<=0.45 has area <=0.34,
      so solid fraction >= 0.66).  Observed ranges: rect_hole 0.656-0.750,
      slot_pair 0.637-0.750, kagome 0.451-0.691, vs 0.45-0.75 for the originals.
      Cross-family error involving those families is therefore partly a
      VOLUME-FRACTION EXTRAPOLATION, not a topology effect -- exactly the
      confound the EML paper was built to exclude.
      FIX: reparametrise by hole AREA and ASPECT RATIO so area (hence solid
      fraction) is controlled independently of anisotropy.

  (2) PERCOLATION DEGENERACY.  89% of rand_lattice cells, 18% of star_hole and
      4% of cross do not percolate in at least one direction, so their stiffness
      is the void stiffness (~1e-6) and C11/C22 is a ratio of two numerical
      zeros.  Any "anisotropy" measured on such a family is an artefact.
      FIX: require the solid phase to percolate in BOTH directions under
      periodic boundary conditions, checked geometrically before solving.

Protocol applied uniformly to all ten families:
    solid fraction in [0.45, 0.75]  AND  percolates in x AND in y
    material: E_solid=1, nu=0.3, void E=1e-6, plane strain, 48x48 Q4.
"""
import sys, time
import numpy as np
from scipy import ndimage
from homogenize2d import homogenize

import matched_sampler as MS          # original 6 (widened ranges)
import families_ext as FE             # helpers _grid, _pd, _strut

N = 48
BAND = (0.45, 0.75)
MAT = [(1.0, 0.3), (1e-6, 0.3)]

_grid, _pd, _strut = FE._grid, FE._pd, FE._strut


# ---------------------------------------------------------------------------
# periodic percolation test
# ---------------------------------------------------------------------------
def percolates(solid):
    """True iff the solid phase spans the cell in BOTH x and y under periodic BCs.

    Tile 3x3 and ask whether one connected component reaches from the first
    column to the last (x) and from the first row to the last (y).
    """
    if solid.sum() == 0:
        return False
    T = np.tile(solid, (3, 3))
    lab, k = ndimage.label(T)                      # 4-connectivity
    if k == 0:
        return False
    left, right = set(lab[:, 0]) - {0}, set(lab[:, -1]) - {0}
    top, bot = set(lab[0, :]) - {0}, set(lab[-1, :]) - {0}
    return bool(left & right) and bool(top & bot)


# ---------------------------------------------------------------------------
# reparametrised extended families: AREA controlled, ASPECT controlled
# ---------------------------------------------------------------------------
def gen_rect_hole(n, rng):
    """Rectangular hole. Hole area A and aspect w/h drawn independently, so
    solid fraction spans the full band at any anisotropy level."""
    A = rng.uniform(0.25, 0.55)
    rho = rng.uniform(1.8, 4.0)                    # w/h  -> anisotropic
    w = min(np.sqrt(A * rho), 0.94)
    h = A / w
    px, py = _grid(n)
    dx, dy = np.abs(_pd(px, .5)), np.abs(_pd(py, .5))
    ph = np.zeros((n, n), dtype=int)
    ph[(dx <= w / 2) & (dy <= h / 2)] = 1
    return ph, dict(w=w, h=h)


def gen_slot_pair(n, rng):
    """Two parallel slots; total void area A controlled independently."""
    A = rng.uniform(0.25, 0.55)
    w = rng.uniform(0.55, 0.94)                    # slot length
    t = A / (2 * w)                                # slot thickness
    if t > 0.30:
        t = 0.30
    sep = rng.uniform(t + 0.12, 0.48)
    px, py = _grid(n)
    Xc = np.abs(_pd(px, .5))
    ph = np.zeros((n, n), dtype=int)
    for yc in (0.5 - sep / 2, 0.5 + sep / 2):
        Yc = np.abs(_pd(py, yc))
        ph[(Xc <= w / 2) & (Yc <= t / 2)] = 1
    return ph, dict(w=w, t=t, sep=sep)


def gen_kagome(n, rng):
    """Triangular strut lattice (isotropic strut family). Widened thickness so
    the strut fraction reaches the top of the band."""
    t = rng.uniform(0.05, 0.20)
    px, py = _grid(n)
    solid = np.zeros((n, n), dtype=bool)
    for cx, cy in [(0., 0.), (0.5, 0.5), (1., 0.), (0., 1.), (1., 1.)]:
        for ang in (0, 60, 120):
            a = np.radians(ang); L = 0.5
            p0 = (cx - L * np.cos(a), cy - L * np.sin(a))
            p1 = (cx + L * np.cos(a), cy + L * np.sin(a))
            solid |= _strut(px, py, p0, p1, t)
    ph = np.ones((n, n), dtype=int)
    ph[solid] = 0
    return ph, dict(t=t)


def gen_star_hole(n, rng):
    """4-fold star perforation (isotropic hole family). Widened R so it reaches
    the bottom of the band."""
    R = rng.uniform(0.25, 0.52)
    amp = rng.uniform(0.15, 0.35)
    px, py = _grid(n)
    Xc, Yc = _pd(px, .5), _pd(py, .5)
    boundary = R * (1 + amp * np.cos(4 * np.arctan2(Yc, Xc)))
    ph = np.zeros((n, n), dtype=int)
    ph[np.hypot(Xc, Yc) <= boundary] = 1
    return ph, dict(R=R, amp=amp)


def gen_rand_lattice(n, rng):
    """Random strut lattice, each node joined to its 2 nearest periodic
    neighbours (as in the EML generator).  Node count raised (8-16 vs 3-5) and
    strut thickness lowered, because the original 3-5 node / thick-strut version
    produced a fat isolated blob that does not percolate in 89% of draws; those
    cells have void-level stiffness and a meaningless C11/C22 ratio."""
    nn = int(rng.integers(8, 17))
    nodes = rng.random((nn, 2))
    t = rng.uniform(0.04, 0.18)
    px, py = _grid(n)
    solid = np.zeros((n, n), dtype=bool)
    for i in range(nn):
        d = [np.hypot(_pd(nodes[i, 0], nodes[j, 0]), _pd(nodes[i, 1], nodes[j, 1]))
             if j != i else 9e9 for j in range(nn)]
        for j in np.argsort(d)[:2]:
            solid |= _strut(px, py, tuple(nodes[i]), tuple(nodes[j]), t)
    ph = np.ones((n, n), dtype=int)
    ph[solid] = 0
    return ph, dict(nnodes=nn, t=t)


GEN = {
    "circle":       MS.gen_circle,
    "square":       MS.gen_square,
    "ellipse":      MS.gen_ellipse,
    "cross":        MS.gen_cross,
    "honeycomb":    MS.gen_honeycomb,
    "rand_lattice": gen_rand_lattice,
    "rect_hole":    gen_rect_hole,
    "slot_pair":    gen_slot_pair,
    "kagome":       gen_kagome,
    "star_hole":    gen_star_hole,
}


# ---------------------------------------------------------------------------
def generate(name, nwant=600, seed=0, outdir="."):
    fn = GEN[name]
    rng = np.random.default_rng(1000 + seed)
    X, Y, F = [], [], []
    t0 = time.time()
    n_geom, n_band, n_perc = 0, 0, 0
    while len(X) < nwant and n_geom < nwant * 400:
        n_geom += 1
        ph, _ = fn(N, rng)
        solid = (ph == 0)
        f = float(solid.mean())
        if not (BAND[0] <= f <= BAND[1]):
            continue
        n_band += 1
        if not percolates(solid):
            continue
        n_perc += 1
        try:
            CH = homogenize(ph, MAT)
        except Exception:
            continue
        if not np.all(np.isfinite(CH)) or min(CH[0, 0], CH[1, 1], CH[2, 2]) <= 1e-3:
            continue
        X.append(solid.astype(np.uint8))
        Y.append([CH[0, 0], CH[0, 1], CH[0, 2], CH[1, 1], CH[1, 2], CH[2, 2]])
        F.append(f)
    X = np.array(X, np.uint8); Y = np.array(Y); F = np.array(F)
    r = Y[:, 0] / Y[:, 3]
    print(f"  {name:14s} N={len(X):4d}  {(time.time()-t0)/60:5.2f} min  "
          f"frac {F.min():.3f}-{F.max():.3f} (med {np.median(F):.3f})  "
          f"C11/C22 med {np.median(r):7.3f} max {r.max():8.3f}  "
          f"| geom {n_geom} -> band {n_band} -> perc {n_perc} -> kept {len(X)}",
          flush=True)
    for k, v in [("X", X), ("y", Y), ("frac", F)]:
        np.save(f"{outdir}/clean48_{name}__{k}.npy", v)


if __name__ == "__main__":
    which = sys.argv[1:] if len(sys.argv) > 1 else list(GEN)
    print("Uniform protocol: frac in [0.45,0.75] AND percolating in x and y\n")
    for nm in which:
        generate(nm, nwant=600, outdir="../data_clean")
