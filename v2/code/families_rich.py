"""
families_rich.py -- re-parameterised versions of the families that were exhausted.

THE PROBLEM.  The distinct-rasterisation audit showed that several families cannot produce
many distinct 48x48 rasterisations because they have almost no degrees of freedom:

    gen_circle : 1 parameter  (radius)                 ->  65 distinct
    gen_square : 1 parameter  (side)                   ->   6 distinct
    gen_cross  : 2 parameters (width, arm length)      ->  47 distinct
    diamond_hole, square_lattice, diag_lattice         ->   7, 6, 3 distinct

A one-parameter family on a 48x48 grid can only rasterise to about as many distinct images
as there are admissible pixel radii inside the density band -- a few dozen.  Sampling
harder cannot help; the family is enumerated.

THE FIX AND ITS COST.  Each family gains shape freedom while keeping the property that
DEFINES it.  This is not cosmetic: adding freedom changes what the family means, so each
change is stated explicitly and the defining invariant is preserved and asserted.

    circle_rich   : one central hole, but ELLIPTICAL with free aspect and orientation.
                    Defining property kept: a single convex hole.
                    NOTE: aspect != 1 breaks exact in-plane isotropy, so this family is no
                    longer guaranteed C11/C22 = 1.  A dedicated isotropic subset is
                    produced separately by circle_iso (aspect exactly 1, radius free),
                    which preserves the exactly-isotropic demonstration cells even though
                    there are only ~65 of them.
    square_rich   : rectangular hole, free aspect AND free rotation angle.
    cross_rich    : cross with independent arm widths/lengths in x and y, plus rotation.
    lattice_rich  : square/diagonal strut lattice with free strut thickness, node jitter
                    and a second strut family at a free angle.

All of them keep the periodic, single-inclusion-or-lattice character of the originals.
Rotation is applied in the periodic coordinate system so periodicity is preserved.
"""
import numpy as np

import families_ext as FE
_grid, _pd, _strut = FE._grid, FE._pd, FE._strut


def _rot(X, Y, ang):
    c, s = np.cos(ang), np.sin(ang)
    return c * X + s * Y, -s * X + c * Y


# ---------------------------------------------------------------- circle
def gen_circle_iso(n, rng):
    """The ORIGINAL exactly-isotropic circular hole: aspect exactly 1.
    Kept unchanged so the C11/C22 = 1.000 demonstration cells still exist."""
    r = rng.uniform(0.20, 0.52)
    px, py = _grid(n)
    ph = np.zeros((n, n), dtype=int)
    ph[np.hypot(_pd(px, .5), _pd(py, .5)) <= r] = 1
    return ph, dict(r=r, aspect=1.0)


def gen_circle_rich(n, rng):
    """Elliptical hole: area, aspect ratio and orientation free (3 DOF vs 1)."""
    A = rng.uniform(0.16, 0.55)               # hole area fraction
    asp = rng.uniform(1.0, 2.6)               # a/b
    th = rng.uniform(0, np.pi)
    b = np.sqrt(A / (np.pi * asp)); a = asp * b
    px, py = _grid(n)
    X, Y = _rot(_pd(px, .5), _pd(py, .5), th)
    ph = np.zeros((n, n), dtype=int)
    ph[(X / a) ** 2 + (Y / b) ** 2 <= 1.0] = 1
    return ph, dict(a=a, b=b, th=th)


# ---------------------------------------------------------------- square
def gen_square_rich(n, rng):
    """Rectangular hole with free area, aspect and rotation (3 DOF vs 1)."""
    A = rng.uniform(0.16, 0.55)
    asp = rng.uniform(1.0, 3.0)
    th = rng.uniform(0, np.pi / 2)
    h = np.sqrt(A / asp); w = asp * h
    px, py = _grid(n)
    X, Y = _rot(_pd(px, .5), _pd(py, .5), th)
    ph = np.zeros((n, n), dtype=int)
    ph[(np.abs(X) <= w / 2) & (np.abs(Y) <= h / 2)] = 1
    return ph, dict(w=w, h=h, th=th)


# ---------------------------------------------------------------- cross
def gen_cross_rich(n, rng):
    """Cross with INDEPENDENT x and y arms plus rotation (5 DOF vs 2)."""
    wx = rng.uniform(0.10, 0.34); wy = rng.uniform(0.10, 0.34)
    Lx = rng.uniform(0.22, 0.50); Ly = rng.uniform(0.22, 0.50)
    th = rng.uniform(0, np.pi / 2)
    px, py = _grid(n)
    X, Y = _rot(_pd(px, .5), _pd(py, .5), th)
    X, Y = np.abs(X), np.abs(Y)
    ph = np.zeros((n, n), dtype=int)
    ph[((Y <= wy / 2) & (X <= Lx)) | ((X <= wx / 2) & (Y <= Ly))] = 1
    return ph, dict(wx=wx, wy=wy, Lx=Lx, Ly=Ly, th=th)


# ---------------------------------------------------------------- lattices
def _lattice(n, rng, angles, jitter, t_range):
    t = rng.uniform(*t_range)
    px, py = _grid(n)
    solid = np.zeros((n, n), dtype=bool)
    for cx, cy in [(0., 0.), (0.5, 0.5), (1., 0.), (0., 1.), (1., 1.)]:
        jx = cx + rng.uniform(-jitter, jitter)
        jy = cy + rng.uniform(-jitter, jitter)
        for ang in angles:
            a = np.radians(ang); L = 0.5
            p0 = (jx - L * np.cos(a), jy - L * np.sin(a))
            p1 = (jx + L * np.cos(a), jy + L * np.sin(a))
            solid |= _strut(px, py, p0, p1, t)
    ph = np.ones((n, n), dtype=int)
    ph[solid] = 0
    return ph, dict(t=t)


def gen_square_lattice_rich(n, rng):
    """Orthogonal struts, free thickness, node jitter and a free skew angle."""
    skew = rng.uniform(-18, 18)
    return _lattice(n, rng, (0 + skew, 90 + skew), rng.uniform(0, 0.09), (0.04, 0.19))


def gen_diag_lattice_rich(n, rng):
    """Diagonal struts, free thickness, node jitter and a free rotation."""
    rot = rng.uniform(-20, 20)
    return _lattice(n, rng, (45 + rot, 135 + rot), rng.uniform(0, 0.09), (0.04, 0.19))


def gen_diamond_hole_rich(n, rng):
    """Diamond (rotated square) hole with free area, aspect and orientation."""
    A = rng.uniform(0.16, 0.55)
    asp = rng.uniform(1.0, 2.6)
    th = rng.uniform(0, np.pi / 2)
    h = np.sqrt(2 * A / asp); w = asp * h
    px, py = _grid(n)
    X, Y = _rot(_pd(px, .5), _pd(py, .5), th)
    ph = np.zeros((n, n), dtype=int)
    ph[np.abs(X) / w + np.abs(Y) / h <= 0.5] = 1
    return ph, dict(w=w, h=h, th=th)


GEN_RICH = {
    "circle_iso": gen_circle_iso,
    "circle_rich": gen_circle_rich,
    "square_rich": gen_square_rich,
    "cross_rich": gen_cross_rich,
    "square_lattice_rich": gen_square_lattice_rich,
    "diag_lattice_rich": gen_diag_lattice_rich,
    "diamond_hole_rich": gen_diamond_hole_rich,
}
