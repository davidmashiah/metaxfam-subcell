"""
families_new.py -- ten additional design families, taking the pool from 10 to 20.

Why: with 10 families the criterion-vs-transfer correlations are estimated from
45 (K=2) or 120 (K=3) subsets and every criterion is significant on only ~4 of 6
target/K jobs.  The effect sits near the noise floor of the design.  More
families widens the subset space AND, more importantly, fills in the anisotropy
axis so that "anisotropy" is sampled continuously rather than at 3 distinct
levels.

Design intent -- each cell of the (topology x anisotropy) grid gets several
members, and two families (rect_lattice, layered) have CONTINUOUSLY TUNABLE
anisotropy so the axis is densely covered:

  isotropic holes      : tri_hole, hex_hole, diamond_hole, star6_hole
  anisotropic holes    : two_holes, cross_aniso
  isotropic struts     : square_lattice, diag_lattice
  anisotropic struts   : rect_lattice (tunable), chevron, reentrant, layered

All are generated under the same protocol as gen_clean.py: solid fraction is
rejection-sampled into [0.45,0.75] and the solid phase must percolate in both
directions under periodic BCs.
"""
import numpy as np
from families_ext import _grid, _pd, _strut


# ---------------------------------------------------------------- holes, isotropic
def gen_tri_hole(n, rng):
    """Triangular hole: 3-fold symmetry -> in-plane isotropic elasticity."""
    R = rng.uniform(0.30, 0.62)
    px, py = _grid(n)
    X, Y = _pd(px, .5), _pd(py, .5)
    th = np.arctan2(Y, X)
    r = np.hypot(X, Y)
    # rounded triangle via 3-fold radius modulation
    b = R * (1 + 0.30 * np.cos(3 * th)) / 1.3
    ph = np.zeros((n, n), dtype=int); ph[r <= b] = 1
    return ph, {}


def gen_hex_hole(n, rng):
    """Hexagonal hole: 6-fold -> isotropic."""
    R = rng.uniform(0.28, 0.55)
    px, py = _grid(n)
    X, Y = _pd(px, .5), _pd(py, .5)
    th = np.arctan2(Y, X); r = np.hypot(X, Y)
    b = R * (1 + 0.12 * np.cos(6 * th))
    ph = np.zeros((n, n), dtype=int); ph[r <= b] = 1
    return ph, {}


def gen_diamond_hole(n, rng):
    """Square hole rotated 45 deg (L1 ball).  4-fold -> isotropic."""
    a = rng.uniform(0.62, 1.04)
    px, py = _grid(n)
    X, Y = np.abs(_pd(px, .5)), np.abs(_pd(py, .5))
    ph = np.zeros((n, n), dtype=int); ph[(X + Y) <= a / 2] = 1
    return ph, {}


def gen_star6_hole(n, rng):
    """6-pointed star perforation -> isotropic."""
    R = rng.uniform(0.28, 0.58)
    amp = rng.uniform(0.20, 0.40)
    px, py = _grid(n)
    X, Y = _pd(px, .5), _pd(py, .5)
    b = R * (1 + amp * np.cos(6 * np.arctan2(Y, X)))
    ph = np.zeros((n, n), dtype=int); ph[np.hypot(X, Y) <= b] = 1
    return ph, {}


# ---------------------------------------------------------------- holes, anisotropic
def gen_two_holes(n, rng):
    """Two circular holes separated along x -> anisotropic, hole topology."""
    r = rng.uniform(0.16, 0.35)
    sep = rng.uniform(0.30, 0.50)
    px, py = _grid(n)
    ph = np.zeros((n, n), dtype=int)
    for xc in (0.5 - sep / 2, 0.5 + sep / 2):
        ph[np.hypot(_pd(px, xc), _pd(py, .5)) <= r] = 1
    return ph, {}


def gen_cross_aniso(n, rng):
    """Cross-shaped hole with unequal arm widths -> tunable anisotropy, hole."""
    wx = rng.uniform(0.12, 0.42)
    wy = rng.uniform(0.12, 0.42)
    L = rng.uniform(0.55, 0.95)
    px, py = _grid(n)
    X, Y = np.abs(_pd(px, .5)), np.abs(_pd(py, .5))
    ph = np.zeros((n, n), dtype=int)
    ph[((Y <= wx / 2) & (X <= L / 2)) | ((X <= wy / 2) & (Y <= L / 2))] = 1
    return ph, {}


# ---------------------------------------------------------------- struts, isotropic
def gen_square_lattice(n, rng):
    """Orthogonal struts of EQUAL thickness -> C11 = C22 by symmetry."""
    t = rng.uniform(0.24, 0.54)
    px, py = _grid(n)
    X, Y = np.abs(_pd(px, .5)), np.abs(_pd(py, .5))
    solid = (X <= t / 2) | (Y <= t / 2)
    ph = np.ones((n, n), dtype=int); ph[solid] = 0
    return ph, {}


def gen_diag_lattice(n, rng):
    """Two diagonal strut directions at +-45 deg -> isotropic strut network."""
    t = rng.uniform(0.09, 0.30)
    px, py = _grid(n)
    solid = np.zeros((n, n), dtype=bool)
    for c in (0.0, 0.5, 1.0):
        solid |= _strut(px, py, (c - 0.75, -0.75), (c + 0.75, 0.75), t)
        solid |= _strut(px, py, (c - 0.75, 0.75), (c + 0.75, -0.75), t)
    ph = np.ones((n, n), dtype=int); ph[solid] = 0
    return ph, {}


# ---------------------------------------------------------------- struts, anisotropic
def gen_rect_lattice(n, rng):
    """Orthogonal struts with INDEPENDENT x and y thickness.  Anisotropy is a
    continuous knob (tx/ty), so this family alone fills the anisotropy axis."""
    tx = rng.uniform(0.10, 0.52)
    ty = rng.uniform(0.10, 0.52)
    px, py = _grid(n)
    X, Y = np.abs(_pd(px, .5)), np.abs(_pd(py, .5))
    solid = (X <= ty / 2) | (Y <= tx / 2)      # horizontal bar thickness tx
    ph = np.ones((n, n), dtype=int); ph[solid] = 0
    return ph, {}


def gen_chevron(n, rng):
    """Zig-zag (chevron) strut rows -> strongly anisotropic strut family."""
    t = rng.uniform(0.07, 0.26)
    amp = rng.uniform(0.12, 0.30)
    px, py = _grid(n)
    solid = np.zeros((n, n), dtype=bool)
    for yc in (0.25, 0.75):
        solid |= _strut(px, py, (0.0, yc - amp), (0.5, yc + amp), t)
        solid |= _strut(px, py, (0.5, yc + amp), (1.0, yc - amp), t)
    # full-height vertical ties so the network percolates in y
    for xc in (0.0, 0.5, 1.0):
        solid |= _strut(px, py, (xc, -0.1), (xc, 1.1), t)
    ph = np.ones((n, n), dtype=int); ph[solid] = 0
    return ph, {}


def gen_reentrant(n, rng):
    """Re-entrant (auxetic-type) honeycomb -> anisotropic strut."""
    t = rng.uniform(0.06, 0.30)
    a = rng.uniform(0.15, 0.35)
    px, py = _grid(n)
    solid = np.zeros((n, n), dtype=bool)
    for xc in (0.25, 0.75):
        solid |= _strut(px, py, (xc, 0.0), (xc, 0.5), t)          # vertical rib
        solid |= _strut(px, py, (xc, 0.5), (xc - a, 0.75), t)     # re-entrant arms
        solid |= _strut(px, py, (xc, 0.5), (xc + a, 0.75), t)
        solid |= _strut(px, py, (xc - a, 0.75), (xc, 1.0), t)
        solid |= _strut(px, py, (xc + a, 0.75), (xc, 1.0), t)
    ph = np.ones((n, n), dtype=int); ph[solid] = 0
    return ph, {}


def gen_layered(n, rng):
    """Horizontal solid layers -> near rank-1 laminate, extreme and continuously
    tunable anisotropy.  Thin vertical ties keep it percolating in y."""
    h = rng.uniform(0.30, 0.70)          # solid layer thickness
    tie = rng.uniform(0.04, 0.12)
    px, py = _grid(n)
    Y = np.abs(_pd(py, .5)); X = np.abs(_pd(px, .5))
    solid = (Y <= h / 2) | (X <= tie / 2)
    ph = np.ones((n, n), dtype=int); ph[solid] = 0
    return ph, {}


GEN_NEW = {
    "tri_hole":       gen_tri_hole,
    "hex_hole":       gen_hex_hole,
    "diamond_hole":   gen_diamond_hole,
    "star6_hole":     gen_star6_hole,
    "two_holes":      gen_two_holes,
    "cross_aniso":    gen_cross_aniso,
    "square_lattice": gen_square_lattice,
    "diag_lattice":   gen_diag_lattice,
    "rect_lattice":   gen_rect_lattice,
    "chevron":        gen_chevron,
    "reentrant":      gen_reentrant,
    "layered":        gen_layered,
}
