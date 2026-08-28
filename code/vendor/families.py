"""
families.py
===========
Parametric generators for distinct 2D metamaterial unit-cell DESIGN FAMILIES.

Each family is a qualitatively different geometric class (different topology or
symmetry), not merely a different parameter value within one class. This is the
distinction that matters for the cross-family generalization study: a surrogate
trained on one family is asked to predict properties for a family it has never
seen.

All cells are periodic, defined on an n x n pixel grid, phase 0 = solid,
phase 1 = void.
"""

import numpy as np

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _grid(n):
    yy, xx = np.mgrid[0:n, 0:n]
    return (xx + 0.5) / n, (yy + 0.5) / n          # cell coords in [0,1]


def _periodic_delta(a, b):
    """Signed minimum-image difference on a unit-periodic axis."""
    d = a - b
    return d - np.round(d)


def _strut(px, py, p0, p1, thickness):
    """Boolean mask of a strut between p0 and p1, with periodic images."""
    mask = np.zeros(px.shape, dtype=bool)
    for sx in (-1.0, 0.0, 1.0):
        for sy in (-1.0, 0.0, 1.0):
            x0, y0 = p0[0] + sx, p0[1] + sy
            x1, y1 = p1[0] + sx, p1[1] + sy
            dx, dy = x1 - x0, y1 - y0
            L2 = dx * dx + dy * dy
            if L2 == 0:
                continue
            t = np.clip(((px - x0) * dx + (py - y0) * dy) / L2, 0.0, 1.0)
            dist = np.hypot(px - (x0 + t * dx), py - (y0 + t * dy))
            mask |= dist <= thickness / 2.0
    return mask


# ---------------------------------------------------------------------------
# FAMILY 1: circular hole
# ---------------------------------------------------------------------------
def circular_hole(n, rng):
    r = rng.uniform(0.12, 0.45)
    px, py = _grid(n)
    d = np.hypot(_periodic_delta(px, 0.5), _periodic_delta(py, 0.5))
    phase = np.zeros((n, n), dtype=int)
    phase[d <= r] = 1
    return phase, dict(r=r)


# ---------------------------------------------------------------------------
# FAMILY 2: square hole
# ---------------------------------------------------------------------------
def square_hole(n, rng):
    s = rng.uniform(0.20, 0.75)
    px, py = _grid(n)
    dx = np.abs(_periodic_delta(px, 0.5))
    dy = np.abs(_periodic_delta(py, 0.5))
    phase = np.zeros((n, n), dtype=int)
    phase[(dx <= s / 2) & (dy <= s / 2)] = 1
    return phase, dict(s=s)


# ---------------------------------------------------------------------------
# FAMILY 3: elliptical hole (anisotropic, rotated)
# ---------------------------------------------------------------------------
def elliptical_hole(n, rng):
    a = rng.uniform(0.12, 0.45)
    b = rng.uniform(0.12, 0.45)
    th = rng.uniform(0, np.pi)
    px, py = _grid(n)
    X = _periodic_delta(px, 0.5)
    Y = _periodic_delta(py, 0.5)
    Xr = X * np.cos(th) + Y * np.sin(th)
    Yr = -X * np.sin(th) + Y * np.cos(th)
    phase = np.zeros((n, n), dtype=int)
    phase[(Xr / a) ** 2 + (Yr / b) ** 2 <= 1.0] = 1
    return phase, dict(a=a, b=b, theta=th)


# ---------------------------------------------------------------------------
# FAMILY 4: cross / plus-shaped void
# ---------------------------------------------------------------------------
def cross_void(n, rng):
    w = rng.uniform(0.10, 0.35)      # arm half-width
    L = rng.uniform(0.30, 0.48)      # arm half-length
    px, py = _grid(n)
    X = np.abs(_periodic_delta(px, 0.5))
    Y = np.abs(_periodic_delta(py, 0.5))
    phase = np.zeros((n, n), dtype=int)
    horiz = (Y <= w / 2) & (X <= L)
    vert = (X <= w / 2) & (Y <= L)
    phase[horiz | vert] = 1
    return phase, dict(w=w, L=L)


# ---------------------------------------------------------------------------
# FAMILY 5: re-entrant honeycomb (strut topology -- entirely different class)
# ---------------------------------------------------------------------------
def reentrant_honeycomb(n, rng):
    theta = rng.uniform(-40.0, -10.0)     # re-entrant angles only
    t = rng.uniform(0.06, 0.16)
    h_over_l = rng.uniform(1.6, 2.6)
    th = np.radians(theta)
    l = 1.0
    h = h_over_l * l
    W = 2.0 * l * np.cos(th)
    H = 2.0 * (h + l * np.sin(th))
    # normalise geometry into the unit square
    ls = l * np.sin(th)
    pts = [((0.0, 0.0), (0.0, h)),
           ((0.0, h), (W / 2, h + ls)),
           ((W / 2, h + ls), (W, h)),
           ((W / 2, h + ls), (W / 2, 2 * h + ls)),
           ((W / 2, 2 * h + ls), (W, H)),
           ((W / 2, 2 * h + ls), (0.0, H))]
    px, py = _grid(n)
    solid = np.zeros((n, n), dtype=bool)
    for (p0, p1) in pts:
        q0 = (p0[0] / W, p0[1] / H)
        q1 = (p1[0] / W, p1[1] / H)
        solid |= _strut(px, py, q0, q1, t)
    phase = np.ones((n, n), dtype=int)
    phase[solid] = 0
    return phase, dict(theta=theta, t=t, h_over_l=h_over_l)


# ---------------------------------------------------------------------------
# FAMILY 6: random lattice of struts (stochastic topology)
# ---------------------------------------------------------------------------
def random_lattice(n, rng):
    nnodes = rng.integers(3, 6)
    nodes = rng.random((nnodes, 2))
    t = rng.uniform(0.06, 0.14)
    px, py = _grid(n)
    solid = np.zeros((n, n), dtype=bool)
    # connect each node to its 2 nearest neighbours (periodic distance)
    for i in range(nnodes):
        d = [np.hypot(_periodic_delta(nodes[i, 0], nodes[j, 0]),
                      _periodic_delta(nodes[i, 1], nodes[j, 1]))
             if j != i else 9e9 for j in range(nnodes)]
        order = np.argsort(d)[:2]
        for j in order:
            solid |= _strut(px, py, tuple(nodes[i]), tuple(nodes[j]), t)
    phase = np.ones((n, n), dtype=int)
    phase[solid] = 0
    return phase, dict(nnodes=int(nnodes), t=t)


FAMILIES = {
    "circle": circular_hole,
    "square": square_hole,
    "ellipse": elliptical_hole,
    "cross": cross_void,
    "honeycomb": reentrant_honeycomb,
    "rand_lattice": random_lattice,
}


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    n = 32
    print("Design families (solid fraction ranges over 20 samples each):\n")
    for name, fn in FAMILIES.items():
        fracs = []
        for _ in range(20):
            ph, _p = fn(n, rng)
            fracs.append(float(np.mean(ph == 0)))
        print(f"  {name:14s} solid frac: {min(fracs):.3f} - {max(fracs):.3f}")
    # print one example each
    print("\nExample cells (# = solid):")
    for name, fn in FAMILIES.items():
        ph, _p = fn(24, rng)
        print(f"\n  --- {name} ---")
        for j in range(23, -1, -2):
            print("    " + "".join("#" if ph[j, i] == 0 else "." for i in range(24)))
