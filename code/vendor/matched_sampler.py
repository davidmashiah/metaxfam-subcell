"""
matched_sampler.py
==================
Sample unit cells from each design family CONDITIONED on a common solid-fraction
band, via rejection sampling.

WHY THIS MATTERS
----------------
Effective stiffness is dominated by solid volume fraction. If design families
occupy different volume-fraction ranges, then a surrogate trained on family A and
tested on family B is really being asked to EXTRAPOLATE IN VOLUME FRACTION, and
any measured failure is confounded -- it says nothing about topology.

By matching the volume-fraction distribution across families, cross-family error
becomes attributable to the geometry/topology itself. This is the control that
makes the study meaningful.
"""

import numpy as np
from families import FAMILIES

# widened parameter ranges so every family can reach the common band
WIDE = {
    "circle":       dict(r=(0.25, 0.45)),
    "square":       dict(s=(0.45, 0.80)),
    "ellipse":      dict(a=(0.15, 0.49), b=(0.15, 0.49)),
    "cross":        dict(w=(0.15, 0.50), L=(0.35, 0.50)),
    "honeycomb":    dict(t=(0.10, 0.42)),
    "rand_lattice": dict(t=(0.10, 0.45)),
}


def _grid(n):
    yy, xx = np.mgrid[0:n, 0:n]
    return (xx + 0.5) / n, (yy + 0.5) / n


def _pd(a, b):
    d = a - b
    return d - np.round(d)


def _strut(px, py, p0, p1, thickness):
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
            mask |= np.hypot(px - (x0 + t * dx), py - (y0 + t * dy)) <= thickness / 2
    return mask


# --- family generators with widened ranges ---------------------------------
def gen_circle(n, rng):
    r = rng.uniform(*WIDE["circle"]["r"])
    px, py = _grid(n)
    ph = np.zeros((n, n), dtype=int)
    ph[np.hypot(_pd(px, .5), _pd(py, .5)) <= r] = 1
    return ph, dict(r=r)


def gen_square(n, rng):
    s = rng.uniform(*WIDE["square"]["s"])
    px, py = _grid(n)
    ph = np.zeros((n, n), dtype=int)
    ph[(np.abs(_pd(px, .5)) <= s / 2) & (np.abs(_pd(py, .5)) <= s / 2)] = 1
    return ph, dict(s=s)


def gen_ellipse(n, rng):
    a = rng.uniform(*WIDE["ellipse"]["a"])
    b = rng.uniform(*WIDE["ellipse"]["b"])
    th = rng.uniform(0, np.pi)
    px, py = _grid(n)
    X, Y = _pd(px, .5), _pd(py, .5)
    Xr = X * np.cos(th) + Y * np.sin(th)
    Yr = -X * np.sin(th) + Y * np.cos(th)
    ph = np.zeros((n, n), dtype=int)
    ph[(Xr / a) ** 2 + (Yr / b) ** 2 <= 1.0] = 1
    return ph, dict(a=a, b=b, theta=th)


def gen_cross(n, rng):
    w = rng.uniform(*WIDE["cross"]["w"])
    L = rng.uniform(*WIDE["cross"]["L"])
    px, py = _grid(n)
    X, Y = np.abs(_pd(px, .5)), np.abs(_pd(py, .5))
    ph = np.zeros((n, n), dtype=int)
    ph[((Y <= w / 2) & (X <= L)) | ((X <= w / 2) & (Y <= L))] = 1
    return ph, dict(w=w, L=L)


def gen_honeycomb(n, rng):
    theta = rng.uniform(-40.0, -10.0)
    t = rng.uniform(*WIDE["honeycomb"]["t"])
    hl = rng.uniform(1.6, 2.6)
    th = np.radians(theta)
    l = 1.0
    h = hl * l
    W = 2 * l * np.cos(th)
    H = 2 * (h + l * np.sin(th))
    ls = l * np.sin(th)
    pts = [((0, 0), (0, h)), ((0, h), (W / 2, h + ls)), ((W / 2, h + ls), (W, h)),
           ((W / 2, h + ls), (W / 2, 2 * h + ls)), ((W / 2, 2 * h + ls), (W, H)),
           ((W / 2, 2 * h + ls), (0, H))]
    px, py = _grid(n)
    solid = np.zeros((n, n), dtype=bool)
    for p0, p1 in pts:
        solid |= _strut(px, py, (p0[0] / W, p0[1] / H), (p1[0] / W, p1[1] / H), t)
    ph = np.ones((n, n), dtype=int)
    ph[solid] = 0
    return ph, dict(theta=theta, t=t, h_over_l=hl)


def gen_rand_lattice(n, rng):
    nn = int(rng.integers(3, 6))
    nodes = rng.random((nn, 2))
    t = rng.uniform(*WIDE["rand_lattice"]["t"])
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
    "circle": gen_circle,
    "square": gen_square,
    "ellipse": gen_ellipse,
    "cross": gen_cross,
    "honeycomb": gen_honeycomb,
    "rand_lattice": gen_rand_lattice,
}


def sample_matched(name, n, rng, band=(0.45, 0.75), max_tries=4000):
    """Rejection-sample a cell from `name` whose solid fraction lies in `band`."""
    fn = GEN[name]
    for _ in range(max_tries):
        ph, p = fn(n, rng)
        f = float(np.mean(ph == 0))
        if band[0] <= f <= band[1]:
            return ph, p, f
    return None, None, None


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    n = 32
    band = (0.45, 0.75)
    print(f"Rejection sampling into common solid-fraction band {band}\n")
    print(f"  {'family':14s} {'accept rate':>12s} {'frac range':>18s} {'mean':>7s}")
    for name in GEN:
        fr, tries = [], 0
        for _ in range(40):
            ph, p, f = sample_matched(name, n, rng, band)
            if f is not None:
                fr.append(f)
        # measure acceptance rate separately
        acc = 0
        N = 300
        for _ in range(N):
            ph, p = GEN[name](n, rng)
            if band[0] <= float(np.mean(ph == 0)) <= band[1]:
                acc += 1
        fr = np.array(fr)
        if len(fr):
            print(f"  {name:14s} {acc/N*100:11.1f}% "
                  f"{fr.min():8.3f}-{fr.max():.3f} {fr.mean():7.3f}")
        else:
            print(f"  {name:14s} FAILED to reach band")
