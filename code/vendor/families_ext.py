"""
families_ext.py
===============
Four ADDITIONAL design families, chosen to break the confound in the original six
(where all hole-families were isotropic and all strut-families were anisotropic).

The selection-rule finding is that training families should cover the physical
degrees of freedom (anisotropy), not maximise geometric diversity. To show this is
about PHYSICS and not merely "holes vs struts", we add:

  rect_hole    : rectangular hole (a PERFORATION that is strongly ANISOTROPIC)
  slot_pair    : two parallel slots (perforation, anisotropic, different topology)
  kagome       : kagome-like strut lattice (STRUT family that is near-ISOTROPIC)
  star_hole    : 4-pointed star perforation (isotropic by 4-fold symmetry)

Now anisotropy is decoupled from holes-vs-struts, so "cover the anisotropic
families" cannot be dismissed as "just pick the strut families".
"""

import numpy as np


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


# --- anisotropic perforations -------------------------------------------------
def gen_rect_hole(n, rng):
    w = rng.uniform(0.25, 0.75)      # width
    h = rng.uniform(0.12, 0.45)      # height (deliberately != w -> anisotropic)
    px, py = _grid(n)
    dx = np.abs(_pd(px, .5)); dy = np.abs(_pd(py, .5))
    ph = np.zeros((n, n), dtype=int)
    ph[(dx <= w/2) & (dy <= h/2)] = 1
    return ph, dict(w=w, h=h)


def gen_slot_pair(n, rng):
    w = rng.uniform(0.35, 0.8)       # slot length
    t = rng.uniform(0.08, 0.22)      # slot thickness
    sep = rng.uniform(0.2, 0.4)      # vertical separation
    px, py = _grid(n)
    X = np.abs(_pd(px, .5))
    ph = np.zeros((n, n), dtype=int)
    for yc in (0.5 - sep/2, 0.5 + sep/2):
        Y = np.abs(_pd(py, yc))
        ph[(X <= w/2) & (Y <= t/2)] = 1
    return ph, dict(w=w, t=t, sep=sep)


# --- near-isotropic strut lattice ---------------------------------------------
def gen_kagome(n, rng):
    """Triangular strut lattice with 6-fold-like symmetry -> near in-plane
    isotropic C11=C22, despite being a STRUT family (the key de-confounder)."""
    t = rng.uniform(0.06, 0.13)
    # symmetric triangular network: horizontal + two diagonal directions,
    # arranged so x and y are statistically equivalent
    px, py = _grid(n)
    solid = np.zeros((n, n), dtype=bool)
    # three strut orientations at 0, 60, 120 deg through a centred hex pattern
    centres = [(0.0,0.0),(0.5,0.5),(1.0,0.0),(0.0,1.0),(1.0,1.0)]
    import numpy as _np
    for cx,cy in centres:
        for ang in (0, 60, 120):
            a=_np.radians(ang); L=0.5
            p0=(cx-L*_np.cos(a), cy-L*_np.sin(a))
            p1=(cx+L*_np.cos(a), cy+L*_np.sin(a))
            solid |= _strut(px, py, p0, p1, t)
    ph = np.ones((n, n), dtype=int)
    ph[solid] = 0
    return ph, dict(t=t)


# --- isotropic star perforation -----------------------------------------------
def gen_star_hole(n, rng):
    R = rng.uniform(0.25, 0.45)
    m = 4                             # 4-fold -> in-plane isotropic C11=C22
    amp = rng.uniform(0.15, 0.35)
    px, py = _grid(n)
    X = _pd(px, .5); Y = _pd(py, .5)
    theta = np.arctan2(Y, X)
    rr = np.hypot(X, Y)
    boundary = R * (1 + amp * np.cos(m * theta))
    ph = np.zeros((n, n), dtype=int)
    ph[rr <= boundary] = 1
    return ph, dict(R=R, amp=amp)


GEN_EXT = {
    "rect_hole": gen_rect_hole,
    "slot_pair": gen_slot_pair,
    "kagome": gen_kagome,
    "star_hole": gen_star_hole,
}


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    n = 48
    print("New families, solid fraction range + anisotropy over 20 samples:\n")
    from homogenize2d import homogenize
    for name, fn in GEN_EXT.items():
        fr, an = [], []
        for _ in range(20):
            ph, _p = fn(n, rng)
            f = float(np.mean(ph == 0))
            if not (0.40 <= f <= 0.80):
                continue
            fr.append(f)
            try:
                CH = homogenize(ph, [(1.0, 0.3), (1e-6, 0.3)])
                an.append(abs(CH[0,0]-CH[1,1])/(CH[0,0]+CH[1,1]+1e-9))
            except Exception:
                pass
        if fr:
            print(f"  {name:11s} frac {min(fr):.2f}-{max(fr):.2f}  "
                  f"anisotropy {np.mean(an):.3f}")
