"""coarse.py -- is the descriptor cheap enough to be worth computing?

The feature is only useful if extracting it costs materially less than the
homogenisation the surrogate exists to replace.  We time (a) a full 48x48
homogenisation, (b) the sub-cell eigensolve at several resolutions, and check
how faithfully coarse descriptors reproduce the 48x48 ones.
"""
import _path  # noqa: F401  (adds code/ and code/vendor/ to sys.path)

import time, sys, numpy as np
sys.path.insert(0, '.')
import datapath
from homogenize2d import homogenize
from subcell import subcell_modes
MAT = [(1.0, 0.3), (1e-6, 0.3)]

def down(a, f):
    n = a.shape[0] // f
    return (a.reshape(n, f, n, f).mean(axis=(1, 3)) > 0.5).astype(int)

fams = ['circle','cross','honeycomb','kagome','chevron','rect_hole','layered','ellipse']
cells = []
for f in fams:
    X = datapath.load('clean48_' + f)['X'][:4]
    for k in range(len(X)):
        cells.append((X[k] == 0).astype(int))

t = time.time()
for ph in cells: homogenize(ph, MAT)
t_hom = (time.time() - t) / len(cells)
print("full 48x48 homogenisation   %7.1f ms/cell  <- the quantity being replaced\n" % (t_hom*1000))

ref = None
print("%-30s%9s%11s   %s" % ("descriptor cost", "ms/cell", "vs homog", "log-corr with 48x48 (w2 w3 w4)"))
for res in (48, 24, 16, 12):
    om = []; t = time.time()
    for ph in cells:
        p = ph if res == 48 else down(ph, 48 // res)
        try:
            o, _, _ = subcell_modes(p, MAT, n_modes=4); om.append(o)
        except Exception:
            om.append([np.nan]*4)
    dt = (time.time() - t) / len(cells); om = np.array(om)
    if res == 48: ref = om.copy()
    g = np.isfinite(om).all(1) & np.isfinite(ref).all(1) & (om > 1e-9).all(1)
    cc = [np.corrcoef(np.log(ref[g, j]), np.log(om[g, j]))[0, 1] for j in range(3)]
    print("%-30s%9.1f%10.2fx   %.3f %.3f %.3f"
          % ("eigensolve %dx%d" % (res, res), dt*1000, t_hom/dt, cc[0], cc[1], cc[2]))
