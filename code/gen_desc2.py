"""Compute sub-cell descriptors at a chosen resolution for all families.
Saved to /mnt/user-data/outputs so they survive container resets."""
import _path  # noqa: F401  (adds code/ and code/vendor/ to sys.path)

import os, sys, time, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import datapath
from subcell import subcell_modes
MAT = [(1.0, 0.3), (1e-6, 0.3)]; N_MODES = 6
RES = int(sys.argv[1]) if len(sys.argv) > 1 else 16
OUT = f"/mnt/user-data/outputs/descriptors_{RES}"
FAMS = ["circle","square","cross","star_hole","star6_hole","tri_hole","hex_hole",
        "diamond_hole","ellipse","rect_hole","slot_pair","two_holes","cross_aniso",
        "kagome","square_lattice","diag_lattice","rand_lattice","honeycomb",
        "rect_lattice","chevron","reentrant","layered"]

def down(a, f):
    n = a.shape[0]//f
    return (a.reshape(n, f, n, f).mean(axis=(1,3)) > 0.5).astype(int)

os.makedirs(OUT, exist_ok=True)
print(f"descriptors at {RES}x{RES}\n", flush=True)
for fam in FAMS:
    fo, fc = f"{OUT}/omega_{fam}.npy", f"{OUT}/char_{fam}.npy"
    if os.path.exists(fo) and os.path.exists(fc):
        print(f"  {fam:16s} cached", flush=True); continue
    X = datapath.load(f"clean48_{fam}")["X"]
    OM = np.zeros((len(X), N_MODES)); CH = np.zeros((len(X), N_MODES, 3)); t0 = time.time()
    for i in range(len(X)):
        ph = (X[i] == 0).astype(int)
        if RES != 48: ph = down(ph, 48//RES)
        try:
            o, c, _ = subcell_modes(ph, MAT, n_modes=N_MODES); OM[i], CH[i] = o, c
        except Exception:
            OM[i] = np.nan; CH[i] = np.nan
    np.save(fo, OM); np.save(fc, CH)
    print(f"  {fam:16s} {(time.time()-t0)/60:5.2f} min  w2 med {np.nanmedian(OM[:,0]):.3f}"
          f"  failed {int(np.isnan(OM[:,0]).sum())}", flush=True)
print("DONE")
