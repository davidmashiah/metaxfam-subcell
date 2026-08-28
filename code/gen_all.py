"""gen_all.py -- regenerate all 22 MetaXFam22 families under the gen_clean protocol.
Skips families already present in ../data_clean (checkpoint per family)."""
import _path  # noqa: F401  (adds code/ and code/vendor/ to sys.path)

import os, sys, time
import numpy as np
import gen_clean as GC
from families_new import GEN_NEW
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data_clean")
GEN = dict(GC.GEN); GEN.update(GEN_NEW)
ORDER = ["circle","square","cross","star_hole","ellipse","rect_hole","slot_pair",
         "kagome","honeycomb","rand_lattice","tri_hole","hex_hole","diamond_hole",
         "star6_hole","two_holes","cross_aniso","square_lattice","diag_lattice",
         "rect_lattice","chevron","reentrant","layered"]
assert set(ORDER)==set(GEN), set(ORDER)^set(GEN)
GC.GEN = GEN
if __name__=="__main__":
    for nm in ORDER:
        if os.path.exists(f"{OUT}/clean48_{nm}__y.npy"):
            print(f"  {nm:14s} exists, skip", flush=True); continue
        GC.generate(nm, nwant=600, outdir=OUT)
    print("ALL DONE", flush=True)
