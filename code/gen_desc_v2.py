"""gen_desc_v2.py -- descriptors from the REBUILT extractor (cluster-averaged character,
optional grey coarsening) for all 600 cells/family, in the same layout as descriptors_48/.
Usage: python gen_desc_v2.py <res> <grey|binary> [index_file.npz]
Output: /mnt/user-data/outputs/descriptors_v2_<res><how>/{omega,char}_<fam>.npy
char columns are reordered to the ORIGINAL convention (rot, dil, shr)."""
import _path  # noqa: F401  (adds code/ and code/vendor/ to sys.path)

import os, sys, time, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import subcell as SB
DATA = "/mnt/user-data/outputs/9_PROJECT_STATE/data"
res = int(sys.argv[1]); how = sys.argv[2]
IDX = np.load(sys.argv[3]) if len(sys.argv) > 3 else None
OUT = f"/mnt/user-data/outputs/descriptors_v2_{res}{how}"; os.makedirs(OUT, exist_ok=True)
FAMS = ["circle","square","cross","star_hole","star6_hole","tri_hole","hex_hole",
        "diamond_hole","ellipse","rect_hole","slot_pair","two_holes","cross_aniso",
        "kagome","square_lattice","diag_lattice","rand_lattice","honeycomb",
        "rect_lattice","chevron","reentrant","layered"]
for fam in FAMS:
    fo, fc = f"{OUT}/omega_{fam}.npy", f"{OUT}/char_{fam}.npy"
    if os.path.exists(fo): print(f"  {fam:16s} cached", flush=True); continue
    X = np.load(f"{DATA}/clean48_{fam}__X.npy").astype(float)
    OM = np.full((len(X), 6), np.nan); CH = np.full((len(X), 6, 3), np.nan); t0 = time.time()
    rows = range(len(X)) if IDX is None else IDX[fam]
    for i in rows:
        cell = X[i] if res == 48 else SB.coarsen(X[i], res, how)
        try:
            d = SB.descriptors(cell)
            OM[i] = d["omega"]; CH[i] = d["char"][:, [2, 0, 1]]      # -> (rot, dil, shr)
        except Exception as e:
            pass
    np.save(fo + ".tmp.npy", OM); os.replace(fo + ".tmp.npy", fo)
    np.save(fc + ".tmp.npy", CH); os.replace(fc + ".tmp.npy", fc)
    n = len(list(rows))
    print(f"  {fam:16s} {n} cells {(time.time()-t0)/n*1000:6.1f} ms/cell  w2 med {np.nanmedian(OM[:,0]):.3f}"
          f"  m2 char med {np.round(np.nanmedian(CH[:,0,:],0),3)}  failed {int(np.isnan(OM[:,0][list(rows)]).sum())}", flush=True)
print("DONE", flush=True)
