"""robust_k.py -- descriptor-only arms at K=1 (22 singletons) and K=2 (3x30 pairs), 2 seeds."""
import _path  # noqa: F401  (adds code/ and code/vendor/ to sys.path)

import json, os, sys, numpy as np
sys.path.insert(0,'.')
import ablate2 as A, robust as RB
from run_feature2 import FAMS
OUT="/mnt/user-data/outputs/p3/results/robust_k.json"
sel=A.selection(); R=json.load(open(OUT)) if os.path.exists(OUT) else {}
def subsK(K):
    if K==1: return [(f,) for f in FAMS]
    out=[]
    for r in (7,8,9):
        rng=np.random.default_rng(r); out+=[tuple(sorted(rng.choice(FAMS,K,replace=False))) for _ in range(30)]
    return out
for K in (1,2):
    subs=subsK(K)
    for s in ("v2_24grey","v2_48","orig_48"):
        C=A.load(RB.SETS[s],sel)
        for arm in ("frac","mode2","frac+mode2"):
            key=f"K{K}|{s if arm!='frac' else 'shared'}|{arm}"
            if key in R: continue
            r22,r11=RB.run(C,arm,subs); R[key]={"C22":r22,"C11":r11}
            print(f"  {key:28s} C22 median {np.median(r22):+.3f} IQR [{np.percentile(r22,25):+.3f},{np.percentile(r22,75):+.3f}]  C11 {np.median(r11):+.3f}",flush=True)
            json.dump(R,open(OUT+".tmp","w")); os.replace(OUT+".tmp",OUT)
print("DONE",flush=True)
