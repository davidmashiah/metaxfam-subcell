"""verify_full.py -- the sub-cell paper's claims across ALL four stiffness components,
two model seeds and 30 subsets, on the verified-clean MetaXFam-D. The earlier check used
C22 only with 20 subsets and one seed, which was too thin to act on."""
import json, os, sys, numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import RidgeCV
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import r2_score
sys.path.insert(0,'/mnt/user-data/outputs/p3/code')
DATA="/mnt/user-data/outputs/metaxfam_d"; DESC="/mnt/user-data/outputs/p4/descriptors_clean24"
OUT="/mnt/user-data/outputs/p4/results/verify_full.json"
FAMS=sorted({f.replace("clean48_","").replace("__X.npy","") for f in os.listdir(DATA) if f.endswith("__X.npy")})
COL={"C11":0,"C12":1,"C22":3,"C33":5}
X={f:np.load(f"{DATA}/clean48_{f}__X.npy").reshape(-1,2304).astype(np.float32) for f in FAMS}
YA={f:np.load(f"{DATA}/clean48_{f}__y.npy") for f in FAMS}
FR={f:np.load(f"{DATA}/clean48_{f}__frac.npy")[:,None].astype(np.float32) for f in FAMS}
M2={f:np.load(f"{DESC}/{f}__m2.npy").astype(np.float32) for f in FAMS}
SD={}
for f in FAMS:
    s=np.log(np.clip(np.load(f"{DESC}/{f}__sdna.npy"),1e-9,None)); b=~np.isfinite(s).all(1)
    if b.any(): s[b]=np.nanmedian(s[~b],0)
    SD[f]=s.astype(np.float32)
ARMS={"pixels":lambda f:X[f],"frac":lambda f:FR[f],
      "frac+sdna":lambda f:np.hstack([FR[f],SD[f]]),
      "frac+mode2":lambda f:np.hstack([FR[f],M2[f]])}
def LEARN(seed):
    return {"RF":lambda:RandomForestRegressor(n_estimators=60,random_state=seed,max_features=0.3,min_samples_leaf=2,n_jobs=1),
            "ridge":lambda:make_pipeline(StandardScaler(with_mean=False),RidgeCV(alphas=np.logspace(-2,4,13))),
            "MLP":lambda:make_pipeline(StandardScaler(with_mean=False),MLPRegressor(hidden_layer_sizes=(64,64),max_iter=800,random_state=seed,early_stopping=True))}
rng=np.random.default_rng(7); subs=[]
while len(subs)<30:
    c=tuple(sorted(rng.choice(FAMS,4,replace=False)))
    if c not in subs: subs.append(c)
R=json.load(open(OUT)) if os.path.exists(OUT) else {}
for tname,col in COL.items():
    for seed in (0,1):
        for ln in ("RF","ridge","MLP"):
            for arm,fn in ARMS.items():
                k=f"{tname}|{seed}|{ln}|{arm}"
                if k in R: continue
                v=[]
                for s in subs:
                    tr=list(s); te=[f for f in FAMS if f not in s]
                    Xt=np.vstack([fn(f) for f in tr]); yt=np.concatenate([YA[f][:,col] for f in tr])
                    Xe=np.vstack([fn(f) for f in te]); ye=np.concatenate([YA[f][:,col] for f in te])
                    v.append(float(r2_score(ye,LEARN(seed)[ln]().fit(Xt,yt).predict(Xe))))
                R[k]=v; json.dump(R,open(OUT+".tmp","w")); os.replace(OUT+".tmp",OUT)
        print(f"  done {tname} seed{seed}",flush=True)
print("\n"+"="*78)
print("CLAIM 1: frac+mode-2 vs raw pixels   (paper: +0.364 gain, wins 100%)")
print("="*78)
print(f"{'target':8s}{'learner':8s}{'mode2':>9s}{'pixels':>9s}{'gain':>9s}{'wins':>7s}")
for t in COL:
    for ln in ("RF","ridge","MLP"):
        a=np.array(R[f"{t}|0|{ln}|frac+mode2"]+R[f"{t}|1|{ln}|frac+mode2"])
        p=np.array(R[f"{t}|0|{ln}|pixels"]+R[f"{t}|1|{ln}|pixels"])
        print(f"{t:8s}{ln:8s}{np.median(a):+9.3f}{np.median(p):+9.3f}{np.median(a-p):+9.3f}{np.mean(a>p):7.0%}")
print("\n"+"="*78); print("CLAIM 2: frac+mode-2 vs shape-DNA   (paper: +0.250 gain, wins 96%)"); print("="*78)
print(f"{'target':8s}{'learner':8s}{'mode2':>9s}{'sdna':>9s}{'gain':>9s}{'wins':>7s}")
for t in COL:
    for ln in ("RF","ridge","MLP"):
        a=np.array(R[f"{t}|0|{ln}|frac+mode2"]+R[f"{t}|1|{ln}|frac+mode2"])
        s=np.array(R[f"{t}|0|{ln}|frac+sdna"]+R[f"{t}|1|{ln}|frac+sdna"])
        print(f"{t:8s}{ln:8s}{np.median(a):+9.3f}{np.median(s):+9.3f}{np.median(a-s):+9.3f}{np.mean(a>s):7.0%}")
json.dump(R,open(OUT,"w")); print("\nDONE")
