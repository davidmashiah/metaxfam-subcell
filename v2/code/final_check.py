"""final_check.py -- pre-registered summary. Aggregation rules fixed BEFORE reading results,
so the headline cannot drift the way it did between the last two runs.

RULES (fixed in advance):
  R1  A claim is SUPPORTED only if it holds for ALL THREE learners on a given component.
  R2  Report every component. No headline may rest on one component or one learner.
  R3  Uncertainty by bootstrap over the 30 subsets x 2 seeds, 95% CI on the paired median.
  R4  A claim is REJECTED if any learner's 95% CI on the paired gain contains or crosses 0
      in a direction opposite the claim.
"""
import json, numpy as np
R=json.load(open('../results/verify_full.json'))
COL=["C11","C12","C22","C33"]; LN=["RF","ridge","MLP"]
rng=np.random.default_rng(0)

def paired(t,ln,a,b):
    x=np.array(R[f"{t}|0|{ln}|{a}"]+R[f"{t}|1|{ln}|{a}"])
    y=np.array(R[f"{t}|0|{ln}|{b}"]+R[f"{t}|1|{ln}|{b}"])
    d=x-y
    bs=[np.median(d[rng.integers(0,len(d),len(d))]) for _ in range(4000)]
    return np.median(d), np.percentile(bs,2.5), np.percentile(bs,97.5), np.mean(x>y)

for title,a,b,note in [
    ("CLAIM 1  frac+mode-2  vs  raw pixels","frac+mode2","pixels","paper: +0.364, wins 100%"),
    ("CLAIM 2  frac+mode-2  vs  shape-DNA","frac+mode2","frac+sdna","paper: +0.250, wins 96%"),
    ("CLAIM 3  frac+mode-2  vs  solid fraction","frac+mode2","frac","paper: +0.274, wins 97%")]:
    print("="*80); print(title,"  (",note,")"); print("="*80)
    print(f"{'target':7s}{'learner':7s}{'gain':>9s}{'95% CI':>20s}{'wins':>7s}{'verdict':>10s}")
    allok=True
    for t in COL:
        for ln in LN:
            g,lo,hi,w=paired(t,ln,a,b)
            ok = lo>0
            allok &= ok
            print(f"{t:7s}{ln:7s}{g:+9.3f}   [{lo:+7.3f},{hi:+7.3f}]{w:7.0%}{'PASS' if ok else 'FAIL':>10s}")
    print(f"  -> holds for every component and learner: {'YES' if allok else 'NO'}\n")

print("="*80); print("STABILITY ACROSS SURROGATE CLASS  (range of median R2 over 12 cells)"); print("="*80)
for arm in ["pixels","frac","frac+sdna","frac+mode2"]:
    v=[np.median(R[f"{t}|0|{ln}|{arm}"]+R[f"{t}|1|{ln}|{arm}"]) for t in COL for ln in LN]
    print(f"  {arm:12s} min {min(v):+9.3f}   max {max(v):+7.3f}   spread {max(v)-min(v):8.3f}")
