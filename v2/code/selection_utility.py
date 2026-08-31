"""
selection_utility.py -- from correlation to a usable rule.

WHERE WE ARE.  Wasserstein distance between the training pool and a held-out family
predicts transfer to that family on MetaXFam-D: 18/18 settings negative, median
rho = -0.484, and the signal is NOT a density artefact (partialling out solid fraction
leaves rho unchanged at -0.495; Wasserstein and the density gap correlate at only +0.10).
Liu et al. (MRS Commun. 16:449-458, 2026) are substantially right, and the result extends
to a de-confounded 18-family metamaterial pool.

BUT A CORRELATION IS NOT A RULE.  The practical question -- the one the training-family
study set out to answer and failed to -- is whether a criterion computed BEFORE training
lets you choose a training set that transfers better than chance. rho ~ -0.5 explains about
a quarter of the rank variance, which may or may not be enough to select on. This script
settles that.

TWO PARTS.

PART A -- CRITERION PANEL. Wasserstein is compared against six alternatives on identical
data, so "Wasserstein is best" is a measured claim rather than an assumption:
    W_pixels, W_twopoint, W_mode2   sliced Wasserstein in three representations
    MMD_pixels                      maximum mean discrepancy, RBF kernel
    energy_pixels                   energy distance
    frac_gap                        difference in median solid fraction (free control)
    coverage                        fraction of held-out cells inside the training support

PART B -- SELECTION UTILITY. For each candidate training subset we compute a criterion
score using ONLY the training families and the unlabelled pool -- no held-out targets, so
it is honestly prospective. The rule is k-centre: pick the subset minimising the WORST
Wasserstein distance to any family not in it. Then:

    picked      transfer achieved by the subset the rule selects
    oracle      best achievable among the candidates
    random      median over candidates (what you get with no rule)
    worst       what an unlucky pick costs you

The rule is worth having only if `picked` beats `random` reliably -- especially on the
WORST-CASE held-out family, which is what actually matters when deploying a surrogate.
"""
import json, os, time
import numpy as np
from scipy.stats import wasserstein_distance, spearmanr
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.decomposition import PCA

DATA = "/mnt/user-data/outputs/metaxfam_d"
OUT = "/mnt/user-data/outputs/p4/results/selection_utility.json"
os.makedirs(os.path.dirname(OUT), exist_ok=True)

FAMS = sorted({f.replace("clean48_", "").replace("__X.npy", "")
               for f in os.listdir(DATA) if f.endswith("__X.npy")})
X = {f: np.load(f"{DATA}/clean48_{f}__X.npy").reshape(-1, 48 * 48).astype(np.float32)
     for f in FAMS}
Yv = {f: np.load(f"{DATA}/clean48_{f}__y.npy")[:, 3] for f in FAMS}          # C22
FR = {f: np.load(f"{DATA}/clean48_{f}__frac.npy") for f in FAMS}


def two_point(a, nbins=16):
    n = 48; a = a.reshape(-1, n, n)
    F = np.fft.rfft2(a - a.mean(axis=(1, 2), keepdims=True))
    C = np.fft.irfft2(F * np.conj(F), s=(n, n)).reshape(len(a), -1) / (n * n)
    yy, xx = np.mgrid[0:n, 0:n]
    r = np.hypot(np.minimum(xx, n - xx), np.minimum(yy, n - yy)).ravel()
    e = np.linspace(0, n / 2, nbins + 1)
    b = np.clip(np.digitize(r, e) - 1, 0, nbins - 1)
    return np.stack([C[:, b == k].mean(1) for k in range(nbins)], 1)


TP = {f: two_point(X[f]) for f in FAMS}
_p = PCA(n_components=16, random_state=0).fit(np.vstack([X[f] for f in FAMS]))
PC = {f: _p.transform(X[f]) for f in FAMS}


def sliced_w(A, B, n_proj=120, seed=0):
    rng = np.random.default_rng(seed)
    mu, sd = A.mean(0), A.std(0) + 1e-9
    A = (A - mu) / sd; B = (B - mu) / sd
    V = rng.normal(size=(A.shape[1], n_proj)); V /= np.linalg.norm(V, axis=0, keepdims=True)
    PA, PB = A @ V, B @ V
    return float(np.mean([wasserstein_distance(PA[:, i], PB[:, i]) for i in range(n_proj)]))


def mmd_rbf(A, B, gamma=None, m=250, seed=0):
    rng = np.random.default_rng(seed)
    A = A[rng.permutation(len(A))[:m]]; B = B[rng.permutation(len(B))[:m]]
    mu, sd = A.mean(0), A.std(0) + 1e-9
    A = (A - mu) / sd; B = (B - mu) / sd
    if gamma is None:
        gamma = 1.0 / A.shape[1]
    def k(P, Q):
        d = ((P[:, None, :] - Q[None, :, :]) ** 2).sum(-1)
        return np.exp(-gamma * d).mean()
    return float(k(A, A) + k(B, B) - 2 * k(A, B))


def energy_dist(A, B, m=250, seed=0):
    rng = np.random.default_rng(seed)
    A = A[rng.permutation(len(A))[:m]]; B = B[rng.permutation(len(B))[:m]]
    mu, sd = A.mean(0), A.std(0) + 1e-9
    A = (A - mu) / sd; B = (B - mu) / sd
    def d(P, Q):
        return np.sqrt(((P[:, None, :] - Q[None, :, :]) ** 2).sum(-1)).mean()
    return float(2 * d(A, B) - d(A, A) - d(B, B))


def coverage(A, B, k=5):
    from sklearn.neighbors import NearestNeighbors
    mu, sd = A.mean(0), A.std(0) + 1e-9
    A = (A - mu) / sd; B = (B - mu) / sd
    nn = NearestNeighbors(n_neighbors=k + 1).fit(A)
    din = nn.kneighbors(A)[0][:, k]
    dout = nn.kneighbors(B, n_neighbors=k)[0][:, k - 1]
    return float(np.mean(dout <= np.percentile(din, 95)))


def mk():
    return RandomForestRegressor(n_estimators=60, random_state=0, max_features=0.3,
                                 min_samples_leaf=2, n_jobs=1)


def main():
    R = json.load(open(OUT)) if os.path.exists(OUT) else {}
    rng = np.random.default_rng(11)
    NSUB = 60
    subs = []
    while len(subs) < NSUB:
        c = tuple(sorted(rng.choice(FAMS, 4, replace=False)))
        if c not in subs:
            subs.append(c)

    if "records" not in R:
        recs = []
        t0 = time.time()
        for si, sub in enumerate(subs):
            tr = list(sub); te = [f for f in FAMS if f not in sub]
            Xtr = np.vstack([X[f] for f in tr]); ytr = np.concatenate([Yv[f] for f in tr])
            m = mk().fit(Xtr, ytr)
            TPtr = np.vstack([TP[f] for f in tr]); PCtr = np.vstack([PC[f] for f in tr])
            frtr = np.median(np.concatenate([FR[f] for f in tr]))
            per = {}
            for f in te:
                per[f] = dict(
                    r2=float(r2_score(Yv[f], m.predict(X[f]))),
                    W_pix=sliced_w(PCtr, PC[f]),
                    W_tp=sliced_w(TPtr, TP[f]),
                    MMD=mmd_rbf(PCtr, PC[f]),
                    energy=energy_dist(PCtr, PC[f]),
                    frac=abs(float(np.median(FR[f])) - frtr),
                    cov=coverage(PCtr, PC[f]),
                )
            recs.append({"set": tr, "per": per})
            if si % 10 == 0:
                print(f"    subset {si+1}/{NSUB}  ({time.time()-t0:.0f}s)", flush=True)
        R["records"] = recs
        json.dump(R, open(OUT, "w"))
    recs = R["records"]

    # ---------------- PART A: criterion panel ----------------
    print("\n" + "=" * 72)
    print("PART A  which criterion predicts transfer? (per held-out family, n=%d)"
          % sum(len(r["per"]) for r in recs))
    print("=" * 72)
    keys = ["W_pix", "W_tp", "MMD", "energy", "cov", "frac"]
    r2all = np.array([v["r2"] for r in recs for v in r["per"].values()])
    panel = {}
    for k in keys:
        x = np.array([v[k] for r in recs for v in r["per"].values()])
        rho = float(spearmanr(x, r2all).statistic)
        panel[k] = rho
        print(f"  {k:10s} rho = {rho:+.3f}")
    R["panel"] = panel

    # ---------------- PART B: selection utility ----------------
    print("\n" + "=" * 72)
    print("PART B  does it let you CHOOSE a training set? (k-centre on Wasserstein)")
    print("=" * 72)
    out = {}
    for k in ["W_pix", "W_tp", "MMD", "cov", "frac"]:
        score = []
        for r in recs:
            v = [p[k] for p in r["per"].values()]
            score.append(max(v) if k != "cov" else -min(v))   # worst-case distance
        score = np.array(score)
        mean_r2 = np.array([np.mean([p["r2"] for p in r["per"].values()]) for r in recs])
        worst_r2 = np.array([min(p["r2"] for p in r["per"].values()) for r in recs])
        pick = int(np.argmin(score))
        out[k] = dict(pick_mean=float(mean_r2[pick]), pick_worst=float(worst_r2[pick]),
                      rand_mean=float(np.median(mean_r2)), rand_worst=float(np.median(worst_r2)),
                      oracle_mean=float(mean_r2.max()), oracle_worst=float(worst_r2.max()),
                      pct_mean=float((mean_r2 < mean_r2[pick]).mean()),
                      pct_worst=float((worst_r2 < worst_r2[pick]).mean()))
        o = out[k]
        print(f"\n  criterion {k}")
        print(f"    mean-R2 :  picked {o['pick_mean']:+.3f}   random {o['rand_mean']:+.3f}"
              f"   oracle {o['oracle_mean']:+.3f}   (picked beats {o['pct_mean']:.0%} of subsets)")
        print(f"    worst-R2:  picked {o['pick_worst']:+.3f}   random {o['rand_worst']:+.3f}"
              f"   oracle {o['oracle_worst']:+.3f}   (picked beats {o['pct_worst']:.0%} of subsets)")
    R["selection"] = out
    json.dump(R, open(OUT, "w"))
    print("\n  A criterion is USEFUL only if 'picked' sits well above 'random', especially")
    print("  on worst-case R2. Percentile near 50% means the rule is no better than chance.")
    print("\nDONE")


if __name__ == "__main__":
    main()
