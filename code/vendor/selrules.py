"""
selrules.py -- candidate rules for choosing WHICH K design families to train on.

Every rule here is CHEAP: it uses either
  (a) the unit-cell IMAGES only (no finite-element solve at all), or
  (b) a PILOT of N_PILOT=20 homogenisations per candidate family
      (~4 s/family, vs 400 solves/family to actually build a training set).

Two of the rules are PRIOR ART and are included as baselines, not as
contributions:
  * mmd   -- maximum-mean-discrepancy source-domain selection.  Same idea as
             MMD-VD (Eng. Res. Express 2025, bearing-fault domain generalization),
             which selects which source domains to include by MMD.
  * wass  -- sliced-Wasserstein distance between train and test input
             distributions, the transferability metric of MRS Communications
             2025 ("Constructing generalizable microstructure-property maps").
  * maxdiv-- greedy max-min geometric diversity, the standard heuristic
             (PATO-style).
"""
import itertools
import numpy as np

import selcore as SC

N_PILOT = 20


# ---------------------------------------------------------------------------
# image-only family descriptors
# ---------------------------------------------------------------------------
_DESC = {}
def descriptor(fam, images):
    """Cheap directional descriptor of a family's typical geometry, averaged
    over the family: solid fraction, row/col mass anisotropy, diagonal
    asymmetry, radial profile."""
    if fam in _DESC:
        return _DESC[fam]
    feats = []
    for img in images[:200]:
        img = img.astype(float)
        n = img.shape[0]
        row, col = img.mean(1), img.mean(0)
        yy, xx = np.mgrid[0:n, 0:n]
        r = np.hypot(xx - n / 2, yy - n / 2) / (n / 2)
        rad = [img[(r >= a) & (r < a + 0.2)].mean() if np.any((r >= a) & (r < a + 0.2)) else 0.
               for a in np.arange(0, 1.0, 0.2)]
        aniso = abs(row.var() - col.var())
        dg = np.mean([img[i, i] for i in range(n)])
        ad = np.mean([img[i, n - 1 - i] for i in range(n)])
        feats.append(np.concatenate([[img.mean(), aniso, abs(dg - ad)], rad]))
    _DESC[fam] = np.mean(feats, axis=0)
    return _DESC[fam]


class Ctx:
    """Precomputes everything the rules need, once."""
    def __init__(self, pool, target="C11", images=None):
        self.pool = list(pool)
        self.target = target
        # --- image-only descriptor space
        D = np.array([descriptor(f, images[f]) for f in self.pool])
        self.desc = (D - D.mean(0)) / (D.std(0) + 1e-9)
        self.di = {f: i for i, f in enumerate(self.pool)}
        # --- PCA features (image-only) for MMD / Wasserstein
        self.Z = {f: SC.feats(f) for f in self.pool}
        allZ = np.vstack([self.Z[f] for f in self.pool])
        # median heuristic bandwidth
        sub = allZ[np.random.default_rng(0).permutation(len(allZ))[:600]]
        d2 = ((sub[:, None, :] - sub[None, :, :]) ** 2).sum(-1)
        self.gamma = 1.0 / (np.median(d2[d2 > 0]) + 1e-12)
        # --- pilot FE quantities (N_PILOT cells per family)
        self.aniso, self.rmu, self.rsd = {}, {}, {}
        for f in self.pool:
            y = SC.yall(f)[:N_PILOT]
            c11, c22 = y[:, 0], y[:, 3]
            self.aniso[f] = float(np.median(np.abs(c11 - c22) / (c11 + c22)))
            t = SC.targ(f, target)[:N_PILOT]
            self.rmu[f], self.rsd[f] = float(t.mean()), float(t.std())

    def gdist(self, a, b):
        return float(np.linalg.norm(self.desc[self.di[a]] - self.desc[self.di[b]]))

    def mmd2(self, a, Bfams):
        """Squared MMD between family a and the union of Bfams (Gaussian kernel)."""
        rng = np.random.default_rng(0)
        A = self.Z[a][:150]
        B = np.vstack([self.Z[b] for b in Bfams])
        B = B[rng.permutation(len(B))[:300]]
        def k(P, Q):
            d2 = ((P[:, None, :] - Q[None, :, :]) ** 2).sum(-1)
            return np.exp(-self.gamma * d2).mean()
        return float(k(A, A) + k(B, B) - 2 * k(A, B))

    def sw(self, a, Bfams, n_proj=64):
        """Sliced-Wasserstein-1 between family a and the union of Bfams."""
        rng = np.random.default_rng(0)
        A = self.Z[a]
        B = np.vstack([self.Z[b] for b in Bfams])
        d = A.shape[1]
        P = rng.normal(size=(d, n_proj)); P /= np.linalg.norm(P, axis=0, keepdims=True)
        pa = np.sort(A @ P, axis=0); pb = np.sort(B @ P, axis=0)
        q = np.linspace(0, 1, 100)
        qa = np.quantile(pa, q, axis=0); qb = np.quantile(pb, q, axis=0)
        return float(np.abs(qa - qb).mean())


# ---------------------------------------------------------------------------
# the rules.  each returns a tuple of K family names.
# ---------------------------------------------------------------------------
def _best_subset(ctx, K, cost):
    """Exhaustive minimisation of a CHEAP cost over K-subsets."""
    best, bc = None, np.inf
    for c in itertools.combinations(ctx.pool, K):
        v = cost(c)
        if v < bc:
            bc, best = v, c
    return best


def rule_maxdiv(ctx, K):
    """Greedy max-min geometric diversity (standard heuristic)."""
    pair = max(itertools.combinations(ctx.pool, 2), key=lambda ab: ctx.gdist(*ab))
    chosen = list(pair)[:K]
    rem = [f for f in ctx.pool if f not in chosen]
    while len(chosen) < K:
        nxt = max(rem, key=lambda f: min(ctx.gdist(f, c) for c in chosen))
        chosen.append(nxt); rem.remove(nxt)
    return tuple(sorted(chosen))


def rule_kcenter_geo(ctx, K):
    """k-center in geometric descriptor space: every unseen family close to a
    chosen one."""
    def cost(c):
        un = [f for f in ctx.pool if f not in c]
        return max(min(ctx.gdist(u, x) for x in c) for u in un) if un else 0.
    return _best_subset(ctx, K, cost)


def rule_mmd(ctx, K):
    """PRIOR ART: minimise the worst MMD from any unseen family to the training
    union (MMD-VD-style source-domain selection)."""
    def cost(c):
        un = [f for f in ctx.pool if f not in c]
        return max(ctx.mmd2(u, c) for u in un) if un else 0.
    return _best_subset(ctx, K, cost)


def rule_wass(ctx, K):
    """PRIOR ART: minimise the worst sliced-Wasserstein input distance from any
    unseen family to the training union."""
    def cost(c):
        un = [f for f in ctx.pool if f not in c]
        return max(ctx.sw(u, c) for u in un) if un else 0.
    return _best_subset(ctx, K, cost)


def rule_aniso_top(ctx, K):
    """PILOT-FE: the K most anisotropic families (anisotropy MAGNITUDE)."""
    return tuple(sorted(sorted(ctx.pool, key=lambda f: -ctx.aniso[f])[:K]))


def rule_aniso_cover(ctx, K):
    """PILOT-FE: k-center on the 1-D anisotropy axis (anisotropy COVERAGE)."""
    def cost(c):
        un = [f for f in ctx.pool if f not in c]
        return max(min(abs(ctx.aniso[u] - ctx.aniso[x]) for x in c) for u in un) if un else 0.
    return _best_subset(ctx, K, cost)


def rule_resp_cover(ctx, K):
    """PILOT-FE: k-center in RESPONSE space (mean, sd of the target)."""
    v = {f: np.array([ctx.rmu[f], ctx.rsd[f]]) for f in ctx.pool}
    V = np.array([v[f] for f in ctx.pool]); mu, sd = V.mean(0), V.std(0) + 1e-12
    v = {f: (v[f] - mu) / sd for f in ctx.pool}
    def cost(c):
        un = [f for f in ctx.pool if f not in c]
        return max(min(np.linalg.norm(v[u] - v[x]) for x in c) for u in un) if un else 0.
    return _best_subset(ctx, K, cost)


def rule_resp_span(ctx, K):
    """PILOT-FE: maximise coverage of the target's VALUE RANGE -- the training
    families' [mu-sd, mu+sd] intervals should cover every unseen family's mean."""
    def cost(c):
        lo = min(ctx.rmu[x] - ctx.rsd[x] for x in c)
        hi = max(ctx.rmu[x] + ctx.rsd[x] for x in c)
        un = [f for f in ctx.pool if f not in c]
        # penalty = worst distance of an unseen mean outside the covered interval
        return max(max(lo - ctx.rmu[u], ctx.rmu[u] - hi, 0.) for u in un) if un else 0.
    return _best_subset(ctx, K, cost)


def rule_hardest(ctx, K):
    """PILOT-FE: the K families with the largest target spread (most
    information-rich), rather than the most spread-out families."""
    return tuple(sorted(sorted(ctx.pool, key=lambda f: -ctx.rsd[f])[:K]))


RULES = {
    "maxdiv":       rule_maxdiv,
    "kcenter_geo":  rule_kcenter_geo,
    "mmd":          rule_mmd,
    "wass":         rule_wass,
    "aniso_top":    rule_aniso_top,
    "aniso_cover":  rule_aniso_cover,
    "resp_cover":   rule_resp_cover,
    "resp_span":    rule_resp_span,
    "hardest":      rule_hardest,
}
PRIOR_ART = {"maxdiv", "mmd", "wass"}
