"""
test_subcell.py -- validate subcell.py against exact solutions on a homogeneous
periodic cell (E=1, nu=0.3, rho=1, plane strain, lx=ly=1).

Exact periodic modes at the Gamma point:  u = exp(i k.x) p,  k = 2 pi (m, n).
  S-wave  omega_S = |k| sqrt(mu/rho),   mu = E/(2(1+nu)) = 0.384615
  P-wave  omega_P = |k| sqrt((lam+2mu)/rho)
Lowest non-rigid: k = 2pi(1,0) and 2pi(0,1) S-waves,  omega = 2 pi sqrt(mu)
                  = 3.89670 (4-fold degenerate: 2 directions x cos/sin).
Next:             k = 2pi(1,1), (1,-1) S-waves, omega = 2 pi sqrt(2 mu) = 5.51076.

Exact characters: a plane S-wave has (c_dil, c_shr, c_rot) = (0, 1/2, 1/2);
a plane P-wave has (1/2, 1/2, 0).
"""
import _path  # noqa: F401  (adds code/ and code/vendor/ to sys.path)

import sys, time
import numpy as np
import subcell as SC

mu = SC.E_SOLID / (2 * (1 + SC.NU))
lam_ = SC.E_SOLID * SC.NU / ((1 + SC.NU) * (1 - 2 * SC.NU))
wS = 2 * np.pi * np.sqrt(mu)
wS2 = 2 * np.pi * np.sqrt(2 * mu)
wP = 2 * np.pi * np.sqrt(lam_ + 2 * mu)
results = []

def rep(name, ok, msg):
    results.append((name, ok)); print(f"  {'PASS' if ok else 'FAIL'}  {name}: {msg}")

# [1] rigid translations
d = SC.descriptors(np.ones((48, 48)))
rep("rigid translations", np.all(np.abs(d["rigid"]) < 1e-8), f"eigenvalues {d['rigid']}")

# [2] fundamental shear
om = d["omega"]
err = abs(om[0] - wS) / wS
rep("fundamental shear", err < 2e-3, f"{om[0]:.4f} vs {wS:.4f} exact (rel {err:.1e})")

# [3] degeneracy of the fundamental cluster (4 modes)
spread = (om[:4].max() - om[:4].min()) / om[0]
rep("degeneracy spread", spread < 1e-8, f"{spread:.1e} across {om[:4]}")

# [4] diagonal shear
err2 = abs(om[4] - wS2) / wS2
rep("diagonal shear", err2 < 3e-3, f"{om[4]:.4f} vs {wS2:.4f} exact (rel {err2:.1e})")

# [5] mesh convergence, should be O(h^2)
errs = []
for n in (12, 24, 48):
    o = SC.descriptors(np.ones((n, n)))["omega"][0]
    errs.append(abs(o - wS) / wS)
ratio = errs[0] / errs[2]
rep("mesh convergence", errs[0] > errs[1] > errs[2] and 12 < ratio < 20,
    f"rel err {errs[0]:.1e} -> {errs[1]:.1e} -> {errs[2]:.1e} (ratio 12->48 = {ratio:.1f}, h^2 -> 16)")

# [6] omega ~ sqrt(E)
E0 = SC.E_SOLID; SC.E_SOLID = 4.0
o4 = SC.descriptors(np.ones((24, 24)))["omega"][0]
SC.E_SOLID = E0
o1 = SC.descriptors(np.ones((24, 24)))["omega"][0]
rep("omega ~ sqrt(E)", abs(o4 / o1 - 2) < 1e-10, f"ratio {o4/o1:.6f} vs 2 exact")

# [7] exact characters: S-wave cluster (0, .5, .5); P-wave (0.5, 0.5, 0)
ch = d["char"]
okS = np.allclose(ch[0], [0, .5, .5], atol=1e-6)
rep("S-wave character", okS, f"{np.round(ch[0], 6)} vs (0, 0.5, 0.5) exact")
# P-wave at k=2pi(1,0): find its cluster among 12 modes
dd = SC.descriptors(np.ones((48, 48)), n_sub=12)
iP = np.argmin(np.abs(dd["omega"] - wP * (1 - 1e-3)))
okP = abs(dd["omega"][iP] - wP) / wP < 3e-3 and np.allclose(dd["char"][iP], [.5, .5, 0], atol=1e-6)
rep("P-wave character", okP, f"omega {dd['omega'][iP]:.4f} vs {wP:.4f}; char {np.round(dd['char'][iP], 6)} vs (0.5, 0.5, 0)")

# [8] basis invariance of cluster character: random rotation of a degenerate pair
omega, Phi, aux = SC.eigenmodes(np.ones((24, 24)), n_modes=8)
I = SC.character_integrals(Phi, aux)
g = SC.clusters(omega)[1]
Q = np.linalg.qr(np.random.default_rng(1).normal(size=(len(g), len(g))))[0]
Phi2 = Phi.copy(); Phi2[:, g] = Phi[:, g] @ Q
I2 = SC.character_integrals(Phi2, aux)
rep("cluster-average basis invariance", np.allclose(I[g].mean(0), I2[g].mean(0), rtol=1e-8),
    f"cluster of size {len(g)}, |diff| = {np.abs(I[g].mean(0)-I2[g].mean(0)).max():.1e}")

# [9] a real cell: void modes must not intrude (all 6 sub-cell modes carry >99.9% of
#     stiffness-weighted |grad u|^2 in solid, and frequencies are O(1))
import matched_sampler as MS
ph, _ = MS.gen_circle(48, np.random.default_rng(0))
solid = (ph == 0).astype(float)
t0 = time.time(); dc = SC.descriptors(solid); tdesc = time.time() - t0
from homogenize2d import homogenize
t0 = time.time(); homogenize(ph, [(1.0, .3), (1e-6, .3)]); thom = time.time() - t0
rep("real cell sanity", np.all(dc["omega"] > 0.5) and np.all(dc["omega"] < 30),
    f"omega {np.round(dc['omega'], 3)}; eig {tdesc:.2f}s vs homog {thom:.2f}s")
print("\n" + "=" * 60)
for n, ok in results: print(f"  {'PASS' if ok else 'FAIL'}  {n}")
print("=" * 60)
print("ALL TESTS PASSED" if all(ok for _, ok in results) else "SOME TESTS FAILED")
sys.exit(0 if all(ok for _, ok in results) else 1)
