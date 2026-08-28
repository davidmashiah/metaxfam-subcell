"""
subcell.py -- sub-cell (intra-cell) elastodynamic eigenmodes of a periodic unit cell.

REBUILT 2026-08-27 after the original was lost in a container reset.  Same
physics as the original (validated against exact plane waves, see
test_subcell.py) but the exact numerical definition of "mode character" below is
a reconstruction and is documented here so nothing is hidden.

Problem solved
--------------
    K phi = omega^2 M phi     on the periodic cell (Gamma point, zero Bloch vector)

K is the same periodic Q4 stiffness matrix homogenize2d.py builds (identical
master/slave DOF map), M is the consistent Q4 mass matrix.  Periodicity admits
exactly two rigid modes (the translations); rotation is not periodic-compatible.
The lowest NON-rigid modes are the sub-cell degrees of freedom that stress
volume-averaging discards.

Material model
--------------
Each element carries a solid fraction f in [0,1] (binary cells: f in {0,1};
coarsened "grey" cells: f = block mean).  E_e = f E_s + (1-f) E_v and
rho_e = f rho_s + (1-f) rho_v.  The void is BOTH nearly stiffness-less
(E_v = 1e-6, same as the homogenisation) AND nearly mass-less (rho_v = 1e-12),
so the void's own cavity modes sit at omega ~ sqrt(E_v/rho_v) ~ 1e3, far above
the solid spectrum, and cannot contaminate the lowest modes.

Mode character (reconstruction -- see note above)
-------------------------------------------------
For an eigenvector phi the displacement gradient at each Gauss point splits
exactly as

    |grad u|_F^2 = 1/2 d^2 + 1/2 [(exx-eyy)^2 + gxy^2] + 2 w^2
                   dilatation  deviatoric shear             rotation

with d = exx+eyy, gxy = du/dy+dv/dx, w = 1/2 (dv/dx - du/dy).  Each term is
integrated over the cell with the element stiffness weight E_e (so the void
does not count) and the three integrals are normalised to sum to one:

    char = (c_dil, c_shr, c_rot),  sum = 1.

Degeneracy: an eigenvector inside a degenerate cluster is only defined up to a
rotation within the cluster, and char is quadratic in phi, so per-vector chars
are basis-dependent.  We therefore average the three integrals over all members
of the cluster (a trace over the eigenspace), which is basis-invariant.  Cluster
= eigenvalues within rel. tol. 1e-4 (ARPACK scatters exact degeneracies at ~1e-6..1e-5).  "Mode k" refers to the k-th sorted
eigenvalue (0,1 = rigid); its character is that of its cluster.
"""
import _path  # noqa: F401  (adds code/ and code/vendor/ to sys.path)

import numpy as np
from scipy.sparse import coo_matrix, csc_matrix
from scipy.sparse.linalg import eigsh

from homogenize2d import constitutive_matrix

E_SOLID, NU, E_VOID = 1.0, 0.3, 1e-6
RHO_SOLID, RHO_VOID = 1.0, 1e-12
N_MODES = 8            # lowest eigenpairs to compute (2 rigid + 6 sub-cell)
CLUSTER_RTOL = 1e-4
N_SPARE = 6            # extra modes computed so the last reported cluster is never truncated


# ---------------------------------------------------------------------------
def _q4_unit(dx, dy, nu, mode):
    """Unit-E stiffness Ke (8x8), consistent unit-density mass Me (8x8), and the
    Gauss-point B matrices (4,3,8) plus rotation rows R (4,8) with
    w = R @ u_e = 1/2 (dv/dx - du/dy)."""
    D = constitutive_matrix(1.0, nu, mode)
    g = 1.0 / np.sqrt(3.0)
    gauss = [(-g, -g), (g, -g), (g, g), (-g, g)]
    detJ = dx * dy / 4.0
    Ke = np.zeros((8, 8)); Me = np.zeros((8, 8))
    Bs = np.zeros((4, 3, 8)); Rs = np.zeros((4, 8))
    for q, (xi, eta) in enumerate(gauss):
        N = np.array([(1 - xi) * (1 - eta), (1 + xi) * (1 - eta),
                      (1 + xi) * (1 + eta), (1 - xi) * (1 + eta)]) / 4.0
        dNdxi = np.array([-(1 - eta), (1 - eta), (1 + eta), -(1 + eta)]) / 4.0
        dNdeta = np.array([-(1 - xi), -(1 + xi), (1 + xi), (1 - xi)]) / 4.0
        dNdx = dNdxi * (2.0 / dx); dNdy = dNdeta * (2.0 / dy)
        B = np.zeros((3, 8))
        B[0, 0::2] = dNdx; B[1, 1::2] = dNdy
        B[2, 0::2] = dNdy; B[2, 1::2] = dNdx
        R = np.zeros(8); R[1::2] = 0.5 * dNdx; R[0::2] = -0.5 * dNdy
        Nm = np.zeros((2, 8)); Nm[0, 0::2] = N; Nm[1, 1::2] = N
        Ke += detJ * (B.T @ D @ B)
        Me += detJ * (Nm.T @ Nm)
        Bs[q] = B; Rs[q] = R
    return Ke, Me, Bs, Rs, detJ


def _edof(nelx, nely):
    ex, ey = np.meshgrid(np.arange(nelx), np.arange(nely))   # (nely, nelx)
    ex = ex.ravel(); ey = ey.ravel()
    def m(i, j): return (j % nely) * nelx + (i % nelx)
    nodes = [m(ex, ey), m(ex + 1, ey), m(ex + 1, ey + 1), m(ex, ey + 1)]
    edof = np.empty((nelx * nely, 8), dtype=np.int64)
    for a, nd in enumerate(nodes):
        edof[:, 2 * a] = 2 * nd; edof[:, 2 * a + 1] = 2 * nd + 1
    return edof


def assemble(frac, lx=1.0, ly=1.0, mode="plane_strain"):
    """frac: (nely, nelx) solid fraction per element, 1 = solid.  Returns K, M
    (csc) and the per-element weights + edof needed for character extraction."""
    frac = np.asarray(frac, dtype=float)
    nely, nelx = frac.shape
    dx, dy = lx / nelx, ly / nely
    Ke, Me, Bs, Rs, detJ = _q4_unit(dx, dy, NU, mode)
    f = frac.ravel()
    Ee = f * E_SOLID + (1 - f) * E_VOID
    re = f * RHO_SOLID + (1 - f) * RHO_VOID
    edof = _edof(nelx, nely)
    ne = len(f); ndof = 2 * nelx * nely
    rows = np.repeat(edof, 8, axis=1).ravel()
    cols = np.tile(edof, (1, 8)).ravel()
    K = coo_matrix(((Ee[:, None] * Ke.ravel()[None, :]).ravel(), (rows, cols)),
                   shape=(ndof, ndof)).tocsc()
    M = coo_matrix(((re[:, None] * Me.ravel()[None, :]).ravel(), (rows, cols)),
                   shape=(ndof, ndof)).tocsc()
    return K, M, dict(edof=edof, Ee=Ee, Bs=Bs, Rs=Rs, detJ=detJ)


def eigenmodes(frac, n_modes=N_MODES, **kw):
    """Lowest n_modes eigenpairs.  Returns omega (sorted, >=0) and Phi (ndof, n)."""
    K, M, aux = assemble(frac, **kw)
    # shift-invert around a slightly negative sigma: K - sigma M is SPD even
    # though K itself has the 2-dim rigid nullspace.
    v0 = np.linspace(0.3, 1.0, K.shape[0])            # deterministic ARPACK start
    lam, Phi = eigsh(K, k=n_modes, M=M, sigma=-1e-3, which="LM", tol=1e-10, v0=v0)
    order = np.argsort(lam); lam = lam[order]; Phi = Phi[:, order]
    omega = np.sqrt(np.clip(lam, 0.0, None))
    return omega, Phi, aux


def character_integrals(Phi, aux):
    """(n_modes, 3) un-normalised integrals of the dilatation / deviatoric-shear /
    rotation parts of |grad phi|^2, stiffness-weighted."""
    edof, Ee, Bs, Rs, detJ = aux["edof"], aux["Ee"], aux["Bs"], aux["Rs"], aux["detJ"]
    n = Phi.shape[1]
    out = np.zeros((n, 3))
    U = Phi[edof, :]                      # (ne, 8, n)
    for q in range(4):
        eps = np.einsum("ij,ejn->ein", Bs[q], U)   # (ne, 3, n) : exx, eyy, gxy
        w = np.einsum("j,ejn->en", Rs[q], U)        # (ne, n)
        d = eps[:, 0, :] + eps[:, 1, :]
        dev = (eps[:, 0, :] - eps[:, 1, :]) ** 2 + eps[:, 2, :] ** 2
        out[:, 0] += detJ * (Ee[:, None] * 0.5 * d ** 2).sum(0)
        out[:, 1] += detJ * (Ee[:, None] * 0.5 * dev).sum(0)
        out[:, 2] += detJ * (Ee[:, None] * 2.0 * w ** 2).sum(0)
    return out


def clusters(omega, rtol=CLUSTER_RTOL):
    """Group indices of (near-)degenerate eigenvalues."""
    lam = omega ** 2
    scale = max(lam.max(), 1e-12)
    groups, cur = [], [0]
    for i in range(1, len(lam)):
        if abs(lam[i] - lam[cur[-1]]) <= rtol * max(lam[i], lam[cur[-1]], 1e-8 * scale):
            cur.append(i)
        else:
            groups.append(cur); cur = [i]
    groups.append(cur)
    return groups


def descriptors(frac, n_sub=6, **kw):
    """Return dict with
         omega : (n_sub,)   frequencies of modes 2 .. 2+n_sub-1   (rigid excluded)
         char  : (n_sub,3)  cluster-averaged (c_dil, c_shr, c_rot), rows sum to 1
         rigid : (2,)       the two rigid eigenvalues (should be ~0)
    """
    omega, Phi, aux = eigenmodes(frac, n_modes=n_sub + 2 + N_SPARE, **kw)
    I = character_integrals(Phi, aux)
    ch = np.zeros_like(I)
    cl = clusters(omega)
    for g in cl:
        m = I[g].mean(0)
        ch[g] = m / max(m.sum(), 1e-300)
    last = 2 + n_sub - 1
    truncated = any((last in g) and (max(g) == len(omega) - 1) for g in cl)
    return dict(omega=omega[2:2 + n_sub], char=ch[2:2 + n_sub], rigid=omega[:2] ** 2,
                clusters=cl, truncated=truncated)


def coarsen(solid, n_out, how="grey"):
    """Block-coarsen a binary (n,n) cell to (n_out,n_out).
    how='grey'  : block mean solid fraction (keeps thin-strut information)
    how='binary': block mean thresholded at 0.5
    """
    solid = np.asarray(solid, dtype=float)
    n = solid.shape[0]; assert n % n_out == 0
    b = n // n_out
    f = solid.reshape(n_out, b, n_out, b).mean(axis=(1, 3))
    return f if how == "grey" else (f >= 0.5).astype(float)


def feature_vector(desc):
    """The 24-vector used in the feature-arm experiments:
       [omega_2, omega_3/omega_2 .. omega_7/omega_2 (5), char (6x3=18)]"""
    om = desc["omega"]
    return np.concatenate([[om[0]], om[1:] / om[0], desc["char"].ravel()])
