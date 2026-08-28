"""
homogenize2d.py
===============
Numerical homogenization of 2D periodic unit cells (linear elasticity).

Given a periodic unit cell described by a pixel grid of material phases, compute
the effective (homogenized) elastic stiffness tensor C^H in Voigt notation:

    [sigma_xx]        [eps_xx  ]
    [sigma_yy]  = C^H [eps_yy  ]
    [sigma_xy]        [gamma_xy]

METHOD
------
Asymptotic / computational homogenization with periodic boundary conditions.
For a macroscopic strain eps_bar, the microscopic displacement field is

    u(x) = eps_bar . x + u~(x),      u~ periodic over the unit cell.

The periodic fluctuation u~ solves the cell problem (weak form):

    int_Y eps(v)^T D eps(u~) dY  =  - int_Y eps(v)^T D eps_bar dY   for all periodic v

which discretizes to  K u~ = -F.  The effective stiffness is then obtained by
volume-averaging the resulting stress:

    C^H[:,k] = (1/|Y|) sum_e [ D_e eps_bar_k V_e + Fe_e^T u~_e ]

with eps_bar_k the k-th unit macroscopic strain.

Discretization: bilinear Q4 quadrilateral elements on a structured grid, 2x2
Gauss quadrature. Periodicity is imposed directly by identifying opposite-edge
nodes to the same degrees of freedom (master/slave DOF mapping), which is exact
and avoids Lagrange multipliers.

Reference method (independent implementation, written from scratch):
Andreassen & Andreasen, "How to determine composite material properties using
numerical homogenization", Computational Materials Science 83:488-495, 2014.
"""

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve


# ---------------------------------------------------------------------------
# Constitutive relations
# ---------------------------------------------------------------------------
def constitutive_matrix(E, nu, mode="plane_strain"):
    """Isotropic 2D constitutive matrix D (3x3) in Voigt notation.

    Voigt convention here: [sxx, syy, sxy] = D [exx, eyy, gxy], gxy = 2*exy.
    """
    if mode == "plane_strain":
        c = E / ((1.0 + nu) * (1.0 - 2.0 * nu))
        D = c * np.array([[1.0 - nu, nu, 0.0],
                          [nu, 1.0 - nu, 0.0],
                          [0.0, 0.0, (1.0 - 2.0 * nu) / 2.0]])
    elif mode == "plane_stress":
        c = E / (1.0 - nu ** 2)
        D = c * np.array([[1.0, nu, 0.0],
                          [nu, 1.0, 0.0],
                          [0.0, 0.0, (1.0 - nu) / 2.0]])
    else:
        raise ValueError("mode must be 'plane_strain' or 'plane_stress'")
    return D


def lame_from_Enu(E, nu):
    """Lame parameters (lambda, mu) for plane strain / 3D."""
    lam = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
    mu = E / (2.0 * (1.0 + nu))
    return lam, mu


# ---------------------------------------------------------------------------
# Q4 element matrices
# ---------------------------------------------------------------------------
def q4_element_matrices(dx, dy, D):
    """Element stiffness Ke (8x8) and macro-strain load matrix Fe (8x3).

    Node ordering (counter-clockwise from bottom-left):
        0:(-1,-1)  1:(+1,-1)  2:(+1,+1)  3:(-1,+1)
    DOF ordering: [u0, v0, u1, v1, u2, v2, u3, v3]

    Fe[:, k] = int_e B^T D e_k dV   (e_k = k-th unit macroscopic strain)
    """
    g = 1.0 / np.sqrt(3.0)
    gauss = [(-g, -g), (g, -g), (g, g), (-g, g)]
    weights = [1.0, 1.0, 1.0, 1.0]

    Ke = np.zeros((8, 8))
    Fe = np.zeros((8, 3))
    detJ = dx * dy / 4.0

    for (xi, eta), w in zip(gauss, weights):
        # bilinear shape function derivatives w.r.t. natural coords
        dNdxi = np.array([-(1 - eta), (1 - eta), (1 + eta), -(1 + eta)]) / 4.0
        dNdeta = np.array([-(1 - xi), -(1 + xi), (1 + xi), (1 - xi)]) / 4.0
        # map to physical coords (rectangular element -> diagonal Jacobian)
        dNdx = dNdxi * (2.0 / dx)
        dNdy = dNdeta * (2.0 / dy)

        B = np.zeros((3, 8))
        B[0, 0::2] = dNdx      # exx = du/dx
        B[1, 1::2] = dNdy      # eyy = dv/dy
        B[2, 0::2] = dNdy      # gxy = du/dy + dv/dx
        B[2, 1::2] = dNdx

        Ke += w * detJ * (B.T @ D @ B)
        Fe += w * detJ * (B.T @ D)

    return Ke, Fe


# ---------------------------------------------------------------------------
# Main homogenization routine
# ---------------------------------------------------------------------------
def homogenize(phase, materials, lx=1.0, ly=1.0, mode="plane_strain"):
    """Compute the homogenized stiffness C^H of a periodic 2D unit cell.

    Parameters
    ----------
    phase : (nely, nelx) int array
        Material index of each pixel/element. phase[j, i] is the element in
        column i, row j (row 0 = bottom).
    materials : list of (E, nu)
        Material properties indexed by the values in `phase`.
    lx, ly : float
        Physical dimensions of the unit cell.
    mode : str
        'plane_strain' or 'plane_stress'.

    Returns
    -------
    CH : (3,3) ndarray
        Homogenized stiffness in Voigt notation.
    """
    phase = np.asarray(phase)
    nely, nelx = phase.shape
    dx, dy = lx / nelx, ly / nely
    cell_volume = lx * ly

    # Precompute element matrices for each material
    Ds = [constitutive_matrix(E, nu, mode) for (E, nu) in materials]
    elem_mats = [q4_element_matrices(dx, dy, D) for D in Ds]

    # --- periodic DOF mapping -------------------------------------------
    # Node (i, j) with i in 0..nelx, j in 0..nely is identified with the
    # master node (i % nelx, j % nely). This enforces periodicity exactly.
    n_master = nelx * nely
    ndof = 2 * n_master

    def master_dofs(i, j):
        m = (j % nely) * nelx + (i % nelx)
        return 2 * m, 2 * m + 1

    # element -> 8 global dofs
    edof = np.zeros((nely * nelx, 8), dtype=int)
    e = 0
    for ey in range(nely):
        for ex in range(nelx):
            nodes = [(ex, ey), (ex + 1, ey), (ex + 1, ey + 1), (ex, ey + 1)]
            dofs = []
            for (i, j) in nodes:
                a, b = master_dofs(i, j)
                dofs.extend([a, b])
            edof[e, :] = dofs
            e += 1

    # --- assemble K and F ------------------------------------------------
    rows, cols, vals = [], [], []
    F = np.zeros((ndof, 3))
    e = 0
    for ey in range(nely):
        for ex in range(nelx):
            p = int(phase[ey, ex])
            Ke, Fe = elem_mats[p]
            d = edof[e]
            rows.append(np.repeat(d, 8))
            cols.append(np.tile(d, 8))
            vals.append(Ke.ravel())
            F[d, :] += Fe
            e += 1
    rows = np.concatenate(rows)
    cols = np.concatenate(cols)
    vals = np.concatenate(vals)
    K = coo_matrix((vals, (rows, cols)), shape=(ndof, ndof)).tocsr()

    # --- solve cell problems ---------------------------------------------
    # Constrain one node to remove the rigid-body translation nullspace.
    free = np.arange(2, ndof)
    chi = np.zeros((ndof, 3))
    Kff = K[free, :][:, free]
    rhs = -F[free, :]
    chi[free, :] = spsolve(Kff, rhs)

    # --- volume-average the stress to get C^H ----------------------------
    CH = np.zeros((3, 3))
    Ve = dx * dy
    for k in range(3):
        eps_bar = np.zeros(3)
        eps_bar[k] = 1.0
        sigma_avg = np.zeros(3)
        e = 0
        for ey in range(nely):
            for ex in range(nelx):
                p = int(phase[ey, ex])
                D = Ds[p]
                Ke, Fe = elem_mats[p]
                d = edof[e]
                u_e = chi[d, k]
                # int_e D (eps_bar + B u~) dV  =  D eps_bar Ve + Fe^T u~_e
                sigma_avg += D @ eps_bar * Ve + Fe.T @ u_e
                e += 1
        CH[:, k] = sigma_avg / cell_volume

    # symmetrize (removes tiny numerical asymmetry)
    CH = 0.5 * (CH + CH.T)
    return CH


# ---------------------------------------------------------------------------
# Effective engineering constants from C^H (orthotropic assumption)
# ---------------------------------------------------------------------------
def engineering_constants(CH):
    """Extract effective E1, E2, nu12, G12 from a 2D C^H by inverting to compliance."""
    S = np.linalg.inv(CH)
    E1 = 1.0 / S[0, 0]
    E2 = 1.0 / S[1, 1]
    nu12 = -S[0, 1] / S[0, 0]
    nu21 = -S[0, 1] / S[1, 1]
    G12 = 1.0 / S[2, 2]
    return dict(E1=E1, E2=E2, nu12=nu12, nu21=nu21, G12=G12)


# ---------------------------------------------------------------------------
# Analytical reference: two-phase rank-1 laminate, layers stacked along y
# ---------------------------------------------------------------------------
def laminate_exact(E1, nu1, E2, nu2, f1, mode="plane_strain"):
    """Exact effective stiffness of a 2-phase laminate with layers normal to y.

    Derivation (plane strain, isotropic phases, Voigt notation):
      Under the laminate kinematics, eps_xx is continuous across layers and
      sigma_yy is continuous. Writing <.> for the volume average:
        C22^H = 1 / <1/C22>
        C12^H = <C12/C22> / <1/C22>
        C11^H = <C11> - <C12^2/C22> + <C12/C22>^2 / <1/C22>
        C33^H = 1 / <1/C33>
    """
    D1 = constitutive_matrix(E1, nu1, mode)
    D2 = constitutive_matrix(E2, nu2, mode)
    f2 = 1.0 - f1

    def avg(a1, a2):
        return f1 * a1 + f2 * a2

    C11_1, C12_1, C22_1, C33_1 = D1[0, 0], D1[0, 1], D1[1, 1], D1[2, 2]
    C11_2, C12_2, C22_2, C33_2 = D2[0, 0], D2[0, 1], D2[1, 1], D2[2, 2]

    inv_C22 = avg(1.0 / C22_1, 1.0 / C22_2)
    C12_over_C22 = avg(C12_1 / C22_1, C12_2 / C22_2)

    C22H = 1.0 / inv_C22
    C12H = C12_over_C22 / inv_C22
    C11H = avg(C11_1, C11_2) - avg(C12_1 ** 2 / C22_1, C12_2 ** 2 / C22_2) \
           + C12_over_C22 ** 2 / inv_C22
    C33H = 1.0 / avg(1.0 / C33_1, 1.0 / C33_2)

    CH = np.array([[C11H, C12H, 0.0],
                   [C12H, C22H, 0.0],
                   [0.0, 0.0, C33H]])
    return CH


# ---------------------------------------------------------------------------
# Voigt / Reuss bounds (must bracket any valid C^H)
# ---------------------------------------------------------------------------
def voigt_reuss(E1, nu1, E2, nu2, f1, mode="plane_strain"):
    D1 = constitutive_matrix(E1, nu1, mode)
    D2 = constitutive_matrix(E2, nu2, mode)
    f2 = 1.0 - f1
    voigt = f1 * D1 + f2 * D2
    reuss = np.linalg.inv(f1 * np.linalg.inv(D1) + f2 * np.linalg.inv(D2))
    return voigt, reuss


# ---------------------------------------------------------------------------
# Convenience geometry generators
# ---------------------------------------------------------------------------
def cell_homogeneous(n):
    return np.zeros((n, n), dtype=int)


def cell_layers_y(nely, nelx, f1):
    """Layers stacked along y: bottom fraction f1 is material 0, rest material 1."""
    phase = np.ones((nely, nelx), dtype=int)
    n1 = int(round(f1 * nely))
    phase[:n1, :] = 0
    return phase


def cell_square_hole(n, hole_frac):
    """Square unit cell with a centred square hole (material 1 = void-ish soft)."""
    phase = np.zeros((n, n), dtype=int)
    h = int(round(hole_frac * n))
    if h > 0:
        s = (n - h) // 2
        phase[s:s + h, s:s + h] = 1
    return phase


def cell_circular_hole(n, radius_frac):
    """Square unit cell with a centred circular hole."""
    phase = np.zeros((n, n), dtype=int)
    yy, xx = np.mgrid[0:n, 0:n]
    cx = cy = (n - 1) / 2.0
    r = radius_frac * n
    phase[((xx - cx) ** 2 + (yy - cy) ** 2) <= r ** 2] = 1
    return phase
