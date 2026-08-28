"""subcell.py -- sub-cell eigenmode descriptors for a periodic 2D unit cell.

Classical homogenisation returns C^H by volume-averaging the microscopic stress.
Averaging is a projection: it retains the two rigid-translation modes of the cell
and discards every other sub-cell degree of freedom.  Roberts (Trans. Math. Appl.
9:tnaf001, 2025) builds multi-continuum homogenisations by retaining the lowest M
eigenmodes instead, the third often being a sub-cell ROTATION.

Here we compute those modes: on the same periodic operator used for C^H we solve
K v = lambda M v, whose two lowest eigenvalues are the rigid translations. The
modes above them are exactly what averaging throws away.

VOID HANDLING. The void has stiffness 1e-6. Giving it a proportional density
would leave void modes at solid-like frequencies, since eigenvalues are invariant
under joint scaling of K and M. We give the void density 1e-12, pushing its modes
to high frequency and leaving the low spectrum to genuine skeleton motion.
"""
import _path  # noqa: F401  (adds code/ and code/vendor/ to sys.path)

import numpy as np
from scipy.sparse import coo_matrix, diags
from scipy.sparse.linalg import eigsh
from homogenize2d import constitutive_matrix, q4_element_matrices

RHO_SOLID, RHO_VOID = 1.0, 1e-12


def _periodic_edof(nelx, nely):
    def master(i, j):
        m = (j % nely) * nelx + (i % nelx)
        return 2 * m, 2 * m + 1
    edof = np.zeros((nely * nelx, 8), dtype=int); e = 0
    for ey in range(nely):
        for ex in range(nelx):
            d = []
            for (i, j) in [(ex, ey), (ex+1, ey), (ex+1, ey+1), (ex, ey+1)]:
                a, b = master(i, j); d += [a, b]
            edof[e, :] = d; e += 1
    return edof


def assemble(phase, materials, lx=1.0, ly=1.0, mode="plane_strain"):
    phase = np.asarray(phase); nely, nelx = phase.shape
    dx, dy = lx / nelx, ly / nely
    elem = [q4_element_matrices(dx, dy, constitutive_matrix(E, nu, mode))
            for (E, nu) in materials]
    rho = [RHO_SOLID, RHO_VOID]
    edof = _periodic_edof(nelx, nely); ndof = 2 * nelx * nely
    rows, cols, vals = [], [], []; mdiag = np.zeros(ndof); Ve = dx * dy; e = 0
    for ey in range(nely):
        for ex in range(nelx):
            p = int(phase[ey, ex]); Ke, _ = elem[p]; d = edof[e]
            rows.append(np.repeat(d, 8)); cols.append(np.tile(d, 8))
            vals.append(Ke.ravel()); mdiag[d] += rho[p] * Ve / 4.0; e += 1
    K = coo_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                   shape=(ndof, ndof)).tocsr()
    return K, diags(mdiag), edof, (nelx, nely, dx, dy)


def mode_character(v, nelx, nely, dx, dy, solid):
    """Rotation / dilatation / shear content of a mode, over the solid phase."""
    u = v[0::2].reshape(nely, nelx); w = v[1::2].reshape(nely, nelx)
    dudx = (np.roll(u,-1,1)-np.roll(u,1,1))/(2*dx); dudy = (np.roll(u,-1,0)-np.roll(u,1,0))/(2*dy)
    dwdx = (np.roll(w,-1,1)-np.roll(w,1,1))/(2*dx); dwdy = (np.roll(w,-1,0)-np.roll(w,1,0))/(2*dy)
    rot = 0.5*(dwdx-dudy); dil = dudx+dwdy
    shr = 0.5*(dudy+dwdx); dev = 0.5*(dudx-dwdy)
    if solid.sum() == 0: return np.zeros(3)
    r = np.sqrt(np.mean(rot[solid]**2)); d = np.sqrt(np.mean(dil[solid]**2))
    s = np.sqrt(np.mean(shr[solid]**2)+np.mean(dev[solid]**2))
    return np.array([r, d, s]) / (r+d+s+1e-30)


def subcell_modes(phase, materials, n_modes=6, lx=1.0, ly=1.0, mode="plane_strain"):
    K, M, _, (nelx, nely, dx, dy) = assemble(phase, materials, lx, ly, mode)
    solid = (np.asarray(phase) == 0)
    vals, vecs = eigsh(K.tocsc(), k=n_modes+2, M=M.tocsc(), sigma=-1e-8, which="LM")
    o = np.argsort(vals); vals, vecs = vals[o], vecs[:, o]
    omega = np.sqrt(np.clip(vals[2:], 0.0, None))
    chars = np.array([mode_character(vecs[:, i+2], nelx, nely, dx, dy, solid)
                      for i in range(n_modes)])
    return omega, chars, vals[:2]
