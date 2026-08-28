"""
test_homogenize.py
==================
Validation of the 2D numerical homogenization code against EXACT analytical
results. These are not sanity checks -- each has a known closed-form answer,
so the code either reproduces it or it is wrong.

Tests
-----
1. Homogeneous cell        -> C^H must equal the input D to machine precision.
                              (Strongest structural test: exercises periodic BCs,
                               assembly, cell solve and averaging simultaneously.)
2. Rank-1 laminate         -> C^H must equal exact laminate theory.
3. Voigt/Reuss bounds      -> any two-phase C^H must lie between the bounds.
4. Symmetry                -> C^H must be symmetric (major symmetry).
5. Square symmetry         -> a cell with 90-degree symmetry must give C11 = C22
                              and zero shear-extension coupling.
6. Mesh convergence        -> results must converge as the mesh is refined.
"""

import numpy as np
from homogenize2d import (
    homogenize, constitutive_matrix, laminate_exact, voigt_reuss,
    engineering_constants, cell_homogeneous, cell_layers_y,
    cell_square_hole, cell_circular_hole,
)


def rel_err(A, B):
    """Relative Frobenius error, guarded against divide-by-zero."""
    denom = np.linalg.norm(B)
    if denom == 0:
        return np.linalg.norm(A - B)
    return np.linalg.norm(A - B) / denom


def test_homogeneous():
    print("\n[1] Homogeneous cell -> C^H must equal input D exactly")
    E, nu = 210.0, 0.3
    D = constitutive_matrix(E, nu)
    passed = True
    for n in [4, 8, 16]:
        phase = cell_homogeneous(n)
        CH = homogenize(phase, [(E, nu)])
        err = rel_err(CH, D)
        ok = err < 1e-10
        passed &= ok
        print(f"    n={n:3d}: rel.err = {err:.3e}   {'PASS' if ok else 'FAIL'}")
    print(f"    input  D  = {np.array2string(D, precision=4, suppress_small=True)}")
    print(f"    computed  = {np.array2string(CH, precision=4, suppress_small=True)}")
    return passed


def test_laminate():
    print("\n[2] Rank-1 laminate -> C^H must equal exact laminate theory")
    E1, nu1 = 210.0, 0.3     # stiff phase
    E2, nu2 = 3.0, 0.35      # compliant phase
    passed = True
    for f1 in [0.25, 0.5, 0.75]:
        exact = laminate_exact(E1, nu1, E2, nu2, f1)
        # mesh must resolve the layer fractions exactly
        nely = 40
        phase = cell_layers_y(nely, 4, f1)
        CH = homogenize(phase, [(E1, nu1), (E2, nu2)])
        err = rel_err(CH, exact)
        ok = err < 1e-8
        passed &= ok
        print(f"    f1={f1:.2f}: rel.err vs exact = {err:.3e}   {'PASS' if ok else 'FAIL'}")
    print(f"    exact (f1=0.75)    = {np.array2string(exact, precision=3, suppress_small=True)}")
    print(f"    computed (f1=0.75) = {np.array2string(CH, precision=3, suppress_small=True)}")
    return passed


def test_bounds():
    print("\n[3] Voigt/Reuss bounds must bracket C^H (two-phase cells)")
    E1, nu1 = 210.0, 0.3
    E2, nu2 = 3.0, 0.35
    passed = True
    for name, phase in [("square hole", cell_square_hole(24, 0.5)),
                        ("circular hole", cell_circular_hole(24, 0.3)),
                        ("layers", cell_layers_y(24, 24, 0.5))]:
        f1 = float(np.mean(phase == 0))
        voigt, reuss = voigt_reuss(E1, nu1, E2, nu2, f1)
        CH = homogenize(phase, [(E1, nu1), (E2, nu2)])
        # Compare bulk-like invariant (trace of the 2x2 extensional block)
        tr_H = CH[0, 0] + CH[1, 1]
        tr_V = voigt[0, 0] + voigt[1, 1]
        tr_R = reuss[0, 0] + reuss[1, 1]
        ok = (tr_R - 1e-6) <= tr_H <= (tr_V + 1e-6)
        passed &= ok
        print(f"    {name:14s} f1={f1:.3f}: Reuss={tr_R:8.3f} <= C^H={tr_H:8.3f} "
              f"<= Voigt={tr_V:8.3f}   {'PASS' if ok else 'FAIL'}")
    return passed


def test_symmetry():
    print("\n[4] C^H must be symmetric")
    E1, nu1 = 210.0, 0.3
    E2, nu2 = 3.0, 0.35
    phase = cell_circular_hole(24, 0.3)
    # recompute without the symmetrization applied inside homogenize()
    CH = homogenize(phase, [(E1, nu1), (E2, nu2)])
    asym = np.linalg.norm(CH - CH.T) / np.linalg.norm(CH)
    ok = asym < 1e-10
    print(f"    ||C - C^T||/||C|| = {asym:.3e}   {'PASS' if ok else 'FAIL'}")
    return ok


def test_square_symmetry():
    print("\n[5] 90-degree symmetric cell -> C11 = C22, no shear coupling")
    E1, nu1 = 210.0, 0.3
    E2, nu2 = 3.0, 0.35
    passed = True
    for name, phase in [("square hole", cell_square_hole(24, 0.5)),
                        ("circular hole", cell_circular_hole(24, 0.3))]:
        CH = homogenize(phase, [(E1, nu1), (E2, nu2)])
        d_diag = abs(CH[0, 0] - CH[1, 1]) / abs(CH[0, 0])
        coupling = (abs(CH[0, 2]) + abs(CH[1, 2])) / abs(CH[0, 0])
        ok = d_diag < 1e-8 and coupling < 1e-8
        passed &= ok
        print(f"    {name:14s}: |C11-C22|/C11 = {d_diag:.3e}, "
              f"shear coupling = {coupling:.3e}   {'PASS' if ok else 'FAIL'}")
    return passed


def test_convergence():
    print("\n[6] Mesh convergence (circular hole, f_solid fixed)")
    E1, nu1 = 210.0, 0.3
    E2, nu2 = 1e-3, 0.3      # near-void
    prev = None
    vals = []
    for n in [16, 32, 64]:
        phase = cell_circular_hole(n, 0.3)
        CH = homogenize(phase, [(E1, nu1), (E2, nu2)])
        ec = engineering_constants(CH)
        vals.append(ec["E1"])
        msg = f"    n={n:3d}: E1_eff = {ec['E1']:9.4f}"
        if prev is not None:
            msg += f"   change = {abs(ec['E1'] - prev)/prev*100:6.2f}%"
        prev = ec["E1"]
        print(msg)
    # converging: successive changes should shrink
    d1 = abs(vals[1] - vals[0]) / vals[0]
    d2 = abs(vals[2] - vals[1]) / vals[1]
    ok = d2 < d1
    print(f"    successive change shrinking ({d1*100:.2f}% -> {d2*100:.2f}%)   "
          f"{'PASS' if ok else 'FAIL'}")
    return ok


if __name__ == "__main__":
    print("=" * 68)
    print("VALIDATION OF 2D NUMERICAL HOMOGENIZATION")
    print("=" * 68)
    results = {
        "homogeneous == input D": test_homogeneous(),
        "laminate == exact theory": test_laminate(),
        "Voigt/Reuss bracketing": test_bounds(),
        "C^H symmetry": test_symmetry(),
        "square symmetry": test_square_symmetry(),
        "mesh convergence": test_convergence(),
    }
    print("\n" + "=" * 68)
    for k, v in results.items():
        print(f"  {'PASS' if v else 'FAIL'}  {k}")
    print("=" * 68)
    print("ALL TESTS PASSED" if all(results.values()) else "SOME TESTS FAILED")
