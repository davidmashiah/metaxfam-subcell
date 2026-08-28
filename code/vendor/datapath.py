"""Locate and load dataset files regardless of where a script is run from.

Data are stored as plain .npy arrays (not .npz) so that the distributed archive
contains no nested archives -- .npz is itself a ZIP container, which some
repository upload pipelines reject.

Each dataset "<name>" is three files:
    <name>__X.npy     unit-cell images   (N, n, n) uint8, 1 = solid
    <name>__y.npy     stiffness targets  (N, 6) float64
    <name>__frac.npy  solid fraction     (N,) float64
"""
import os
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_DIRS = [
    os.getcwd(),
    os.path.join(os.getcwd(), "data"),
    os.path.join(_HERE, "..", "data"),
    os.path.join(_HERE, "data"),
    _HERE,
]


def find(filename):
    """Return a usable path to `filename`, or raise a clear error."""
    for d in _DIRS:
        p = os.path.join(d, filename)
        if os.path.exists(p):
            return p
    raise FileNotFoundError(
        f"Could not find {filename}. Looked in: "
        + ", ".join(os.path.normpath(d) for d in _DIRS)
    )


def load(name):
    """Load a dataset by name (with or without a .npz suffix).

    Returns a dict with keys 'X', 'y', 'frac' -- the same interface the code
    used when the data were stored as .npz, so call sites are unchanged.
    Falls back to reading a real .npz if one is present.
    """
    if name.endswith(".npz"):
        name = name[:-4]
    # preferred: plain .npy triplet
    try:
        return {
            "X": np.load(find(f"{name}__X.npy")),
            "y": np.load(find(f"{name}__y.npy")),
            "frac": np.load(find(f"{name}__frac.npy")),
        }
    except FileNotFoundError:
        pass
    # fallback: legacy .npz
    return dict(np.load(find(f"{name}.npz"), allow_pickle=True))


def load_any(name):
    """Load an arbitrary saved result (e.g. a transfer matrix) as a dict."""
    if name.endswith(".npz"):
        stem = name[:-4]
    else:
        stem = name
    try:
        return dict(np.load(find(f"{stem}.npz"), allow_pickle=True))
    except FileNotFoundError:
        import glob
        out = {}
        for p in glob.glob(os.path.join(os.path.dirname(find(f"{stem}__r2.npy")),
                                        f"{stem}__*.npy")):
            key = os.path.basename(p).split("__", 1)[1][:-4]
            out[key] = np.load(p, allow_pickle=True)
        if not out:
            raise
        return out
