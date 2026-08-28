"""_path.py -- put code/ and code/vendor/ on sys.path.

The 18 modules in vendor/ are unmodified copies of the solver and family generators
from github.com/davidmashiah/metaxfam22 (Zenodo 10.5281/zenodo.21734948).  They are
vendored, not forked: byte-identical to that repo, so this repository never becomes a
maintenance fork of it.  Importing this module makes them resolvable by plain name,
exactly as the original scripts expect.
"""
import os, sys
_H = os.path.dirname(os.path.abspath(__file__))
for p in (_H, os.path.join(_H, "vendor")):
    if p not in sys.path:
        sys.path.insert(0, p)
