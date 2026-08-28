# metaxfam-subcell

Sub-cell eigenmode descriptors for metamaterial homogenisation surrogates.

Code and results for:

> **The target, not the learner: an identifiability limit on cross-topology transfer of
> homogenisation surrogates.** David Mashiah, School of Mechanical Engineering, Tel Aviv
> University.

Preprint, descriptors and archived results: **Zenodo DOI — to be added once the record is
published.**

---

## The idea in one paragraph

Classical homogenisation returns the effective stiffness `C^H` by volume-averaging the
microscopic stress. Averaging is a projection: it keeps the two rigid-translation modes of
the unit cell and discards every other sub-cell degree of freedom. This repository measures
what gets discarded, by solving the Γ-point elastodynamic eigenproblem on the *same*
periodic operator that produces `C^H`, and asks whether that content explains — and partly
repairs — the failure of learned surrogates to transfer to unit-cell topologies they were
never trained on.

Four families in the benchmark have `C11/C22 = 1.000` to machine precision, so their
stiffness tensors carry no directional information at all. Their sub-cell mode characters
are nevertheless well separated. Three numbers describing how the softest sub-cell mode
divides its motion between rotation, dilatation and shear predict `C22` on 18 unseen
families at **R² = +0.356**, where 2304 raw pixels give **−0.008** and a free
solid-fraction control gives **+0.082** — at a quarter of the cost of one homogenisation.

Negative results are here too, deliberately: the descriptor does not rescue a pixel
surrogate, it fails as a training-family selection criterion, and an earlier version of the
extractor produced a false negative that is analysed rather than buried.

## Requirements

Python 3.10+, NumPy, SciPy, scikit-learn, Matplotlib (`pip install -r requirements.txt`).
No GPU, no commercial FE package. Everything was developed and run on one CPU core.

## The dataset is not in this repository

The 22-family MetaXFam22 unit-cell dataset (13,200 cells with stiffness tensors) lives at
**Zenodo `10.5281/zenodo.21734948`**, with its generators at
[davidmashiah/metaxfam22](https://github.com/davidmashiah/metaxfam22). Unpack the 66 `.npy`
files into `data/`, or regenerate them:

```bash
cd code && python gen_all.py      # ~30 min on one core, seeded and checkpointed
```

The generator is seeded, so regenerated cells match the archived ones; this was verified
against the stored descriptors to five decimal places.

## Layout

```
code/            analysis for this paper
  subcell.py             the sub-cell eigenmode extractor — the core new tool
  test_subcell.py        its validation suite (10 tests)
  subcell_v1_original.py the superseded extractor, kept on purpose (see below)
  gen_desc_v2.py         descriptors at any resolution / coarsening
  robust.py              main result: 120 paired evaluations per arm
  robust_k.py            K = 1 and K = 2
  learners.py            RF / GBR / kernel ridge / MLP, and the per-family breakdown
  baselines.py           shape-DNA, Hu moments, two-point correlation
  d4base.py              exact D4 orbit-averaged pixel surrogate
  predictor.py           mode character as a selection criterion (a negative result)
  cnn_test.py            from-scratch CNN vs the four-number feature
  ablate2.py             feature-subset ablation
  summarize.py           prints the paper's main table
  figs3.py, figs3b.py    the six figures
  vendor/                UNMODIFIED copies of the metaxfam22 solver and generators
results/         every JSON behind the numbers in the paper, plus numbers.txt
figures/         the six paper figures
```

## Reproducing

Validate the solvers first. Nothing downstream means anything if these fail.

```bash
cd code
python vendor/test_homogenize.py   # 6 tests: laminate, Voigt–Reuss, symmetry, convergence
python test_subcell.py             # 10 tests: exact plane-wave frequencies AND characters,
                                   #  h² convergence, degenerate-cluster basis invariance
```

Then:

```bash
python gen_desc_v2.py 24 grey   # descriptors (or download them from the Zenodo record)
python robust.py                # the main result
python baselines.py             # named baselines
python d4base.py                # symmetry baseline
python predictor.py             # selection-criterion test
python cnn_test.py              # CNN comparison
python summarize.py             # main table
python figs3.py && python figs3b.py
```

Long runs checkpoint into `results/*.json` and are safe to interrupt. Rough wall-clock on
one core: descriptors ~15 min, `robust.py` ~45 min, `cnn_test.py` ~30 min, the rest minutes.

## About `code/vendor/`

Those 18 files are byte-identical copies of the homogenisation solver and family generators
from [metaxfam22](https://github.com/davidmashiah/metaxfam22). They are vendored rather than
forked so this repository never drifts from that one — if you change something there, copy
it here rather than editing in place. `code/_path.py` puts them on `sys.path` so the scripts
import them by plain name.

## About `subcell_v1_original.py`

The original extractor computed mode character from a single eigenvector. Every
four-fold-symmetric family has a degenerate second mode, and mode character is quadratic in
the eigenvector, so that quantity is basis-dependent — in part a random draw from inside the
degenerate eigenspace. It produced the conclusion that the descriptor collapses under
coarsening (+0.388 → +0.051). `subcell.py` averages over the degenerate cluster, which is
basis-invariant, and the collapse disappears (+0.409 → +0.340 at 24×24). Both are shipped so
the comparison in the paper can be re-run rather than taken on trust.

## Citing

See `CITATION.cff`. Please cite the preprint and the Zenodo record.

## Licence

Code MIT (`LICENSE`). Results and figures CC BY 4.0.

## Contact

David Mashiah — davidmashiah@mail.tau.ac.il — ORCID
[0009-0004-4684-955X](https://orcid.org/0009-0004-4684-955X)
