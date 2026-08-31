# metaxfam-subcell

Sub-cell eigenmode descriptors for metamaterial homogenisation surrogates.

> ## ⚠ v1 SUPERSEDED — its central claim did not survive recomputation
>
> The original version of this work was computed on a dataset later found to be
> **61.6 % duplicated** (92.3 % at the 24×24 resolution at which the descriptors were
> computed). Recomputing every number on duplicate-free data **removed its main result**:
>
> | claim | v1 (duplicated data) | v2 (clean data) |
> |---|---|---|
> | 4 numbers beat 2304 pixels | +0.364, wins 100 % | **+0.045 under RF; loses on C11** |
> | beats shape-DNA (prior art) | +0.250, wins 96 % | +0.161 on C22; **fails on C11** |
> | beats solid fraction | +0.274, wins 97 % | holds on C22/C12; **fails on C11** |
>
> Duplication does not only inflate in-distribution scores — it **depresses
> high-dimensional baselines** on cross-family evaluation, and so flatters low-dimensional
> competitors. The pixel baseline on C22 recovers from −0.008 to **+0.37** once duplicates
> are removed.
>
> **The current work is in [`v2/`](v2/)**, archived at
> [`10.5281/zenodo.22137052`](https://doi.org/10.5281/zenodo.22137052) (see the record for
> the current version DOI). v1 code and results below are retained unchanged.

---

## What v2 reports

**1. The evaluation protocol matters more than the model.** Same surrogate, same training
size: R² between 0.90 and 0.96 under a random split, −0.73 to +0.14 when whole topology
families are held out. Gaps of 0.58 to 1.63 across four stiffness components.

**2. The descriptor's advantage is stability, not accuracy.** Across twelve
component-by-learner combinations:

| representation | dim | min R² | max R² |
|---|---|---|---|
| raw pixels | 2304 | **−42.99** | +0.34 |
| solid fraction + shape-DNA | 7 | −1.56 | +0.33 |
| solid fraction | 1 | +0.04 | +0.24 |
| **solid fraction + mode-2 character** | **4** | **+0.02** | **+0.41** |

It is the only representation tested that never fails.

**3. The advantage is component-specific — and that is the point.** It beats the geometric
spectrum and the free solid-fraction control on the shear-coupled C12 and C22 for every
learner, and does not beat solid fraction at all on C11. Rotation couples to shear, so a
rotational-mode explanation predicts exactly this pattern.

Aggregation rules were fixed *before* results were read: a claim counts as supported only
if it holds for every learner on a component, with a bootstrap 95 % CI on the paired median
gain excluding zero.

## The benchmark

v2 uses **MetaXFam-D** — 18 families × 172 cells, verified zero duplicates —
archived at [`10.5281/zenodo.22167146`](https://doi.org/10.5281/zenodo.22167146), with
generators and the duplication audit at
[davidmashiah/metaxfam22](https://github.com/davidmashiah/metaxfam22).
The superseded duplicated dataset remains at
[`10.5281/zenodo.21734948`](https://doi.org/10.5281/zenodo.21734948) because published work
refers to it.

## Reproducing v2

```bash
cd v2/code
python test_subcell.py     # 10 tests, incl. exact plane-wave frequencies AND characters
python verify_full.py      # 4 components × 3 learners × 2 seeds × 30 subsets
python final_check.py      # pre-registered summary with bootstrap CIs
```

NumPy, SciPy, scikit-learn only. One CPU core. `v2/results/` holds the raw JSON behind
every table and figure in the paper.

## About `code/subcell_v1_original.py`

The original extractor computed mode character from a single eigenvector. Every
four-fold-symmetric family has a degenerate second mode, and character is quadratic in the
eigenvector, so that quantity is basis-dependent. `code/subcell.py` averages over the
degenerate cluster, which is basis-invariant. Both are shipped so the comparison can be
re-run.

## Citing

See `CITATION.cff`. Cite the Zenodo record by version.

## Licence

Code MIT (`LICENSE`). Results and figures CC BY 4.0.

## Contact

David Mashiah — davidmashiah@mail.tau.ac.il — ORCID
[0009-0004-4684-955X](https://orcid.org/0009-0004-4684-955X)
