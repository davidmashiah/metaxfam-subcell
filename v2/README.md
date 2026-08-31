# A rotational sub-cell descriptor for cross-topology stiffness prediction

**Version 2.0.0 — a substantial revision, not an update.** The conclusions of version 1
changed when every number was recomputed on duplicate-free data. Read the next section
before citing either version.

---

## What changed, and why

Version 1 ("The target, not the learner: an identifiability limit on cross-topology
transfer of homogenization surrogates") was computed on MetaXFam22, a dataset later found
to be **61.6% duplicated** at 48x48 and 92.3% duplicated at the 24x24 resolution at which
the descriptors were computed.

Recomputing on MetaXFam-D (18 families, 3,096 cells, verified zero duplicates,
doi 10.5281/zenodo.22167146) **removed the central claim of v1**:

| claim | v1 (duplicated data) | v2 (clean data) |
|---|---|---|
| 4 numbers beat 2304 pixels | +0.364, wins 100% | **+0.045 under RF, wins 67%; loses on C11** |
| beats shape-DNA (prior art) | +0.250, wins 96% | +0.161 on C22, wins 90%; **fails on C11** |
| beats solid fraction | +0.274, wins 97% | holds on C22/C12, **fails on C11** |

The direction of the bias is instructive: duplication does not only inflate
in-distribution scores, it **depresses high-dimensional baselines on cross-family
evaluation**, and so systematically flatters low-dimensional competitors. The pixel
baseline on C22 recovers from -0.008 to +0.37 once duplicates are removed.

## What v2 claims instead

1. **The evaluation protocol matters more than the model.** Same surrogate, same training
   size: R2 0.90-0.96 under a random split, -0.73 to +0.14 family-disjoint. Gaps of 0.58
   to 1.63 across four components.
2. **Stability, not accuracy, is the descriptor's advantage.** Across twelve
   component-by-learner combinations the 4-number feature stays in [+0.02, +0.41]. Raw
   pixels span [-42.99, +0.34]; shape-DNA spans [-1.56, +0.33]. It is the only
   representation tested that never fails.
3. **The advantage is component-specific, and that is the point.** It wins on the
   shear-coupled C12 and C22 for every learner and does not beat solid fraction at all on
   C11 — the signature a rotational-mode explanation predicts.

Aggregation rules were fixed before reading results: a claim counts as supported only if it
holds for every learner on a component, with a bootstrap 95% CI on the paired median gain
excluding zero. This is stated because an earlier analysis summarised on one learner and
reached a different headline from the same measurements.

## Layout

```
paper/        subcell_v2.pdf and its LaTeX source (venue-neutral, stock article class)
code/         extractor + validation suite, and every analysis script behind the numbers
results/      the raw JSON each table and figure is computed from
descriptors/  mode-2 characters and shape-DNA recomputed on the clean benchmark
figures/      the four paper figures at 300 dpi
```

## Reproducing

The benchmark is a separate record: doi 10.5281/zenodo.22167146. Place its
`data_benchmark/` where the scripts expect it, then:

```bash
python code/test_subcell.py     # 10 validation tests, incl. exact plane-wave characters
python code/verify_full.py      # 4 components x 3 learners x 2 seeds x 30 subsets
python code/final_check.py      # pre-registered summary with bootstrap CIs
```

NumPy, SciPy, scikit-learn only. One CPU core.

## Superseded record

Version 1 is retained at doi 10.5281/zenodo.22137052 and is not withdrawn. Its numerical
results are reproducible from its own deposit; they are simply computed on a dataset now
known to be duplicated.

## Licence

Code MIT; paper, figures and results CC BY 4.0.

David Mashiah — davidmashiah@mail.tau.ac.il — ORCID 0009-0004-4684-955X
