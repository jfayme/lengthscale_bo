# Rewrite — module 3: `lsab/reduce.py`

Inputs as merged: `lsab.datasets.Pool` (`frame`, `components`, `numeric`) and
`lsab.featurize.embed_component(rep, smiles, **kw) -> (X float32 with NaN
rows, used_rep)`.

## Job
Turn a `Pool` + a representation name + a reduction name into the numeric
matrix the GP will see, one row per candidate, and the metadata the sweep
records about it. Pure numpy/pandas; no torch, no BayBE, no sklearn.

This module is the seam that decides `d`, and `d` is the whole argument of the
chen rule, so every step below is deterministic and written down. Read the old
`gollum_pipeline.descriptor_frame`, `representations.descriptor_frame`,
`botorch_bo.load_pool` / `REDUCTIONS`, and BayBE's `df_uncorrelated_features`
(in the installed baybe, `baybe/utils/dataframe.py`) before writing.

## Contract

```python
@dataclass(frozen=True)
class Block:                      # one column block of the pool matrix
    name: str                     # component or numeric column name
    kind: Literal["component", "numeric"]
    columns: slice                # its columns in X
    rep: str | None               # requested rep (component blocks)
    used_rep: str | None          # after coverage fallback
    d_raw: int                    # embedding width before anything
    d_constant_dropped: int
    d_reduced: int                # == columns.stop - columns.start
    n_unique: int                 # unique molecules in the component (or unique values, numeric)
    n_failed: int                 # NaN rows imputed

@dataclass(frozen=True, eq=False)
class FeaturePool:
    pool: Pool
    rep: str
    reduction: str
    X: np.ndarray                 # (pool.n, D) float64, row i <-> pool.frame.iloc[i]
    blocks: tuple[Block, ...]     # in column order
    @property
    def d(self) -> int            # X.shape[1]; THIS is the d the chen rule sees
    def meta(self) -> dict        # flat, CSV-friendly: d, and per block "<name>.used_rep", "<name>.d_reduced", "<name>.n_failed"

REDUCTIONS: dict[str, Reduction] = {
    "decorr0.7": Decorrelate(threshold=0.7),   # house default
    "pca64":     PCA(variance=0.98, cap=64),   # the paper's
    "none":      Identity(),
}

def build(pool: Pool, rep: str, reduction: str = "decorr0.7", *,
          device: str = "cpu", cache_dir=None) -> FeaturePool
```

Keep the three reduction NAMES: they are an axis of the A/B and appear in
result files and the analysis.

## The pipeline, per component, in this order

1. `X, used_rep = embed_component(rep, unique_smiles)` — one row per UNIQUE
   molecule of the component, in `pool.components[name]` order.
2. **Impute** NaN rows with the per-column median of the finite rows (all-NaN
   column → 0.0). If `n_failed / n_unique > 0.01`, raise with the failed
   SMILES listed. Rationale: the candidate pool must be identical for every
   representation; dropping candidates would make the A/B compare different
   pools across reps.
3. **Drop constant columns**: `max - min <= 1e-8` on the raw features. Before
   any reduction, because a reduction fit on constant columns is ill-posed.
4. **Reduce** with `REDUCTIONS[reduction]` (below).
5. **Expand** to candidates: `X_component[index_of(candidate SMILES)]`.

Numeric blocks (Shields `Temp_C`, `Concentration`): raw values, one column
each, no reduction, no scaling. BoTorch `Normalize` in module 5 maps every
column to [0,1] against the pool bounds; nothing upstream scales anything.

Column blocks are concatenated in `pool.spec.components` order, then numeric
in `pool.spec.numeric` order. `X` is float64 (BoTorch runs in double).

## The reductions — implement exactly this

```python
class Decorrelate:
    """Greedy: visit columns in DESCENDING VARIANCE (stable sort), keep a column
    unless |corr| > threshold with an already-kept column. Output keeps the
    original column order.

    Why variance order: embedding columns have no meaning to their index, so
    'keep the first' (what BayBE does) lets the survivor of a correlated cluster
    be whichever came first. Descending variance keeps the most informative one.
    This is a deliberate difference from the old runs; README line."""
    threshold: float
    def fit_transform(self, X) -> np.ndarray   # X (n_unique, d) -> (n_unique, d')
```
Edge cases, all deterministic: `n_unique == 1` → every column is constant and
was dropped in step 3, so `d' == 0`; raise a clear error naming the component
(a one-molecule component is not a search dimension). `n_unique == 2` → every
pair of columns has |corr| = 1, so exactly one column survives; that is the
method's behaviour, log it at info level. Correlation computed with
`np.corrcoef` on float64; NaN cannot occur after step 3.

```python
class PCA:
    """Centred SVD (numpy, full_matrices=False). Keep the smallest k with
    cumulative explained variance >= `variance`, capped at `cap` and at
    rank = min(n_unique - 1, d). Deterministic sign: flip each component so its
    largest-|loading| entry is positive (BLAS-independent output)."""
    variance: float
    cap: int
    def fit_transform(self, X) -> np.ndarray

class Identity:
    def fit_transform(self, X) -> np.ndarray   # returns X
```
`k = int(np.searchsorted(cumulative, variance) + 1)` as before, then the caps.
No whitening. No sklearn.

## Dropped, on purpose
- BayBE `CustomDiscreteParameter` / `SearchSpace.from_product` / `comp_rep`:
  the product space is never built; rows are the measured candidates from
  module 1.
- BayBE's duplicate-row check and the 1e-7 tie-breaking noise it forced on
  the old code. Two candidates with identical features are a fact about the
  representation; the GP handles them (jitter). README line.
- `normalize=` global/local, the z-score for phys/mordred (both reps gone),
  `prefix` / `_pca{i}` column naming (column identity lives in `Block`, not
  in strings), `descriptor_frame`'s `df.attrs`.
- The "drop labels whose embedding failed" path: replaced by impute-or-raise
  above, one rule.

## CLI
`python -m lsab.reduce --dataset bh_reaction_1 shields --rep morgan mace_mp0 --reduction decorr0.7 pca64 none`
prints one line per (dataset, rep, reduction): `D`, then per block
`name: used_rep d_raw→d_reduced (n_unique, n_failed)`. This replaces the
`n, d` half of the old `preflight.py`; the geometry half comes in module 4.

Run it once for `morgan` on every dataset under `decorr0.7` and paste the `D`
column next to the old `preflight_before.txt` values in the module report. Not
a gate — nothing is binding — but it tells us how far the variance-ordered
decorrelation moves `d` from BayBE's, which is worth knowing before the sweep.

## Tests (`tests/test_reduce.py`) — synthetic, no models
Register a fake rep in `featurize.REPS` via monkeypatch that maps a SMILES to
a deterministic vector (e.g. seeded from `hash(smiles)`), so `build` runs end
to end on a real `Pool` without weights.

1. `test_decorrelate_drops_correlated`: 6 columns where c1 = 0.99·c0 + noise,
   c3 = -c2, c5 = c4 + tiny noise, and c0, c2 have the larger variances →
   survivors are `{0, 2, 4}` (c4 vs c5: the higher-variance one), output in
   original order; threshold 0.99999 keeps all six.
2. `test_decorrelate_edge_cases`: `n_unique == 2` → exactly one column;
   a one-row component raises with the component name in the message.
3. `test_pca`: an X with 3 dominant directions + small noise → k = 3 at
   98%; `cap` binds when set to 2; `k <= n_unique - 1` for `n_unique = 5,
   d = 100`; sign convention: `fit_transform(X)` equals `fit_transform(X)` on a
   copy with rows permuted, up to the permutation (i.e. no sign flips between
   runs); variance explained is monotone.
4. `test_constant_and_impute`: a constant column is removed; a NaN row is
   imputed with the column median and `n_failed == 1`; two NaN rows in a
   50-molecule component raise (4% > 1%).
5. `test_assembly_reaction_pool`: on `bh_reaction_1` with the fake rep,
   `reduction="none"`: `X.shape == (pool.n, sum of block widths)`; for 5
   random candidates, the row equals the concatenation of the component
   vectors looked up by SMILES; `blocks` slices tile `range(D)` exactly, in
   spec order.
6. `test_assembly_shields_numeric`: numeric blocks are raw values equal to
   `pool.frame[col]`, width 1, `kind == "numeric"`; component blocks carry
   `used_rep` (fake both reps and force a fallback to check it is recorded).
7. `test_meta_is_flat_and_csv_safe`: `meta()` values are str/int/float only;
   keys stable across two builds.
8. `test_reductions_registry`: the three names exist; `build` with an unknown
   name raises listing the valid ones.

## Definition of done
Tests 1–8 pass in a few seconds; the CLI runs for morgan on all datasets in
under a minute; `reduce.py` under ~220 lines; no old-tree imports; README
gains two "differences" lines (variance-ordered decorrelation; no duplicate
noise) and one paragraph on the impute-or-raise rule.

## When it's reviewed
Send me the merged `Block` / `FeaturePool` and the morgan `D` table
(new vs old `preflight_before.txt`). Module 4 (lengthscale rules, model space,
prior parameterisation, the geometry probe) is written against `FeaturePool`.
