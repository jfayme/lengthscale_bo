# Rewrite — module 4: `lsab/lengthscale.py`

Input: `lsab.reduce.FeaturePool` as merged (`X` float64 `(n, D)`, `d`,
`blocks`, `meta()`). Output: a fully resolved `Prior` that module 5's GP
builder consumes as plain numbers, plus the per-cell diagnostics that replace
the old `preflight.py`.

Pure numpy + scipy `pdist`. No torch, no sklearn, no module-level RNG, no
state. The one test that compares against BoTorch's `Normalize` imports
botorch inside the test.

Read the old `lengthscale_rules.py` (the specification for chen/geom and the
model-space rule), `botorch_bo.compute_pool_geometry` / `LengthscaleSpec` /
`make_lengthscale_spec`, and `preflight.py` before writing.

## Decisions in force
- Rules: `chen`, `geom`. **diamgate is not ported** (see the chat note); the
  interface below would admit a per-iteration rule later, but nothing in
  module 5 should branch on it now.
- One geometry probe, one subsample size (2000), one seed (0), a fresh
  `np.random.default_rng(seed)` inside the function. The old code had two
  probes (1500-point mean for geom, 2000-point median/p95 for diamgate) fed
  by two module-level RNGs; that trap is gone with the state.
- Everything geometric is computed in **model space**: the raw `X` mapped
  through the same per-column min-max map BoTorch's `Normalize` applies with
  the pool bounds. Module 5 must hand `Normalize` the bounds from
  `pool_bounds(X)` here, so there is exactly one definition of the space the
  kernel sees.

## Contract

```python
USTAR = {                    # u* = argmax_u u*|k'(u)|, per kernel; the pool's D_bar / u* is the geom centre
    "matern12": 1.0,
    "matern32": 2.0 / sqrt(3.0),
    "matern52": (1.0 + sqrt(3.0)) / sqrt(5.0),    # 1.221810 -- the kernel this repo uses
    "rbf": sqrt(2.0),
}
KERNEL = "matern52"          # the one module 5 builds; ell_star() defaults to it
DEGENERATE_CV = 1e-6

def pool_bounds(X) -> np.ndarray            # (2, D) float64: [min; max] per column. Module 5 uses THIS for Normalize.
def to_model_space(X, bounds) -> np.ndarray # (X - lo) / where(hi - lo > 0, hi - lo, 1.0); replicates botorch Normalize incl. zero-range columns

@dataclass(frozen=True)
class Geometry:
    n: int                  # pool rows
    d: int                  # columns (== FeaturePool.d)
    n_used: int             # rows in the distance sample (min(n, max_points))
    mean: float             # D_bar -- the geom rule's input
    std: float
    cv: float               # std / mean (0.0 if mean == 0)
    median: float
    p95: float
    degenerate: bool        # cv < DEGENERATE_CV: every distance equal, the cell carries no geometric information

def geometry(X_model, *, max_points=2000, seed=0) -> Geometry
    # pdist(euclidean) on a without-replacement subsample when n > max_points, else on all rows.
    # n_used == n means exact. Deterministic for a given seed.

def rule_chen(g: Geometry) -> float         # 0.4 * sqrt(g.d) + 4.0 -- the published rule; geometry-blind by construction
def rule_geom(g: Geometry) -> float         # g.mean / USTAR[KERNEL]
RULES = {"chen": rule_chen, "geom": rule_geom}
def ell_star(g: Geometry, kernel: str = KERNEL) -> float   # g.mean / USTAR[kernel]; == rule_geom for KERNEL; logged as the diagnostic denominator

PRIOR_MODES = ("match_parameterisation", "match_concentration")
def gamma_parameters(ell_0, prior_mode="match_concentration", cv=0.3) -> tuple[float, float]
    # match_parameterisation : Gamma(2*ell_0, 2)                 -- Chen's tied form; CV = 1/sqrt(2*ell_0) follows the centre
    # match_concentration    : a = 1/cv^2, Gamma(a, a/ell_0)     -- fixed CV; only the centre differs between arms
    # raises unless ell_0 is finite and > 0
def prior_cv_of(concentration) -> float     # 1/sqrt(concentration)

@dataclass(frozen=True)
class Prior:
    rule: str
    prior_mode: str
    cv: float                # the requested cv (meaningful for match_concentration only)
    ell_0: float
    concentration: float
    rate: float
    def describe(self) -> str            # e.g. "geom  l0=1.83  match_concentration  cv=0.30"

def make_prior(fp: FeaturePool, rule: str, prior_mode: str = "match_concentration",
               cv: float = 0.3, *, geometry: Geometry | None = None) -> Prior
    # geometry(to_model_space(fp.X, pool_bounds(fp.X))) unless one is passed in (the sweep computes it once per cell)

def cell_summary(fp: FeaturePool, *, geometry: Geometry | None = None, rank: bool = False) -> dict
    # flat, CSV-safe: dataset, rep, reduction, n, d, [rank], n_used, D_bar, cv, degenerate,
    # ell_geom, ell_chen, ratio_chen_geom, plus everything from fp.meta()
```

`rank`, when requested, is `np.linalg.matrix_rank(X_model)` — an SVD of the
whole pool matrix, so it is opt-in (seconds for the 10k-row pools). It exists
because `d` counts columns and the chen rule is a function of `d`, while the
pool can have far fewer independent directions (module 3 found 1,145 columns
for 720 additives, 6 for 4 ligands). The analysis will want `d`, `rank` and
`ratio_chen_geom` side by side.

## Dropped, on purpose
- diamgate, `_data_lengthscale`, the sklearn dependency, `NMIN` / `R_RHO`.
- `LengthscaleSpec` as a *lazy* spec with a rule string resolved inside the GP
  builder. `Prior` is resolved here; module 5 receives numbers.
- `lengthscale_ab_factory` and the BayBE kernel factory; all of
  `hyperprior_general.py`.
- `pool_mean_distance` / `pool_distance_stats` as separate probes → one
  `geometry`.
- Any `rule(X, d)` signature: rules take a `Geometry`, so a rule cannot
  accidentally be handed raw-space features.

## CLI — the new preflight
```
python -m lsab.lengthscale                                  # DEFAULT_DATASETS x morgan x decorr0.7
python -m lsab.lengthscale --dataset shields bh_full --rep morgan mace_mp0 --reduction decorr0.7 pca64 --rank
```
One row per cell: `dataset rep reduction n d [rank] D_bar cv degenerate ell_geom ell_chen ratio`.
Same column names as `cell_summary`. Nothing else.

## Tests (`tests/test_lengthscale.py`)
1. `test_ustar_is_the_argmax`: for Matern-5/2, `k(u) = (1 + sqrt5 u + 5u^2/3) exp(-sqrt5 u)`;
   maximise `u * |k'(u)|` numerically on a fine grid (or with
   `scipy.optimize.minimize_scalar`) and assert the argmax equals
   `USTAR["matern52"]` to 5 places; same for matern12 and rbf. This pins the
   constant to its definition, not to a number someone typed.
2. `test_model_space_matches_botorch_normalize` (imports botorch inside):
   random `X` with offset and scale and one constant column; `to_model_space(X,
   pool_bounds(X))` equals `Normalize(d, bounds=torch.tensor(pool_bounds(X)))
   .transform(torch.tensor(X))` to 1e-12, constant column included.
3. `test_geom_scale_equivariance`: `rule_geom(geometry(c*Xm)) == c * rule_geom(geometry(Xm))`
   to 10 places for c in (0.25, 3, 17.5); `rule_chen` is unchanged.
4. `test_geometry_is_model_space`: rescaling raw columns by random factors
   changes the raw-space D_bar but leaves `geometry(to_model_space(...)).mean`
   unchanged to 10 places.
5. `test_geometry_exact_and_subsampled`: for n < max_points, `n_used == n`
   and `mean` equals `pdist(X).mean()` exactly; for n > max_points, two calls
   with the same seed are identical and two seeds differ; `n_used ==
   max_points`.
6. `test_degenerate`: `np.eye(40)` → `degenerate`, `mean == sqrt(2)` to 10
   places; a random pool is not degenerate.
7. `test_prior_modes`: match_concentration → `prior_cv_of(concentration) == cv`
   for several centres; match_parameterisation → CV `== 1/sqrt(2*ell_0)`,
   decreasing in `ell_0`; both modes: `concentration / rate == ell_0`;
   `ell_0 <= 0` or NaN raises; unknown mode raises.
8. `test_make_prior_on_a_pool`: with the module-3 fake rep on
   `bh_reaction_1`, `make_prior` for both rules gives finite positive
   `ell_0`, the two differ, `describe()` contains the rule name, and passing a
   precomputed `geometry` gives the identical `Prior`.
9. `test_cell_summary_flat`: values are str/int/float/bool; keys are the
   documented set plus `fp.meta()` keys; `--rank` adds exactly `rank`, with
   `rank <= min(n, d)`.

## Definition of done
Tests 1–9 pass (2 needs botorch, present in `aimnet-bo`); the default CLI
runs in under a minute; the module imports only numpy/scipy/dataclasses and
`lsab.reduce`; under ~180 lines; README gets one "differences" line (single
2000-point geometry probe, seed 0, instead of two probes) and the diamgate
note ("not ported; chen and geom are the A/B").

## When it's reviewed
Send me the merged `Geometry` / `Prior` / `make_prior` / `pool_bounds`
signatures and the default CLI table (with `--rank` if it finishes in
reasonable time). Module 5 (the GP + the pool-BO loop, qLogEI only) is written
against `Prior` and `pool_bounds`.
