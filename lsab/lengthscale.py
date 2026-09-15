"""
lsab/lengthscale.py
===================

Module 4 of the lengthscale-A/B rewrite: WHERE the GP lengthscale prior is centred
and how hard it pulls, resolved into plain numbers (`Prior`) for module 5. Pure
numpy + scipy `pdist`: no torch, no module-level RNG, no state.

  chen : ell_0 = 0.4*sqrt(d) + 4, the published prior (Chen, Fleck & Stuyver, JCTC
         2026, DOI 10.1021/acs.jctc.6c00251). Blind to the pool. The reference
         implementation was the tutorial's AdaptiveKernelFactory, which set
         `GammaPrior(2*ell_0, 2.0)` and started the kernel at `ell_0` -- i.e. exactly
         `gamma_parameters(ell_0, "match_parameterisation")` below. That file is no
         longer in the repo; this docstring is the spec now.
  geom : ell_0 = D_bar / u*. D_bar is the pool's mean pairwise distance; u* maximises
         u*|k'(u)|, so the kernel varies most across the distances the pool holds.
         u* belongs to the KERNEL: change the kernel and USTAR must change with it.
diamgate, the old per-iteration house prior, is not ported: chen and geom are the A/B.

All geometry is in MODEL SPACE: the pool through the per-column min-max map BoTorch's
`Normalize` applies. Module 5 hands `Normalize` the bounds from `pool_bounds`, so
there is ONE definition of the space the kernel sees; D_bar on raw features is
meaningless.

    python -m lsab.lengthscale     # the new preflight: DEFAULT_DATASETS x morgan x decorr0.7
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial.distance import pdist

from lsab.reduce import FeaturePool

USTAR = {   # u* = argmax_u u*|k'(u)|, per kernel; the pool's D_bar / u* is the geom centre
    "matern12": 1.0,
    "matern32": float(2.0 / np.sqrt(3.0)),
    "matern52": float((1.0 + np.sqrt(3.0)) / np.sqrt(5.0)),   # 1.221810 -- the kernel this repo uses
    "rbf": float(np.sqrt(2.0)),
}
KERNEL = "matern52"     # the one module 5 builds; ell_star() defaults to it
DEGENERATE_CV = 1e-6    # distance CV below which a pool has no geometry (one-hot: all sqrt(2))
PRIOR_MODES = ("match_parameterisation", "match_concentration")


# =============================================================================
# 1. MODEL SPACE AND THE ONE GEOMETRY PROBE
# =============================================================================
def pool_bounds(X) -> np.ndarray:
    """(2, D) float64 [min; max] per column, for BoTorch's Normalize in module 5.
    A zero-range column gets max = min + 1: Normalize divides by the raw range, so
    [c, c] would give 0/0 = NaN (botorch 0.18); widened, the column maps to 0."""
    X = np.asarray(X, dtype=np.float64)
    lo, hi = X.min(axis=0), X.max(axis=0)
    return np.stack([lo, np.where(hi > lo, hi, lo + 1.0)])


def to_model_space(X, bounds) -> np.ndarray:
    """(X - lo) / (hi - lo), a zero range read as 1: BoTorch's Normalize(d, bounds=bounds)."""
    X, bounds = np.asarray(X, dtype=np.float64), np.asarray(bounds, dtype=np.float64)
    width = bounds[1] - bounds[0]
    return (X - bounds[0]) / np.where(width > 0, width, 1.0)


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
    degenerate: bool        # cv < DEGENERATE_CV: every distance equal, no geometric information


def geometry(X_model, *, max_points: int = 2000, seed: int = 0) -> Geometry:
    """Pairwise Euclidean distances of a model-space pool: every row when n <= max_points
    (exact), else a without-replacement subsample from a FRESH default_rng(seed)."""
    X = np.asarray(X_model, dtype=np.float64)
    n, d = X.shape
    if n < 2:
        raise ValueError(f"geometry needs at least two rows, got {n}")
    sample = X[np.random.default_rng(seed).choice(n, max_points, replace=False)] if n > max_points else X
    distances = pdist(sample)
    mean, std = float(distances.mean()), float(distances.std())
    cv = std / mean if mean > 0 else 0.0
    return Geometry(n=n, d=d, n_used=len(sample), mean=mean, std=std, cv=cv,
                    median=float(np.median(distances)), p95=float(np.percentile(distances, 95)),
                    degenerate=bool(cv < DEGENERATE_CV))


def pool_geometry(fp: FeaturePool) -> Geometry:
    """`geometry` of the space the kernel sees for `fp`: what the sweep computes once per cell."""
    return geometry(to_model_space(fp.X, pool_bounds(fp.X)))


# =============================================================================
# 2. THE RULES AND THE PRIOR  (rules take a Geometry, never raw features)
# =============================================================================
def rule_chen(g: Geometry) -> float:
    """Arm A: 0.4*sqrt(d) + 4, the published rule. Geometry-blind by construction."""
    return float(0.4 * np.sqrt(g.d) + 4.0)


def rule_geom(g: Geometry) -> float:
    """Arm B: D_bar / u* for the kernel module 5 builds."""
    return g.mean / USTAR[KERNEL]


RULES = {"chen": rule_chen, "geom": rule_geom}


def ell_star(g: Geometry, kernel: str = KERNEL) -> float:
    """D_bar / u* for `kernel`: rule_geom for KERNEL, and the diagnostic denominator for chen."""
    return g.mean / USTAR[kernel]


def gamma_parameters(ell_0, prior_mode: str = "match_concentration", cv: float = 0.3) -> tuple[float, float]:
    """(concentration, rate) of the Gamma lengthscale prior; its mean is ell_0.
    match_parameterisation: Chen's tied Gamma(2*ell_0, 2). Its CV, 1/sqrt(2*ell_0), FOLLOWS
        the centre, so re-centring also changes how hard the prior pulls.
    match_concentration: a = 1/cv^2, Gamma(a, a/ell_0). One CV for both arms, so ONLY the
        centre differs: the clean comparison."""
    ell_0 = float(ell_0)
    if not (np.isfinite(ell_0) and ell_0 > 0):
        raise ValueError(f"ell_0 must be finite and positive, got {ell_0}")
    if prior_mode == "match_parameterisation":
        return 2.0 * ell_0, 2.0
    if prior_mode == "match_concentration":
        if not (np.isfinite(cv) and cv > 0):
            raise ValueError(f"cv must be finite and positive, got {cv}")
        concentration = 1.0 / (cv * cv)
        return concentration, concentration / ell_0
    raise ValueError(f"unknown prior_mode {prior_mode!r}; expected one of {', '.join(PRIOR_MODES)}")


def prior_cv_of(concentration) -> float:
    """Coefficient of variation of Gamma(concentration, rate); independent of rate."""
    return float(1.0 / np.sqrt(concentration))


@dataclass(frozen=True)
class Prior:
    rule: str
    prior_mode: str
    cv: float                # the requested cv (meaningful for match_concentration only)
    ell_0: float
    concentration: float
    rate: float

    def describe(self) -> str:
        return f"{self.rule}  l0={self.ell_0:.3g}  {self.prior_mode}  cv={self.cv:.2f}"


def make_prior(fp: FeaturePool, rule: str, prior_mode: str = "match_concentration",
               cv: float = 0.3, *, geometry: Geometry | None = None) -> Prior:
    """Resolve `rule` on this pool into the numbers module 5 consumes. `geometry`
    defaults to `pool_geometry(fp)`; the sweep passes the one it computed for the cell."""
    if rule not in RULES:
        raise ValueError(f"unknown rule {rule!r}; expected one of {', '.join(RULES)}")
    g = geometry if geometry is not None else pool_geometry(fp)
    if (g.n, g.d) != fp.X.shape:
        raise ValueError(f"geometry is of a {g.n}x{g.d} pool, not this {fp.X.shape[0]}x{fp.d} one")
    ell_0 = RULES[rule](g)
    return Prior(rule, prior_mode, float(cv), ell_0, *gamma_parameters(ell_0, prior_mode, cv))


# =============================================================================
# 3. THE NEW PREFLIGHT
# =============================================================================
def cell_summary(fp: FeaturePool, *, geometry: Geometry | None = None, rank: bool = False) -> dict:
    """One flat, CSV-safe row per (dataset, rep, reduction) cell, then fp.meta().
    `rank` (opt-in, an SVD of the whole pool) is the rank of the CENTRED model-space
    matrix: the dimension the distances live in, which the uncentred rank can
    overcount by one. It sits next to d because chen counts columns, not directions."""
    g = geometry if geometry is not None else pool_geometry(fp)
    row = {"dataset": fp.pool.name, "rep": fp.rep, "reduction": fp.reduction, "n": g.n, "d": g.d}
    if rank:
        X_model = to_model_space(fp.X, pool_bounds(fp.X))
        row["rank"] = int(np.linalg.matrix_rank(X_model - X_model.mean(axis=0)))
    ell_geom, ell_chen = rule_geom(g), rule_chen(g)
    row.update(n_used=g.n_used, D_bar=g.mean, cv=g.cv, degenerate=g.degenerate, ell_geom=ell_geom,
               ell_chen=ell_chen, ratio_chen_geom=ell_chen / ell_geom if ell_geom > 0 else float("inf"))
    row.update({key: value for key, value in fp.meta().items() if key not in row})
    return row


COLUMNS = ("dataset", "rep", "reduction", "n", "d", "rank", "D_bar", "cv", "degenerate",
           "ell_geom", "ell_chen", "ratio_chen_geom")


def main(argv: list[str] | None = None) -> None:
    import argparse

    from lsab import datasets
    from lsab.featurize import REPS
    from lsab.reduce import REDUCTIONS, build

    parser = argparse.ArgumentParser(prog="python -m lsab.lengthscale", description="Per cell: geometry, both centres.")
    parser.add_argument("--dataset", nargs="+", default=list(datasets.DEFAULT_DATASETS), choices=datasets.names())
    parser.add_argument("--rep", nargs="+", default=["morgan"], choices=list(REPS))
    parser.add_argument("--reduction", nargs="+", default=["decorr0.7"], choices=list(REDUCTIONS))
    parser.add_argument("--rank", action="store_true", help="add the centred model-space rank (an SVD per cell)")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--cache-dir", default=None)
    args = parser.parse_args(argv)
    columns = [c for c in COLUMNS if c != "rank" or args.rank]
    width = {"dataset": 18, "rep": 12, "reduction": 10}

    def cell(value, column):
        pad = width.get(column, 10)
        return f"{value:<{pad}.4g}" if isinstance(value, float) else f"{value!s:<{pad}}"

    print("  ".join(cell(c, c) for c in columns))
    for name in args.dataset:
        pool = datasets.load(name)
        for rep in args.rep:
            for reduction in args.reduction:
                row = cell_summary(build(pool, rep, reduction, device=args.device, cache_dir=args.cache_dir),
                                   rank=args.rank)
                print("  ".join(cell(row[c], c) for c in columns), flush=True)


if __name__ == "__main__":
    main()
