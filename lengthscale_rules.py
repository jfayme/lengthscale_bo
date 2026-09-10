"""
lengthscale_rules.py
====================
Lengthscale prior centres. Each rule returns a scalar `ell_0` for one pool.

Two rules are compared in the A/B (see `lengthscale_ab.py`):

  chen : ell_0 = 0.4*sqrt(d) + 4        -- the published dimension-aware prior
         ("Adaptive_ours", Chen, Fleck & Stuyver, J. Chem. Theory Comput. 2026,
         DOI 10.1021/acs.jctc.6c00251). Implemented verbatim in
         HSF-ChemBO-tutorial/base/kernels.py::AdaptiveKernelFactory; that
         implementation is the specification. A function of the feature
         dimension ONLY -- blind to how the pool is actually spread out.

  geom : ell_0 = D_bar / u*             -- pool geometry
         D_bar = mean pairwise Euclidean distance of the pool, u* = the value of
         u = D_bar/ell that maximises u*|k'(u)| for the kernel in use, i.e. the
         lengthscale at which the kernel output varies most across the distances
         the pool actually contains. u* is a property of the KERNEL, not a fitted
         constant: for Matern-5/2, k(u) = (1 + sqrt5 u + 5u^2/3) exp(-sqrt5 u),
         so u*|k'(u)| is maximised at the positive root of 5u^2 - 2*sqrt5*u - 2,
         u* = (1 + sqrt3)/sqrt5 = 1.221810.  IF THE KERNEL CHANGES, u* MUST TOO
         (see USTAR).

Both rules are parameter-free given the pool. Nothing here is tuned per dataset.

CRITICAL: `X_model_space` must be the matrix THE KERNEL SEES -- post
dimensionality reduction AND post input transform (BoTorch `Normalize`, i.e. the
per-column min-max [0,1] cube). Use `to_model_space` to get there from raw pool
features and the model's own bounds. Computing D_bar on raw descriptors silently
produces a meaningless ell_star.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial.distance import pdist

# u* = argmax_u u*|k'(u)|, per kernel. Matern-5/2 is what this repo uses
# (botorch_bo.build_gp, base/kernels.py); the others are here so a kernel switch
# cannot silently keep the wrong constant.
USTAR = {
    "matern12": 1.0,                                   # k = exp(-u): maximal at u = 1
    "matern32": 2.0 / np.sqrt(3.0),                    # 1.154701
    "matern52": (1.0 + np.sqrt(3.0)) / np.sqrt(5.0),   # 1.221810
    "rbf": np.sqrt(2.0),                               # 1.414214
}
USTAR_MATERN52 = USTAR["matern52"]

DEGENERATE_CV = 1e-6   # coefficient of variation of the pairwise distances below
#                        which a pool carries no geometric information at all
#                        (e.g. a one-hot identity pool: every distance = sqrt(2)).


# ---------------------------------------------------------------------------
# The space the kernel sees
# ---------------------------------------------------------------------------
def to_model_space(X, bounds):
    """Map raw pool features into the space the GP kernel operates in.

    Replicates BoTorch `Normalize(d, bounds=bounds)` exactly: an affine per-column
    map to [0, 1] using the SAME bounds handed to the model (the full-pool
    per-column min/max in `botorch_bo.run_comparison`). Zero-range columns are
    passed through unscaled rather than producing NaN; the pipeline drops constant
    columns upstream, so this is a guard, not a policy.
    """
    X = np.asarray(X, dtype=float)
    bounds = np.asarray(bounds, dtype=float)
    offset = bounds[0]
    coefficient = bounds[1] - offset
    coefficient = np.where(np.abs(coefficient) > 0.0, coefficient, 1.0)
    return (X - offset) / coefficient


def _subsample(X, max_points, seed):
    X = np.asarray(X, dtype=float)
    if len(X) > max_points:
        idx = np.random.default_rng(seed).choice(len(X), max_points, replace=False)
        X = X[idx]
    return X


def pool_mean_distance(X_model_space, max_points=1500, seed=0):
    """Mean pairwise Euclidean distance, computed in THE SPACE THE KERNEL SEES.

    X_model_space must be post-reduction AND post-input-transform. Subsample for
    tractability; the estimate is stable well below 1500 points.
    """
    return float(pdist(_subsample(X_model_space, max_points, seed)).mean())


def pool_distance_stats(X_model_space, max_points=1500, seed=0):
    """(mean, std, cv, degenerate) of the pool's pairwise distances, model space.

    `degenerate` flags a pool whose distances are all identical (cv < 1e-6): the
    one-hot encoding of a single-component search space is the identity matrix,
    every pairwise distance is exactly sqrt(2), expected improvement ties across
    the whole pool and the campaign selects in file order. Such a cell carries no
    information about either lengthscale rule and must not enter an average.
    """
    distances = pdist(_subsample(X_model_space, max_points, seed))
    mean = float(distances.mean()) if distances.size else 0.0
    std = float(distances.std()) if distances.size else 0.0
    cv = std / mean if mean > 0 else 0.0
    return mean, std, cv, bool(cv < DEGENERATE_CV)


# ---------------------------------------------------------------------------
# The rules
# ---------------------------------------------------------------------------
def rule_chen(X_model_space, d):
    """Arm A: the published dimension-aware prior. Geometry-blind by construction."""
    return 0.4 * np.sqrt(d) + 4.0


def rule_geom(X_model_space, d):
    """Arm B: pool geometry, D_bar / u* for the Matern-5/2 kernel."""
    return pool_mean_distance(X_model_space) / USTAR_MATERN52


RULES = {"chen": rule_chen, "geom": rule_geom}


def ell_star(X_model_space, kernel="matern52"):
    """The kernel-matched geometric lengthscale D_bar / u*.

    Numerically identical to `rule_geom` for Matern-5/2; kept separate because it
    is also the DIAGNOSTIC denominator (`ell_0 / ell_star`) logged for the chen
    arm, where it is not the rule.
    """
    return pool_mean_distance(X_model_space) / USTAR[kernel]


# ---------------------------------------------------------------------------
# Prior parameterisation: where the centre points vs how hard the prior pulls
# ---------------------------------------------------------------------------
def gamma_prior_parameters(ell_0, prior_mode="match_concentration", cv=0.3):
    """(concentration, rate) of the Gamma lengthscale prior, mean = ell_0.

    prior_mode="match_parameterisation"
        Chen's tied form, reused verbatim with ell_0 swapped:
            Gamma(2*ell_0, 2)  ->  mean ell_0, CV = 1/sqrt(2*ell_0).
        The width DEPENDS ON THE CENTRE, so re-centring the prior also changes how
        tightly it is held. This answers "what happens if you drop the new rule
        into the existing code?" -- and it confounds centre with strength.

    prior_mode="match_concentration"
        Fixed coefficient of variation:
            concentration = 1/cv^2,  rate = concentration/ell_0.
        Identical prior tightness in both arms, so ONLY the centre differs. This
        is the scientifically clean comparison, and `cv=0.3` is the value
        `botorch_bo.build_gp` already uses.
    """
    ell_0 = float(ell_0)
    if not np.isfinite(ell_0) or ell_0 <= 0:
        raise ValueError(f"ell_0 must be finite and positive, got {ell_0}")
    if prior_mode == "match_parameterisation":
        return 2.0 * ell_0, 2.0
    if prior_mode == "match_concentration":
        concentration = 1.0 / (cv * cv)
        return concentration, concentration / ell_0
    raise ValueError(f"unknown prior_mode: {prior_mode}")


def prior_cv_of(concentration):
    """Coefficient of variation of Gamma(concentration, rate) -- independent of rate."""
    return float(1.0 / np.sqrt(concentration))


PRIOR_MODES = ("match_parameterisation", "match_concentration")
