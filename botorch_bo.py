"""
botorch_bo.py  --  THE WORKHORSE
================================
Pool-based Bayesian optimisation in pure BoTorch, used to test our dynamic
acquisition-function policy (the hand-editable rule book in `bo_selection_v2.py`)
against a static qLogEI reference, on a fixed candidate pool.

WHY PURE BOTORCH (and not BayBE):
    BayBE silently collapses the entropy-search acquisition functions (MES/GIBBON,
    PES, JES) to a qNIPV proxy, so it cannot test the *real* ones. Here the
    acquisition function is just an argmax over the (fixed) candidate pool -- no
    `optimize_acqf` -- so every BoTorch acquisition function works as intended,
    including the genuine entropy-search family.

WHAT A "RUN" DOES:
    For each Monte-Carlo repeat we draw a random set of initial experiments, then
    run TWO campaigns from that SAME initial set so the comparison is paired:
        * TESTED     : our dynamic v2 rule book chooses the acquisition function
                       afresh every iteration (full per-iteration rule trace kept).
        * REFERENCE  : a fixed static acquisition function every iteration
                       (default qLogEI -- the strong, standard baseline).
    We track, per iteration and averaged over the Monte-Carlo repeats:
        * AUC               : area under the normalised cumulative-best-objective curve.
        * Top-5% coverage   : fraction of the pool's top-5% experiments discovered.
    After the run, `bo_report.py` writes a self-contained HTML report into
    `run_output/` with the settings, the metric tables, the per-iteration graphs,
    and the dynamic policy's per-iteration rule-trigger table.

FIXED DEFAULT SETTINGS (our house style -- applied to every campaign here):
    * decorrelation 0.7 on the featurisation         (see load_pool)
    * Matern-5/2 ARD kernel                           (one lengthscale per dim)
    * the "diamgate" lengthscale prior                (median-anchored centre that
      interpolates toward a smooth target as the data proves the objective smooth,
      with a NARROW cv=0.3 Gamma prior so the prior, not noise, sets the scale)
    * Input Normalize([0,1]) + outcome Standardize
    * 10 Monte-Carlo repeats x 50 iterations (5 random-init), shared seeds.

COMMAND LINE (see `python botorch_bo.py --help` for the full list):
    # default run: dynamic v2 vs static qLogEI on shields/morgan, 10 MC x 50 iter
    python botorch_bo.py

    # a different dataset + representation, more iterations
    python botorch_bo.py --dataset bh_full --rep mace_mp0 --iter 60

    # change the reference acquisition function, fewer MC repeats for a quick look
    python botorch_bo.py --dataset photoswitches --reference EI --mc 4

    # run the rules from a saved config file instead of the in-file defaults
    python botorch_bo.py --config v13_noceil_win15

    # just list what is available
    python botorch_bo.py --list-datasets
    python botorch_bo.py --list-afs
"""

from __future__ import annotations
import os
import sys
import time
import argparse
import warnings

warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, cdist

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "HSF-ChemBO-tutorial"))

import torch

from dataclasses import dataclass
from typing import Optional

torch.set_default_dtype(torch.double)

import hyperprior_general as H
from hyperprior_general import _data_lengthscale  # the standalone roughness probe
import lengthscale_rules as LR  # the lengthscale-rule seam (chen / geom / diamgate)

from botorch.models import SingleTaskGP
from botorch.models.transforms.input import Normalize
from botorch.models.transforms.outcome import Standardize
from botorch.fit import fit_gpytorch_mll
from gpytorch.kernels import MaternKernel
from gpytorch.priors import GammaPrior
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.acquisition import (
    qLogExpectedImprovement,
    qExpectedImprovement,
    qUpperConfidenceBound,
    qProbabilityOfImprovement,
    PosteriorMean,
)
from botorch.acquisition.max_value_entropy_search import qLowerBoundMaxValueEntropy
from botorch.acquisition.predictive_entropy_search import qPredictiveEntropySearch
from botorch.acquisition.joint_entropy_search import qJointEntropySearch
from botorch.generation import MaxPosteriorSampling
from af_selection import AFSelector, HeuristicConfig, load_config

OUTPUT_DIR = os.path.join(HERE, "botorch_out")  # trace CSVs (machine-readable)
os.makedirs(OUTPUT_DIR, exist_ok=True)
_GLOBAL_RNG = np.random.default_rng(0)  # only for sub-sampling the geometry probe

# Acquisition functions this workhorse can score over the pool (CLI: --reference, and
# the names the v2 rule book may emit). Listed for --list-afs and validation.
SUPPORTED_ACQUISITION_FUNCTIONS = [
    "qLogEI",
    "LogEI",
    "EI",  # expected improvement family
    "UCB",
    "PI",  # upper confidence bound / probability of improvement
    "PosMean",
    "PosSTD",  # pure exploit (posterior mean) / pure explore (posterior std)
    "MES",
    "GIBBON",
    "PES",
    "JES",  # information / entropy search family (the real ones)
    "TS",  # Thompson sampling
]

# Cap on the fixed reference set used for the GP posterior-uncertainty signal.
# Mean posterior std converges as 1/sqrt(M) (independent of dimension) and the set
# is frozen, so a few thousand points give a clean, comparable cross-iteration
# trend; the whole pool is used when smaller than this.
UNCERTAINTY_REFERENCE_CAP = 5000

# Beta for UCB when the rule book selects it as the AGGRESSIVE explorer. Outcomes
# are Standardize'd (signal std ~1), so this is ~ mean + sqrt(beta)*std: beta=9
# ~= mean + 3*std (strongly exploratory). Lower it to make UCB less aggressive.
UCB_BETA = 9.0


@dataclass
class BOState:
    N: int
    N_rem: int
    D: int
    f_worst_so_far: float
    f_best_so_far: float
    shortest_distance: float
    ls_min: float = 0.0
    ls_max: float = 0.0
    ls_mean: float = 0.0
    ls_std: float = 0.0
    outputscale: float = 0.0
    current_af: Optional[str] = None
    f_mean: float = 0.0  # mean of observed objective values
    f_std: float = 0.0  # std of observed objective values
    f_latest: float = 0.0  # objective value of the most recent pick (for AF trust)
    posterior_uncertainty: float = 0.0  # mean posterior std over unsampled pool
    noise: float = 0.0  # fitted GaussianLikelihood noise

    @property
    def range_width(self) -> float:
        return max(self.f_best_so_far - self.f_worst_so_far, 1e-12)

    @property
    def total_budget(self) -> int:
        return self.N + self.N_rem

    @property
    def progress(self) -> float:
        return self.N / max(self.total_budget, 1)


# =============================================================================
# 1. POOL EXTRACTION  --  features + matched objective values for a dataset/representation
# =============================================================================
# Dimensionality-reduction modes -> (pca_cap, decorrelate) for `H._campaign`.
# The reduction is what sets `d`, and the chen rule is a function of `d` alone, so
# it is an axis of the lengthscale A/B, not an implementation detail.
REDUCTIONS = {
    "decorr0.7": (None, 0.7),   # the house default: BayBE decorrelation, per parameter
    "pca64": (64, False),       # the paper's: PCA to 98% variance, capped at 64
    "none": (None, False),      # native dimensionality (large for text/MLIP reps)
}


def load_pool(dataset, representation, reduction="decorr0.7"):
    """Return (X_pool, y_pool, n_dimensions) for one dataset + representation.

    Reuses BayBE's exact featurisation (the post-reduction computational
    representation, `searchspace.discrete.comp_rep`) so results stay comparable to
    the rest of the project, and matches each pool row to its measured yield.
    Rows whose yield is unmeasured (some Buchwald-Hartwig product combinations) are
    dropped. `reduction` defaults to the house decorrelation-0.7 setting, i.e.
    unchanged behaviour; see `REDUCTIONS`.
    """
    H._GenFactory.STRATEGY = "diamgate"  # our house lengthscale strategy
    loaded = H._load(dataset)
    lookup_table = loaded["lookup"]
    if reduction not in REDUCTIONS:
        raise ValueError(f"unknown reduction {reduction!r}; expected one of "
                         f"{', '.join(REDUCTIONS)}")
    pca_cap, decorrelate = REDUCTIONS[reduction]
    campaign, n_dimensions = H._campaign(
        representation, dataset, loaded, pca_cap, decorrelate
    )
    searchspace = campaign.searchspace

    parameter_names = [p.name for p in searchspace.parameters]
    experiment_labels = searchspace.discrete.exp_rep.reset_index(
        drop=True
    )  # SMILES / numeric labels
    computational_features = searchspace.discrete.comp_rep.reset_index(
        drop=True
    )  # aligned feature matrix

    join_columns = [c for c in parameter_names if c in lookup_table.columns]
    matched_objective_values = experiment_labels.merge(
        lookup_table[join_columns + ["yield"]].drop_duplicates(join_columns),
        on=join_columns,
        how="left",
    )["yield"].values.astype(float)

    feature_matrix = computational_features.values.astype(float)
    measured = ~np.isnan(matched_objective_values)  # keep only measured combinations
    # `problem_dim` = number of experimental parameters (NOT the feature-space
    # dimension); the rule book uses it to set the base AF-switch patience.
    problem_dim = len(parameter_names)
    return (
        feature_matrix[measured],
        matched_objective_values[measured],
        int(n_dimensions),
        problem_dim,
    )


# =============================================================================
# 2. THE "DIAMGATE" KERNEL  --  our default GP prior (median-anchored, narrow)
# =============================================================================
def compute_pool_geometry(feature_matrix, max_points=2000):
    """Summarise the pool's geometry in the per-column min-max-normalised space.

    Returns (median_distance, p95_distance, per_column_min, per_column_range):
        * median_distance : the rough lengthscale of the pool.
        * p95_distance    : a stable diameter, used as the "smooth" target.
        * per_column_min / per_column_range : bounds, reused to map train points
          into the same normalised space as the geometry probe.
    """
    X = feature_matrix
    if X.shape[0] > max_points:
        X = X[_GLOBAL_RNG.choice(X.shape[0], max_points, replace=False)]
    column_min, column_max = X.min(0), X.max(0)
    column_range = np.where(column_max > column_min, column_max - column_min, 1.0)
    normalised_distances = pdist((X - column_min) / column_range, "euclidean")
    return (
        float(np.median(normalised_distances)),
        float(np.percentile(normalised_distances, 95)),
        column_min,
        column_range,
    )


def diamgate_lengthscale_centre(geometry, train_x, train_y, min_points=10):
    """The diamgate prior-mean lengthscale.

    l0 = median + (smooth_target - median) * smoothness
        * median        : geometry-anchored, rough-safe starting scale.
        * smooth_target : p95 / 0.50  (the kernel-saturated "smooth objective" scale).
        * smoothness    : in [0,1], grown from a quick MLE roughness probe on the
                          measured points once at least `min_points` are in. Until
                          then we stay at the rough geometry scale.
    """
    median_distance, p95_distance, column_min, column_range = geometry
    geometry_scale = max(median_distance, 1e-3)
    smooth_target = p95_distance / 0.50  # 1 / R_RHO = 2.0
    if train_y is None or len(train_y) < min_points:
        return geometry_scale
    probed_lengthscale = _data_lengthscale(train_x, train_y, column_min, column_range)
    if probed_lengthscale is None:
        return geometry_scale
    smoothness = min(max((probed_lengthscale / geometry_scale - 1.0) / 2.0, 0.0), 1.0)
    return geometry_scale + (smooth_target - geometry_scale) * smoothness


@dataclass
class LengthscaleSpec:
    """WHERE the lengthscale prior is centred -- the only thing an A/B arm varies.

    This is the single seam for the lengthscale rule; nothing else in the BO loop
    branches on it (see `lengthscale_rules.py` for the rules themselves).

        rule       : "diamgate" -- the house default: a median-anchored centre
                     recomputed EVERY iteration from the measured points;
                     "chen" / "geom" -- per-pool constants, computed once from the
                     model-space pool by `lengthscale_rules.RULES`.
        ell_0      : the fixed centre for chen/geom; None for diamgate.
        prior_mode : how the Gamma prior's width is set around that centre
                     ("match_concentration" fixes the CV, "match_parameterisation"
                     reuses Chen's tied Gamma(2*l0, 2) whose width follows the
                     centre -- see `lengthscale_rules.gamma_prior_parameters`).
        cv         : the coefficient of variation for "match_concentration".

    The default instance reproduces the incumbent house prior exactly (diamgate
    centre, narrow cv=0.3 Gamma), so `build_gp` with no spec is unchanged.
    """

    rule: str = "diamgate"
    ell_0: Optional[float] = None
    prior_mode: str = "match_concentration"
    cv: float = 0.3

    def centre(self, geometry, train_x, train_y):
        """The prior-mean lengthscale for this iteration."""
        if self.rule == "diamgate":
            return diamgate_lengthscale_centre(geometry, train_x, train_y)
        if self.ell_0 is None:
            raise ValueError(f"rule {self.rule!r} needs a precomputed ell_0")
        return float(self.ell_0)

    def gamma_parameters(self, centre):
        """(concentration, rate) of the Gamma lengthscale prior at `centre`."""
        return LR.gamma_prior_parameters(centre, self.prior_mode, self.cv)

    def describe(self):
        centre = "per-iteration" if self.ell_0 is None else f"{self.ell_0:.4g}"
        return f"{self.rule} (l0={centre}, {self.prior_mode}, cv={self.cv})"


DEFAULT_LENGTHSCALE_SPEC = LengthscaleSpec()  # the incumbent: diamgate, cv=0.3


def make_lengthscale_spec(
    feature_matrix,
    bounds,
    n_dimensions,
    lengthscale_rule=None,
    prior_mode="match_concentration",
    prior_cv=0.3,
):
    """Build a campaign's `LengthscaleSpec` from the pool.

    `lengthscale_rule=None` (or "diamgate") -> the incumbent house prior, unchanged.
    "chen" / "geom" -> a constant centre computed ONCE, in the space the kernel
    sees: the raw features pushed through the model's own `Normalize` bounds
    (`lengthscale_rules.to_model_space`). Computing it on raw descriptors instead
    would silently produce a meaningless centre for the geom arm.
    """
    if lengthscale_rule in (None, "diamgate"):
        return LengthscaleSpec(
            rule="diamgate", ell_0=None, prior_mode=prior_mode, cv=prior_cv
        )
    if lengthscale_rule not in LR.RULES:
        raise ValueError(
            f"unknown lengthscale rule {lengthscale_rule!r}; "
            f"expected one of diamgate, {', '.join(LR.RULES)}"
        )
    model_space_features = LR.to_model_space(feature_matrix, np.asarray(bounds))
    ell_0 = float(LR.RULES[lengthscale_rule](model_space_features, n_dimensions))
    return LengthscaleSpec(
        rule=lengthscale_rule, ell_0=ell_0, prior_mode=prior_mode, cv=prior_cv
    )


def build_gp(train_x, train_y, n_dimensions, geometry, bounds, lengthscale_spec=None):
    """A BoTorch SingleTaskGP with our default diamgate kernel.

    Matern-5/2 ARD kernel; lengthscale centre from `diamgate_lengthscale_centre`
    wrapped in a NARROW Gamma prior (coefficient of variation 0.3) so the prior,
    not sampling noise, pins the scale. NO ScaleKernel / outputscale: outcomes are
    Standardize'd to unit variance, so the signal variance is ~1 by construction
    (BoTorch's modern default). This removes the mis-centred amplitude factor that
    otherwise drifts the GP posterior uncertainty between iterations. Inputs
    Normalize([0,1]) on the pool bounds.

    `lengthscale_spec` swaps ONLY the prior centre (and, explicitly, the prior's
    width parameterisation); omitted, it is the incumbent diamgate prior above.
    """
    spec = lengthscale_spec if lengthscale_spec is not None else DEFAULT_LENGTHSCALE_SPEC
    lengthscale_centre = spec.centre(
        geometry, train_x.numpy(), train_y.numpy().ravel()
    )
    concentration, rate = spec.gamma_parameters(lengthscale_centre)

    base_kernel = MaternKernel(
        nu=2.5,
        ard_num_dims=n_dimensions,  # ARD: one lengthscale per dimension
        lengthscale_prior=GammaPrior(concentration, rate),
    )
    base_kernel.lengthscale = lengthscale_centre

    return SingleTaskGP(
        train_x,
        train_y,
        covar_module=base_kernel,  # no ScaleKernel: rely on Standardize for unit signal variance
        input_transform=Normalize(n_dimensions, bounds=bounds),
        outcome_transform=Standardize(1),
    )


def fit_gp(gp):
    """Fit the GP hyperparameters by exact marginal-likelihood maximisation."""
    fit_gpytorch_mll(ExactMarginalLogLikelihood(gp.likelihood, gp))
    return gp


# =============================================================================
# 3. ACQUISITION SCORING  --  score every pool candidate, then argmax
# =============================================================================
def score_candidates(acquisition_name, gp, candidate_x, pool_x, best_f):
    """Return a 1-D tensor scoring every candidate under `acquisition_name`.
    The next experiment is the candidate with the highest score in this
    tensor."""

    candidate_q = candidate_x.unsqueeze(1)  # (n_candidates, q=1, d)

    # Expected improvement family
    # Higher = better expected improvement over the current best (best_f).
    if acquisition_name in ("qLogEI", "LogEI", "EI"):
        acq = (
            qExpectedImprovement(gp, best_f=best_f)
            if acquisition_name == "EI"
            else qLogExpectedImprovement(gp, best_f=best_f)
        )
        return acq(candidate_q)

    # Upper confidence bound
    # Higher = more optimistic estimate of the candidate's potential.
    if acquisition_name == "UCB":
        return qUpperConfidenceBound(gp, beta=UCB_BETA)(candidate_q)

    # Posterior mean
    # Higher = higher predicted mean value.
    if acquisition_name in ("PosMean", "PM"):
        return PosteriorMean(gp)(candidate_q)

    # Max-value Entropy Search
    # Higher = more informative for finding the global optimum.
    if acquisition_name in ("MES", "GIBBON"):
        return qLowerBoundMaxValueEntropy(gp, candidate_set=pool_x)(candidate_q)

    # Predictive Entropy Search
    # Higher = more reduction in entropy (uncertainty) about the location of
    # the optimum.
    if acquisition_name == "PES":
        with torch.no_grad():
            top_pool = (
                gp.posterior(pool_x)
                .mean.view(-1)
                .topk(min(10, pool_x.shape[0]))
                .indices
            )
        return qPredictiveEntropySearch(gp, optimal_inputs=pool_x[top_pool])(
            candidate_q
        )

    # Thompson sampling
    # The candidate chosen by Thompson Sampling gets the highest score
    # (closest to zero).
    if acquisition_name == "TS":
        with torch.no_grad():
            chosen = MaxPosteriorSampling(gp, replacement=False)(
                candidate_x, num_samples=1
            )
        # Mark the Thompson-chosen candidate as the argmax via a one-hot-like score.
        squared_distance_to_choice = (candidate_x - chosen).pow(2).sum(-1)
        return -squared_distance_to_choice

    raise ValueError(f"unknown acquisition function: {acquisition_name}")


# =============================================================================
# 4. ACQUISITION FUNCTIONS  --  uniform interface so one loop drives both campaigns
# =============================================================================
class StaticAcquisitionFunction:
    """Always returns the same fixed acquisition function (the reference baseline)."""

    needs_gp_state = False  # no GP-state stats required

    def __init__(self, acquisition_name):
        self.acquisition_name = acquisition_name

    def choose(self, selection_state, f_best_so_far):
        # return the fixed acquisition function
        return self.acquisition_name, None


class DynamicAcquisitionFunction:
    """Our hand-editable rule book (bo_selection_v2.BOSelectionV2), wrapped to the
    uniform interface and emitting a per-iteration trace row."""

    needs_gp_state = True  # the rules read GP-state stats

    def __init__(self, selector):
        self.selector = selector

    def choose(self, selection_state, f_best_so_far):
        # call select function from af_selection.AFSelector
        acquisition_name, _rule_tag, _reason, diagnostics = self.selector.select(
            selection_state, f_best_so_far
        )

        # trace_row records all the meta-data (e.g. iteration number, MC
        # repeat...)
        trace_row = dict(af=acquisition_name, **diagnostics)
        return acquisition_name, trace_row


def build_state(
    gp,
    reference_x,
    sampled_indices,
    y_pool,
    normalised_features,
    n_dimensions,
    total_iterations,
    current_af,
):
    """Assemble the BOState the rule book reads, in the MAXIMISATION frame.

    The objective is maximised directly: f_best_so_far is the incumbent (best observed) and
    should go UP. Lengthscale comes from the fitted GP; `shortest_distance` is the
    nearest-neighbour distance of the most recent pick to the rest, in the
    normalised feature space. `f_latest` is the most recent pick's objective value,
    which the rule book uses to score the trust of the AF that chose it.

    `posterior_uncertainty` is the mean GP posterior std over a FIXED reference set
    (`reference_x`, frozen once per campaign), so the value is comparable across
    iterations (the support does not change). There is no ScaleKernel, so the
    signal variance is ~1 by construction (Standardize); we report outputscale=1.0
    for consumers that still reference it.
    """
    lengthscales = gp.covar_module.lengthscale.detach().numpy().ravel()
    outputscale = 1.0  # no ScaleKernel: signal variance pinned to ~1 by Standardize

    objective_values = y_pool[sampled_indices]

    # compute the nearest-neighbour distance (minimum euclidean distance) of
    # the most recent pick to the rest of the sampled points in the normalised
    # feature space
    sampled_features = normalised_features[sampled_indices]
    if len(sampled_features) > 1:
        nearest_neighbour_distance = float(
            cdist(sampled_features[-1:], sampled_features[:-1]).min()
        )
    else:
        nearest_neighbour_distance = float("nan")

    # mean GP posterior std over the FIXED reference set, plus the fitted noise
    with torch.no_grad():
        post = gp.posterior(reference_x)
        posterior_uncertainty = float(post.variance.sqrt().mean())
    noise = float(gp.likelihood.noise.detach().mean())

    return BOState(
        N=len(sampled_indices),
        N_rem=total_iterations - len(sampled_indices),
        D=n_dimensions,
        f_worst_so_far=float(objective_values.min()),
        f_best_so_far=float(objective_values.max()),
        f_mean=float(objective_values.mean()),
        f_std=float(objective_values.std()),
        f_latest=float(objective_values[-1]),
        shortest_distance=nearest_neighbour_distance,
        ls_min=float(lengthscales.min()),
        ls_max=float(lengthscales.max()),
        ls_mean=float(lengthscales.mean()),
        ls_std=float(lengthscales.std()),
        outputscale=outputscale,
        current_af=current_af,
        posterior_uncertainty=posterior_uncertainty,
        noise=noise,
    )


# =============================================================================
# 5. ONE CAMPAIGN  --  run a single MC repeat for a single policy
# =============================================================================
def run_one_campaign(
    policy,
    pool_x,
    y_pool,
    normalised_features,
    n_dimensions,
    geometry,
    bounds,
    initial_indices,
    total_iterations,
    mc_index,
    record_trace,
    lengthscale_spec=None,
    record_fit=False,
):
    """Run one Bayesian-optimisation campaign from a fixed initial set.

    Returns a dict with the per-iteration trajectories (cumulative-best, running
    AUC, running coverage will be computed by the caller) plus the sampled-objective-value
    sequence and -- if `record_trace` -- the dynamic acquisition function's rule-trace rows.

    `lengthscale_spec` selects the lengthscale prior centre (None = the incumbent
    diamgate prior). `record_fit` additionally logs the POST-marginal-likelihood
    lengthscales each iteration, which is what tells you whether the prior did
    anything at all: if the fit lands in the same place from both starting points,
    the arms should tie, and that is a result rather than a null.
    """
    sampled_indices = list(initial_indices)
    last_acquisition = None
    trace_rows = []
    fit_rows = []

    # Fixed reference set for the GP posterior-uncertainty signal: the whole pool
    # if small enough, else a frozen random subset capped at UNCERTAINTY_REFERENCE_CAP.
    # Sampled once per campaign and never changed, so mean posterior std over it is
    # comparable across iterations (constant support, incl. points later selected).
    if len(pool_x) <= UNCERTAINTY_REFERENCE_CAP:
        reference_x = pool_x
    else:
        generator = torch.Generator().manual_seed(mc_index)
        reference_idx = torch.randperm(len(pool_x), generator=generator)[
            :UNCERTAINTY_REFERENCE_CAP
        ]
        reference_x = pool_x[reference_idx]

    for iteration in range(len(sampled_indices), total_iterations):
        index_tensor = torch.tensor(sampled_indices)
        train_x = pool_x[index_tensor]
        train_y = torch.tensor(y_pool[sampled_indices]).unsqueeze(-1)
        gp = fit_gp(
            build_gp(
                train_x, train_y, n_dimensions, geometry, bounds, lengthscale_spec
            )
        )

        if record_fit:
            fitted_lengthscales = gp.covar_module.lengthscale.detach().numpy().ravel()
            log_fitted = np.log(fitted_lengthscales)
            fit_rows.append(
                dict(
                    mc=mc_index,
                    n_exp=iteration,
                    # geometric mean across the ARD dimensions + the spread of the
                    # per-dimension lengthscales in log space
                    ell_fitted_mean=float(np.exp(log_fitted.mean())),
                    ell_fitted_log_sd=float(log_fitted.std()),
                )
            )

        f_best_so_far = float(y_pool[sampled_indices].max())

        # Ask the policy which acquisition function to use this iteration.
        if policy.needs_gp_state:
            selection_state = build_state(
                gp,
                reference_x,
                sampled_indices,
                y_pool,
                normalised_features,
                n_dimensions,
                total_iterations,
                last_acquisition,
            )
            acquisition_name, trace_row = policy.choose(selection_state, f_best_so_far)
        else:
            acquisition_name, trace_row = policy.choose(None, f_best_so_far)

        # Score every not-yet-sampled candidate and pick the argmax.
        candidate_indices = [
            i for i in range(len(y_pool)) if i not in set(sampled_indices)
        ]
        candidate_x = pool_x[torch.tensor(candidate_indices)]
        with torch.no_grad():
            scores = score_candidates(
                acquisition_name, gp, candidate_x, pool_x, f_best_so_far
            )
        picked = candidate_indices[int(torch.argmax(scores.view(-1)))]

        improved = bool(y_pool[picked] > f_best_so_far)
        sampled_indices.append(picked)
        last_acquisition = acquisition_name

        if record_trace and trace_row is not None:
            trace_rows.append(
                dict(
                    mc=mc_index,
                    n_exp=iteration + 1,
                    improved=improved,
                    f_picked=round(float(y_pool[picked]), 3),
                    **trace_row,
                )
            )

    return dict(
        sampled_objective_values=y_pool[sampled_indices].copy(),
        trace_rows=trace_rows,
        fit_rows=fit_rows,
        sampled_indices=list(sampled_indices),
    )


# =============================================================================
# 6. METRIC TRAJECTORIES  --  per-iteration AUC and top-5% coverage
# =============================================================================
def trajectory_metrics(sampled_objective_values, y_min, y_max, top_threshold, n_top5):
    """Per-iteration metric curves for one campaign.

    Returns (running_auc, running_coverage), each an array of length = #experiments:
        * running_auc[k]      : normalised area under the cumulative-best curve up to
                                experiment k (equals the headline AUC at the last k).
        * running_coverage[k] : fraction of the pool's top-5% experiments discovered
                                within the first k experiments.
    """
    sampled_objective_values = np.asarray(sampled_objective_values, float)
    cumulative_best = np.maximum.accumulate(sampled_objective_values)
    normalised_best = (cumulative_best - y_min) / (y_max - y_min)
    x = np.arange(1, len(sampled_objective_values) + 1)

    running_auc = np.empty(len(sampled_objective_values))
    for k in range(len(sampled_objective_values)):
        if k == 0:
            running_auc[k] = normalised_best[0]
        else:
            running_auc[k] = np.trapz(normalised_best[: k + 1], x[: k + 1]) / (
                x[k] - x[0]
            )

    discovered_top = np.cumsum(sampled_objective_values >= top_threshold)
    running_coverage = (
        np.minimum(discovered_top / n_top5, 1.0)
        if n_top5
        else np.full(len(sampled_objective_values), np.nan)
    )
    return running_auc, running_coverage


# =============================================================================
# 7. THE RUN  --  paired tested-vs-reference comparison over many MC repeats
# =============================================================================
def run_comparison(
    dataset,
    representation,
    mc_runs,
    total_iterations,
    switch_after=5,
    reference_af="qLogEI",
    rule_config=None,
    ceiling=None,
    seed_base=1337,
    lengthscale_rule=None,
    prior_mode="match_concentration",
    prior_cv=0.3,
):
    """Run the dynamic v2 policy and the static reference, paired by MC seed, and
    return everything the HTML report needs.

    Both campaigns in a given MC repeat start from the IDENTICAL random initial set
    (same seed), so any AUC/coverage difference is attributable to the policy.
    """
    feature_matrix, y_pool, n_dimensions, problem_dim = load_pool(
        dataset, representation
    )
    pool_x = torch.tensor(feature_matrix)
    bounds = torch.stack([pool_x.min(0).values, pool_x.max(0).values])
    geometry = compute_pool_geometry(feature_matrix)
    column_min, column_range = geometry[2], geometry[3]
    normalised_features = (feature_matrix - column_min) / column_range
    lengthscale_spec = make_lengthscale_spec(
        feature_matrix, bounds, n_dimensions, lengthscale_rule, prior_mode, prior_cv
    )

    y_min, y_max = float(y_pool.min()), float(y_pool.max())
    top_threshold = float(np.quantile(y_pool, 0.95))
    n_top5 = int((y_pool >= top_threshold).sum())
    if ceiling is None:  # default ceiling for the rule book
        ceiling = 100.0 if _is_yield_dataset(dataset) else y_max

    # The rule config: the in-file V2Config() defaults, or a saved named config.
    config = load_config(rule_config) if rule_config else HeuristicConfig()

    tested_auc_curves, tested_cov_curves = [], []
    reference_auc_curves, reference_cov_curves = [], []
    tested_final_auc, tested_final_cov = [], []
    reference_final_auc, reference_final_cov = [], []
    all_trace_rows = []

    for mc_index in range(mc_runs):
        rng = np.random.default_rng(seed_base + mc_index)
        initial_indices = list(rng.choice(len(y_pool), switch_after, replace=False))

        # --- REFERENCE: static acquisition function from this initial set ---
        reference = run_one_campaign(
            StaticAcquisitionFunction(reference_af),
            pool_x,
            y_pool,
            normalised_features,
            n_dimensions,
            geometry,
            bounds,
            initial_indices,
            total_iterations,
            mc_index,
            record_trace=False,
            lengthscale_spec=lengthscale_spec,
        )

        # --- TESTED: dynamic v2 rule book from the SAME initial set ---
        selector = AFSelector(
            total_budget=total_iterations,
            n_init=switch_after,
            problem_dim=problem_dim,
            config=config,
        )
        tested = run_one_campaign(
            DynamicAcquisitionFunction(selector),
            pool_x,
            y_pool,
            normalised_features,
            n_dimensions,
            geometry,
            bounds,
            initial_indices,
            total_iterations,
            mc_index,
            record_trace=True,
            lengthscale_spec=lengthscale_spec,
        )
        all_trace_rows.extend(tested["trace_rows"])

        for campaign, auc_curves, cov_curves, final_auc, final_cov in [
            (
                tested,
                tested_auc_curves,
                tested_cov_curves,
                tested_final_auc,
                tested_final_cov,
            ),
            (
                reference,
                reference_auc_curves,
                reference_cov_curves,
                reference_final_auc,
                reference_final_cov,
            ),
        ]:
            auc, coverage = trajectory_metrics(
                campaign["sampled_objective_values"],
                y_min,
                y_max,
                top_threshold,
                n_top5,
            )
            auc_curves.append(auc)
            cov_curves.append(coverage)
            final_auc.append(auc[-1])
            final_cov.append(coverage[-1])

    experiment_axis = np.arange(1, total_iterations + 1)
    trace_frame = pd.DataFrame(all_trace_rows)
    # Persist the machine-readable trace alongside the report.
    trace_frame.to_csv(
        os.path.join(
            OUTPUT_DIR, f"{dataset}__{representation}__{config.name}__trace.csv"
        ),
        index=False,
    )

    return dict(
        settings=dict(
            dataset=dataset,
            representation=representation,
            mc_runs=mc_runs,
            total_iterations=total_iterations,
            switch_after=switch_after,
            reference_af=reference_af,
            rule_config=config.name,
            ceiling=ceiling,
            seed_base=seed_base,
            n_dimensions=n_dimensions,
            pool_size=len(y_pool),
            n_top5=n_top5,
            top_threshold=round(top_threshold, 4),
            decorrelation=0.7,
            kernel="Matern-5/2 ARD",
            lengthscale_prior=lengthscale_spec.describe(),
            lengthscale_rule=lengthscale_spec.rule,
            prior_mode=lengthscale_spec.prior_mode,
        ),
        experiment_axis=experiment_axis,
        tested=_summarise(
            tested_auc_curves, tested_cov_curves, tested_final_auc, tested_final_cov
        ),
        reference=_summarise(
            reference_auc_curves,
            reference_cov_curves,
            reference_final_auc,
            reference_final_cov,
        ),
        trace_frame=trace_frame,
        config=config,
    )


def _summarise(auc_curves, cov_curves, final_auc, final_cov):
    """Mean per-iteration curves + final scalar metrics across the MC repeats."""
    return dict(
        auc_curve=np.mean(auc_curves, axis=0),
        coverage_curve=np.mean(cov_curves, axis=0),
        final_auc_mean=float(np.mean(final_auc)),
        final_auc_std=float(np.std(final_auc)),
        final_coverage_mean=float(np.mean(final_cov)),
        final_coverage_std=float(np.std(final_cov)),
    )


# Yield datasets have a natural 0-100% ceiling; property datasets use the pool max.
_YIELD_DATASETS = {
    "shields",
    "bh_full",
    "bh_reaction_1",
    "bh_reaction_2",
    "bh_reaction_3",
    "bh_reaction_4",
    "bh_reaction_5",
    "cpa_thiol_imine",
    "suzuki_perera",
    "suzuki_miyaura",
    "additives_plate_1",
    "additives_plate_2",
    "additives_plate_3",
    "additives_plate_4",
}


def _is_yield_dataset(dataset):
    return dataset in _YIELD_DATASETS


# =============================================================================
# 8. COMMAND-LINE INTERFACE
# =============================================================================
def available_datasets():
    """All datasets the workhorse can load: 'shields' plus the gollum registry."""
    return ["shields"] + [d["name"] for d in H.GP.DATASETS]


def available_representations():
    """Featurisations available (fingerprints + MLIP/LM embeddings)."""
    return ["morgan", "ohe", "mordred"] + list(H.GP.REPS)


def _build_arg_parser():
    parser = argparse.ArgumentParser(
        prog="botorch_bo.py",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="Pool BO workhorse: our dynamic v2 rule book vs a static reference, "
        "with an HTML report (settings, metric tables, per-iteration graphs, "
        "rule-trigger table) written to run_output/.",
        epilog="Datasets:  run --list-datasets   |   Acquisition functions: run --list-afs\n"
        "Edit bo_selection_v2.py to tune the dynamic rules before a run.",
    )
    parser.add_argument(
        "--dataset",
        default="shields",
        help="dataset to optimise on (default: shields; see --list-datasets)",
    )
    parser.add_argument(
        "--rep",
        "--representation",
        dest="rep",
        default="morgan",
        help="featurisation (default: morgan; see --list-afs/--list-datasets)",
    )
    parser.add_argument(
        "--mc", type=int, default=10, help="number of Monte-Carlo repeats (default: 10)"
    )
    parser.add_argument(
        "--iter",
        type=int,
        default=50,
        help="experiments per campaign, including the random init (default: 50)",
    )
    parser.add_argument(
        "--switch-after",
        type=int,
        default=5,
        help="number of random initial experiments (default: 5)",
    )
    parser.add_argument(
        "--reference",
        default="qLogEI",
        help="static reference acquisition function (default: qLogEI)",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="run a saved rule config from rule_versions/<name>.json "
        "instead of the in-file V2Config defaults",
    )
    parser.add_argument(
        "--ceiling",
        type=float,
        default=None,
        help="override the rule-book ceiling (default: 100 for yield "
        "datasets, pool-max otherwise)",
    )
    parser.add_argument(
        "--seed", type=int, default=1337, help="base RNG seed (default: 1337)"
    )
    parser.add_argument(
        "--lengthscale-rule",
        default=None,
        choices=["diamgate", "chen", "geom"],
        help="lengthscale prior centre: diamgate (default, the house prior), "
        "chen (0.4*sqrt(d)+4), geom (pool mean distance / 1.2218)",
    )
    parser.add_argument(
        "--prior-mode",
        default="match_concentration",
        choices=["match_concentration", "match_parameterisation"],
        help="how the Gamma prior width is set around the centre "
        "(default: match_concentration, a fixed CV in every arm)",
    )
    parser.add_argument(
        "--prior-cv",
        type=float,
        default=0.3,
        help="coefficient of variation for --prior-mode match_concentration "
        "(default: 0.3, the house value)",
    )
    parser.add_argument(
        "--no-report",
        action="store_true",
        help="skip writing the HTML report (only print the summary)",
    )
    parser.add_argument(
        "--open",
        action="store_true",
        help="open the HTML report in the default browser when done",
    )
    parser.add_argument(
        "--list-datasets", action="store_true", help="list datasets and exit"
    )
    parser.add_argument(
        "--list-afs",
        action="store_true",
        help="list supported acquisition functions and exit",
    )
    return parser


def main():
    args = _build_arg_parser().parse_args()

    if args.list_datasets:
        print("Datasets:\n  " + "\n  ".join(available_datasets()))
        print("\nRepresentations:\n  " + "\n  ".join(available_representations()))
        return
    if args.list_afs:
        print(
            "Acquisition functions (for --reference and emitted by the v2 rules):\n  "
            + "\n  ".join(SUPPORTED_ACQUISITION_FUNCTIONS)
        )
        return

    if args.reference not in SUPPORTED_ACQUISITION_FUNCTIONS:
        sys.exit(f"unknown --reference {args.reference!r}; see --list-afs")

    print(
        f"Pool BO  |  {args.dataset}/{args.rep}  |  {args.mc} MC x {args.iter} iter\n"
        f"  tested    : dynamic v2 rule book "
        f"({args.config or 'in-file V2Config defaults'})\n"
        f"  reference : static {args.reference}\n",
        flush=True,
    )

    start = time.time()
    results = run_comparison(
        args.dataset,
        args.rep,
        args.mc,
        args.iter,
        switch_after=args.switch_after,
        reference_af=args.reference,
        rule_config=args.config,
        ceiling=args.ceiling,
        seed_base=args.seed,
        lengthscale_rule=args.lengthscale_rule,
        prior_mode=args.prior_mode,
        prior_cv=args.prior_cv,
    )
    elapsed = time.time() - start

    tested, reference = results["tested"], results["reference"]
    print(
        f"  tested    AUC={tested['final_auc_mean']:.3f} +/- {tested['final_auc_std']:.3f}  "
        f"cov={tested['final_coverage_mean']:.2f}"
    )
    print(
        f"  reference AUC={reference['final_auc_mean']:.3f} +/- {reference['final_auc_std']:.3f}  "
        f"cov={reference['final_coverage_mean']:.2f}"
    )
    print(
        f"  dAUC (tested - reference) = "
        f"{tested['final_auc_mean'] - reference['final_auc_mean']:+.3f}   ({elapsed:.0f}s)\n",
        flush=True,
    )

    if not args.no_report:
        import bo_report

        report_path = bo_report.write_report(results, elapsed_seconds=elapsed)
        print(f"wrote HTML report -> {report_path}", flush=True)
        if args.open:
            import webbrowser

            webbrowser.open(report_path)


if __name__ == "__main__":
    main()
