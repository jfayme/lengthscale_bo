"""
lsab/bo.py
==========

Module 5 of the lengthscale-A/B rewrite: ONE pool-BO campaign. A Matern-5/2 ARD GP
whose lengthscale prior is the `Prior` it is handed; analytic LogEI over the
not-yet-sampled candidates; the argmax is the next experiment. Plus where the
fitted lengthscales landed after every fit (the A/B's "did the prior do anything"
diagnostic) and the trajectory metrics. Nothing here knows what a rule is.

The GP: inputs through `Normalize` with `pool_bounds` (module 4's single definition
of model space), outcomes through `Standardize`, and NO ScaleKernel. Standardised
outcomes have unit variance, so the signal variance is ~1 by construction; an
outputscale and its prior would re-introduce an amplitude factor the A/B holds fixed.

No global side effects: tensors are float64 by construction, never through
`torch.set_default_dtype`; the per-campaign torch seed lives inside `fork_rng`; and
fit warnings are COUNTED into the result rather than silenced process-wide (the old
module filtered every warning, which is how fit failures went unseen). CPU only: the
GP never has more than n_iter training points.

    python -m lsab.bo --dataset bh_reaction_1 --rep morgan --rule geom --iter 20
"""
from __future__ import annotations

import time
import warnings
from dataclasses import dataclass

import numpy as np
import torch
from botorch.acquisition import LogExpectedImprovement
from botorch.fit import fit_gpytorch_mll
from botorch.models import SingleTaskGP
from botorch.models.transforms import Normalize, Standardize
from gpytorch.kernels import MaternKernel
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.priors import GammaPrior

from lsab.lengthscale import Prior, pool_bounds
from lsab.reduce import FeaturePool

F64 = torch.float64


class CampaignError(RuntimeError):
    """A campaign that failed; `iteration` is the 0-based experiment it was choosing."""

    def __init__(self, iteration: int):
        super().__init__(f"campaign failed while choosing experiment {iteration}")
        self.iteration = iteration


@dataclass(frozen=True, eq=False)
class Campaign:
    fp: FeaturePool
    prior: Prior
    n_init: int = 5
    n_iter: int = 50          # TOTAL experiments, including the n_init random ones
    seed: int = 0
    seed_base: int = 1337
    record_fit: bool = True


@dataclass(frozen=True, eq=False)   # eq=False: arrays have no single truth value
class CampaignResult:
    sampled_indices: np.ndarray       # (n_iter,) int64, row indices into fp.X / pool.frame
    sampled_objective: np.ndarray     # (n_iter,) float64
    fitted_ell_mean: np.ndarray       # (n_iter - n_init,) exp(mean log ell) over ARD dims per fit; NaN if not recorded
    fitted_ell_log_sd: np.ndarray     # (n_iter - n_init,) std of log ell over ARD dims
    n_fit_warnings: int
    seconds: float


# =============================================================================
# 1. THE PIECES
# =============================================================================
def initial_design(n_pool: int, n_init: int, seed: int, seed_base: int = 1337) -> np.ndarray:
    """The random start. It depends on nothing but its arguments, so every arm of a
    (cell, seed) pair starts from the same experiments: the pairing the A/B rests on."""
    return np.random.default_rng(seed_base + seed).choice(n_pool, n_init, replace=False).astype(np.int64)


def build_gp(train_x: torch.Tensor, train_y: torch.Tensor, prior: Prior, bounds: torch.Tensor) -> SingleTaskGP:
    """Matern-5/2 ARD, lengthscale prior Gamma(prior.concentration, prior.rate), every
    ARD dimension starting at prior.ell_0. Prior parameters and the start are float64
    tensors: a Python float would pass through torch's float32 default first."""
    d = train_x.shape[-1]
    ls_prior = GammaPrior(torch.tensor(prior.concentration, dtype=F64), torch.tensor(prior.rate, dtype=F64))
    kernel = MaternKernel(nu=2.5, ard_num_dims=d, lengthscale_prior=ls_prior).to(F64)
    kernel.lengthscale = torch.full((1, d), prior.ell_0, dtype=F64)
    return SingleTaskGP(train_x, train_y, covar_module=kernel,
                        input_transform=Normalize(d, bounds=bounds), outcome_transform=Standardize(1))


def fit_gp(gp: SingleTaskGP) -> tuple[SingleTaskGP, int]:
    """Exact marginal-likelihood fit. Every warning it raises is counted, not shown."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        fit_gpytorch_mll(ExactMarginalLogLikelihood(gp.likelihood, gp))
    return gp, len(caught)


def score_logei(gp: SingleTaskGP, candidate_x: torch.Tensor, best_f: float) -> torch.Tensor:
    """Analytic LogEI for each candidate on its own (q = 1): (n_candidates,)."""
    return LogExpectedImprovement(gp, best_f=best_f)(candidate_x.unsqueeze(1))


# =============================================================================
# 2. THE LOOP
# =============================================================================
def run_campaign(c: Campaign) -> CampaignResult:
    """n_init random experiments, then one GP fit + LogEI argmax per experiment.

    Every iteration builds a FRESH GP whose ARD lengthscales start at prior.ell_0, so
    each fit is warm-started from the prior centre, not from the previous fit. That is
    intended: it is what makes the centre matter at every step, not only the first."""
    start = time.perf_counter()
    X = torch.tensor(c.fp.X, dtype=F64)
    y = torch.tensor(c.fp.pool.objective, dtype=F64)
    bounds = torch.tensor(pool_bounds(c.fp.X), dtype=F64)
    sampled = initial_design(len(X), c.n_init, c.seed, c.seed_base).tolist()
    n_fits = c.n_iter - c.n_init
    ell_mean, ell_log_sd = np.full(n_fits, np.nan), np.full(n_fits, np.nan)
    n_warnings = 0
    with torch.random.fork_rng(devices=[]):          # the seed stays inside this campaign
        torch.manual_seed(c.seed_base + c.seed)      # identical fit retries across paired arms
        for it in range(c.n_init, c.n_iter):
            try:
                gp, caught = fit_gp(build_gp(X[sampled], y[sampled, None], c.prior, bounds))
                n_warnings += caught
                if c.record_fit:
                    log_ell = np.log(gp.covar_module.lengthscale.detach().numpy().ravel())
                    ell_mean[it - c.n_init], ell_log_sd[it - c.n_init] = np.exp(log_ell.mean()), log_ell.std()
                unsampled = np.ones(len(X), dtype=bool)
                unsampled[sampled] = False
                candidates = np.flatnonzero(unsampled)
                with torch.no_grad():
                    scores = score_logei(gp, X[torch.from_numpy(candidates)], best_f=float(y[sampled].max()))
                sampled.append(int(candidates[np.argmax(scores.numpy())]))   # first index on ties
            except Exception as error:
                raise CampaignError(it) from error
    indices = np.asarray(sampled, dtype=np.int64)
    return CampaignResult(indices, c.fp.pool.objective[indices], ell_mean, ell_log_sd,
                          n_warnings, time.perf_counter() - start)


# =============================================================================
# 3. THE METRICS
# =============================================================================
def trajectory(sampled_objective, pool_objective) -> tuple[np.ndarray, np.ndarray]:
    """(running_auc, running_coverage), each of length n_iter.

    running_auc[k]: the area under the normalised cumulative-best curve up to
        experiment k (trapezoids, unit spacing) over its width; [0] is the first
        normalised value. The last entry is the campaign's AUC.
    running_coverage[k]: the share of the pool's top-5% candidates sampled so far.
    Normalisation and the top-5% threshold come from the POOL, not the samples.
    """
    sampled, pool = np.asarray(sampled_objective, dtype=float), np.asarray(pool_objective, dtype=float)
    y_min, y_max = pool.min(), pool.max()
    if y_max == y_min:
        raise ValueError("the pool's objective is constant: there is nothing to optimise")
    best = (np.maximum.accumulate(sampled) - y_min) / (y_max - y_min)
    area = np.concatenate([[0.0], np.cumsum((best[1:] + best[:-1]) / 2.0)])
    running_auc = np.concatenate([best[:1], area[1:] / np.arange(1, len(best))])
    threshold = np.quantile(pool, 0.95)
    running_coverage = np.minimum(np.cumsum(sampled >= threshold) / np.sum(pool >= threshold), 1.0)
    return running_auc, running_coverage


RANDOM_AUC_DRAWS = 400     # random campaigns behind auc_random; the count the old sweep used
RANDOM_AUC_OFFSET = 90210  # added to seed_base, so the baseline draws a stream no campaign uses


def random_auc(pool_objective, n_iter: int, *, n_draws: int = RANDOM_AUC_DRAWS,
               seed_base: int = 1337) -> float:
    """Mean final AUC of `n_draws` random-selection campaigns of `n_iter` experiments on
    this pool: the zero of the lift scale.

    It depends on the objective and the budget ALONE -- not on the representation, the
    reduction, the rule, the prior or the seed -- so one value serves every cell and arm
    of a dataset, and module 6 computes it once per dataset. Same estimator as the old
    tree's `lengthscale_ab.Cell._random_auc` (git history), down to the draw count and
    the seed offset.
    """
    pool = np.asarray(pool_objective, dtype=float)
    rng = np.random.default_rng(seed_base + RANDOM_AUC_OFFSET)
    budget = min(int(n_iter), len(pool))     # a pool shorter than the budget cannot fill it
    return float(np.mean([trajectory(pool[rng.choice(len(pool), budget, replace=False)], pool)[0][-1]
                          for _ in range(n_draws)]))


def lift(auc, auc_random) -> float:
    """`auc` rescaled so this pool's random baseline is 0 and a perfect campaign is 1.

    THE HEADLINE METRIC of the A/B: raw AUC is dominated by how easy a dataset is, which
    is held fixed inside a pair but not across the matrix. NaN when `auc_random` is None
    (it was not computed) or 1 (no headroom above random).
    """
    if auc_random is None:
        return float("nan")
    headroom = 1.0 - float(auc_random)
    return float((float(auc) - float(auc_random)) / headroom) if headroom > 0 else float("nan")


def metrics(result: CampaignResult, pool_objective: np.ndarray, n_init: int,
            *, auc_random: float | None = None) -> dict:
    """The flat, CSV-safe outcome of one campaign.

    `auc_random` is this pool's random-selection baseline from `random_auc`; it is what
    turns `auc` into `lift`. Module 6 passes the value it memoised for the dataset.
    Omitted, `lift` is NaN: the baseline costs 400 campaigns, too much to compute behind
    the caller's back once per campaign.
    """
    if len(result.fitted_ell_mean) != len(result.sampled_objective) - n_init:
        raise ValueError(f"n_init={n_init} does not match this result's fit record")
    pool = np.asarray(pool_objective, dtype=float)
    sampled = result.sampled_objective
    running_auc, running_coverage = trajectory(sampled, pool)
    hits = np.flatnonzero(sampled >= np.quantile(pool, 0.95))
    ell, log_sd = (result.fitted_ell_mean, result.fitted_ell_log_sd) if n_init < len(sampled) else ([np.nan],) * 2
    return {
        "auc": float(running_auc[-1]),
        "lift": lift(running_auc[-1], auc_random),
        "coverage_top5": float(running_coverage[-1]),
        "simple_regret": float((pool.max() - sampled.max()) / (pool.max() - pool.min())),
        "best_found": float(sampled.max()),
        "first_top5_hit": int(hits[0]) + 1 if hits.size else float("nan"),   # 1-based; NaN if never
        "ell_fitted_final": float(ell[-1]),
        "ell_fitted_traj": float(np.mean(ell)),
        "ell_fitted_log_sd_final": float(log_sd[-1]),
        "n_fit_warnings": int(result.n_fit_warnings),
        "seconds": float(result.seconds),
    }


# =============================================================================
# 4. CLI  -- one campaign, for eyeballing (the sweep is module 6)
# =============================================================================
def main(argv: list[str] | None = None) -> None:
    import argparse

    from lsab import datasets
    from lsab.featurize import REPS
    from lsab.lengthscale import PRIOR_MODES, RULES, make_prior
    from lsab.reduce import REDUCTIONS, build

    parser = argparse.ArgumentParser(prog="python -m lsab.bo", description="Run one pool-BO campaign.")
    parser.add_argument("--dataset", default="bh_reaction_1", choices=datasets.names())
    parser.add_argument("--rep", default="morgan", choices=list(REPS))
    parser.add_argument("--reduction", default="decorr0.7", choices=list(REDUCTIONS))
    parser.add_argument("--rule", default="geom", choices=list(RULES))
    parser.add_argument("--prior-mode", default="match_concentration", choices=PRIOR_MODES)
    parser.add_argument("--cv", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--iter", type=int, default=20, help="total experiments, initial design included")
    parser.add_argument("--init", type=int, default=5)
    parser.add_argument("--cache-dir", default=None, help="embedding cache (lsab.featurize)")
    args = parser.parse_args(argv)
    fp = build(datasets.load(args.dataset), args.rep, args.reduction, cache_dir=args.cache_dir)
    prior = make_prior(fp, args.rule, args.prior_mode, args.cv)
    result = run_campaign(Campaign(fp, prior, n_init=args.init, n_iter=args.iter, seed=args.seed))
    print(prior.describe())
    print(metrics(result, fp.pool.objective, args.init,
                  auc_random=random_auc(fp.pool.objective, args.iter)))
    print(f"{'k':>4} {'index':>7} {'objective':>12} {'cum_best':>12} {'ell_fitted':>11}")
    for k, (index, value) in enumerate(zip(result.sampled_indices, result.sampled_objective), 1):
        ell = result.fitted_ell_mean[k - 1 - args.init] if k > args.init else float("nan")
        print(f"{k:>4} {index:>7} {value:>12.5g} {result.sampled_objective[:k].max():>12.5g} {ell:>11.4g}")


if __name__ == "__main__":
    main()
