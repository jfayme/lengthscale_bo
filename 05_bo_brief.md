# Rewrite — module 5: `lsab/bo.py`

Inputs as merged: `FeaturePool` (`X` float64 `(n, D)`, `pool.objective`),
`Prior` (`ell_0`, `concentration`, `rate`, `describe()`), `pool_bounds(X)`,
`pool_geometry(fp)` from `lsab.lengthscale`.

## Job
One pool-BO campaign: a Matern-5/2 ARD GP whose lengthscale prior is the
`Prior` it is handed, analytic LogEI over the not-yet-sampled candidates, the
argmax is the next experiment. Plus the per-iteration record of where the
fitted lengthscales landed (the A/B's "did the prior do anything" diagnostic)
and the trajectory metrics. Nothing here knows what a rule is.

Read the old `botorch_bo.build_gp` / `fit_gp` / `run_one_campaign` /
`trajectory_metrics` and `lengthscale_ab.run_campaign` / `Cell.initial_indices`
before writing. Everything about the dynamic policy (`build_state`,
`DynamicAcquisitionFunction`, `UNCERTAINTY_REFERENCE_CAP`, the rule-book
ceiling, `bo_report`) is gone.

## Decisions in force
- **Acquisition: analytic `botorch.acquisition.LogExpectedImprovement`**, one
  candidate at a time over the pool (`candidate_x.unsqueeze(1)`), under
  `torch.no_grad()`. No MC sampler. README "differences" line: the old runs
  used qLogEI with 1 candidate, which is the MC approximation of this.
- **No global side effects.** No `torch.set_default_dtype`, no
  `warnings.filterwarnings("ignore")` at module scope. Tensors are built with
  `dtype=torch.float64` explicitly. The old module suppressed every warning in
  the process, which is how fit failures went unseen.
- **Fit warnings are counted, not hidden.** `fit_gpytorch_mll` may emit
  `OptimizationWarning` / `BadInitialCandidatesWarning`; catch them with
  `warnings.catch_warnings(record=True)` around the fit, count them into the
  result. Any exception propagates as `CampaignError(iteration, cause)`.
- **Pairing.** The initial design depends on `(seed_base, seed, n_pool,
  n_init)` only; `torch.manual_seed(seed_base + seed)` at the start of every
  campaign so any torch randomness inside `fit_gpytorch_mll`'s retries is
  identical across arms. Nothing else is random.
- CPU only. The GP has at most 50 training points; no device plumbing.
- `Normalize` gets `pool_bounds(fp.X)` — the single definition of model space
  (module 4). `Standardize(1)` on the outcome. **No `ScaleKernel`**: outcomes
  are standardised to unit variance, so the signal variance is ~1 by
  construction; this is what the old `build_gp` did and the README should say
  why (an outputscale prior would re-introduce the amplitude factor the A/B
  holds fixed).

## Contract

```python
class CampaignError(RuntimeError):
    iteration: int            # experiment index at which it failed

@dataclass(frozen=True, eq=False)
class Campaign:
    fp: FeaturePool
    prior: Prior
    n_init: int = 5
    n_iter: int = 50          # TOTAL experiments, including the n_init random ones
    seed: int = 0
    seed_base: int = 1337
    record_fit: bool = True

@dataclass(frozen=True)
class CampaignResult:
    sampled_indices: np.ndarray       # (n_iter,) int64, row indices into fp.X / pool.frame
    sampled_objective: np.ndarray     # (n_iter,) float64
    fitted_ell_mean: np.ndarray       # (n_iter - n_init,) exp(mean(log ell_i)) over ARD dims after each fit; NaN if not recorded
    fitted_ell_log_sd: np.ndarray     # (n_iter - n_init,) std of log ell_i
    n_fit_warnings: int
    seconds: float

def initial_design(n_pool: int, n_init: int, seed: int, seed_base: int = 1337) -> np.ndarray
    # np.random.default_rng(seed_base + seed).choice(n_pool, n_init, replace=False); no other input

def build_gp(train_x: torch.Tensor, train_y: torch.Tensor, prior: Prior, bounds: torch.Tensor) -> SingleTaskGP
    # MaternKernel(nu=2.5, ard_num_dims=D, lengthscale_prior=GammaPrior(prior.concentration, prior.rate))
    # kernel.lengthscale = prior.ell_0     (every ARD dim starts at the centre)
    # SingleTaskGP(train_x, train_y, covar_module=kernel, input_transform=Normalize(D, bounds=bounds),
    #              outcome_transform=Standardize(1))

def fit_gp(gp) -> tuple[SingleTaskGP, int]        # fit_gpytorch_mll(ExactMarginalLogLikelihood(...)); returns (gp, n_warnings)
def score_logei(gp, candidate_x: torch.Tensor, best_f: float) -> torch.Tensor   # (n_candidates,)
def run_campaign(c: Campaign) -> CampaignResult

def trajectory(sampled_objective, pool_objective) -> tuple[np.ndarray, np.ndarray]
    # (running_auc, running_coverage), both length n_iter — the old trajectory_metrics, verbatim semantics:
    #   normalised_best[k] = (cummax[k] - y_min) / (y_max - y_min), y_min/max over the POOL
    #   running_auc[0] = normalised_best[0]; running_auc[k] = trapz(normalised_best[:k+1], x[:k+1]) / (x[k] - x[0]), x = 1..n
    #   top5: threshold = quantile(pool, 0.95); n_top5 = count(pool >= threshold); coverage[k] = min(cumsum(sampled >= threshold)[k] / n_top5, 1)

def metrics(result: CampaignResult, pool_objective: np.ndarray, n_init: int) -> dict
    # flat, CSV-safe:
    #   auc                 running_auc[-1]
    #   coverage_top5       running_coverage[-1]
    #   simple_regret       (y_max - max(sampled)) / (y_max - y_min)
    #   best_found          max(sampled)
    #   first_top5_hit      1-based experiment index of the first top-5% candidate sampled, NaN if never
    #   ell_fitted_final    fitted_ell_mean[-1]
    #   ell_fitted_traj     mean(fitted_ell_mean)
    #   ell_fitted_log_sd_final
    #   n_fit_warnings, seconds
```

## The loop (`run_campaign`), so there is no ambiguity

```
torch.manual_seed(seed_base + seed)
X = torch.tensor(fp.X, dtype=float64); y_pool = fp.pool.objective
bounds = torch.tensor(pool_bounds(fp.X), dtype=float64)
sampled = list(initial_design(n, n_init, seed, seed_base))
for it in range(n_init, n_iter):
    gp, w = fit_gp(build_gp(X[sampled], y[sampled, None], prior, bounds))
    record fitted lengthscales if record_fit
    candidates = the indices not in sampled (a boolean mask, not a set-diff in a list comprehension)
    with no_grad: scores = score_logei(gp, X[candidates], best_f = y[sampled].max())
    sampled.append(candidates[argmax(scores)])          # first index on ties; deterministic
```
Wrap the body so any exception becomes `CampaignError(iteration=it)` with the
original as `__cause__`.

## Dropped, on purpose
`StaticAcquisitionFunction` / `DynamicAcquisitionFunction` and the policy
interface; `build_state`, `shortest_distance`, `posterior_uncertainty` and the
frozen reference set; `score_candidates` and every acquisition other than
LogEI (UCB / PM / MES / PES / TS, plus the qPI / JES imports that had no
branch); `run_comparison`, `available_*`, `_is_yield_dataset`, the whole CLI
and HTML report of `botorch_bo.py`; `load_pool` (module 3) and
`compute_pool_geometry` / `diamgate_lengthscale_centre` /
`LengthscaleSpec` / `make_lengthscale_spec` (module 4); `botorch_out/` trace
CSVs; the module-level `_GLOBAL_RNG`.

## CLI
`python -m lsab.bo --dataset bh_reaction_1 --rep morgan --reduction decorr0.7 --rule geom --prior-mode match_concentration --cv 0.3 --seed 0 --iter 20 --init 5`
runs ONE campaign and prints `prior.describe()`, the metrics dict, and a
per-iteration table (`k`, `index`, `objective`, `cum_best`, `ell_fitted`).
It exists for eyeballing one run; the sweep is module 6.

## Tests (`tests/test_bo.py`) — botorch required (it is in `aimnet-bo`)
Use a synthetic `FeaturePool` (n = 80, D = 3, a smooth objective such as
`-||x - x*||^2 + noise`, built by hand without models) for 1–8; the module-3
fake rep on `bh_reaction_1` for 9.

1. `test_initial_design`: deterministic; no duplicates; `seed` changes it;
   `seed_base` changes it; it does not depend on anything else (call it
   before and after constructing two different `Prior`s — trivially true, but
   the test documents the contract).
2. `test_build_gp_reads_the_prior`: `covar_module` is a `MaternKernel` (no
   `ScaleKernel`); `ard_num_dims == D`; `lengthscale_prior.concentration /
   rate` equal `prior.concentration / rate`; every initial lengthscale equals
   `prior.ell_0`; `input_transform.bounds` equals `pool_bounds(X)`;
   `outcome_transform` is `Standardize`.
3. `test_paired_arms_share_the_start`: two priors (chen, geom), same seed →
   `sampled_indices[:n_init]` identical.
4. `test_deterministic`: the same `Campaign` twice → identical
   `sampled_indices` and `fitted_ell_mean`.
5. `test_beats_random`: over 10 seeds, mean final AUC of a campaign
   (`n_iter = 20`) exceeds the mean AUC of 20 random draws with the same
   seeds by a margin (say 0.05) on the smooth synthetic pool. Loose, but it
   catches a sign error in `best_f` or LogEI.
6. `test_trajectory_by_hand`: hand-built sequences: a sequence that hits the
   pool max at experiment 1 has `running_auc == 1` throughout and
   `simple_regret == 0`; coverage reaches exactly 1.0 when every top-5%
   candidate has been sampled and never exceeds it; `first_top5_hit` is the
   right 1-based index and NaN when none.
7. `test_fit_record`: `fitted_ell_mean` has length `n_iter - n_init`, all
   finite, positive; `record_fit=False` → all NaN; `n_fit_warnings >= 0`.
8. `test_campaign_error_carries_iteration`: monkeypatch `fit_gp` to raise at
   the 3rd fit → `CampaignError` with `iteration == n_init + 2` and
   `__cause__` set.
9. `test_real_pool_smoke`: `bh_reaction_1`, fake rep, `decorr0.7`, `geom`,
   `n_iter = 10`: runs in under 30 s, metrics are finite, `sampled_indices`
   are unique and within range.

## Definition of done
Tests 1–9 pass; the CLI runs 20 iterations on `bh_reaction_1` / morgan in
well under a minute; `bo.py` under ~220 lines; imports: numpy, torch,
botorch, gpytorch, warnings, time, dataclasses, `lsab.lengthscale`; no old-tree
imports; README gains the LogEI line and the no-ScaleKernel paragraph.

## When it's reviewed
Send me the merged `Campaign` / `CampaignResult` / `metrics` keys, the
seconds per campaign for `bh_reaction_1` and `enamine10k` (morgan,
decorr0.7, 50 iterations, one seed) and the `n_fit_warnings` you see. Module
6 (the resumable, shardable sweep with a new row schema) is sized from those
timings.
