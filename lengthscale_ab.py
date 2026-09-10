"""
lengthscale_ab.py  --  the paired lengthscale A/B
=================================================
A clean, paired, side-by-side comparison of two rules for where to centre the GP
lengthscale prior, with everything else in the pipeline held fixed.

    chen : ell_0 = 0.4*sqrt(d) + 4          (the published dimension-aware prior)
    geom : ell_0 = D_bar / 1.221810         (pool geometry, Matern-5/2 matched)

crossed with how the prior's WIDTH is set around that centre:

    match_parameterisation : Gamma(2*ell_0, 2)  -- Chen's tied form. Its CV is
        1/sqrt(2*ell_0), so re-centring silently changes how hard the prior pulls.
        Answers "what if you drop the new rule into the existing code?"
    match_concentration    : Gamma(1/cv^2, 1/(cv^2 * ell_0)) -- the same CV in
        both arms, so ONLY the centre differs. The scientifically clean comparison.

If the two prior modes disagree, the difference is a prior-STRENGTH effect, not a
centre effect, and must be reported as such.

WHAT IS HELD FIXED (identical in every arm):
    static qLogEI (NOT the dynamic v2 rule book -- that would confound), Matern-5/2
    ARD, no outputscale, Normalize([0,1]) on pool bounds + Standardize, 5 random
    initial points, 50 experiments, the same reduction, and -- per (dataset, rep,
    reduction, seed) -- the IDENTICAL initial design and torch seed. The initial
    indices are drawn from the seed BEFORE any rule is applied, so pairing cannot
    depend on the arm.

OUTPUT: one row per campaign (not pre-aggregated) -> run_output/lengthscale_ab.csv.
Resumable: rows are appended as they finish and completed cells are skipped.

USAGE
    python lengthscale_ab.py --dry-run                  # matrix + cost estimate
    python lengthscale_ab.py --datasets bh_full bh_reaction_1 --reps morgan
    python lengthscale_ab.py --shard 0/4                # run in 4 parallel shells
    python lengthscale_ab.py --reductions decorr0.7 pca64
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
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import botorch_bo as BO
import lengthscale_rules as LR

OUT_DIR = os.path.join(HERE, "run_output")
os.makedirs(OUT_DIR, exist_ok=True)
DEFAULT_CSV = os.path.join(OUT_DIR, "lengthscale_ab.csv")

# The chemistry, not the file: Buchwald-Hartwig sub-grids are correlated slices of
# ONE system, as are the four additive plates. Aggregate to family before any
# significance test -- runs within a family are not independent replicates.
FAMILY = {
    "bh_full": "buchwald_hartwig",
    "bh_reaction_1": "buchwald_hartwig",
    "bh_reaction_2": "buchwald_hartwig",
    "bh_reaction_3": "buchwald_hartwig",
    "bh_reaction_4": "buchwald_hartwig",
    "bh_reaction_5": "buchwald_hartwig",
    "additives_plate_1": "additives",
    "additives_plate_2": "additives",
    "additives_plate_3": "additives",
    "additives_plate_4": "additives",
    "suzuki_miyaura": "suzuki",
    "suzuki_perera": "suzuki",
    "shields": "direct_arylation",
    "cpa_thiol_imine": "cpa",
    "photoswitches": "photoswitches",
    "redox_mer": "redox",
    "pce10k": "photovoltaics",
    "enamine10k": "enamine",
}

DEFAULT_DATASETS = [
    "bh_full",          # standing preference: bh_full + bh1 in every benchmark run
    "bh_reaction_1",
    "cpa_thiol_imine",
    "shields",
    "photoswitches",
    "redox_mer",
]
DEFAULT_REPS = ["morgan", "mace_mp0", "aimnet2_all", "ohe"]
RULES = ["chen", "geom"]
PRIOR_MODES = list(LR.PRIOR_MODES)

ROW_COLUMNS = [
    "dataset", "family", "representation", "reduction", "lengthscale_rule",
    "prior_mode", "seed", "d", "n_pool", "D_bar", "ell_star", "ell_0",
    "ell_0_over_ell_star", "ell_fitted_mean", "ell_fitted_log_sd",
    "ell_fitted_over_ell_star", "auc", "simple_regret", "top5_coverage",
    "auc_random", "lift",
    # diagnostics beyond the required set
    "degenerate", "dist_cv", "prior_concentration", "prior_rate",
    "prior_cv_effective", "ell_fitted_traj_mean", "n_init", "n_iter",
    "acquisition", "seconds",
]

KEY_COLUMNS = ["dataset", "representation", "reduction", "lengthscale_rule",
               "prior_mode", "seed"]


# ---------------------------------------------------------------------------
# Per-cell setup: everything that does not depend on the arm or the seed
# ---------------------------------------------------------------------------
class Cell:
    """One (dataset, representation, reduction) pool, plus its geometry.

    Loaded once and reused by every arm and seed, so the arms cannot differ in
    anything but the prior centre.
    """

    def __init__(self, dataset, representation, reduction, n_iter, n_init,
                 n_random_draws=400, seed_base=1337):
        self.dataset = dataset
        self.representation = representation
        self.reduction = reduction
        self.n_iter = n_iter
        self.n_init = n_init

        features, y_pool, n_dimensions, _ = BO.load_pool(
            dataset, representation, reduction
        )
        self.features = features
        self.y_pool = y_pool
        self.d = int(n_dimensions)
        self.n_pool = int(len(y_pool))

        self.pool_x = torch.tensor(features)
        self.bounds = torch.stack([self.pool_x.min(0).values, self.pool_x.max(0).values])
        self.geometry = BO.compute_pool_geometry(features)
        column_min, column_range = self.geometry[2], self.geometry[3]
        self.normalised_features = (features - column_min) / column_range

        # D_bar in THE SPACE THE KERNEL SEES: the model's own Normalize bounds.
        self.model_space = LR.to_model_space(features, self.bounds.numpy())
        self.D_bar, dist_sd, self.dist_cv, self.degenerate = LR.pool_distance_stats(
            self.model_space
        )
        self.ell_star = LR.ell_star(self.model_space)

        self.y_min, self.y_max = float(y_pool.min()), float(y_pool.max())
        self.top_threshold = float(np.quantile(y_pool, 0.95))
        self.n_top5 = int((y_pool >= self.top_threshold).sum())
        self.auc_random = self._random_auc(n_random_draws, seed_base)

    def _random_auc(self, n_draws, seed_base):
        """AUC of a random-selection campaign of the same budget on the same pool."""
        rng = np.random.default_rng(seed_base + 90210)
        budget = min(self.n_iter, self.n_pool)
        aucs = []
        for _ in range(n_draws):
            picks = rng.choice(self.n_pool, budget, replace=False)
            auc, _ = BO.trajectory_metrics(
                self.y_pool[picks], self.y_min, self.y_max,
                self.top_threshold, self.n_top5,
            )
            aucs.append(auc[-1])
        return float(np.mean(aucs))

    def initial_indices(self, seed, seed_base=1337):
        """The initial design for one seed -- drawn BEFORE any rule is applied, so
        it is identical across every arm by construction."""
        rng = np.random.default_rng(seed_base + seed)
        return list(rng.choice(self.n_pool, self.n_init, replace=False))

    def ell_0(self, rule):
        return float(LR.RULES[rule](self.model_space, self.d))


# ---------------------------------------------------------------------------
# One campaign = one row
# ---------------------------------------------------------------------------
def run_campaign(cell, rule, prior_mode, seed, prior_cv=0.3, acquisition="qLogEI",
                 seed_base=1337):
    """Run one arm at one seed and return the tidy row."""
    ell_0 = cell.ell_0(rule)
    spec = BO.LengthscaleSpec(
        rule=rule, ell_0=ell_0, prior_mode=prior_mode, cv=prior_cv
    )
    concentration, rate = spec.gamma_parameters(ell_0)
    initial = cell.initial_indices(seed, seed_base)

    # Same acquisition-sampler stream in every arm, so a paired difference is the
    # prior and not qLogEI's Monte-Carlo noise.
    torch.manual_seed(seed_base + seed)

    start = time.time()
    campaign = BO.run_one_campaign(
        BO.StaticAcquisitionFunction(acquisition),
        cell.pool_x,
        cell.y_pool,
        cell.normalised_features,
        cell.d,
        cell.geometry,
        cell.bounds,
        initial,
        cell.n_iter,
        seed,
        record_trace=False,
        lengthscale_spec=spec,
        record_fit=True,
    )
    seconds = time.time() - start

    values = campaign["sampled_objective_values"]
    auc_curve, coverage_curve = BO.trajectory_metrics(
        values, cell.y_min, cell.y_max, cell.top_threshold, cell.n_top5
    )
    auc = float(auc_curve[-1])
    simple_regret = float(
        (cell.y_max - values.max()) / (cell.y_max - cell.y_min)
    )
    fit_rows = campaign["fit_rows"]
    final_fit = fit_rows[-1] if fit_rows else {"ell_fitted_mean": np.nan,
                                               "ell_fitted_log_sd": np.nan}
    traj_mean = (float(np.mean([r["ell_fitted_mean"] for r in fit_rows]))
                 if fit_rows else np.nan)

    return dict(
        dataset=cell.dataset,
        family=FAMILY.get(cell.dataset, cell.dataset),
        representation=cell.representation,
        reduction=cell.reduction,
        lengthscale_rule=rule,
        prior_mode=prior_mode,
        seed=seed,
        d=cell.d,
        n_pool=cell.n_pool,
        D_bar=cell.D_bar,
        ell_star=cell.ell_star,
        ell_0=ell_0,
        ell_0_over_ell_star=ell_0 / cell.ell_star if cell.ell_star else np.nan,
        ell_fitted_mean=final_fit["ell_fitted_mean"],
        ell_fitted_log_sd=final_fit["ell_fitted_log_sd"],
        ell_fitted_over_ell_star=(final_fit["ell_fitted_mean"] / cell.ell_star
                                  if cell.ell_star else np.nan),
        auc=auc,
        simple_regret=simple_regret,
        top5_coverage=float(coverage_curve[-1]),
        auc_random=cell.auc_random,
        lift=(auc - cell.auc_random) / (1.0 - cell.auc_random),
        degenerate=cell.degenerate,
        dist_cv=cell.dist_cv,
        prior_concentration=concentration,
        prior_rate=rate,
        prior_cv_effective=LR.prior_cv_of(concentration),
        ell_fitted_traj_mean=traj_mean,
        n_init=cell.n_init,
        n_iter=cell.n_iter,
        acquisition=acquisition,
        seconds=round(seconds, 1),
    )


# ---------------------------------------------------------------------------
# The sweep
# ---------------------------------------------------------------------------
def _done_keys(csv_path):
    if not os.path.exists(csv_path):
        return set()
    done = pd.read_csv(csv_path)
    if done.empty:
        return set()
    return set(map(tuple, done[KEY_COLUMNS].astype(str).values.tolist()))


def _append_row(csv_path, row):
    frame = pd.DataFrame([row], columns=ROW_COLUMNS)
    frame.to_csv(csv_path, mode="a", header=not os.path.exists(csv_path), index=False)


def failures_path(csv_path):
    stem, extension = os.path.splitext(csv_path)
    return f"{stem}_failures{extension}"


def _log_failure(csv_path, dataset, rep, reduction, rule, mode, seed, error):
    """Record a campaign that could not be run.

    A GP fit can genuinely fail to converge (BoTorch `ModelFittingError`) on an
    ill-conditioned cell. Dropping the campaign breaks that (cell, seed, prior
    mode) PAIR, so the analysis discards its partner too -- and the failure counts
    are reported per arm, because a rule whose lengthscale makes fits fail more
    often has a real disadvantage that must not be silently swallowed.
    """
    path = failures_path(csv_path)
    row = dict(dataset=dataset, representation=rep, reduction=reduction,
               lengthscale_rule=rule, prior_mode=mode, seed=seed,
               error=f"{type(error).__name__}: {error}")
    pd.DataFrame([row]).to_csv(path, mode="a", header=not os.path.exists(path),
                               index=False)


def sweep(datasets, reps, reductions, seeds, rules, prior_modes, n_iter, n_init,
          prior_cv, acquisition, csv_path, seed_base, shard=None,
          skip_degenerate=False):
    done = _done_keys(csv_path)
    cells = [(ds, rep, red) for ds in datasets for rep in reps for red in reductions]
    if shard is not None:
        index, count = shard
        cells = [c for i, c in enumerate(cells) if i % count == index]

    for dataset, rep, reduction in cells:
        # Seed-major, so an interrupted run still has BALANCED arms: every seed
        # that started is finished in all four arms before the next one begins.
        pending = [
            (rule, mode, seed)
            for seed in seeds for rule in rules for mode in prior_modes
            if (dataset, rep, reduction, rule, mode, str(seed)) not in done
        ]
        if not pending:
            print(f"[skip] {dataset}/{rep}/{reduction}: complete", flush=True)
            continue
        try:
            cell = Cell(dataset, rep, reduction, n_iter, n_init, seed_base=seed_base)
        except Exception as exc:  # a rep that cannot embed a dataset, etc.
            print(f"[FAIL] {dataset}/{rep}/{reduction}: {exc}", flush=True)
            continue

        flag = "  *** DEGENERATE POOL (excluded from aggregates) ***" if cell.degenerate else ""
        print(
            f"\n=== {dataset} / {rep} / {reduction} ===  d={cell.d} n={cell.n_pool} "
            f"D_bar={cell.D_bar:.3f} ell*={cell.ell_star:.3f} "
            f"chen={cell.ell_0('chen'):.3f} (chen/ell*={cell.ell_0('chen')/cell.ell_star:.2f}) "
            f"auc_random={cell.auc_random:.3f}{flag}",
            flush=True,
        )
        if cell.degenerate and skip_degenerate:
            print("       skipped (--skip-degenerate)", flush=True)
            continue

        for rule, mode, seed in pending:
            try:
                row = run_campaign(cell, rule, mode, seed, prior_cv, acquisition,
                                   seed_base)
            except Exception as exc:
                _log_failure(csv_path, dataset, rep, reduction, rule, mode, seed, exc)
                print(f"  [FAIL] {rule:5s} {mode:22s} seed={seed:<3d} "
                      f"{type(exc).__name__}: {exc}", flush=True)
                continue
            _append_row(csv_path, row)
            print(
                f"  {rule:5s} {mode:22s} seed={seed:<3d} l0={row['ell_0']:.3f} "
                f"l_fit={row['ell_fitted_mean']:.3f} auc={row['auc']:.3f} "
                f"lift={row['lift']:+.3f} sr={row['simple_regret']:.3f} "
                f"cov={row['top5_coverage']:.2f} ({row['seconds']:.0f}s)",
                flush=True,
            )
    print(f"\nwrote {csv_path}", flush=True)


def _dry_run(datasets, reps, reductions, seeds, rules, prior_modes, n_iter):
    n_cells = len(datasets) * len(reps) * len(reductions)
    n_campaigns = n_cells * len(rules) * len(prior_modes) * len(seeds)
    seconds = n_campaigns * 100.0  # ~100 s/campaign at 50 iterations, observed
    print(f"cells      : {n_cells}  ({len(datasets)} datasets x {len(reps)} reps "
          f"x {len(reductions)} reductions)")
    print(f"arms       : {len(rules)} rules x {len(prior_modes)} prior modes")
    print(f"seeds      : {len(seeds)}")
    print(f"campaigns  : {n_campaigns} x {n_iter} iterations")
    print(f"estimate   : ~{seconds/3600:.1f} h single process "
          f"(~{seconds/3600/4:.1f} h across 4 shards)")


def main():
    parser = argparse.ArgumentParser(
        prog="lengthscale_ab.py",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="Paired chen-vs-geom lengthscale A/B; one row per campaign.",
    )
    parser.add_argument("--datasets", nargs="+", default=DEFAULT_DATASETS)
    parser.add_argument("--reps", nargs="+", default=DEFAULT_REPS)
    parser.add_argument("--reductions", nargs="+", default=["decorr0.7"],
                        choices=list(BO.REDUCTIONS))
    parser.add_argument("--rules", nargs="+", default=RULES, choices=RULES)
    parser.add_argument("--prior-modes", nargs="+", default=PRIOR_MODES,
                        choices=PRIOR_MODES)
    parser.add_argument("--seeds", type=int, default=20,
                        help="number of shared seeds per arm (default 20)")
    parser.add_argument("--iter", type=int, default=50,
                        help="experiments per campaign, including the init")
    parser.add_argument("--init", type=int, default=5,
                        help="random initial experiments (default 5)")
    parser.add_argument("--prior-cv", type=float, default=0.3)
    parser.add_argument("--acquisition", default="qLogEI")
    parser.add_argument("--seed-base", type=int, default=1337)
    parser.add_argument("--csv", default=DEFAULT_CSV)
    parser.add_argument("--shard", default=None,
                        help="run a slice of the cells, as i/n (e.g. 0/4)")
    parser.add_argument("--skip-degenerate", action="store_true",
                        help="do not run cells whose pool distances are all equal "
                             "(they are flagged in the CSV either way)")
    parser.add_argument("--dry-run", action="store_true",
                        help="print the matrix and a cost estimate, then exit")
    args = parser.parse_args()

    seeds = list(range(args.seeds))
    if args.dry_run:
        _dry_run(args.datasets, args.reps, args.reductions, seeds, args.rules,
                 args.prior_modes, args.iter)
        return

    shard = None
    if args.shard:
        index, count = (int(v) for v in args.shard.split("/"))
        shard = (index, count)

    sweep(args.datasets, args.reps, args.reductions, seeds, args.rules,
          args.prior_modes, args.iter, args.init, args.prior_cv, args.acquisition,
          args.csv, args.seed_base, shard, args.skip_degenerate)


if __name__ == "__main__":
    main()
