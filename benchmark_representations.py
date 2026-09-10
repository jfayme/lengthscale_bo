"""
benchmark_representations.py
============================

Benchmark molecular representations in a BayBE Bayesian-optimization campaign on
the Shields dataset, reproducing the HSF-ChemBO paper's setup exactly for the
baselines so AIMNet2 / MACE drop in for a like-for-like comparison.

Representations (``--reps``):
    aimnet2, mace_off23, mace_mp0   (this work)
    chemeleon, chemberta            (paper baselines, tutorial base/ code)

Paper / tutorial setup reproduced (see tutorial_advanced.ipynb):
    * Search space  = product of {Solvent, Base, Ligand} (molecular, custom
      descriptors) x {Temp_C, Concentration} (numeric, min-max normalized).
    * Kernel        = base.kernels.AdaptiveKernelFactory  ->  ScaleKernel(
                        Matern nu=2.5, lengthscale ~ Gamma(2*l0, 2),
                        l0 = 0.4*sqrt(d) + 4.0 ), the "Adaptive_ours" hyperprior.
    * Surrogate     = GaussianProcessSurrogate(kernel_or_factory=factory).
    * Acquisition   = qLogEI via BotorchRecommender.
    * Init          = RandomRecommender for the first ``switch_after`` (=5) points
                      (TwoPhaseMetaRecommender), then BO.
    * simulate_scenarios: 50 DoE iterations, 20 Monte-Carlo runs, batch_size=1.

Outputs, per representation:
    * AUC            -- area under the normalized cumulative-best-yield curve
                        (1.0 == oracle finds the global optimum at iteration 0).
    * Top5%_coverage -- fraction of MC runs that have discovered a top-5%
                        (>= 95th percentile) yield by the final iteration, and
                        the AUC of that coverage curve over iterations.

Usage
-----
Full paper setting (slow, hours):
    python benchmark_representations.py --reps aimnet2 mace_off23 mace_mp0 chemeleon chemberta
Fast smoke test:
    python benchmark_representations.py --reps aimnet2 --n-iter 8 --mc-runs 2 --quick
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
TUTORIAL = HERE / "HSF-ChemBO-tutorial"
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(TUTORIAL))


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------


def load_shields():
    """Load the Shields dataset and the per-component {label: SMILES} maps."""
    lookup = pd.read_excel(TUTORIAL / "shields_dataset.xlsx", index_col=0)

    from base.utils import _normalize
    lookup = lookup.copy()
    lookup["Temp_C"] = _normalize(lookup["Temp_C"])
    lookup["Concentration"] = _normalize(lookup["Concentration"])

    component_smiles = {
        "Solvent": dict(sorted(set(zip(lookup.Solvent, lookup.Solvent_SMILES)))),
        "Base": dict(sorted(set(zip(lookup.Base, lookup.Base_SMILES)))),
        "Ligand": dict(sorted(set(zip(lookup.Ligand, lookup.Ligand_SMILES)))),
    }
    return lookup, component_smiles


# ---------------------------------------------------------------------------
# Campaign construction (one per representation)
# ---------------------------------------------------------------------------


def lengthscale_ab_factory(lengthscale_rule, prior_mode="match_concentration",
                           prior_cv=0.3):
    """A BayBE kernel factory identical to the paper's `AdaptiveKernelFactory`
    except for WHERE the lengthscale prior is centred (and, explicitly, how its
    width is parameterised) -- the single seam of the lengthscale A/B.

    ``lengthscale_rule="chen"`` with ``prior_mode="match_parameterisation"``
    reproduces `AdaptiveKernelFactory` exactly: centre 0.4*sqrt(d)+4, lengthscale
    prior Gamma(2*l0, 2), outputscale prior Gamma(l0, 1). The outputscale is held
    at the PAPER value in every arm so only the lengthscale centre moves.

    The centre is computed in the space the kernel sees: the discrete search
    space's computational representation pushed through the same per-column
    min-max map BayBE's `Normalize` applies.
    """
    from baybe.surrogates.gaussian_process.kernel_factory import KernelFactory
    from baybe.kernels import ScaleKernel, MaternKernel
    from baybe.priors.basic import GammaPrior

    import lengthscale_rules as LR

    class _LengthscaleABFactory(KernelFactory):
        def __call__(self, searchspace, train_x, train_y):
            d = len(searchspace.comp_rep_columns)
            pool = searchspace.discrete.comp_rep.values.astype(float)
            bounds = np.stack([pool.min(0), pool.max(0)])
            ell_0 = float(LR.RULES[lengthscale_rule](LR.to_model_space(pool, bounds), d))
            concentration, rate = LR.gamma_prior_parameters(ell_0, prior_mode, prior_cv)
            outputscale = 0.4 * np.sqrt(d) + 4.0   # the paper value, both arms
            return ScaleKernel(
                MaternKernel(nu=2.5,
                             lengthscale_prior=GammaPrior(concentration, rate),
                             lengthscale_initial_value=ell_0),
                outputscale_prior=GammaPrior(outputscale, 1.0),
                outputscale_initial_value=outputscale)

    return _LengthscaleABFactory()


def build_campaign(rep_name, lookup, component_smiles, *, decorrelate=0.7,
                   switch_after=5, fingerprinter_kwargs=None,
                   coverage_fallback="mace_mp0", lengthscale_rule=None,
                   prior_mode="match_concentration", prior_cv=0.3):
    """Build a BayBE Campaign whose molecular parameters use representation ``rep_name``.

    Element coverage is checked per component. If ``rep_name`` cannot represent a
    component's molecules (e.g. AIMNet2 / MACE-OFF23 on the Shields Cs/K bases),
    that component falls back to ``coverage_fallback`` (default ``mace_mp0``,
    which covers 89 elements) and a clear warning is printed. Set
    ``coverage_fallback=None`` to raise instead.
    """
    from baybe import Campaign
    from baybe.objectives import SingleTargetObjective
    from baybe.targets import NumericalTarget
    from baybe.parameters import NumericalDiscreteParameter
    from baybe.searchspace import SearchSpace
    from baybe.surrogates import GaussianProcessSurrogate
    from baybe.recommenders import (
        BotorchRecommender, RandomRecommender, TwoPhaseMetaRecommender,
    )
    from base.kernels import AdaptiveKernelFactory  # paper's adaptive hyperprior

    import representations as R

    # The lengthscale-rule seam. None -> the paper's factory, byte-for-byte the
    # behaviour this file has always had.
    kernel_factory = (AdaptiveKernelFactory() if lengthscale_rule is None
                      else lengthscale_ab_factory(lengthscale_rule, prior_mode, prior_cv))

    objective = SingleTargetObjective(target=NumericalTarget(name="yield", mode="MAX"))

    params = []
    if rep_name in ("ohe", "morgan", "mordred"):
        # Reference baselines: no NN, no element coverage / fallback. OHE is a pure
        # identity (decorrelate off so the one-hot is preserved); Morgan/Mordred use
        # the same decorrelation dim-control as the embedding parameters.
        for comp in ("Solvent", "Base", "Ligand"):
            dec = False if rep_name == "ohe" else decorrelate
            params.append(R.baybe_parameter(rep_name, comp,
                                            component_smiles[comp], decorrelate=dec))
        params.append(NumericalDiscreteParameter(name="Temp_C",
                                                 values=set(lookup.Temp_C)))
        params.append(NumericalDiscreteParameter(name="Concentration",
                                                 values=set(lookup.Concentration)))
        searchspace = SearchSpace.from_product(parameters=params)
        feat_dim = len(searchspace.comp_rep_columns)
        surrogate = GaussianProcessSurrogate(kernel_or_factory=kernel_factory)
        recommender = TwoPhaseMetaRecommender(
            initial_recommender=RandomRecommender(),
            recommender=BotorchRecommender(surrogate_model=surrogate,
                                           acquisition_function="qLogEI"),
            switch_after=switch_after)
        campaign = Campaign(searchspace=searchspace, objective=objective,
                            recommender=recommender)
        return campaign, feat_dim

    # One shared fingerprinter per representation (loads the NN once).
    fps = {rep_name: R.make_fingerprinter(rep_name, **(fingerprinter_kwargs or {}))}

    for comp in ("Solvent", "Base", "Ligand"):
        smis = list(component_smiles[comp].values())
        bad = R.uncovered_elements(rep_name, smis)
        use_rep = rep_name
        if bad:
            if coverage_fallback is None:
                raise ValueError(
                    f"[{rep_name}] cannot cover component '{comp}' "
                    f"(elements Z={sorted(bad)}); no coverage_fallback set.")
            print(f"  [coverage] {rep_name} cannot embed '{comp}' "
                  f"(Z={sorted(bad)}) -> falling back to '{coverage_fallback}' "
                  f"for this component", flush=True)
            use_rep = coverage_fallback
            if use_rep not in fps:
                fps[use_rep] = R.make_fingerprinter(use_rep)
        params.append(
            R.baybe_parameter(use_rep, comp, component_smiles[comp],
                              fingerprinter=fps[use_rep], decorrelate=decorrelate)
        )
    params.append(NumericalDiscreteParameter(name="Temp_C",
                                             values=set(lookup.Temp_C)))
    params.append(NumericalDiscreteParameter(name="Concentration",
                                             values=set(lookup.Concentration)))

    searchspace = SearchSpace.from_product(parameters=params)
    feat_dim = len(searchspace.comp_rep_columns)

    surrogate = GaussianProcessSurrogate(kernel_or_factory=kernel_factory)
    recommender = TwoPhaseMetaRecommender(
        initial_recommender=RandomRecommender(),
        recommender=BotorchRecommender(
            surrogate_model=surrogate,
            acquisition_function="qLogEI",
        ),
        switch_after=switch_after,
    )
    campaign = Campaign(searchspace=searchspace, objective=objective,
                        recommender=recommender)
    return campaign, feat_dim


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def _cumbest_column(df):
    for c in df.columns:
        if c.endswith("_CumBest"):
            return c
    raise KeyError(f"No *_CumBest column in simulate_scenarios output: {list(df.columns)}")


def _parse_measurement(v):
    """Extract a scalar yield from a simulate_scenarios measurement cell."""
    if isinstance(v, (list, tuple, np.ndarray)):
        return float(np.max(v)) if len(v) else np.nan
    if isinstance(v, str):
        s = v.strip().lstrip("[").rstrip("]")
        if not s:
            return np.nan
        return float(s.split(",")[0])
    return float(v)


def per_run_metrics(curves, lookup, *, top_frac=0.05):
    """Per-(representation, Monte-Carlo-run) metrics -> tidy DataFrame.

    One row per MC run so the 20 runs stay *paired across representations*
    (same Monte_Carlo_Run index == same random seed), enabling a Wilcoxon
    signed-rank test.

    Metrics
    -------
    AUC : area under the cumulative-best-yield curve, normalized to [0,1]
          via (cumbest - y_min)/(y_max - y_min), integrated over iterations.
    Top5pct_coverage : fraction of the dataset's top-5% (>=95th pct) reactions
          actually *sampled* during the campaign. Discrete BayBE samples without
          replacement, so #measurements >= threshold == #distinct top reactions
          found; divided by the number of top-5% reactions.
    """
    cb = _cumbest_column(curves)
    meas = cb.replace("_CumBest", "_Measurements")
    y = lookup["yield"].astype(float)
    f_best, f_min = float(y.max()), float(y.min())
    thr = float(y.quantile(1.0 - top_frac))
    n_top5 = int((y >= thr).sum())

    rows = []
    keys = ["representation", "Monte_Carlo_Run"]
    for (rep, mc), g in curves.groupby(keys):
        g = g.sort_values("Num_Experiments")
        x = g["Num_Experiments"].values.astype(float)
        cbv = g[cb].values.astype(float)
        auc = float(np.trapz((cbv - f_min) / (f_best - f_min), x)
                    / (x.max() - x.min())) if x.max() > x.min() else float("nan")
        if meas in g:
            mvals = g[meas].map(_parse_measurement).values
            discovered = int(np.sum(mvals >= thr))
        else:  # fall back to cum-best crossing
            discovered = int(cbv[-1] >= thr)
        coverage = min(discovered / n_top5, 1.0) if n_top5 else float("nan")
        rows.append({"representation": rep, "mc_run": int(mc),
                     "AUC": auc, "Top5pct_coverage": coverage,
                     "final_best": float(cbv[-1])})
    return pd.DataFrame(rows)


def pairwise_wilcoxon(per_run, metric="AUC", baseline=None):
    """Paired Wilcoxon signed-rank test on a per-run metric across representations.

    Runs are paired by ``mc_run``. Returns a DataFrame of
    (rep_a, rep_b, median_diff, W, p_value). If ``baseline`` is given, only
    ``rep vs baseline`` comparisons are returned.
    """
    from itertools import combinations
    from scipy.stats import wilcoxon

    wide = per_run.pivot(index="mc_run", columns="representation", values=metric)
    reps = list(wide.columns)
    pairs = ([(r, baseline) for r in reps if r != baseline]
             if baseline else list(combinations(reps, 2)))
    out = []
    for a, b in pairs:
        da, db = wide[a].values, wide[b].values
        mask = ~(np.isnan(da) | np.isnan(db))
        da, db = da[mask], db[mask]
        diff = da - db
        if np.allclose(diff, 0):
            W, p = float("nan"), 1.0
        else:
            W, p = wilcoxon(da, db)
        out.append({"metric": metric, "rep_a": a, "rep_b": b,
                    "median_a": float(np.median(da)), "median_b": float(np.median(db)),
                    "median_diff": float(np.median(diff)), "W": float(W),
                    "p_value": float(p), "n_pairs": int(mask.sum())})
    return pd.DataFrame(out)


def compute_metrics(result, lookup, *, top_frac=0.05):
    """AUC of normalized cumulative-best, and top-5% coverage (final + AUC)."""
    cb = _cumbest_column(result)
    f_best = float(lookup["yield"].max())
    f_min = float(lookup["yield"].min())
    thr = float(lookup["yield"].quantile(1.0 - top_frac))  # 95th percentile

    # Mean cumulative-best curve across MC runs, per iteration.
    per_iter = result.groupby("Num_Experiments")[cb].mean().sort_index()
    iters = per_iter.index.values.astype(float)
    norm_curve = (per_iter.values - f_min) / (f_best - f_min)
    # Normalized AUC over iterations (trapezoid, scaled to [0,1] in x).
    auc = float(np.trapz(norm_curve, iters) / (iters.max() - iters.min()))

    # Top-5% coverage: per (MC run, iteration) whether cum-best >= threshold.
    result = result.copy()
    result["_hit"] = (result[cb] >= thr).astype(float)
    cov_curve = result.groupby("Num_Experiments")["_hit"].mean().sort_index()
    cov_final = float(cov_curve.values[-1])
    cov_auc = float(np.trapz(cov_curve.values, cov_curve.index.values.astype(float))
                    / (cov_curve.index.max() - cov_curve.index.min()))

    return {
        "AUC_cumbest": auc,
        "Top5pct_coverage_final": cov_final,
        "Top5pct_coverage_AUC": cov_auc,
        "F_best": f_best,
        "top5_threshold": thr,
    }


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


def run(reps, *, n_iter=50, mc_runs=20, switch_after=5, batch_size=1,
        seed=1337, decorrelate=0.7, outdir=HERE / "bench_out",
        fingerprinter_kwargs=None, coverage_fallback="mace_mp0", resume=True,
        lengthscale_rule=None, prior_mode="match_concentration", prior_cv=0.3):
    """Run the benchmark, saving each representation's curves AS IT FINISHES.

    Interruption-proof: per-rep results are written to
    ``outdir/curves__<rep>.csv`` immediately, and with ``resume=True`` (default)
    a re-run skips any representation whose file already exists. So if the
    machine sleeps / the run dies, just re-run the same command to continue.
    """
    from baybe.simulation import simulate_scenarios
    from baybe.utils.random import set_random_seed

    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    lookup, component_smiles = load_shields()

    curves = []
    for rep in reps:
        rep_csv = outdir / f"curves__{rep}.csv"
        if resume and rep_csv.exists():
            print(f"\n=== Representation: {rep} === (resuming from {rep_csv.name})", flush=True)
            curves.append(pd.read_csv(rep_csv))
            continue
        print(f"\n=== Representation: {rep} ===", flush=True)
        t0 = time.time()
        campaign, feat_dim = build_campaign(
            rep, lookup, component_smiles, decorrelate=decorrelate,
            switch_after=switch_after,
            fingerprinter_kwargs=(fingerprinter_kwargs or {}).get(rep),
            coverage_fallback=coverage_fallback,
            lengthscale_rule=lengthscale_rule, prior_mode=prior_mode,
            prior_cv=prior_cv,
        )
        build_t = time.time() - t0
        print(f"  search-space dim d={feat_dim}  (featurize+build {build_t:.1f}s)", flush=True)

        set_random_seed(seed)
        t0 = time.time()
        result = simulate_scenarios(
            {rep: campaign}, lookup,
            batch_size=batch_size, n_doe_iterations=n_iter,
            n_mc_iterations=mc_runs, impute_mode="ignore",
        )
        sim_t = time.time() - t0

        m = compute_metrics(result, lookup)
        result["representation"] = rep
        result["dim"] = feat_dim
        result.to_csv(rep_csv, index=False)   # <-- persist immediately
        curves.append(result)
        print(f"  AUC={m['AUC_cumbest']:.4f}  top5%cov_final={m['Top5pct_coverage_final']:.3f}"
              f"  top5%cov_AUC={m['Top5pct_coverage_AUC']:.3f}  sim={sim_t:.1f}s"
              f"  -> saved {rep_csv.name}", flush=True)

    all_curves = pd.concat(curves, ignore_index=True)
    pr = per_run_metrics(all_curves, lookup)
    dims = all_curves.groupby("representation")["dim"].first() if "dim" in all_curves else {}
    summary = (pr.groupby("representation")
               .agg(AUC_cumbest=("AUC", "mean")).reset_index())
    summary["dim"] = summary["representation"].map(dims)

    summary.to_csv(outdir / "summary.csv", index=False)
    all_curves.to_csv(outdir / "curves.csv", index=False)
    pr.to_csv(outdir / "per_run_metrics.csv", index=False)

    # mean +/- std across MC runs (paper reports mean +/- variance)
    agg = pr.groupby("representation").agg(
        AUC_mean=("AUC", "mean"), AUC_std=("AUC", "std"),
        Top5_mean=("Top5pct_coverage", "mean"), Top5_std=("Top5pct_coverage", "std"))
    print("\n================ SUMMARY (mean +/- std over MC runs) ================")
    print(agg.round(4).to_string())
    print(f"\nWrote summary.csv, curves.csv, per_run_metrics.csv to {outdir}")
    return summary, all_curves, pr


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--reps", nargs="+", default=list(
        ("aimnet2", "mace_off23", "mace_mp0", "chemeleon", "chemberta", "t5")))
    ap.add_argument("--n-iter", type=int, default=50)
    ap.add_argument("--mc-runs", type=int, default=20)
    ap.add_argument("--switch-after", type=int, default=5,
                    help="random init points before BO (paper: 5)")
    ap.add_argument("--decorrelate", type=float, default=0.7)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--outdir", default=str(HERE / "bench_out"))
    ap.add_argument("--coverage-fallback", default="mace_mp0",
                    help="representation to use for components an MLIP can't cover "
                         "(Shields Cs/K bases); 'none' to raise instead. "
                         "'chemeleon' makes it most comparable to the paper.")
    ap.add_argument("--quick", action="store_true",
                    help="use small MACE models for speed")
    ap.add_argument("--t5-model", default=None,
                    help="HF model for the 't5' rep. Default = chemistry-T5 "
                         "(GT4SD/...). Pass 't5-base' if the ~1GB chemistry-T5 "
                         "download stalls (t5-base is the cached, verified fallback).")
    ap.add_argument("--lengthscale-rule", default=None,
                    choices=["chen", "geom"],
                    help="lengthscale prior centre (default: the paper's "
                         "AdaptiveKernelFactory, i.e. chen with its tied prior)")
    ap.add_argument("--prior-mode", default="match_concentration",
                    choices=["match_concentration", "match_parameterisation"],
                    help="how the Gamma prior width is set around the centre")
    ap.add_argument("--prior-cv", type=float, default=0.3,
                    help="CV for --prior-mode match_concentration (default 0.3)")
    ap.add_argument("--fresh", action="store_true",
                    help="ignore cached per-rep CSVs (don't resume). Use when "
                         "changing --n-iter/--mc-runs or starting over.")
    args = ap.parse_args()
    cov_fb = None if str(args.coverage_fallback).lower() == "none" else args.coverage_fallback

    fk = {}
    if args.quick:
        fk.update({"mace_off23": {"model_size": "small"},
                   "mace_mp0": {"model_size": "small"}})
    if args.t5_model:
        fk["t5"] = {"model_name": args.t5_model}
    fk = fk or None
    run(args.reps, n_iter=args.n_iter, mc_runs=args.mc_runs,
        switch_after=args.switch_after, decorrelate=args.decorrelate,
        seed=args.seed, outdir=args.outdir, fingerprinter_kwargs=fk,
        coverage_fallback=cov_fb, resume=not args.fresh,
        lengthscale_rule=args.lengthscale_rule, prior_mode=args.prior_mode,
        prior_cv=args.prior_cv)


if __name__ == "__main__":
    main()
