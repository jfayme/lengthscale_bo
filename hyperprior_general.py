"""
hyperprior_general.py
=====================
Does a *more general* lengthscale prior beat the paper's fixed 0.4*sqrt(d)+4
formula across representations and tasks?

Motivation (from the final report's section 6 + the distance diagnostic):
  * The paper formula ell0 = 0.4*sqrt(d)+4 depends only on the raw dimension d.
  * But BoTorch min-max Normalizes every input to [0,1], and the *actual* median
    pairwise distance in that space is 0.5-2.0 for our pools -- i.e. the formula
    lengthscale (~5) sits 5-10x ABOVE the real geometry, and raw d even points the
    WRONG way across tasks (photoswitches d=13 wants a SHORTER ell than redox d=4).
  * Classic fix: the *median heuristic* ell0 = c * median||x_i - x_j|| (computed on
    the candidate pool, which pool-based BO has in full at t=0). It is
    representation- and dataset-adaptive by construction and reduces to the sqrt(d)
    law when every coordinate is unit-variance/well-spread.

We hold EVERYTHING else fixed (Matern-5/2, qLogEI, 5 init, 50 iter, 20 MC, common
outputscale) and vary only the lengthscale-prior CENTER:

  paper     : ell0 = 0.4*sqrt(d)+4                         (baseline / the formula)
  sqrtd     : ell0 = sqrt(d)                               (raw curse-of-dim scaling)
  median0.5 : ell0 = 0.5 * median_pairwise_dist(pool_norm)
  median1   : ell0 = 1.0 * median_pairwise_dist(pool_norm)
  median2   : ell0 = 2.0 * median_pairwise_dist(pool_norm)

Lengthscale prior Gamma(2*ell0, 2) (mean ell0, the paper's shape); outputscale held
at the paper value o0 = 0.4*sqrt(d)+4 for every variant so only the lengthscale
center moves.

Usage:
  python hyperprior_general.py --reps aimnet2_all --datasets photoswitches redox_mer
  python hyperprior_general.py --reps aimnet2_all mace_mp0 morgan t5 \
      --datasets photoswitches redox_mer bh_full shields
"""
from __future__ import annotations
import os, sys, math, time, argparse, warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from scipy.spatial.distance import pdist

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, "HSF-ChemBO-tutorial"))
import base.kernels as BK
from baybe.kernels import ScaleKernel, MaternKernel
from baybe.priors.basic import GammaPrior

OUT = os.path.join(HERE, "hyperprior_gen_out"); os.makedirs(OUT, exist_ok=True)
_RNG = np.random.default_rng(0)


def _pool_geom(searchspace, max_n=2000):
    """(median dist, p95 dist, per-col min, per-col range) over the candidate pool, in
    the per-column min-max-normalized space. median = rough scale; p95 = stable diameter
    (smooth scale). Bounds reused to put train points into the SAME space for the fit."""
    X = searchspace.discrete.comp_rep.values.astype(float)
    if X.shape[0] > max_n:
        X = X[_RNG.choice(X.shape[0], max_n, replace=False)]
    lo, hi = X.min(0), X.max(0); rng = np.where(hi > lo, hi - lo, 1.0)
    dd = pdist((X - lo) / rng, "euclidean")
    return float(np.median(dd)), float(np.percentile(dd, 95)), lo, rng


def _data_lengthscale(train_x, train_y, lo, rng):
    """Quick isotropic Matern-5/2 MLE lengthscale on the measured points, normalized
    by the POOL bounds so it is comparable to the pool median (GOLLuM ratio = this/median).
    This is the 'roughness probe': smooth objective -> long ell, rough -> short ell."""
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import Matern, ConstantKernel as C, WhiteKernel as W
    y = np.asarray(train_y, float).ravel()
    if len(y) < 3 or np.ptp(y) < 1e-9:
        return None
    X = (np.asarray(train_x, float) - lo) / rng
    ys = (y - y.mean()) / (y.std() + 1e-12)
    k = C(1.0) * Matern(length_scale=1.0, nu=2.5, length_scale_bounds=(1e-2, 1e2)) + W(0.1, (1e-4, 10.0))
    try:
        gp = GaussianProcessRegressor(kernel=k, n_restarts_optimizer=1, normalize_y=False).fit(X, ys)
        return float(gp.kernel_.k1.k2.length_scale)
    except Exception:
        return None


class _GenFactory(BK.KernelFactory):
    """Strategy string is "<center>" or "<center>@cv<float>".

    <center> sets the prior-MEAN lengthscale:
        paper      -> 0.4*sqrt(d)+4          (the formula; geometry-blind +4 floor)
        sqrtd      -> sqrt(d)                 (raw curse-of-dim scaling, no floor)
        median<c>  -> c * median_pairwise_dist(pool)   (geometry-anchored; "geom"==median1)
    @cv<float> sets the prior WIDTH independently (coefficient of variation):
        omitted    -> paper's tied shape Gamma(2*l0, 2)  (CV falls out of l0 -> CONFOUNDED)
        @cv0.3     -> narrow / confident prior   (data can't move the lengthscale)
        @cv1.2     -> wide / permissive prior     (data slides the lengthscale freely)
    The center knob tests Q3-point-1 (anchor to geometry); the @cv knob tests
    Q3-point-2 (widen so the MAP fit finds the objective's roughness). Use a
    {paper,median1} x {cv0.3,cv1.2} factorial to DISENTANGLE the two effects.
    """
    STRATEGY = "paper"           # set per run
    NMIN = 10                    # min measured points before trusting the roughness probe
    def __init__(self, *a, **k): self._geom = None
    def _pool(self, searchspace):
        if self._geom is None:
            self._geom = _pool_geom(searchspace)
        return self._geom
    R_RHO = 0.50                 # Matern-5/2 dist at corr rho~0.83 -> 1/R_RHO=2.0 (data-validated:
    #                              smooth-cell ell*/p95 median = 2.03)
    def _center(self, name, d, searchspace):
        if name == "paper": return 0.4 * math.sqrt(d) + 4.0
        if name == "sqrtd": return math.sqrt(d)
        if name == "geom":  name = "median1"
        if name.startswith("median"):
            c = float(name.replace("median", "") or "1")
            return max(c * self._pool(searchspace)[0], 1e-3)
        raise ValueError(name)
    def _diamgate_center(self, searchspace, train_x, train_y, d):
        """FINAL formula: l0 = median + ((1/r_rho)*p95 - median)*s.  Geometry (rough) ->
        kernel-saturated diameter (smooth), interpolated by data smoothness s=w(ratio)."""
        med, p95, lo, rng = self._pool(searchspace)
        geom = max(med, 1e-3)
        smooth_target = p95 / _GenFactory.R_RHO            # = 1.82 * p95 (kernel saturation)
        if train_y is None or len(train_y) < _GenFactory.NMIN:
            return geom
        ell = _data_lengthscale(train_x, train_y, lo, rng)
        if ell is None:
            return geom
        w = min(max((ell / geom - 1.0) / (3.0 - 1.0), 0.0), 1.0)   # smoothness s in [0,1]
        return geom + (smooth_target - geom) * w
    def _adaptive_center(self, searchspace, train_x, train_y, d, gate=False):
        """Roughness-probe center. Start at geometry (median dist, rough-safe); once
        >=NMIN points are in, fit a quick GP for the data lengthscale ell.
          gate=False ('adaptive')    : center = clip(ell, geometry, floor)
                                       (FIT-based; lands ~median, misses the smooth-task
                                        exploration bonus the +4 floor provides)
          gate=True  ('adaptivegate'): center = log-interpolate(geometry -> floor) by a
                                       smoothness gate w(ratio), ratio = ell/median.
                                       Rough (low ratio) -> geometry; smooth (high ratio)
                                       -> long floor (BO exploration). Thresholds [1,3]
                                       are provisional -- validate on bh grids, don't trust
                                       on these 2 pools alone."""
        med, p95, lo, rng = self._pool(searchspace)
        geom, floor = max(med, 1e-3), 0.4 * math.sqrt(d) + 4.0
        if train_y is None or len(train_y) < _GenFactory.NMIN:
            return geom
        ell = _data_lengthscale(train_x, train_y, lo, rng)
        if ell is None:
            return geom
        if not gate:
            return min(max(ell, geom), floor)
        ratio = ell / geom
        w = min(max((ratio - 1.0) / (3.0 - 1.0), 0.0), 1.0)      # rough->0, smooth->1
        return math.exp((1.0 - w) * math.log(geom) + w * math.log(floor))
    def __call__(self, searchspace, train_x, train_y):
        if _GenFactory.STRATEGY == "baybe_default":   # BayBE's own default prior (BayBE_adaptive)
            from baybe.surrogates.gaussian_process.presets import DefaultKernelFactory
            return DefaultKernelFactory()(searchspace, train_x, train_y)
        d = len(searchspace.comp_rep_columns)
        o0 = 0.4 * math.sqrt(d) + 4.0                 # common outputscale (isolate ls)
        s, cv = _GenFactory.STRATEGY, None
        if "@cv" in s:
            s, cvs = s.split("@cv"); cv = float(cvs)
        if s == "diamgate":                            # the FINAL formula
            l0 = self._diamgate_center(searchspace, train_x, train_y, d)
            if cv is None: cv = 0.3
        elif s in ("adaptive", "adaptivegate"):
            l0 = self._adaptive_center(searchspace, train_x, train_y, d,
                                       gate=(s == "adaptivegate"))
            if cv is None: cv = 0.3                    # adaptive default: NARROW (per factorial)
        else:
            l0 = self._center(s, d, searchspace)
        if cv is None:                                 # paper's tied shape (width ~ 1/sqrt(2 l0))
            ls_prior = GammaPrior(2.0 * l0, 2.0)
        else:                                          # independent width: mean=l0, CV=cv
            alpha = 1.0 / (cv * cv); ls_prior = GammaPrior(alpha, alpha / l0)
        return ScaleKernel(
            MaternKernel(nu=2.5, lengthscale_prior=ls_prior,
                         lengthscale_initial_value=l0),
            outputscale_prior=GammaPrior(o0, 1.0), outputscale_initial_value=o0)


BK.AdaptiveKernelFactory = _GenFactory                # monkeypatch (pipeline uses this)

import gollum_pipeline as GP                          # noqa: E402  (after patch)
import benchmark_representations as B                 # noqa: E402
CFG = {d["name"]: d for d in GP.DATASETS}


def _load(ds):
    if ds == "shields":
        lookup, comps = B.load_shields()
        return {"kind": "shields", "lookup": lookup}
    return GP.load_dataset(CFG[ds])


def _campaign(rep, ds, loaded, pca_cap, decorr):
    # pca_cap given -> PCA mode (decorrelate off); pca_cap None -> decorrelation-`decorr` mode.
    if ds == "shields":
        camp, dim = B.build_campaign(rep, loaded["lookup"], B.load_shields()[1],
                                     decorrelate=decorr)
        return camp, dim
    dec = decorr if pca_cap is None else False
    return GP.build_campaign(rep, loaded, pca_cap=pca_cap, decorrelate=dec)[:2]


def run(reps, datasets, strategies, mc, niter, pca_cap, decorr=0.7):
    from baybe.simulation import simulate_scenarios
    from baybe.utils.random import set_random_seed
    tag = "" if pca_cap else "__nopca"
    summary = []
    for ds in datasets:
        loaded = _load(ds)
        lk = loaded["lookup"]
        print(f"\n==== {ds} (pca_cap={pca_cap}) ====", flush=True)
        for rep in reps:
            rows = []
            for s in strategies:
                csv = os.path.join(OUT, f"{ds}__{rep}__{s}{tag}.csv")
                if os.path.exists(csv):
                    res = pd.read_csv(csv)
                else:
                    _GenFactory.STRATEGY = s
                    camp, dim = _campaign(rep, ds, loaded, pca_cap, decorr)
                    set_random_seed(1337); t = time.time()
                    res = simulate_scenarios({s: camp}, lk, batch_size=1,
                                             n_doe_iterations=niter, n_mc_iterations=mc,
                                             impute_mode="ignore")
                    res["representation"] = s
                    res.to_csv(csv, index=False)
                    print(f"  {rep:12s} {s:10s} dim={dim} (sim {time.time()-t:.0f}s)", flush=True)
                pr = B.per_run_metrics(res, lk)
                rows.append((s, pr.AUC.mean(), pr.AUC.std()))
                summary.append(dict(dataset=ds, rep=rep, strategy=s,
                                    AUC=pr.AUC.mean(), std=pr.AUC.std()))
            best = max(rows, key=lambda r: r[1])[0]
            print(f"  -- {rep} (best: {best}) --")
            for s, a, sd in sorted(rows, key=lambda r: -r[1]):
                star = " <-- paper" if s == "paper" else ("  *BEST*" if s == best else "")
                print(f"     {s:10s} AUC={a:.3f} +/- {sd:.3f}{star}", flush=True)
    pd.DataFrame(summary).to_csv(os.path.join(OUT, "summary.csv"), index=False)
    print("\nwrote", os.path.join(OUT, "summary.csv"))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", nargs="+", default=["aimnet2_all"])
    # bh_full + bh_reaction_1 included by default in every run (user standing preference)
    ap.add_argument("--datasets", nargs="+",
                    default=["photoswitches", "redox_mer", "bh_full", "bh_reaction_1"])
    ap.add_argument("--strategies", nargs="+",
                    default=["paper", "sqrtd", "median0.5", "median1", "median2"])
    ap.add_argument("--mc", type=int, default=20)
    ap.add_argument("--iter", type=int, default=50)
    ap.add_argument("--pca-cap", type=int, default=64, help="0 = no PCA (use decorrelation)")
    ap.add_argument("--decorrelate", type=float, default=0.7,
                    help="decorrelation threshold when --pca-cap 0")
    a = ap.parse_args()
    run(a.reps, a.datasets, a.strategies, a.mc, a.iter,
        pca_cap=(a.pca_cap or None), decorr=a.decorrelate)
