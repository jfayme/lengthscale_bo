"""
tests/test_bo.py
================

Acceptance tests for module 5, `lsab/bo.py` (stdlib unittest; needs botorch):

    python -m unittest tests.test_bo -v

Tests 1-8 run on a hand-built pool: 80 candidates in 3 dimensions with a smooth
objective and no model anywhere. The objective is a Gaussian bump,
exp(-||x - x*||^2 / 0.08) plus a little noise, rather than the bowl -||x - x*||^2:
on the bowl, 20 random draws already reach AUC 0.92 because only one far corner is
low, which leaves no room to see BO beat random by 0.05. On the bump, BO averages
0.77 against random's 0.55. Test 9 runs on bh_reaction_1 with module 3's fake
representation.
"""

import sys
import time
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd
import torch
from gpytorch.kernels import MaternKernel
from botorch.models.transforms import Standardize

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from lsab import bo, datasets, featurize  # noqa: E402
from lsab.bo import (Campaign, CampaignError, CampaignResult, build_gp, initial_design,  # noqa: E402
                     metrics, run_campaign, trajectory)
from lsab.datasets import DatasetSpec, Pool  # noqa: E402
from lsab.lengthscale import make_prior, pool_bounds  # noqa: E402
from lsab.reduce import FeaturePool, build  # noqa: E402
from tests.test_reduce import _fake_spec  # noqa: E402


def _smooth_pool(n=80, d=3, seed=0) -> FeaturePool:
    rng = np.random.default_rng(seed)
    X = rng.random((n, d))
    objective = np.exp(-np.sum((X - 0.7) ** 2, axis=1) / (2 * 0.2 ** 2)) + 0.01 * rng.normal(size=n)
    frame = pd.DataFrame({"candidate": [f"c{i}" for i in range(n)], "objective": objective})
    spec = DatasetSpec("smooth", "test", "unused.csv", "y", "max", ("candidate",))
    pool = Pool(spec, frame, {"candidate": frame["candidate"].tolist()}, {})
    return FeaturePool(pool, "synthetic", "none", X, blocks=())


class TestPieces(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fp = _smooth_pool()
        cls.chen, cls.geom = make_prior(cls.fp, "chen"), make_prior(cls.fp, "geom")

    def test_initial_design(self):
        first = initial_design(80, 5, seed=3)
        np.testing.assert_array_equal(first, initial_design(80, 5, seed=3))
        self.assertEqual(len(set(first.tolist())), 5)
        self.assertFalse(np.array_equal(first, initial_design(80, 5, seed=4)))
        self.assertFalse(np.array_equal(first, initial_design(80, 5, seed=3, seed_base=1)))
        make_prior(self.fp, "chen", "match_parameterisation")   # building priors in between...
        np.testing.assert_array_equal(first, initial_design(80, 5, seed=3))   # ...changes nothing

    def test_build_gp_reads_the_prior(self):
        X = torch.tensor(self.fp.X, dtype=torch.float64)
        y = torch.tensor(self.fp.pool.objective, dtype=torch.float64)[:10, None]
        bounds = torch.tensor(pool_bounds(self.fp.X), dtype=torch.float64)
        gp = build_gp(X[:10], y, self.geom, bounds)
        kernel = gp.covar_module
        self.assertIs(type(kernel), MaternKernel)                  # no ScaleKernel around it
        self.assertEqual(kernel.ard_num_dims, 3)
        self.assertEqual(kernel.lengthscale_prior.concentration.item(), self.geom.concentration)
        self.assertEqual(kernel.lengthscale_prior.rate.item(), self.geom.rate)
        np.testing.assert_allclose(kernel.lengthscale.detach().numpy(), self.geom.ell_0, rtol=1e-12)
        np.testing.assert_allclose(gp.input_transform.bounds.numpy(), pool_bounds(self.fp.X), rtol=1e-12)
        self.assertIsInstance(gp.outcome_transform, Standardize)

    def test_trajectory_by_hand(self):
        pool = np.array([0.0, 1.0, 2.0, 3.0, 10.0] + [5.0] * 15)   # 20 candidates; top 5% = {10.0}
        running_auc, running_coverage = trajectory([10.0, 1.0, 2.0], pool)
        np.testing.assert_array_equal(running_auc, [1.0, 1.0, 1.0])      # the max at experiment 1
        found = metrics(_result([10.0, 1.0, 2.0]), pool, n_init=1)
        self.assertEqual((found["simple_regret"], found["first_top5_hit"]), (0.0, 1))
        np.testing.assert_array_equal(running_coverage, [1.0, 1.0, 1.0])  # capped at exactly 1
        self.assertEqual(metrics(_result([0.0, 1.0, 10.0]), pool, n_init=1)["first_top5_hit"], 3)
        self.assertTrue(np.isnan(metrics(_result([0.0, 1.0, 2.0]), pool, n_init=1)["first_top5_hit"]))
        np.testing.assert_allclose(trajectory([0.0, 10.0], pool)[0], [0.0, 0.5])   # trapezoid over one step
        two_top = np.array([0.0] * 38 + [9.0, 10.0])                     # 40 values: top 5% = {9, 10}
        np.testing.assert_array_equal(trajectory([9.0, 0.0, 10.0, 9.0], two_top)[1], [0.5, 0.5, 1.0, 1.0])


def _result(sampled, n_init=1):
    sampled = np.asarray(sampled, dtype=float)
    fits = np.ones(len(sampled) - n_init)
    return CampaignResult(np.arange(len(sampled)), sampled, fits, fits, 0, 0.0)


class TestCampaigns(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fp = _smooth_pool()
        cls.chen, cls.geom = make_prior(cls.fp, "chen"), make_prior(cls.fp, "geom")

    def test_paired_arms_share_the_start(self):
        a = run_campaign(Campaign(self.fp, self.chen, n_iter=8, seed=2))
        b = run_campaign(Campaign(self.fp, self.geom, n_iter=8, seed=2))
        np.testing.assert_array_equal(a.sampled_indices[:5], b.sampled_indices[:5])

    def test_deterministic(self):
        first, again = (run_campaign(Campaign(self.fp, self.geom, n_iter=10, seed=1)) for _ in range(2))
        np.testing.assert_array_equal(first.sampled_indices, again.sampled_indices)
        np.testing.assert_array_equal(first.fitted_ell_mean, again.fitted_ell_mean)

    def test_beats_random(self):
        pool = self.fp.pool.objective
        bo_auc, random_auc = [], []
        for seed in range(10):
            result = run_campaign(Campaign(self.fp, self.geom, n_iter=20, seed=seed))
            bo_auc.append(metrics(result, pool, n_init=5)["auc"])
            draws = np.random.default_rng(1337 + seed).choice(len(pool), 20, replace=False)
            random_auc.append(trajectory(pool[draws], pool)[0][-1])
        self.assertGreater(np.mean(bo_auc), np.mean(random_auc) + 0.05, (np.mean(bo_auc), np.mean(random_auc)))

    def test_fit_record(self):
        recorded = run_campaign(Campaign(self.fp, self.geom, n_iter=9, seed=0))
        self.assertEqual(len(recorded.fitted_ell_mean), 4)
        self.assertTrue(np.all(np.isfinite(recorded.fitted_ell_mean) & (recorded.fitted_ell_mean > 0)))
        self.assertGreaterEqual(recorded.n_fit_warnings, 0)
        silent = run_campaign(Campaign(self.fp, self.geom, n_iter=9, seed=0, record_fit=False))
        self.assertTrue(np.all(np.isnan(silent.fitted_ell_mean)))

    def test_campaign_error_carries_iteration(self):
        real_fit, calls = bo.fit_gp, []

        def fails_on_the_third(gp):
            calls.append(1)
            if len(calls) == 3:
                raise RuntimeError("synthetic fit failure")
            return real_fit(gp)

        with mock.patch.object(bo, "fit_gp", fails_on_the_third):
            with self.assertRaises(CampaignError) as raised:
                run_campaign(Campaign(self.fp, self.geom, n_iter=10, seed=0))
        self.assertEqual(raised.exception.iteration, 5 + 2)
        self.assertIsInstance(raised.exception.__cause__, RuntimeError)

    def test_real_pool_smoke(self):
        with mock.patch.dict(featurize.REPS, {"fake": _fake_spec()}):
            fp = build(datasets.load("bh_reaction_1"), "fake", "decorr0.7")
        start = time.perf_counter()
        result = run_campaign(Campaign(fp, make_prior(fp, "geom"), n_iter=10, seed=0))
        self.assertLess(time.perf_counter() - start, 30.0)
        found = metrics(result, fp.pool.objective, n_init=5)
        finite = {k: v for k, v in found.items() if k != "first_top5_hit"}   # NaN when nothing top-5% was hit
        self.assertTrue(all(np.isfinite(v) for v in finite.values()), found)
        self.assertEqual(len(set(result.sampled_indices.tolist())), 10)
        self.assertTrue(np.all((result.sampled_indices >= 0) & (result.sampled_indices < fp.pool.n)))


if __name__ == "__main__":
    unittest.main()
