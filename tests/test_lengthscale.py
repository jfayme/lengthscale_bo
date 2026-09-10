"""
tests/test_lengthscale.py
=========================

Acceptance tests for module 4, `lsab/lengthscale.py` (stdlib unittest):

    python -m unittest tests.test_lengthscale -v

No models: the pool tests reuse module 3's fake representation. Test 2 compares
against BoTorch's own `Normalize` and is skipped where botorch is not installed.
"""

import importlib.util
import sys
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.spatial.distance import pdist

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from lsab import datasets, featurize  # noqa: E402
from lsab.lengthscale import (PRIOR_MODES, USTAR, cell_summary, gamma_parameters, geometry,  # noqa: E402
                              make_prior, pool_bounds, pool_geometry, prior_cv_of, rule_chen,
                              rule_geom, to_model_space)
from lsab.reduce import build  # noqa: E402
from tests.test_reduce import _fake_spec  # noqa: E402

# Each kernel as a function of u = distance / lengthscale, straight from its definition.
KERNELS = {
    "matern12": lambda u: np.exp(-u),
    "matern32": lambda u: (1 + np.sqrt(3) * u) * np.exp(-np.sqrt(3) * u),
    "matern52": lambda u: (1 + np.sqrt(5) * u + 5 * u ** 2 / 3) * np.exp(-np.sqrt(5) * u),
    "rbf": lambda u: np.exp(-u ** 2 / 2),
}


def _derivative(k, u, h=1e-20):
    """k'(u) by the complex step: exact to machine precision, no derivation to get wrong."""
    return k(u + 1j * h).imag / h


class TestRules(unittest.TestCase):
    def test_ustar_is_the_argmax(self):
        for name, k in KERNELS.items():
            with self.subTest(name):
                best = minimize_scalar(lambda u: -u * abs(_derivative(k, u)), bounds=(0.05, 5.0),
                                       method="bounded", options={"xatol": 1e-10})
                self.assertAlmostEqual(best.x, USTAR[name], places=5)

    @unittest.skipUnless(importlib.util.find_spec("botorch"), "botorch not installed")
    def test_model_space_matches_botorch_normalize(self):
        import torch
        from botorch.models.transforms.input import Normalize
        rng = np.random.default_rng(0)
        X = rng.normal(size=(50, 5)) * [1.0, 10.0, 100.0, 0.1, 3.0] + [0.0, -5.0, 50.0, 2.0, 7.0]
        X[:, 2] = 4.2                                                        # a constant column
        bounds = pool_bounds(X)
        expected = Normalize(5, bounds=torch.tensor(bounds)).transform(torch.tensor(X)).numpy()
        np.testing.assert_allclose(to_model_space(X, bounds), expected, rtol=0, atol=1e-12)
        self.assertTrue(np.all(to_model_space(X, bounds)[:, 2] == 0.0))     # not NaN

    def test_geom_scale_equivariance(self):
        Xm = np.random.default_rng(1).random((300, 8))
        base = geometry(Xm)
        for c in (0.25, 3.0, 17.5):
            scaled = geometry(c * Xm)
            self.assertAlmostEqual(rule_geom(scaled), c * rule_geom(base), places=10)
            self.assertEqual(rule_chen(scaled), rule_chen(base))

    def test_geometry_is_model_space(self):
        rng = np.random.default_rng(2)
        X = rng.normal(size=(200, 6))
        rescaled = X * rng.uniform(0.01, 100.0, size=6) + 10 * rng.normal(size=6)
        self.assertNotAlmostEqual(geometry(X).mean, geometry(rescaled).mean, places=3)
        in_model_space = lambda A: geometry(to_model_space(A, pool_bounds(A))).mean  # noqa: E731
        self.assertAlmostEqual(in_model_space(X), in_model_space(rescaled), places=10)

    def test_geometry_exact_and_subsampled(self):
        rng = np.random.default_rng(3)
        small = rng.random((500, 4))
        exact = geometry(small)
        self.assertEqual(exact.n_used, 500)
        self.assertEqual(exact.mean, float(pdist(small).mean()))
        big = rng.random((2500, 4))
        first, again, other = geometry(big, seed=0), geometry(big, seed=0), geometry(big, seed=1)
        self.assertEqual(first, again)
        self.assertNotEqual(first.mean, other.mean)
        self.assertEqual(first.n_used, 2000)

    def test_degenerate(self):
        one_hot = geometry(np.eye(40))
        self.assertTrue(one_hot.degenerate)
        self.assertAlmostEqual(one_hot.mean, np.sqrt(2.0), places=10)
        self.assertFalse(geometry(np.random.default_rng(4).random((40, 5))).degenerate)

    def test_prior_modes(self):
        centres = (0.3, 1.0, 2.7, 9.0)
        for ell_0 in centres:
            for cv in (0.1, 0.3, 1.2):
                concentration, rate = gamma_parameters(ell_0, "match_concentration", cv)
                self.assertAlmostEqual(prior_cv_of(concentration), cv, places=12)
                self.assertAlmostEqual(concentration / rate, ell_0, places=12)
        tied = []
        for ell_0 in centres:
            concentration, rate = gamma_parameters(ell_0, "match_parameterisation")
            self.assertAlmostEqual(prior_cv_of(concentration), 1.0 / np.sqrt(2.0 * ell_0), places=12)
            self.assertAlmostEqual(concentration / rate, ell_0, places=12)
            tied.append(prior_cv_of(concentration))
        self.assertTrue(np.all(np.diff(tied) < 0))          # Chen's width shrinks as the centre grows
        for bad in (0.0, -1.0, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                gamma_parameters(bad)
        with self.assertRaises(ValueError):
            gamma_parameters(1.0, "bogus")
        self.assertEqual(set(PRIOR_MODES), {"match_parameterisation", "match_concentration"})


class TestOnAPool(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with mock.patch.dict(featurize.REPS, {"fake": _fake_spec()}):
            cls.fp = build(datasets.load("bh_reaction_1"), "fake", "decorr0.7")

    def test_make_prior_on_a_pool(self):
        chen, geom = make_prior(self.fp, "chen"), make_prior(self.fp, "geom")
        for prior in (chen, geom):
            self.assertTrue(np.isfinite(prior.ell_0) and prior.ell_0 > 0)
            self.assertIn(prior.rule, prior.describe())
        self.assertNotEqual(chen.ell_0, geom.ell_0)
        self.assertEqual(make_prior(self.fp, "geom", geometry=pool_geometry(self.fp)), geom)
        with self.assertRaises(ValueError):
            make_prior(self.fp, "diamgate")                  # not ported

    def test_cell_summary_flat(self):
        plain, ranked = cell_summary(self.fp), cell_summary(self.fp, rank=True)
        documented = ["dataset", "rep", "reduction", "n", "d", "n_used", "D_bar", "cv", "degenerate",
                      "ell_geom", "ell_chen", "ratio_chen_geom"]
        self.assertEqual(list(plain), documented + [k for k in self.fp.meta() if k != "d"])
        self.assertTrue(all(type(v) in (str, int, float, bool) for v in plain.values()), plain)
        self.assertEqual(set(ranked) - set(plain), {"rank"})
        self.assertEqual(type(ranked["rank"]), int)
        self.assertLessEqual(ranked["rank"], min(ranked["n"], ranked["d"]))


if __name__ == "__main__":
    unittest.main()
