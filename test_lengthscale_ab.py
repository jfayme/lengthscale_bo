"""
test_lengthscale_ab.py
======================
Acceptance tests for the paired lengthscale A/B (`lengthscale_rules.py`,
`lengthscale_ab.py`, and the seam threaded through `botorch_bo.py`).

Stdlib `unittest` only -- no new dependency beyond numpy/scipy/pandas.

    python -m unittest test_lengthscale_ab -v          # everything
    python -m unittest test_lengthscale_ab.TestRules   # the fast, data-free ones

The data-backed tests load a real pool (bh_reaction_1 / morgan) through the same
code path the sweep uses; they take a few seconds and are skipped with a clear
message if the featurisation stack or its caches are unavailable.

`test_default_unchanged` compares against `tests/reference_default_run.json`,
regenerate ONLY when the default path is deliberately changed:

    python test_lengthscale_ab.py --write-reference
"""
from __future__ import annotations

import os
import sys
import json
import math
import unittest

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import lengthscale_rules as LR

REFERENCE_PATH = os.path.join(HERE, "tests", "reference_default_run.json")

# The reference cell for the data-backed tests: small, fast, non-degenerate.
REF_DATASET = "bh_reaction_1"
REF_REP = "morgan"
REF_REDUCTION = "decorr0.7"
REF_ITER = 12          # > 10, so the diamgate roughness probe is exercised
REF_INIT = 5
REF_SEED = 0
REF_SEED_BASE = 1337


# ---------------------------------------------------------------------------
# Fast, data-free
# ---------------------------------------------------------------------------
class TestRules(unittest.TestCase):
    def test_ustar_root(self):
        """u* = (1+sqrt3)/sqrt5 is the positive root of 5u^2 - 2*sqrt5*u - 2."""
        u = LR.USTAR_MATERN52
        self.assertAlmostEqual(u, 1.221810, places=6)
        residual = 5 * u**2 - 2 * math.sqrt(5) * u - 2
        self.assertLess(abs(residual), 1e-12)

    def test_ustar_maximises_kernel_sensitivity(self):
        """u* really is argmax of u*|k'(u)| for Matern-5/2, not a fitted constant."""
        u = np.linspace(1e-4, 6.0, 400001)
        # k(u) = (1 + sqrt5 u + 5u^2/3) exp(-sqrt5 u); |k'(u)| = 5/3 u (1 + sqrt5 u) e^{-sqrt5 u}
        sensitivity = u * (5.0 / 3.0) * u * (1 + math.sqrt(5) * u) * np.exp(-math.sqrt(5) * u)
        self.assertAlmostEqual(u[int(sensitivity.argmax())], LR.USTAR_MATERN52, places=4)

    def test_other_kernel_constants(self):
        """A kernel switch must move u* with it."""
        self.assertAlmostEqual(LR.USTAR["matern32"], 1.154701, places=6)
        self.assertAlmostEqual(LR.USTAR["rbf"], 1.414214, places=6)

    def test_rule_geom_scale_equivariance(self):
        """Scaling all features by c scales ell_star by exactly c; chen is invariant."""
        X = np.random.default_rng(0).random((300, 7))
        for c in (0.25, 3.0, 17.5):
            self.assertAlmostEqual(
                LR.rule_geom(c * X, 7), c * LR.rule_geom(X, 7), places=10
            )
            self.assertEqual(LR.rule_chen(c * X, 7), LR.rule_chen(X, 7))

    def test_degenerate_detection(self):
        """A one-hot single-component pool is flagged degenerate."""
        one_hot = np.eye(40)
        mean, sd, cv, degenerate = LR.pool_distance_stats(one_hot)
        self.assertTrue(degenerate)
        self.assertAlmostEqual(mean, math.sqrt(2.0), places=10)  # every distance = sqrt(2)
        self.assertLess(cv, LR.DEGENERATE_CV)
        # ... and an ordinary pool is not
        self.assertFalse(LR.pool_distance_stats(
            np.random.default_rng(1).random((200, 6)))[3])

    def test_prior_modes(self):
        """match_concentration fixes the CV; match_parameterisation does not."""
        centres = [1.0, 3.0, 7.0]
        cvs_matched = [
            LR.prior_cv_of(LR.gamma_prior_parameters(c, "match_concentration", 0.3)[0])
            for c in centres
        ]
        self.assertTrue(all(abs(v - 0.3) < 1e-12 for v in cvs_matched))

        cvs_tied = [
            LR.prior_cv_of(LR.gamma_prior_parameters(c, "match_parameterisation")[0])
            for c in centres
        ]
        self.assertEqual(cvs_tied, sorted(cvs_tied, reverse=True))  # CV = 1/sqrt(2 l0)
        for centre, cv in zip(centres, cvs_tied):
            self.assertAlmostEqual(cv, 1.0 / math.sqrt(2 * centre), places=12)
        # both parameterisations put the MEAN at the centre
        for mode in LR.PRIOR_MODES:
            concentration, rate = LR.gamma_prior_parameters(7.0, mode)
            self.assertAlmostEqual(concentration / rate, 7.0, places=12)

    def test_model_space_matches_botorch_normalize(self):
        """D_bar must be computed in the space the kernel sees, so `to_model_space`
        has to agree with the model's own input transform."""
        import torch
        from botorch.models.transforms.input import Normalize

        X = np.random.default_rng(2).normal(size=(50, 5)) * 10 + 3
        pool = torch.tensor(X)
        bounds = torch.stack([pool.min(0).values, pool.max(0).values])
        expected = Normalize(5, bounds=bounds).transform(pool).numpy()
        np.testing.assert_allclose(LR.to_model_space(X, bounds.numpy()), expected,
                                   rtol=0, atol=1e-12)


class TestBaybePassThrough(unittest.TestCase):
    """The BayBE-side seam in `benchmark_representations.py`."""

    class _StubSearchSpace:
        """The two attributes a kernel factory reads: the column list and the pool."""

        def __init__(self, pool):
            import pandas as pd

            self.comp_rep_columns = [f"x{i}" for i in range(pool.shape[1])]
            self.discrete = type("D", (), {})()
            self.discrete.comp_rep = pd.DataFrame(pool, columns=self.comp_rep_columns)

    def setUp(self):
        try:
            import baybe  # noqa: F401
        except Exception as exc:
            self.skipTest(f"baybe unavailable: {exc}")

    def test_chen_tied_reproduces_the_paper_factory(self):
        """chen + match_parameterisation IS the paper's AdaptiveKernelFactory."""
        sys.path.insert(0, os.path.join(HERE, "HSF-ChemBO-tutorial"))
        from base.kernels import AdaptiveKernelFactory
        import benchmark_representations as B

        searchspace = self._StubSearchSpace(np.random.default_rng(4).random((60, 9)))
        paper = AdaptiveKernelFactory()(searchspace, None, None)
        ours = B.lengthscale_ab_factory("chen", "match_parameterisation")(
            searchspace, None, None)
        self.assertEqual(paper, ours)

    def test_geom_moves_only_the_lengthscale_prior(self):
        """Swapping the rule changes the lengthscale prior and NOTHING else."""
        import benchmark_representations as B

        searchspace = self._StubSearchSpace(np.random.default_rng(5).random((60, 9)))
        chen = B.lengthscale_ab_factory("chen", "match_concentration")(
            searchspace, None, None)
        geom = B.lengthscale_ab_factory("geom", "match_concentration")(
            searchspace, None, None)
        self.assertEqual(chen.outputscale_prior, geom.outputscale_prior)
        self.assertEqual(chen.outputscale_initial_value, geom.outputscale_initial_value)
        self.assertEqual(chen.base_kernel.nu, geom.base_kernel.nu)
        self.assertNotEqual(chen.base_kernel.lengthscale_initial_value,
                            geom.base_kernel.lengthscale_initial_value)
        # same tightness, different centre
        self.assertAlmostEqual(chen.base_kernel.lengthscale_prior.concentration,
                               geom.base_kernel.lengthscale_prior.concentration,
                               places=12)


# ---------------------------------------------------------------------------
# Data-backed: the real pool, the real BO loop
# ---------------------------------------------------------------------------
def _load_reference_cell():
    import lengthscale_ab as AB

    return AB.Cell(REF_DATASET, REF_REP, REF_REDUCTION, REF_ITER, REF_INIT,
                   n_random_draws=20, seed_base=REF_SEED_BASE)


class TestPairedRun(unittest.TestCase):
    cell = None

    @classmethod
    def setUpClass(cls):
        try:
            cls.cell = _load_reference_cell()
        except Exception as exc:  # missing data / featurisation stack
            raise unittest.SkipTest(f"cannot load {REF_DATASET}/{REF_REP}: {exc}")

    def test_paired_initial_design(self):
        """All four arms start from the IDENTICAL initial design."""
        import torch
        import botorch_bo as BO
        import lengthscale_ab as AB

        expected = self.cell.initial_indices(REF_SEED, REF_SEED_BASE)
        for rule in AB.RULES:
            for prior_mode in AB.PRIOR_MODES:
                drawn = self.cell.initial_indices(REF_SEED, REF_SEED_BASE)
                self.assertEqual(list(drawn), list(expected))

                spec = BO.LengthscaleSpec(rule=rule, ell_0=self.cell.ell_0(rule),
                                          prior_mode=prior_mode, cv=0.3)
                torch.manual_seed(REF_SEED_BASE + REF_SEED)
                campaign = BO.run_one_campaign(
                    BO.StaticAcquisitionFunction("qLogEI"),
                    self.cell.pool_x, self.cell.y_pool,
                    self.cell.normalised_features, self.cell.d, self.cell.geometry,
                    self.cell.bounds, drawn, REF_INIT + 1, REF_SEED,
                    record_trace=False, lengthscale_spec=spec,
                )
                self.assertEqual(campaign["sampled_indices"][:REF_INIT], list(expected),
                                 f"{rule}/{prior_mode} did not start from the shared design")

    def test_arms_differ_only_in_the_prior_centre(self):
        """The two rules put the prior in genuinely different places, and (under
        match_concentration) hold it with exactly the same tightness."""
        import botorch_bo as BO

        centres = {rule: self.cell.ell_0(rule) for rule in ("chen", "geom")}
        self.assertNotAlmostEqual(centres["chen"], centres["geom"], places=3)

        concentrations = {
            rule: BO.LengthscaleSpec(rule=rule, ell_0=centre,
                                     prior_mode="match_concentration",
                                     cv=0.3).gamma_parameters(centre)[0]
            for rule, centre in centres.items()
        }
        self.assertAlmostEqual(concentrations["chen"], concentrations["geom"], places=12)

    def test_dbar_is_model_space(self):
        """D_bar/ell_star come from the [0,1] cube, not the raw descriptors.

        The signature of that: a per-column rescaling of the raw features changes
        the raw mean distance but leaves the model-space D_bar untouched, because
        `Normalize` divides the rescaling straight back out. (Morgan bits are
        already 0/1, so raw and model space coincide for THIS cell -- which is
        exactly why the invariance, not a numeric difference, is what to assert.)
        """
        self.assertAlmostEqual(self.cell.ell_star,
                               self.cell.D_bar / LR.USTAR_MATERN52, places=12)
        self.assertTrue(0.0 <= self.cell.model_space.min())
        self.assertTrue(self.cell.model_space.max() <= 1.0 + 1e-12)
        self.assertAlmostEqual(self.cell.D_bar,
                               LR.pool_mean_distance(self.cell.model_space), places=12)

        scale = np.random.default_rng(3).uniform(0.5, 20.0, self.cell.features.shape[1])
        rescaled = self.cell.features * scale
        rescaled_bounds = np.stack([rescaled.min(0), rescaled.max(0)])
        self.assertAlmostEqual(
            LR.pool_mean_distance(LR.to_model_space(rescaled, rescaled_bounds)),
            self.cell.D_bar, places=10)
        self.assertNotAlmostEqual(LR.pool_mean_distance(rescaled), self.cell.D_bar,
                                  places=3)

    def test_default_unchanged(self):
        """The default path reproduces the stored reference result exactly."""
        if not os.path.exists(REFERENCE_PATH):
            self.skipTest(f"no reference at {REFERENCE_PATH}; "
                          "run `python test_lengthscale_ab.py --write-reference`")
        reference = json.load(open(REFERENCE_PATH))
        current = _default_run(self.cell)
        self.assertEqual(reference["settings"], current["settings"])
        np.testing.assert_array_equal(
            np.array(reference["sampled_objective_values"]),
            np.array(current["sampled_objective_values"]),
        )
        np.testing.assert_array_equal(
            np.array(reference["sampled_indices"]), np.array(current["sampled_indices"])
        )

    def test_default_spec_equals_no_spec(self):
        """Passing the default LengthscaleSpec is the same as passing none, i.e. the
        refactor did not move the incumbent prior."""
        import torch
        import botorch_bo as BO

        train_index = torch.tensor(self.cell.initial_indices(REF_SEED, REF_SEED_BASE))
        train_x = self.cell.pool_x[train_index]
        train_y = torch.tensor(self.cell.y_pool[train_index.numpy()]).unsqueeze(-1)

        without = BO.build_gp(train_x, train_y, self.cell.d, self.cell.geometry,
                              self.cell.bounds)
        with_spec = BO.build_gp(train_x, train_y, self.cell.d, self.cell.geometry,
                                self.cell.bounds, BO.LengthscaleSpec())

        for attribute in ("concentration", "rate"):
            self.assertEqual(
                float(getattr(without.covar_module.lengthscale_prior, attribute)),
                float(getattr(with_spec.covar_module.lengthscale_prior, attribute)),
            )
        # ... and it is still the historical narrow cv=0.3 Gamma
        self.assertAlmostEqual(
            float(without.covar_module.lengthscale_prior.concentration),
            1.0 / 0.3**2, places=12)
        np.testing.assert_array_equal(
            without.covar_module.lengthscale.detach().numpy(),
            with_spec.covar_module.lengthscale.detach().numpy(),
        )


def _default_run(cell):
    """One campaign on the DEFAULT (no lengthscale_rule) path -> the reference."""
    import torch
    import botorch_bo as BO

    initial = cell.initial_indices(REF_SEED, REF_SEED_BASE)
    torch.manual_seed(REF_SEED_BASE + REF_SEED)
    campaign = BO.run_one_campaign(
        BO.StaticAcquisitionFunction("qLogEI"),
        cell.pool_x, cell.y_pool, cell.normalised_features, cell.d, cell.geometry,
        cell.bounds, initial, REF_ITER, REF_SEED, record_trace=False,
    )
    return dict(
        settings=dict(dataset=REF_DATASET, representation=REF_REP,
                      reduction=REF_REDUCTION, n_iter=REF_ITER, n_init=REF_INIT,
                      seed=REF_SEED, seed_base=REF_SEED_BASE, d=cell.d,
                      n_pool=cell.n_pool, acquisition="qLogEI"),
        sampled_objective_values=[float(v) for v in campaign["sampled_objective_values"]],
        sampled_indices=[int(i) for i in campaign["sampled_indices"]],
    )


def write_reference():
    os.makedirs(os.path.dirname(REFERENCE_PATH), exist_ok=True)
    reference = _default_run(_load_reference_cell())
    with open(REFERENCE_PATH, "w") as handle:
        json.dump(reference, handle, indent=2)
    print(f"wrote {REFERENCE_PATH}")


if __name__ == "__main__":
    if "--write-reference" in sys.argv:
        write_reference()
    else:
        unittest.main()
