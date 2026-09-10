"""
tests/test_reduce.py
====================

Acceptance tests for module 3, `lsab/reduce.py` (stdlib unittest, no models):

    python -m unittest tests.test_reduce -v

A fake representation is registered in `featurize.REPS`. It maps a SMILES to a
deterministic vector seeded from a STABLE hash of the string (crc32, not
Python's per-process `hash`), so `build` runs end to end on real pools.
"""

import sys
import unittest
import zlib
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from lsab import datasets, featurize  # noqa: E402
from lsab.datasets import DatasetSpec, Pool  # noqa: E402
from lsab.featurize import RepSpec  # noqa: E402
from lsab.reduce import PCA, REDUCTIONS, Decorrelate, build  # noqa: E402

D_FAKE = 16


def _fake_spec(name="fake", elements=None, fallback=None, fail=(), constant_column=False):
    """A RepSpec with no model: a 16-d vector per SMILES, raising for `fail`."""
    class Fake:
        def __call__(self, smiles):
            if any(s in fail for s in smiles):
                raise ValueError("cannot embed")
            rows = [np.random.default_rng(zlib.crc32(s.encode())).normal(size=D_FAKE) for s in smiles]
            X = np.array(rows, dtype=np.float32)
            if constant_column:
                X[:, 0] = 1.0
            return X
    return RepSpec(name, lambda device: Fake(), elements, fallback, cached=False)


def _fake_vector(smiles):
    return np.random.default_rng(zlib.crc32(smiles.encode())).normal(size=D_FAKE).astype(np.float32)


def _synthetic_pool(components: dict[str, list[str]]) -> Pool:
    """A pool whose rows pair the i-th molecule of every component (shorter lists cycle)."""
    n = max(len(v) for v in components.values())
    frame = pd.DataFrame({c: [v[i % len(v)] for i in range(n)] for c, v in components.items()})
    frame["objective"] = np.arange(n, dtype=float)
    spec = DatasetSpec("synthetic", "test", "unused.csv", "y", "max", tuple(components))
    return Pool(spec, frame, {c: list(dict.fromkeys(v)) for c, v in components.items()}, {})


class TestReductions(unittest.TestCase):
    def test_decorrelate_drops_correlated(self):
        rng = np.random.default_rng(0)
        z = rng.normal(size=(200, 3))
        noise = rng.normal(size=(200, 3))
        c0, c2, c4 = 3 * z[:, 0], 2 * z[:, 1], z[:, 2]
        # c3 is -0.99*c2 + noise, not exactly -c2: |corr| = 1 would be dropped at ANY
        # threshold below 1, so "0.99999 keeps all six" could not hold.
        X = np.column_stack([c0, 0.99 * c0 + 0.1 * noise[:, 0], c2, -0.99 * c2 + 0.05 * noise[:, 1],
                             c4, 0.99 * c4 + 0.02 * noise[:, 2]])
        variance = X.var(axis=0)
        self.assertTrue(variance[0] > variance[1] and variance[2] > variance[3] and variance[4] > variance[5])
        np.testing.assert_array_equal(Decorrelate(0.7).fit_transform(X), X[:, [0, 2, 4]])
        # variance order, not index order: reversing the columns keeps the SAME survivors
        np.testing.assert_array_equal(Decorrelate(0.7).fit_transform(X[:, ::-1]), X[:, [4, 2, 0]])
        np.testing.assert_array_equal(Decorrelate(0.99999).fit_transform(X), X)

    def test_decorrelate_edge_cases(self):
        X = np.random.default_rng(1).normal(size=(2, 5))
        with self.assertLogs("lsab.reduce", "INFO"):
            kept = Decorrelate(0.7).fit_transform(X)
        np.testing.assert_array_equal(kept, X[:, [np.argmax(X.var(axis=0))]])   # one column survives
        pool = _synthetic_pool({"solo_component": ["CCO"], "other": ["C", "CC", "CCC"]})
        with mock.patch.dict(featurize.REPS, {"fake": _fake_spec()}):
            with self.assertRaisesRegex(ValueError, "solo_component"):
                build(pool, "fake", "decorr0.7")

    def test_pca(self):
        rng = np.random.default_rng(2)
        basis = np.linalg.qr(rng.normal(size=(20, 3)))[0].T                  # 3 orthonormal directions
        X = (rng.normal(size=(100, 3)) * [10.0, 5.0, 3.0]) @ basis + 0.01 * rng.normal(size=(100, 20))
        scores = PCA(0.98, 64).fit_transform(X)
        self.assertEqual(scores.shape, (100, 3))
        self.assertEqual(PCA(0.98, 2).fit_transform(X).shape, (100, 2))      # the cap binds
        self.assertEqual(PCA(0.98, 64).fit_transform(rng.normal(size=(5, 100))).shape, (5, 4))
        order = rng.permutation(100)                                          # no sign flips between runs
        np.testing.assert_allclose(PCA(0.98, 64).fit_transform(X[order]), scores[order], atol=1e-9)
        variance = scores.var(axis=0)
        self.assertTrue(np.all(np.diff(variance) <= 0))                      # explained variance is monotone

    def test_constant_and_impute(self):
        smiles = [f"M{i}" for i in range(100)]
        with mock.patch.dict(featurize.REPS, {"fake": _fake_spec(fail={"M7"}, constant_column=True)}):
            features = build(_synthetic_pool({"mol": smiles}), "fake", "none")
        block = features.blocks[0]
        self.assertEqual((block.d_raw, block.d_constant_dropped, block.n_failed), (D_FAKE, D_FAKE - 1, 1))
        others = np.array([_fake_vector(s) for s in smiles if s != "M7"], dtype=np.float64)[:, 1:]
        np.testing.assert_array_equal(features.X[7], np.median(others, axis=0))
        fifty = [f"M{i}" for i in range(50)]
        with mock.patch.dict(featurize.REPS, {"fake": _fake_spec(fail={"M1", "M2"})}):
            with self.assertRaisesRegex(ValueError, "M1"):                    # 2 of 50 = 4% > 1%
                build(_synthetic_pool({"mol": fifty}), "fake", "none")


class TestAssembly(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bh = datasets.load("bh_reaction_1")
        cls.shields = datasets.load("shields")

    def test_assembly_reaction_pool(self):
        with mock.patch.dict(featurize.REPS, {"fake": _fake_spec()}):
            features = build(self.bh, "fake", "none")
        self.assertEqual(features.X.shape, (self.bh.n, sum(b.d_reduced for b in features.blocks)))
        self.assertEqual(features.X.dtype, np.float64)
        for i in np.random.default_rng(3).choice(self.bh.n, 5, replace=False):
            row = self.bh.frame.iloc[i]
            expected = np.concatenate([_fake_vector(row[c]) for c in self.bh.spec.components])
            np.testing.assert_array_equal(features.X[i], expected.astype(np.float64))
        tiled = [c for b in features.blocks for c in range(b.columns.start, b.columns.stop)]
        self.assertEqual(tiled, list(range(features.d)))
        self.assertEqual([b.name for b in features.blocks], list(self.bh.spec.components))

    def test_assembly_shields_numeric(self):
        fakes = {"mace_off23": _fake_spec("mace_off23", featurize.REPS["mace_off23"].elements, "mace_mp0"),
                 "mace_mp0": _fake_spec("mace_mp0", featurize.REPS["mace_mp0"].elements)}
        with mock.patch.dict(featurize.REPS, fakes), self.assertLogs("lsab.featurize", "WARNING"):
            features = build(self.shields, "mace_off23", "none")
        used = {b.name: b.used_rep for b in features.blocks if b.kind == "component"}
        self.assertEqual(used, {"Solvent_SMILES": "mace_off23", "Base_SMILES": "mace_mp0",
                                "Ligand_SMILES": "mace_off23"})      # the Cs/K bases fell back
        for name in self.shields.spec.numeric:
            block = next(b for b in features.blocks if b.name == name)
            self.assertEqual((block.kind, block.d_reduced), ("numeric", 1))
            np.testing.assert_array_equal(features.X[:, block.columns][:, 0],
                                          self.shields.frame[name].to_numpy(dtype=np.float64))

    def test_meta_is_flat_and_csv_safe(self):
        with mock.patch.dict(featurize.REPS, {"fake": _fake_spec()}):
            first, second = (build(self.bh, "fake", "decorr0.7").meta() for _ in range(2))
        self.assertTrue(all(type(v) in (str, int, float) for v in first.values()), first)
        self.assertEqual(list(first), list(second))
        self.assertEqual(first, second)

    def test_reductions_registry(self):
        self.assertEqual(set(REDUCTIONS), {"decorr0.7", "pca64", "none"})
        with self.assertRaises(ValueError) as raised:
            build(self.bh, "morgan", "bogus")
        for name in REDUCTIONS:
            self.assertIn(name, str(raised.exception))


if __name__ == "__main__":
    unittest.main()
