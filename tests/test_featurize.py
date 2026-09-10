"""
tests/test_featurize.py
=======================

Acceptance tests for module 2, `lsab/featurize.py` + `lsab/featurizers/` (stdlib unittest).

    python -m unittest tests.test_featurize -v                  # tests 1-6: no model weights
    LSAB_SLOW=1 python -m unittest tests.test_featurize -v      # + test 7, the real models

Tests 4-6 register FAKE featurizers in `REPS`, so the cache and coverage logic
run without any model. Test 7 is opt-in because it loads six models (and, if
their weights are not cached yet, downloads them); within it, a rep whose stack
does not import is skipped rather than failed.
"""

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
from rdkit.rdBase import BlockLogs

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from lsab.featurize import REPS, RepSpec, embed, embed_component, uncovered_elements  # noqa: E402


def _fake_rep(name="fake", elements=None, fallback=None, interrupt_at=None):
    """A RepSpec whose featurizer needs no model, plus counters of what it did.

    The fake returns a deterministic 3-vector per SMILES, raises on "BAD", and
    raises KeyboardInterrupt on its `interrupt_at`-th molecule.
    """
    calls = {"build": 0, "embedded": []}

    class Fake:
        def __init__(self):
            calls["build"] += 1
            self.name = name

        def __call__(self, smiles):
            calls["embedded"].extend(smiles)
            if len(calls["embedded"]) == interrupt_at:
                raise KeyboardInterrupt
            if "BAD" in smiles:
                raise ValueError("cannot embed BAD")
            return np.array([[len(s), sum(map(ord, s)) % 97, s.count("C")] for s in smiles],
                            dtype=np.float32)

    return RepSpec(name, lambda device: Fake(), elements, fallback, cached=True), calls


class TestFeaturize(unittest.TestCase):
    def test_featurize_import_is_light(self):
        heavy = ["torch", "mace", "aimnet", "transformers", "rdkit"]
        code = f"import sys, lsab.featurize; print([m for m in {heavy!r} if m in sys.modules])"
        run = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True,
                             text=True, check=True)   # a fresh interpreter: other tests import rdkit
        self.assertEqual(run.stdout.strip(), "[]")

    def test_morgan(self):
        from lsab.featurizers.morgan import MorganFeaturizer
        featurizer = MorganFeaturizer()
        X = featurizer(["CCO", "c1ccccc1"])
        self.assertEqual(X.shape, (2, 2048))
        self.assertEqual(X.dtype, np.float32)
        self.assertTrue(set(np.unique(X)) <= {0.0, 1.0})
        np.testing.assert_array_equal(X, featurizer(["CCO", "c1ccccc1"]))
        self.assertFalse(np.array_equal(X[0], X[1]))
        with BlockLogs(), self.assertRaises(ValueError):   # BlockLogs: RDKit's own parse error is noise
            featurizer(["C1CC("])

    def test_conformer_helper(self):
        from lsab.featurizers.conformer import smiles_to_atoms
        numbers, coords, charge = smiles_to_atoms("CCO")
        self.assertEqual(int((numbers == 1).sum()), 6)   # hydrogens added
        self.assertEqual(numbers.dtype, np.int64)
        self.assertEqual((coords.dtype, coords.shape), (np.float32, (9, 3)))
        self.assertEqual(charge, 0)
        np.testing.assert_array_equal(coords, smiles_to_atoms("CCO")[1])
        flexible = "CCCCCCCCO"
        self.assertFalse(np.array_equal(smiles_to_atoms(flexible)[1],
                                        smiles_to_atoms(flexible, optimize=False)[1]))
        with BlockLogs(), self.assertRaises(ValueError):
            smiles_to_atoms("C1CC(")
        self.assertEqual(smiles_to_atoms("[NH4+]")[2], 1)

    def test_cache_roundtrip(self):
        spec, calls = _fake_rep()
        with tempfile.TemporaryDirectory() as tmp, mock.patch.dict(REPS, {"fake": spec}):
            first = embed("fake", ["CCO", "CC", "BAD", "CCO"], cache_dir=tmp)
            self.assertEqual(calls["build"], 1)
            self.assertEqual((first.shape, first.dtype), ((4, 3), np.float32))
            self.assertEqual(calls["embedded"].count("CCO"), 1)   # a duplicate is embedded once...
            np.testing.assert_array_equal(first[0], first[3])      # ...and fills both rows
            self.assertTrue(np.isnan(first[2]).all())              # a failure is a NaN row...
            failures = json.loads((Path(tmp) / "fake" / "failures.json").read_text())
            self.assertIn("BAD", failures)                         # ...recorded on disk

            second = embed("fake", ["BAD", "CC", "CCO"], cache_dir=tmp)
            self.assertEqual(calls["build"], 1)                    # all hits: no model built
            self.assertEqual(calls["embedded"].count("BAD"), 1)    # a cached failure is not retried
            np.testing.assert_array_equal(second, first[[2, 1, 0]])   # same values, input order
            self.assertEqual(list(Path(tmp).rglob("*.tmp")), [])

    def test_cache_survives_interruption(self):
        spec, _ = _fake_rep(interrupt_at=150)
        smiles = ["C" * n for n in range(1, 201)]
        with tempfile.TemporaryDirectory() as tmp, mock.patch.dict(REPS, {"fake": spec}):
            with self.assertRaises(KeyboardInterrupt):
                embed("fake", smiles, cache_dir=tmp)
            with np.load(Path(tmp) / "fake" / "vectors.npz") as saved:   # no allow_pickle needed
                self.assertEqual(saved["smiles"].tolist(), smiles[:100])   # the periodic flush
                self.assertEqual(saved["X"].shape, (100, 3))
            self.assertEqual(list(Path(tmp).rglob("*.tmp")), [])

    def test_coverage(self):
        self.assertEqual(uncovered_elements("mace_off23", ["[Cs+].[O-]C(=O)C"]), {55})
        self.assertEqual(uncovered_elements("t5", ["[Cs+].[O-]C(=O)C"]), set())

        from lsab.datasets import load
        bases = load("shields").components["Base_SMILES"]
        off23, off23_calls = _fake_rep("mace_off23", REPS["mace_off23"].elements, "mace_mp0")
        mp0, _ = _fake_rep("mace_mp0", REPS["mace_mp0"].elements, None)
        strict, _ = _fake_rep("strict", frozenset({1, 6, 8}), None)
        fakes = {"mace_off23": off23, "mace_mp0": mp0, "strict": strict}
        with tempfile.TemporaryDirectory() as tmp, mock.patch.dict(REPS, fakes):
            with self.assertLogs("lsab.featurize", "WARNING") as logged:
                X, used = embed_component("mace_off23", bases, cache_dir=tmp)
            self.assertEqual(used, "mace_mp0")
            self.assertEqual(len(logged.records), 1)
            self.assertIn("55", logged.output[0])            # the warning names the elements
            self.assertEqual(off23_calls["build"], 0)         # the uncovered rep never loads
            self.assertEqual(X.shape, (len(bases), 3))
            with self.assertRaises(ValueError):               # no fallback -> raise
                embed_component("strict", bases, cache_dir=tmp)


# The d each real featurizer must produce. The cache path does not encode the
# model configuration, so this is what catches a registry entry that changed.
EXPECTED_D = {"mace_mp0": 256, "mace_off23": 256, "aimnet2": 256, "aimnet2_all": 772,
              "t5": 768, "chemberta": 768}


@unittest.skipUnless(os.environ.get("LSAB_SLOW"), "slow: set LSAB_SLOW=1 where the model weights are")
class TestRealFeaturizers(unittest.TestCase):
    def test_real_featurizers(self):
        for rep, d in EXPECTED_D.items():
            with self.subTest(rep):
                try:
                    featurizer = REPS[rep].build("cpu")
                except ImportError as error:
                    self.skipTest(f"{rep}: model stack not installed ({error})")
                X = featurizer(["CCO", "c1ccccc1"])
                self.assertEqual((X.shape, X.dtype), ((2, d), np.float32))
                self.assertTrue(np.isfinite(X).all())
                self.assertFalse(np.array_equal(X[0], X[1]))
                once = featurizer(["CCO"])
                np.testing.assert_array_equal(once, featurizer(["CCO"]))   # deterministic on CPU
                # batch padding must not move a row (the masked mean ignores pad tokens)
                np.testing.assert_allclose(once[0], X[0], rtol=1e-4, atol=1e-5)
                self.assertEqual(getattr(featurizer, "d", d), d)


if __name__ == "__main__":
    unittest.main()
