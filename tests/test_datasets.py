"""
tests/test_datasets.py
======================

Acceptance tests for module 1, `lsab/datasets.py` (stdlib `unittest`, ~20 s):

    python -m unittest tests.test_datasets -v

`test_matches_old_snapshot` compares against `tests/old_pool_snapshot.json`, written
once from the OLD tree (now deleted; `tools/snapshot_pools.py` is in git history, and
the snapshot cannot be regenerated). It is a sanity net for a loader
that silently drops rows or flips a sign, not a bit-identity gate. The other tests
read the source files themselves, so the loader is never trusted to check itself.
"""

import json
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from lsab.datasets import DATA_ROOT, DATASETS, load, names, validate  # noqa: E402


def _rdkit_elements(smiles):
    """Every atomic number RDKit sees in `smiles`, or None if it cannot parse it."""
    from rdkit import Chem
    mol = Chem.MolFromSmiles(smiles)
    return None if mol is None else {atom.GetAtomicNum() for atom in mol.GetAtoms()}


class TestDatasets(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.pools = {name: load(name) for name in names()}   # shared by the tests below

    def test_every_dataset_loads(self):
        for name in names():
            with self.subTest(name):
                validate(load(name))

    def test_matches_old_snapshot(self):
        snapshot = json.loads((ROOT / "tests" / "old_pool_snapshot.json").read_text())
        for name, old in snapshot.items():
            with self.subTest(name):
                pool = self.pools[name]
                self.assertEqual(pool.n, old["n"])
                for stat, reduce in (("obj_min", np.min), ("obj_max", np.max), ("obj_mean", np.mean)):
                    self.assertAlmostEqual(float(reduce(pool.objective)), old[stat], places=6)
                self.assertEqual({c: len(v) for c, v in pool.components.items()}, old["components"])

    def test_min_target_is_negated(self):
        specs = [spec for spec in DATASETS.values() if spec.direction == "min"]
        self.assertTrue(specs, "no 'min' dataset: this test would pass vacuously")
        for spec in specs:
            with self.subTest(spec.name):
                key = list(spec.components)
                source = (pd.read_csv(DATA_ROOT / spec.path)
                          .dropna(subset=key + [spec.target]).drop_duplicates(key))
                pool = self.pools[spec.name]
                matched = pool.frame[key].merge(source, on=key, how="left")[spec.target]
                np.testing.assert_array_equal(pool.objective, -matched.to_numpy(dtype=float))

    def test_element_filter(self):
        specs = [spec for spec in DATASETS.values() if spec.element_filter is not None]
        self.assertTrue(specs, "no filtered dataset: this test would pass vacuously")
        shrunk = []
        for spec in specs:
            with self.subTest(spec.name):
                (column,) = spec.components
                raw = set(pd.read_csv(DATA_ROOT / spec.path)
                          .dropna(subset=[column, spec.target])[column])
                pooled = set(self.pools[spec.name].components[column])
                self.assertTrue(all(_rdkit_elements(s) <= spec.element_filter for s in pooled))
                # both ways: the loader's regex keeps and drops exactly what RDKit would
                allowed = {s for s in raw if (elements := _rdkit_elements(s)) is not None
                           and elements <= spec.element_filter}
                self.assertEqual(pooled, allowed)
                shrunk.append(len(raw) > len(pooled))
        self.assertTrue(any(shrunk), "the filter removed nothing anywhere, so it is untested")

    def test_registry_is_consistent(self):
        for name, spec in DATASETS.items():
            with self.subTest(name):
                self.assertEqual(spec.name, name)
                self.assertTrue(spec.family)
                self.assertTrue((DATA_ROOT / spec.path).exists(), spec.path)
                self.assertTrue(spec.components)
                self.assertTrue(set(spec.numeric).isdisjoint(spec.components))
                self.assertIn(spec.direction, ("max", "min"))


if __name__ == "__main__":
    unittest.main()
