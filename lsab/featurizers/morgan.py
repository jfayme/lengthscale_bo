"""
lsab/featurizers/morgan.py
==========================

ECFP4 / Morgan bit vector (radius 2, folded to 2048 bits): the standard
cheminformatics BO baseline. Milliseconds per molecule, so `lsab.featurize`
never caches it. Deterministic on every device.

Unlike the old `MorganFingerprint`, an unparsable SMILES RAISES instead of
returning an all-zero row: a zero vector is a real point in the feature space,
and the featurizer protocol reserves failure for the cache layer to record.
"""
from __future__ import annotations

import numpy as np
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator


class MorganFeaturizer:
    name = "morgan"

    def __init__(self, radius: int = 2, n_bits: int = 2048):
        self.n_bits = n_bits
        self._generator = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits)

    def __call__(self, smiles: list[str]) -> np.ndarray:
        X = np.zeros((len(smiles), self.n_bits), dtype=np.float32)
        for row, s in enumerate(smiles):
            mol = Chem.MolFromSmiles(s)
            if mol is None:
                raise ValueError(f"RDKit could not parse SMILES: {s!r}")
            X[row] = self._generator.GetFingerprintAsNumPy(mol)
        return X
