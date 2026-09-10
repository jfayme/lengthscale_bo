"""
lsab/featurizers/mace.py
========================

Mean-pooled INVARIANT (l=0) per-atom descriptors from a MACE foundation model.

MACE node features are SO(3) irreducible representations. Only the l=0 scalar
channels are rotation invariant, so only those are safe to feed a GP kernel;
l>=1 channels rotate with the molecule. MACE's own
`MACECalculator.get_descriptors(invariants_only=True)` slices exactly those
scalars out of every interaction layer (`num_layers=-1` keeps them all), so this
module never touches model internals.

Two departures from the generic featurizer rules, both forced by MACE:
  * NO `torch.no_grad()`. `get_descriptors` runs the full forward pass, which
    computes forces by autograd; under no_grad it raises. The calculator already
    freezes the weights, so nothing trains.
  * `eval()` matters here: the calculator leaves its networks in training mode,
    where MACE also builds the second-order graph it would need to train on forces.

One dtype, float32, for BOTH models. The old code ran OFF23 in float64 and MP-0
in float32 for no recorded reason; the cache is float32 anyway.

Deterministic on CPU (seeded conformer, eval mode). GPU results may differ in
the last bits between runs; that is not worth fixing for a GP feature.
"""
from __future__ import annotations

import os

# torch >= 2.6 defaults torch.load(weights_only=True), which rejects the MACE
# foundation-model checkpoints (they pickle a `slice` and e3nn objects). MACE's
# own loader honours this env var; set it BEFORE importing mace so the trusted,
# locally cached foundation models load.
os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")

from typing import Literal  # noqa: E402

import numpy as np  # noqa: E402
from ase import Atoms  # noqa: E402
from mace.calculators import mace_mp, mace_off  # noqa: E402

from lsab.featurizers.conformer import smiles_to_atoms  # noqa: E402

_LOADERS = {"mp0": mace_mp, "off23": mace_off}


class MaceFeaturizer:
    """Mean-pooled per-atom INVARIANT (l=0) descriptor from a MACE foundation model.
    model="mp0" (Materials Project, Z=1..89) or "off23" (organic, 10 elements)."""

    def __init__(self, model: Literal["mp0", "off23"], size: str = "medium",
                 device: str = "cpu", dtype: str = "float32", seed: int = 42):
        if model not in _LOADERS:
            raise ValueError(f"model must be 'mp0' or 'off23', got {model!r}")
        self.name = f"mace_{model}"
        self.seed = seed
        self.calc = _LOADERS[model](model=size, device=device, default_dtype=dtype)
        for network in self.calc.models:
            network.eval()

    def __call__(self, smiles: list[str]) -> np.ndarray:
        rows = []
        for s in smiles:
            numbers, coords, _charge = smiles_to_atoms(s, seed=self.seed)
            per_atom = self.calc.get_descriptors(Atoms(numbers=numbers, positions=coords),
                                                 invariants_only=True, num_layers=-1)
            rows.append(per_atom.mean(axis=0, dtype=np.float64))
        return np.asarray(rows, dtype=np.float32)
