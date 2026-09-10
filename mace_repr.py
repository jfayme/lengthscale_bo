"""
mace_repr.py
============

Invariant molecular embeddings from MACE foundation models (MACE-OFF23 for
organic molecules, MACE-MP-0 universal), following the same SMILES -> ETKDG
conformer -> per-atom hidden state -> mean-pool recipe as ``aimnet2_embedding``.

The equivariance question (and the clean answer)
------------------------------------------------
MACE node features are SO(3) irreducible representations: scalars (l=0, "0e"),
vectors (l=1, "1o"), tensors (l=2, "2e"), ...  Only l=0 channels are rotation
invariant, so only those are safe to feed a GP kernel; l>=1 channels rotate with
the molecule and would make the kernel orientation dependent.

Two ways to get invariants:
  (a) take the l=0 ("0e") channels directly, or
  (b) take norms of the l>=1 blocks.

Reading the actual MACE code, (a) is clearly the cleanest and is loss-less for
the invariant content:

  * Every MACE layer keeps a block of l=0 scalar channels; MACE's own
    ``mace.modules.utils.extract_invariant`` slices exactly those out of the
    concatenated ``node_feats``.
  * The **last** interaction/product layer is built with
    ``hidden_irreps_out = hidden_irreps[0]`` (the ``Nx0e`` scalar block only) --
    i.e. the final per-atom layer is *already* purely invariant by construction
    (see ``mace/modules/models.py`` and ``MACE.__init__``).
  * ``MACECalculator.get_descriptors(invariants_only=True)`` packages this:
    it runs the model, reads ``output["node_feats"]`` and calls
    ``extract_invariant`` -> per-atom invariant descriptor
    ``(n_atoms, num_layers * num_scalar_features)``.

So we do NOT touch model internals and do NOT compute norms of higher-l
irreps -- we use the official ``get_descriptors`` and mean-pool over atoms.
``num_layers=-1`` (default) concatenates every layer's scalars (the standard
MACE descriptor); pass ``num_layers=1``-style slicing via ``last_layer_only``
to keep only the final scalar layer if a smaller vector is wanted.

Dependencies:  ``mace-torch`` (-> torch, e3nn), ``ase``, ``rdkit``, ``numpy``.
"""

from __future__ import annotations

import os
from functools import lru_cache

import numpy as np

# torch >= 2.6 defaults torch.load(weights_only=True), which rejects the
# MACE foundation-model checkpoints (they pickle a `slice` and e3nn objects).
# MACE's own loader honors this env var; set it BEFORE importing mace so the
# trusted, locally-cached foundation models load. (Equivalent scoped fix:
# torch.serialization.add_safe_globals([slice, ...]).)
os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
# Avoid the Windows libiomp5md.dll double-init abort seen with conda+pip torch.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from aimnet2_repr import smiles_to_atoms  # reuse the RDKit ETKDG featurizer

# ---------------------------------------------------------------------------
# Element coverage (atomic numbers).  Used to flag out-of-scope molecules.
# ---------------------------------------------------------------------------
# AIMNet2 wb97m-d3:  H B C N O F Si P S Cl As Se Br I  (14 elements)
AIMNET2_ELEMENTS = {1, 5, 6, 7, 8, 9, 14, 15, 16, 17, 33, 34, 35, 53}
# MACE-OFF23:  H C N O F P S Cl Br I  (10 organic elements)
MACEOFF_ELEMENTS = {1, 6, 7, 8, 9, 15, 16, 17, 35, 53}
# MACE-MP-0:  Z = 1..89 (Materials Project, 89 elements)
MACEMP_ELEMENTS = set(range(1, 90))

_SYMBOL = {
    1: "H", 5: "B", 6: "C", 7: "N", 8: "O", 9: "F", 14: "Si", 15: "P",
    16: "S", 17: "Cl", 33: "As", 34: "Se", 35: "Br", 53: "I",
}


def _check_coverage(numbers, allowed, model_label, *, also_flag_vs_aimnet2=False):
    zs = set(int(z) for z in numbers)
    missing = sorted(zs - allowed)
    if missing:
        syms = [_SYMBOL.get(z, f"Z{z}") for z in missing]
        raise ValueError(
            f"{model_label} does not cover element(s) {syms} (Z={missing})."
        )
    if also_flag_vs_aimnet2:
        outside = sorted(zs - AIMNET2_ELEMENTS)
        if outside:
            syms = [_SYMBOL.get(z, f"Z{z}") for z in outside]
            import warnings
            warnings.warn(
                f"Molecule contains element(s) {syms} (Z={outside}) OUTSIDE the "
                f"AIMNet2 element set -- AIMNet2 cannot embed this molecule, only "
                f"MACE-MP-0 can. Do not compare these rows against AIMNet2.",
                stacklevel=2,
            )


# ---------------------------------------------------------------------------
# MACE calculator loading (cached) + descriptor extraction
# ---------------------------------------------------------------------------


@lru_cache(maxsize=4)
def _load_mace_off(model_size: str, device: str, dtype: str):
    from mace.calculators import mace_off
    return mace_off(model=model_size, device=device, default_dtype=dtype)


@lru_cache(maxsize=4)
def _load_mace_mp(model_size: str, device: str, dtype: str):
    from mace.calculators import mace_mp
    return mace_mp(model=model_size, device=device, default_dtype=dtype)


@lru_cache(maxsize=2)
def _load_mace_mh(head: str, device: str, dtype: str):
    """MACE-MH-1 multi-head foundation model (1024-d invariant descriptor).

    Needs an explicit head; the descriptor (node_feats) is head-independent, so
    head only affects the energy. spice_wB97M is the organic wB97M head.
    """
    import os
    from mace.calculators import mace_mp
    path = os.path.expanduser("~/.cache/mace/mace-mh-1.model")
    return mace_mp(model=path, device=device, default_dtype=dtype, head=head)


def _mace_per_atom_descriptor(numbers, coords, calc, *, last_layer_only=False):
    """Run a MACE calculator and return the (n_atoms, D) invariant descriptors."""
    from ase import Atoms

    atoms = Atoms(numbers=numbers, positions=coords)
    # (n_atoms, num_layers * num_scalar_features), invariant (l=0) channels only.
    desc = calc.get_descriptors(atoms, invariants_only=True, num_layers=-1)
    if last_layer_only:
        # Keep only the final layer's scalar block (it is purely invariant).
        n_layers = int(calc.models[0].num_interactions)
        per_layer = desc.shape[1] // n_layers
        desc = desc[:, -per_layer:]
    return np.asarray(desc, dtype=np.float64)


def _mace_pooled_descriptor(numbers, coords, calc, *, last_layer_only=False):
    """Run a MACE calculator and mean-pool per-atom invariant descriptors."""
    return _mace_per_atom_descriptor(
        numbers, coords, calc, last_layer_only=last_layer_only).mean(axis=0)


# ---------------------------------------------------------------------------
# The two requested functions
# ---------------------------------------------------------------------------


def mace_off23_embedding(
    smiles: str,
    model_size: str = "medium",
    device: str = "cpu",
    *,
    last_layer_only: bool = False,
    seed: int = 42,
) -> np.ndarray:
    """MACE-OFF23 (organic molecules) invariant molecular embedding.

    Extracts the per-atom l=0 scalar components from MACE's interaction layers
    (via the official ``get_descriptors(invariants_only=True)``) and mean-pools
    across atoms.  Conformer from RDKit ETKDG.

    Coverage: H, C, N, O, F, P, S, Cl, Br, I (10 elements); raises on others.
    Returns float32 ndarray, shape ``(num_layers * num_scalar_features,)``.
    """
    numbers, coords, _charge = smiles_to_atoms(smiles, seed=seed)
    _check_coverage(numbers, MACEOFF_ELEMENTS, "MACE-OFF23")
    calc = _load_mace_off(model_size, device, "float64")
    return _mace_pooled_descriptor(
        numbers, coords, calc, last_layer_only=last_layer_only
    ).astype(np.float32)


def mace_mp0_embedding(
    smiles: str,
    model_size: str = "medium",
    device: str = "cpu",
    *,
    last_layer_only: bool = False,
    seed: int = 42,
) -> np.ndarray:
    """MACE-MP-0 (universal, 89 elements) invariant molecular embedding.

    Same extraction approach as ``mace_off23_embedding``.  Because MACE-MP-0
    covers 89 elements vs AIMNet2's 14, this also warns when the molecule
    contains elements outside the AIMNet2 scope (so benchmark rows that only
    MACE-MP-0 can produce are not silently compared against AIMNet2).

    Returns float32 ndarray, shape ``(num_layers * num_scalar_features,)``.
    """
    numbers, coords, _charge = smiles_to_atoms(smiles, seed=seed)
    _check_coverage(numbers, MACEMP_ELEMENTS, "MACE-MP-0",
                    also_flag_vs_aimnet2=True)
    calc = _load_mace_mp(model_size, device, "float32")
    return _mace_pooled_descriptor(
        numbers, coords, calc, last_layer_only=last_layer_only
    ).astype(np.float32)


# ---------------------------------------------------------------------------
# CheMeleon-style batch fingerprinters (drop into PretrainedWrapper / custom_*)
# ---------------------------------------------------------------------------


class MACEOFF23Fingerprint:
    """Batch interface mirroring ``CheMeleonFingerprint`` for MACE-OFF23."""

    def __init__(self, model_size: str = "medium", device: str = "cpu",
                 last_layer_only: bool = False, seed: int = 42,
                 pooling: str = "mean", **rff_cfg):
        self.calc = _load_mace_off(model_size, device, "float64")
        self.last_layer_only = last_layer_only
        self.seed = seed
        self.pooling = pooling
        if pooling == "rff":
            from pooling import RFFBatchPooler
            self._rff = RFFBatchPooler(**rff_cfg)
        elif pooling != "mean":
            raise ValueError(f"pooling must be 'mean' or 'rff', got {pooling!r}")

    def _per_atom(self, smiles):
        numbers, coords, _ = smiles_to_atoms(smiles, seed=self.seed)
        _check_coverage(numbers, MACEOFF_ELEMENTS, "MACE-OFF23")
        return _mace_per_atom_descriptor(numbers, coords, self.calc,
                                         last_layer_only=self.last_layer_only)

    def __call__(self, molecules) -> np.ndarray:
        if isinstance(molecules, str):
            molecules = [molecules]
        if self.pooling == "rff":
            return self._rff(molecules, self._per_atom)
        return np.stack(
            [mace_off23_embedding(m, last_layer_only=self.last_layer_only,
                                  seed=self.seed) for m in molecules],
            axis=0,
        )


class MACEMP0Fingerprint:
    """Batch interface mirroring ``CheMeleonFingerprint`` for MACE-MP-0."""

    def __init__(self, model_size: str = "medium", device: str = "cpu",
                 last_layer_only: bool = False, seed: int = 42,
                 pooling: str = "mean", **rff_cfg):
        self.calc = _load_mace_mp(model_size, device, "float32")
        self.last_layer_only = last_layer_only
        self.seed = seed
        self.pooling = pooling
        if pooling == "rff":
            from pooling import RFFBatchPooler
            self._rff = RFFBatchPooler(**rff_cfg)
        elif pooling != "mean":
            raise ValueError(f"pooling must be 'mean' or 'rff', got {pooling!r}")

    def _per_atom(self, smiles):
        numbers, coords, _ = smiles_to_atoms(smiles, seed=self.seed)
        _check_coverage(numbers, MACEMP_ELEMENTS, "MACE-MP-0",
                        also_flag_vs_aimnet2=True)
        return _mace_per_atom_descriptor(numbers, coords, self.calc,
                                         last_layer_only=self.last_layer_only)

    def __call__(self, molecules) -> np.ndarray:
        if isinstance(molecules, str):
            molecules = [molecules]
        if self.pooling == "rff":
            return self._rff(molecules, self._per_atom)
        return np.stack(
            [mace_mp0_embedding(m, last_layer_only=self.last_layer_only,
                                seed=self.seed) for m in molecules],
            axis=0,
        )


def mace_mh1_embedding(smiles: str, head: str = "spice_wB97M", device: str = "cpu",
                       *, last_layer_only: bool = False, seed: int = 42) -> np.ndarray:
    """MACE-MH-1 (multi-head foundation model) invariant molecular embedding.

    1024-d per-atom invariant descriptor, mean-pooled. Covers a broad element
    set (incl. Cs/K), so no fallback is needed for the reaction datasets.
    """
    numbers, coords, _ = smiles_to_atoms(smiles, seed=seed)
    calc = _load_mace_mh(head, device, "float64")
    return _mace_pooled_descriptor(
        numbers, coords, calc, last_layer_only=last_layer_only).astype(np.float32)


class MACEMH1Fingerprint:
    """Batch interface for MACE-MH-1 (head-selectable multi-head model)."""

    def __init__(self, head: str = "spice_wB97M", device: str = "cpu",
                 last_layer_only: bool = False, seed: int = 42,
                 pooling: str = "mean", **rff_cfg):
        self.calc = _load_mace_mh(head, device, "float64")
        self.head = head
        self.last_layer_only = last_layer_only
        self.seed = seed
        self.pooling = pooling
        if pooling == "rff":
            from pooling import RFFBatchPooler
            self._rff = RFFBatchPooler(**rff_cfg)
        elif pooling != "mean":
            raise ValueError(f"pooling must be 'mean' or 'rff', got {pooling!r}")

    def _per_atom(self, smiles):
        numbers, coords, _ = smiles_to_atoms(smiles, seed=self.seed)
        return _mace_per_atom_descriptor(numbers, coords, self.calc,
                                         last_layer_only=self.last_layer_only)

    def __call__(self, molecules) -> np.ndarray:
        if isinstance(molecules, str):
            molecules = [molecules]
        if self.pooling == "rff":
            return self._rff(molecules, self._per_atom)
        return np.stack(
            [mace_mh1_embedding(m, head=self.head, last_layer_only=self.last_layer_only,
                                seed=self.seed) for m in molecules], axis=0)


if __name__ == "__main__":
    for name, fn in [("MACE-OFF23", mace_off23_embedding),
                     ("MACE-MP-0", mace_mp0_embedding)]:
        v = fn("CCO")
        print(f"{name} ethanol embedding: shape={v.shape} dtype={v.dtype} "
              f"first4={np.round(v[:4], 4)}")
