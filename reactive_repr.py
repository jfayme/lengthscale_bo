"""
reactive_repr.py
================
Reactive-site (local-atom) pooling of MACE-MP-0 per-atom invariant descriptors,
to test the hypothesis that a *reactivity-aligned* similarity (local electronic
structure at the functional atoms) makes the 3D-MLIP embedding more useful on
reaction grids than whole-molecule mean pooling.

Reactive-atom rule (uniform, role-agnostic, chemically grounded):
    reactive set = { heavy heteroatoms (non-C, non-H) }
                 ∪ { heavy atoms directly bonded to a heteroatom }
This captures the C–X carbon (halogen + ipso C), the phosphine P and its bonded
carbons, the basic N/O of bases/additives — i.e. the functional region where
reactivity lives — while excluding the inert hydrocarbon scaffold. If a molecule
has no heteroatoms, it falls back to all heavy atoms (= heavy-atom mean pooling).

Frozen: no training, same MACE-MP-0 model and same ETKDG conformer as the
whole-molecule baseline; only the set of atoms that is pooled changes.
"""
from __future__ import annotations

import numpy as np

import mace_repr as MR
from mace_repr import MACEMP_ELEMENTS, MACEOFF_ELEMENTS, _check_coverage, _mace_per_atom_descriptor


def _embed_and_mask(smiles: str, seed: int = 42, rule: str = "hetero_nbr",
                    include_h: bool = False):
    """Return (numbers, coords, mask) for one ETKDG conformer.

    mask : boolean (n_atoms,) over the AddHs atom order, True for reactive atoms.
    Built from the *same* AddHs(MolFromSmiles) mol that drives the geometry, so
    indices align with the per-atom MACE descriptor.
    """
    from rdkit import Chem
    from rdkit.Chem import AllChem

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"RDKit could not parse SMILES: {smiles!r}")
    mol = Chem.AddHs(mol)
    params = AllChem.ETKDGv3(); params.randomSeed = seed
    if AllChem.EmbedMolecule(mol, params) != 0:
        params.useRandomCoords = True
        if AllChem.EmbedMolecule(mol, params) != 0:
            raise RuntimeError(f"ETKDG failed for {smiles!r}")
    try:
        AllChem.MMFFOptimizeMolecule(mol, maxIters=500)
    except Exception:
        pass

    conf = mol.GetConformer()
    coords = conf.GetPositions().astype(np.float32)
    numbers = np.array([a.GetAtomicNum() for a in mol.GetAtoms()], dtype=np.int64)

    reactive = set()
    if rule == "hetero_nbr":
        for a in mol.GetAtoms():
            z = a.GetAtomicNum()
            if z not in (1, 6):                      # heavy heteroatom
                reactive.add(a.GetIdx())
                for nbr in a.GetNeighbors():
                    if include_h or nbr.GetAtomicNum() != 1:
                        reactive.add(nbr.GetIdx())
    elif rule == "hetero_only":
        reactive = {a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() not in (1, 6)}
    else:
        raise ValueError(f"unknown rule {rule!r}")

    mask = np.zeros(len(numbers), dtype=bool)
    if reactive:
        mask[list(reactive)] = True
    else:                                            # no heteroatoms -> heavy-atom fallback
        mask = numbers != 1
    return numbers, coords, mask


def _shell_masks(smiles, seed=42, radii=(1, 2)):
    """Return (numbers, coords, {scale: mask}) for multi-scale pooling.

    scale 'whole' = all atoms; scale f'r{R}' = heavy atoms within R bonds of any
    heteroatom (the functional region, grown in shells). Built from the same
    AddHs mol that drives the geometry, so indices align with the descriptor.
    """
    from rdkit import Chem
    from rdkit.Chem import AllChem
    import numpy as np

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"RDKit could not parse SMILES: {smiles!r}")
    mol = Chem.AddHs(mol)
    params = AllChem.ETKDGv3(); params.randomSeed = seed
    if AllChem.EmbedMolecule(mol, params) != 0:
        params.useRandomCoords = True
        if AllChem.EmbedMolecule(mol, params) != 0:
            raise RuntimeError(f"ETKDG failed for {smiles!r}")
    try:
        AllChem.MMFFOptimizeMolecule(mol, maxIters=500)
    except Exception:
        pass

    coords = mol.GetConformer().GetPositions().astype(np.float32)
    numbers = np.array([a.GetAtomicNum() for a in mol.GetAtoms()], dtype=np.int64)
    n = len(numbers)
    hetero = [a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() not in (1, 6)]
    heavy = numbers != 1

    masks = {"whole": np.ones(n, dtype=bool)}
    if hetero:
        D = Chem.GetDistanceMatrix(mol)
        dmin = D[hetero, :].min(axis=0)            # bond distance to nearest heteroatom
        for R in radii:
            m = (dmin <= R) & heavy
            masks[f"r{R}"] = m if m.any() else heavy
    else:                                          # pure hydrocarbon -> shells = heavy atoms
        for R in radii:
            masks[f"r{R}"] = heavy
    return numbers, coords, masks


class MultiScaleMACEMP0:
    """Per-molecule MACE-MP-0 descriptors pooled at multiple scales (whole + shells).

    ``pools(smiles)`` returns {scale: vector}; the per-atom descriptor is computed
    once and pooled several ways, so all scales share the same model/conformer.
    """

    def __init__(self, model_size="medium", device="cpu", seed=42,
                 last_layer_only=False, radii=(1, 2)):
        self.calc = MR._load_mace_mp(model_size, device, "float32")
        self.seed = seed
        self.last_layer_only = last_layer_only
        self.radii = radii

    def pools(self, smiles):
        import numpy as np
        numbers, coords, masks = _shell_masks(smiles, seed=self.seed, radii=self.radii)
        _check_coverage(numbers, MACEMP_ELEMENTS, "MACE-MP-0")
        per_atom = _mace_per_atom_descriptor(numbers, coords, self.calc,
                                             last_layer_only=self.last_layer_only)
        return {scale: per_atom[m].mean(axis=0).astype(np.float32)
                for scale, m in masks.items()}


class ReactiveMACEMP0Fingerprint:
    """MACE-MP-0 invariant descriptor pooled over reactive atoms only."""

    def __init__(self, model_size: str = "medium", device: str = "cpu",
                 last_layer_only: bool = False, seed: int = 42,
                 rule: str = "hetero_nbr"):
        self.calc = MR._load_mace_mp(model_size, device, "float32")
        self.last_layer_only = last_layer_only
        self.seed = seed
        self.rule = rule

    def _one(self, smiles):
        numbers, coords, mask = _embed_and_mask(smiles, seed=self.seed, rule=self.rule)
        _check_coverage(numbers, MACEMP_ELEMENTS, "MACE-MP-0")
        per_atom = _mace_per_atom_descriptor(numbers, coords, self.calc,
                                             last_layer_only=self.last_layer_only)
        return per_atom[mask].mean(axis=0).astype(np.float32)

    def __call__(self, molecules):
        if isinstance(molecules, str):
            molecules = [molecules]
        return np.stack([self._one(m) for m in molecules], axis=0)


class TwoScaleMACEMP0Fingerprint:
    """Two-scale MACE-MP-0 descriptor: [whole-molecule mean ‖ reactive-site mean].

    The per-atom invariant descriptor is computed once, then pooled at two scales
    and concatenated. PCA + the GP's ARD lengthscales then keep whichever scale is
    informative per component (e.g. local C–X electronics for the aryl halide,
    whole-molecule sterics for a bulky ligand) — the disentanglement Morgan gets
    for free, given here as an explicit local⊕global basis. Frozen: no training.
    """

    def __init__(self, model_size: str = "medium", device: str = "cpu",
                 last_layer_only: bool = False, seed: int = 42,
                 rule: str = "hetero_nbr"):
        self.calc = MR._load_mace_mp(model_size, device, "float32")
        self.last_layer_only = last_layer_only
        self.seed = seed
        self.rule = rule

    def _one(self, smiles):
        numbers, coords, mask = _embed_and_mask(smiles, seed=self.seed, rule=self.rule)
        _check_coverage(numbers, MACEMP_ELEMENTS, "MACE-MP-0")
        per_atom = _mace_per_atom_descriptor(numbers, coords, self.calc,
                                             last_layer_only=self.last_layer_only)
        whole = per_atom.mean(axis=0)             # global pool (= baseline MACE-MP-0)
        local = per_atom[mask].mean(axis=0)       # reactive-site pool
        return np.concatenate([whole, local]).astype(np.float32)

    def __call__(self, molecules):
        if isinstance(molecules, str):
            molecules = [molecules]
        return np.stack([self._one(m) for m in molecules], axis=0)


class ReactiveMACEOFF23Fingerprint:
    """MACE-OFF23 invariant descriptor pooled over reactive atoms only."""

    def __init__(self, model_size: str = "medium", device: str = "cpu",
                 last_layer_only: bool = False, seed: int = 42,
                 rule: str = "hetero_nbr"):
        self.calc = MR._load_mace_off(model_size, device, "float64")
        self.last_layer_only = last_layer_only
        self.seed = seed
        self.rule = rule

    def _one(self, smiles):
        numbers, coords, mask = _embed_and_mask(smiles, seed=self.seed, rule=self.rule)
        _check_coverage(numbers, MACEOFF_ELEMENTS, "MACE-OFF23")
        per_atom = _mace_per_atom_descriptor(numbers, coords, self.calc,
                                             last_layer_only=self.last_layer_only)
        return per_atom[mask].mean(axis=0).astype(np.float32)

    def __call__(self, molecules):
        if isinstance(molecules, str):
            molecules = [molecules]
        return np.stack([self._one(m) for m in molecules], axis=0)


if __name__ == "__main__":
    # Sanity: show which atoms the rule selects on representative BH components.
    from rdkit import Chem
    tests = {
        "aryl halide (Br-pyridine)": "Brc1cccnc1",
        "phosphine ligand (PCy3-like)": "CC(C)c1cc(C(C)C)c(-c2ccccc2P(C2CCCCC2)C2CCCCC2)c(C(C)C)c1",
        "isoxazole additive": "Cc1cc(C)on1",
        "phosphazene base": "CN(C)P(N(C)C)(N(C)C)=NP(N(C)C)(N(C)C)=NCC",
    }
    for label, smi in tests.items():
        numbers, coords, mask = _embed_and_mask(smi)
        mol = Chem.AddHs(Chem.MolFromSmiles(smi))
        picked = [mol.GetAtomWithIdx(i).GetSymbol() for i in range(len(numbers)) if mask[i]]
        from collections import Counter
        print(f"{label}: {int(mask.sum())}/{len(numbers)} atoms pooled  {dict(Counter(picked))}")
