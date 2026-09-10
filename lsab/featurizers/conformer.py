"""SMILES -> ONE 3-D conformer, the recipe every 3-D representation shares.

Factored as "make the mol" + "read atoms off the mol" so a caller that also
needs the RDKit graph can get geometry and graph from the SAME AddHs mol.
"""
from __future__ import annotations

import numpy as np


def embed_single_conformer(smiles: str, seed: int = 42, optimize: bool = True):
    """RDKit mol with explicit hydrogens and one ETKDGv3 conformer (seeded);
    one retry with random initial coordinates for hard cases; MMFF relaxation
    capped at 500 iterations, failures swallowed (the geometry only feeds a
    feature extractor, not an energy)."""
    from rdkit import Chem
    from rdkit.Chem import AllChem

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"RDKit could not parse SMILES: {smiles!r}")
    mol = Chem.AddHs(mol)
    params = AllChem.ETKDGv3()
    params.randomSeed = seed
    if AllChem.EmbedMolecule(mol, params) != 0:
        params.useRandomCoords = True
        if AllChem.EmbedMolecule(mol, params) != 0:
            raise RuntimeError(f"ETKDG failed to embed a conformer for {smiles!r}")
    if optimize:
        try:
            AllChem.MMFFOptimizeMolecule(mol, maxIters=500)
        except Exception:
            pass
    return mol


def mol_to_atoms(mol):
    """(numbers int64 (N,), coords float32 (N, 3) Angstrom, net formal charge)."""
    from rdkit import Chem

    coords = mol.GetConformer().GetPositions().astype(np.float32)
    numbers = np.array([a.GetAtomicNum() for a in mol.GetAtoms()], dtype=np.int64)
    return numbers, coords, Chem.GetFormalCharge(mol)


def smiles_to_atoms(smiles: str, seed: int = 42, optimize: bool = True):
    return mol_to_atoms(embed_single_conformer(smiles, seed=seed, optimize=optimize))
