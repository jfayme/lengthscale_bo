"""
lsab/datasets.py
================

Module 1 of the lengthscale-A/B rewrite: dataset name -> `Pool`, i.e. the MEASURED
candidates, each component's SMILES (or numeric settings), and a maximise-oriented
objective. No BO, no features, no torch. The product space is never built: pool-based
BO can only recommend a candidate whose outcome it can look up.

Every dataset quirk is a `DatasetSpec` field in ONE registry, with a WHY comment where
it is not obvious. That replaces the old `if ds == "shields"` branches spread over four
modules; `load` has no per-name branches.

Row rules, applied to the source file in this order:
  1. the target is coerced to numeric and a "min" target negated, so `objective` is
     always maximise;
  2. rows missing the target, a component or a numeric setting are dropped -- an
     unmeasured candidate, or a component with no molecule to embed (e.g. the
     ligand-free and reagent-free arms of suzuki_miyaura, as the old code did);
  3. a repeated candidate keeps its FIRST measurement (no replicate averaging,
     matching the old loader's `drop_duplicates`);
  4. `element_filter`, when set, keeps rows whose SMILES use only those elements.
Candidate order is source-file order after these rules: deterministic, but nothing
downstream may rely on it.

    python -m lsab.datasets      # one line per dataset
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

DATA_ROOT = Path(__file__).resolve().parents[1]   # repo root: holds gollum/data/ and HSF-ChemBO-tutorial/

# AIMNet2 ∩ MACE-OFF23 = {H,C,N,O,F,P,S,Cl,Br,I}, the elements every 3D representation covers.
COMMON_ELEMENTS = frozenset({1, 6, 7, 8, 9, 15, 16, 17, 35, 53})


# =============================================================================
# 1. THE CONTRACT
# =============================================================================
@dataclass(frozen=True)
class DatasetSpec:
    name: str
    family: str                        # the chemistry, not the file: runs within a family are not independent
    path: str                          # relative to DATA_ROOT
    target: str                        # source column
    direction: Literal["max", "min"]   # source direction; "min" is negated on load
    components: tuple[str, ...]        # SMILES columns; a molecule dataset has exactly one
    numeric: tuple[str, ...] = ()      # numeric setting columns (Shields: Temp_C, Concentration)
    element_filter: frozenset[int] | None = None   # keep only candidates whose SMILES use these elements


@dataclass(frozen=True, eq=False)      # eq=False: a DataFrame has no truth value, so pools compare by identity
class Pool:
    spec: DatasetSpec
    frame: pd.DataFrame                # one row per MEASURED candidate: components + numeric + ["objective"]
    components: dict[str, list[str]]   # component -> unique SMILES, first-seen order
    numeric: dict[str, list[float]]    # numeric column -> sorted unique values

    @property
    def name(self) -> str:
        return self.spec.name

    @property
    def n(self) -> int:
        return len(self.frame)

    @property
    def objective(self) -> np.ndarray:
        """float64, maximise. A copy, so callers cannot edit the pool through it."""
        return self.frame["objective"].to_numpy(dtype=np.float64, copy=True)


# =============================================================================
# 2. THE REGISTRY  (families ported verbatim from the old lengthscale_ab.FAMILY)
# =============================================================================
_BH_SUBSCREEN = ("ligand", "additive", "base", "aryl halide")

_SPECS = [
    # --- Buchwald-Hartwig: the complete Ahneman/Doyle grid, and five sub-screens of it
    DatasetSpec("bh_full", "buchwald_hartwig", "gollum/data/reasoning/Buchwald_Hartwig.csv",
                "yield", "max", ("Ligand", "Additive", "Base", "Aryl halide")),
    *[DatasetSpec(f"bh_reaction_{i}", "buchwald_hartwig",
                  f"gollum/data/buchwald-hartwig/bh_reaction_{i}.csv", "objective", "max",
                  _BH_SUBSCREEN) for i in range(1, 6)],
    # --- additive screening plates (the objective is a raw assay signal, not a %)
    *[DatasetSpec(f"additives_plate_{i}", "additives",
                  f"gollum/data/additives/additive_rxn_screening_plate_{i}.csv", "objective",
                  "max", ("additives",)) for i in range(1, 5)],
    # --- Suzuki-Miyaura. catalyst_smiles is left out: it is constant across the file,
    #     so it carries no information (the old loader dropped it at run time).
    DatasetSpec("suzuki_miyaura", "suzuki", "gollum/data/suzuki-miyaura/suzuki_miyaura_data.csv",
                "objective", "max", ("reactant_1_smiles", "reactant_2_smiles", "ligand_smiles",
                                     "reagent_1_smiles", "solvent_1_smiles")),
    DatasetSpec("suzuki_perera", "suzuki", "gollum/data/reasoning/suzuki.csv", "yield", "max",
                ("Electrophile_SMILES", "Nucleophile_SMILES", "Ligand_SMILES", "Base_SMILES",
                 "Solvent_SMILES")),
    # --- Shields direct arylation. Not element-filtered: its Cs/K bases stay in the pool.
    #     Temperature and concentration are raw settings; scaling them is featurisation.
    DatasetSpec("shields", "direct_arylation", "HSF-ChemBO-tutorial/shields_dataset.xlsx",
                "yield", "max", ("Solvent_SMILES", "Base_SMILES", "Ligand_SMILES"),
                numeric=("Temp_C", "Concentration")),
    # --- CPA thiol/imine. The source column is called "yield" but runs -0.42..3.13: not a %.
    DatasetSpec("cpa_thiol_imine", "cpa", "gollum/data/reasoning/CPA.csv", "yield", "max",
                ("Catalyst", "Imine", "Thiol")),
    # --- single-molecule property pools, filtered to COMMON_ELEMENTS so every
    #     representation embeds the identical pool (a property of the pool, not of a rep)
    DatasetSpec("photoswitches", "photoswitches", "gollum/data/molecules/photoswitches.csv.gz",
                "Pi-Pi* Transition Wavelength", "max", ("SMILES",), element_filter=COMMON_ELEMENTS),
    DatasetSpec("redox_mer", "redox", "gollum/data/molecules/redox_mer_with_iupac.csv.gz",
                "Ered", "min", ("SMILES",), element_filter=COMMON_ELEMENTS),
    DatasetSpec("pce10k", "photovoltaics", "gollum/data/molecules/photovoltaics_pce10k.csv.gz",
                "pce", "max", ("SMILES",), element_filter=COMMON_ELEMENTS),
    DatasetSpec("enamine10k", "enamine", "gollum/data/molecules/enamine10k.csv.gz",
                "score", "min", ("SMILES",), element_filter=COMMON_ELEMENTS),
]
DATASETS: dict[str, DatasetSpec] = {spec.name: spec for spec in _SPECS}

# The sweep's default matrix; bh_full + bh_reaction_1 are in every run by standing preference.
DEFAULT_DATASETS = ("bh_full", "bh_reaction_1", "cpa_thiol_imine", "shields",
                    "photoswitches", "redox_mer")


def names() -> list[str]:
    return list(DATASETS)


# =============================================================================
# 3. LOADING
# =============================================================================
# Index == atomic number ("*", the SMILES wildcard, is 0).
_ATOMIC_NUMBER = {symbol: z for z, symbol in enumerate((
    "* H He Li Be B C N O F Ne Na Mg Al Si P S Cl Ar K Ca Sc Ti V Cr Mn Fe Co Ni Cu Zn Ga Ge "
    "As Se Br Kr Rb Sr Y Zr Nb Mo Tc Ru Rh Pd Ag Cd In Sn Sb Te I Xe Cs Ba La Ce Pr Nd Pm Sm "
    "Eu Gd Tb Dy Ho Er Tm Yb Lu Hf Ta W Re Os Ir Pt Au Hg Tl Pb Bi Po At Rn Fr Ra Ac Th Pa U "
    "Np Pu Am Cm Bk Cf Es Fm Md No Lr Rf Db Sg Bh Hs Mt Ds Rg Cn Nh Fl Mc Lv Ts Og").split())}
# Group 1: a bracket atom -- any element after an optional isotope, consuming the rest of
# the bracket so "@SP1" or "H" inside it never reads as an atom. Group 2: the organic subset,
# the only atoms legal outside brackets, which stops the "Sc" in "CSc1ccccc1" reading as Sc.
_ATOM = re.compile(r"\[\d*([A-Z][a-z]?|se|as|te|[bcnops*])[^\]]*\]|(Cl|Br|[BCNOPSFI]|[bcnops*])")


def _heavy_elements(smiles: str) -> set[int]:
    """Atomic numbers of the heavy atoms written in `smiles` (unknown symbol -> -1).

    A regex, not RDKit, keeps `load` dependency-free and fast; the tests check it keeps
    and drops exactly what RDKit would. Hydrogen is skipped: most SMILES leave it
    implicit, so it cannot be filtered consistently.
    """
    return {_ATOMIC_NUMBER.get(symbol.capitalize(), -1)
            for match in _ATOM.findall(smiles) for symbol in match if symbol} - {1}


def _read(path: Path) -> pd.DataFrame:
    return pd.read_excel(path) if path.suffix == ".xlsx" else pd.read_csv(path)


def load(name: str, root: Path = DATA_ROOT) -> Pool:
    """The measured pool for `name`, built by the row rules in the module docstring."""
    spec = DATASETS[name]
    source = _read(Path(root) / spec.path)
    sign = 1.0 if spec.direction == "max" else -1.0
    key = list(spec.components + spec.numeric)
    frame = (source[key]
             .assign(objective=sign * pd.to_numeric(source[spec.target], errors="coerce"))
             .dropna()
             .drop_duplicates(subset=key)
             .astype({column: float for column in spec.numeric}))
    if spec.element_filter is not None:
        allowed = frame[list(spec.components)].map(
            lambda smiles: _heavy_elements(smiles) <= spec.element_filter)
        frame = frame[allowed.all(axis=1)]
    frame = frame.reset_index(drop=True)
    return Pool(spec=spec, frame=frame,
                components={c: frame[c].unique().tolist() for c in spec.components},
                numeric={c: sorted(frame[c].unique().tolist()) for c in spec.numeric})


def validate(pool: Pool) -> None:
    """Raise ValueError on a non-finite objective, a duplicate candidate row, an
    unparsable (or empty) SMILES, or an empty component or pool."""
    from rdkit import Chem
    from rdkit.rdBase import BlockLogs

    problems = []
    if pool.n == 0:
        problems.append("no measured candidates")
    if not np.isfinite(pool.objective).all():
        problems.append("non-finite objective")
    n_duplicates = int(pool.frame.duplicated(list(pool.spec.components + pool.spec.numeric)).sum())
    if n_duplicates:
        problems.append(f"{n_duplicates} duplicate candidate rows")
    with BlockLogs():   # failures are collected below; RDKit's stderr would only repeat them
        for column, smiles in pool.components.items():
            if not smiles:
                problems.append(f"component {column!r} is empty")
            mols = [Chem.MolFromSmiles(s) if isinstance(s, str) else None for s in smiles]
            bad = [s for s, mol in zip(smiles, mols) if mol is None or mol.GetNumAtoms() == 0]
            if bad:
                problems.append(f"{column!r}: {len(bad)} unparsable SMILES, e.g. {bad[:3]}")
    if problems:
        raise ValueError(f"{pool.name}: " + "; ".join(problems))


# =============================================================================
# 4. CLI
# =============================================================================
if __name__ == "__main__":
    for dataset in names():
        pool = load(dataset)
        sizes = " ".join(f"{c}:{len(v)}" for c, v in pool.components.items())
        numeric = " ".join(f"{c}:{len(v)}" for c, v in pool.numeric.items()) or "-"
        print(f"{dataset:18s} {pool.spec.family:17s} n={pool.n:<6d} {sizes}  "
              f"numeric={numeric}  {pool.spec.direction}")
