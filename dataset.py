"""Registry of the reaction datasets used for benchmarking.

Each dataset is described by a DatasetSpec rather than by loading code, so
that adding a benchmark means adding one entry here: the spec records where
the file sits, which column holds the objective, whether that objective is to
be maximised or minimised, and which columns carry the SMILES and the numeric
reaction settings.

The paths are relative to the data root set in the settings file, so a
checkout can keep its datasets wherever suits it.

"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass
class DatasetSpec:
    """Where a benchmark dataset lives and how its columns are read.

    Attributes
    ----------
    name
        Unique identifier for the dataset, and the key it is registered
        under.
    path
        Location of the data file, relative to the data root.
    target
        Source column holding the objective value.
    direction
        Whether the objective is to be maximised or minimised.
    compounds
        Columns holding the SMILES of the reaction components.
    numeric
        Columns holding numeric reaction settings, such as temperature or
        concentration. Empty for datasets whose conditions are fixed.
    element_filter
        Atomic numbers the candidates are restricted to; a candidate whose
        SMILES uses any element outside the set is dropped. If None, no
        candidate is filtered out.

    Notes
    -----
    This describes the dataset, not its contents: nothing is read from disk
    until a loader is handed the spec.

    """

    name: str  # unique identifier for the dataset
    path: str  # relative to DATA_ROOT
    target: str  # source column for the objective value
    direction: Literal["max", "min"]  # goal for the objective value
    compounds: tuple[str, ...]  # SMILES columns
    numeric: tuple[
        str, ...
    ] = ()  # numeric setting columns (e.g. temperature, concentration)
    element_filter: frozenset[int] | None = (
        None  # keep only candidates whose SMILES use these elements
    )


# Every dataset the benchmarks know how to read.
SPECS = [
    DatasetSpec(
        name="bh_full",
        path="data/Buchwald_Hartwig.csv",
        target="yield",
        direction="max",
        compounds=("Ligand", "Additive", "Base", "Aryl halide"),
    ),
    DatasetSpec(
        name="bh_1",
        path="data/bh_reaction_1.csv",
        target="objective",
        direction="max",
        compounds=("Ligand", "Additive", "Base", "Aryl halide"),
    ),
    DatasetSpec(
        name="shields",
        path="data/shields_dataset.csv",
        target="yield",
        direction="max",
        compounds=("Solvent_SMILES", "Base_SMILES", "Ligand_SMILES"),
        numeric=("Temp_C", "Concentration"),
    ),
]

# The registry the rest of the code looks datasets up in, keyed by spec name.
DATASETS: dict[str, DatasetSpec] = {spec.name: spec for spec in SPECS}

# Datasets a sweep runs over when it is not told which to use.
DEFAULT_DATASETS = ("bh_full", "bh_1", "shields")
