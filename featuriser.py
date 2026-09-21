"""Molecular featurisers and the ensemble aggregation they share.

Two kinds of descriptor feed the length-scale study. Topological ones read
connectivity straight off the SMILES and need no geometry; geometry-derived
ones are computed per conformer and therefore arrive as a matrix with one row
per conformer, which has to be collapsed to a single vector before it can be
handed to a GP.

`aggregate` is that collapse, and it is deliberately a function of an
already-computed descriptor matrix rather than of coordinates. Averaging
Cartesian coordinates across conformers produces a structure that does not
exist -- stretched bonds, collapsed rings -- and any descriptor taken from it
is meaningless. Featurise each conformer first, aggregate afterwards.

The mode is a setting rather than a hard-coded choice because it is one of
the axes the study varies: whether an ensemble-aware descriptor moves the
fitted length-scale is a question to be measured, not assumed.

"""

import numpy as np
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator

from conformer import ConformerEnsemble
from logging_config import get_logger
from settings import SETTINGS

logger = get_logger(__name__)

MORGAN_SETTINGS = SETTINGS["morgan"]
FEATURIZER_SETTINGS = SETTINGS["featurizer"]

# How a per-conformer descriptor matrix is collapsed to a single vector.
AGGREGATIONS = ("lowest", "boltzmann", "ensemble_stats")

# Order of the blocks ensemble_stats concatenates, for labelling columns.
STAT_NAMES = ("min", "max", "mean", "std")


def aggregate(
    features: np.ndarray,
    ensemble: ConformerEnsemble,
    mode: str = FEATURIZER_SETTINGS["aggregation"],
    temperature: float = FEATURIZER_SETTINGS["temperature_k"],
) -> np.ndarray:
    """Collapse a per-conformer descriptor matrix to one vector.

    Parameters
    ----------
    features
        Descriptors of each conformer, shape (K, D), in the same conformer
        order as the ensemble.
    ensemble
        The ensemble the descriptors were computed from; supplies the
        energies that weight them.
    mode
        One of the names in AGGREGATIONS. Defaults to the settings value.
    temperature
        Temperature in Kelvin, used by the `boltzmann` mode only.

    Returns
    -------
    vector
        Shape (D,) for `lowest` and `boltzmann`, or (4 * D,) for
        `ensemble_stats`, which concatenates the blocks named in
        STAT_NAMES.

    Raises
    ------
    ValueError
        If the mode is unknown, or the descriptor matrix and the ensemble
        disagree on how many conformers there are.

    Notes
    -----
    `lowest` reproduces the single-conformer behaviour: it takes row zero,
    which the ensemble guarantees is the lowest-energy conformer. It is
    discontinuous in chemical space, because near-identical molecules can
    swap which basin wins; the other two modes are continuous in the
    energies and do not have that failure.

    """
    if mode not in AGGREGATIONS:
        raise ValueError(
            f"unknown aggregation {mode!r}, expected one of {AGGREGATIONS}"
        )
    if features.ndim != 2:
        raise ValueError(
            f"features must be (K, D), got shape {features.shape}"
        )
    if len(features) != len(ensemble):
        raise ValueError(
            f"{len(features)} descriptor rows for "
            f"{len(ensemble)} conformers"
        )

    if mode == "lowest":
        return features[0]

    if mode == "boltzmann":
        weights = ensemble.boltzmann_weights(temperature)
        return weights @ features

    return np.concatenate(
        [
            features.min(axis=0),
            features.max(axis=0),
            features.mean(axis=0),
            features.std(axis=0),
        ]
    )


def aggregated_width(n_descriptors: int, mode: str) -> int:
    """Width of the vector `aggregate` returns for a given mode.

    Parameters
    ----------
    n_descriptors
        Number of columns in the per-conformer descriptor matrix.
    mode
        One of the names in AGGREGATIONS.

    Returns
    -------
    width
        Length of the aggregated vector.

    Raises
    ------
    ValueError
        If the mode is unknown.

    """
    if mode not in AGGREGATIONS:
        raise ValueError(
            f"unknown aggregation {mode!r}, expected one of {AGGREGATIONS}"
        )
    if mode == "ensemble_stats":
        return n_descriptors * len(STAT_NAMES)
    return n_descriptors


class MorganFeaturizer:
    """Morgan (ECFP) fingerprints of a list of molecules.

    A topological baseline featuriser: it reads connectivity straight off the
    SMILES and needs no 3D structure, so it costs nothing next to the
    conformer search and gives the length-scale study something to compare
    geometry-derived descriptors against.

    Parameters
    ----------
    radius
        Number of bonds the circular substructures are grown over; a radius
        of 2 corresponds to ECFP4.
    n_bits
        Length of the folded bit vector, and so the width of the feature
        matrix.

    Attributes
    ----------
    name
        Short identifier used to label the featuriser in results.

    Notes
    -----
    Both defaults are read from the `[morgan]` table of the settings file.
    Being topological, this featuriser has no conformer axis and so never
    goes through `aggregate`.

    """

    name = "morgan"

    def __init__(
        self,
        radius: int = MORGAN_SETTINGS["radius"],
        n_bits: int = MORGAN_SETTINGS["n_bits"],
    ) -> None:
        self.n_bits = n_bits
        self.generator = rdFingerprintGenerator.GetMorganGenerator(
            radius=radius, fpSize=n_bits
        )

    def __call__(self, smiles: list[str]) -> np.ndarray:
        """Featurise a list of molecules.

        Parameters
        ----------
        smiles
            SMILES strings of the molecules to featurise.

        Returns
        -------
        features
            Fingerprint matrix of shape (len(smiles), n_bits), dtype
            float32, with one row per input molecule.

        Raises
        ------
        ValueError
            If RDKit cannot parse one of the SMILES strings.

        """
        # this batch generating check how other featurizer work
        features = np.zeros((len(smiles), self.n_bits), dtype=np.float32)
        for row, s in enumerate(smiles):
            mol = Chem.MolFromSmiles(s)
            if mol is None:
                raise ValueError(f"RDKit could not parse SMILES: {s!r}")
            features[row] = self.generator.GetFingerprintAsNumPy(mol)
        return features
