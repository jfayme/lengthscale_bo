"""
representations.py
==================

One drop-in interface for FIVE molecular representations so they can be
benchmarked identically in a BayBE BO campaign:

    * "aimnet2"     -- AIMNet2 mean-pooled per-atom `aim` hidden state  (this work)
    * "mace_off23"  -- MACE-OFF23 mean-pooled invariant descriptors      (this work)
    * "mace_mp0"    -- MACE-MP-0 mean-pooled invariant descriptors        (this work)
    * "chemeleon"   -- CheMeleon MPNN fingerprint     (HSF-ChemBO-tutorial baseline)
    * "chemberta"   -- ChemBERTa hidden-state pooling  (HSF-ChemBO-tutorial baseline)

Every representation is exposed two ways:

    make_fingerprinter(name) -> callable(list[str|Mol]) -> np.ndarray (n, d)
        Same call signature as the tutorial's ``CheMeleonFingerprint`` so it
        plugs straight into ``base.utils.custom_fingerprinter`` /
        ``custom_PCA_fingerprinter``.

    baybe_parameter(name, param_name, smiles_dict, ...) -> CustomDiscreteParameter
        Ready to drop into a BayBE ``SearchSpace`` exactly like the tutorial's
        ``CustomDiscreteParameter(name=..., data=custom_fingerprinter(...))``.

The 3 NNP fingerprinters reuse ``aimnet2_repr`` / ``mace_repr``; the 2 baselines
reuse the tutorial's own ``base/`` code unchanged, so the comparison against the
paper is apples-to-apples.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Make the tutorial's `base` package importable for the CheMeleon/ChemBERTa
# baselines (PretrainedWrapper, CheMeleonFingerprint, ChemBERTa_Fingerprint).
_TUTORIAL_DIR = Path(__file__).resolve().parent / "HSF-ChemBO-tutorial"
if _TUTORIAL_DIR.exists() and str(_TUTORIAL_DIR) not in sys.path:
    sys.path.insert(0, str(_TUTORIAL_DIR))

REPRESENTATIONS = ("aimnet2", "mace_off23", "mace_mp0",
                   "chemeleon", "chemberta", "t5")

# Element coverage per representation (atomic numbers). None == no 3D element
# constraint (2D-graph / text models can encode any element).
COVERAGE_ELEMENTS = {
    "aimnet2": {1, 5, 6, 7, 8, 9, 14, 15, 16, 17, 33, 34, 35, 53},      # 14
    "mace_off23": {1, 6, 7, 8, 9, 15, 16, 17, 35, 53},                  # 10 organic
    "mace_mp0": set(range(1, 90)),                                       # 89
    "chemeleon": None,
    "chemberta": None,
    "t5": None,
    "mace_mh1": None,   # multi-head foundation model, broad element coverage (incl Cs)
    "aimnet2_all": {1, 5, 6, 7, 8, 9, 14, 15, 16, 17, 33, 34, 35, 53},  # = aimnet2 (all 3 passes)
    "phys": set(range(1, 90)),   # MH-1 backbone (broad); POLAR dims zero-filled when unsafe
    "morgan": None,     # 2D ECFP bit-vector -- any element, no 3D geometry
    "mordred": None,    # 2D Mordred QSAR descriptors -- any element, no geometry
    "ohe": None,        # one-hot identity baseline -- no chemistry at all
    "mace_mp0_rxn": set(range(1, 90)),                       # MACE-MP-0, reactive-site pooled
    "mace_off23_rxn": {1, 6, 7, 8, 9, 15, 16, 17, 35, 53},   # MACE-OFF23, reactive-site pooled
    "mace_mp0_2scale": set(range(1, 90)),                    # MACE-MP-0, whole-molecule ‖ reactive-site
}


def uncovered_elements(name: str, smiles_list) -> set[int]:
    """Atomic numbers in ``smiles_list`` NOT covered by representation ``name``."""
    from rdkit import Chem
    allowed = COVERAGE_ELEMENTS.get(name.lower())
    if allowed is None:
        return set()
    bad: set[int] = set()
    for smi in smiles_list:
        m = Chem.MolFromSmiles(smi)
        if m is None:
            continue
        bad |= {a.GetAtomicNum() for a in m.GetAtoms()} - allowed
    return bad


# ---------------------------------------------------------------------------
# Fingerprinter factory  (callable: list[smiles] -> (n, d) float array)
# ---------------------------------------------------------------------------


class MorganFingerprint:
    """ECFP/Morgan bit-vector fingerprinter (the standard cheminformatics BO baseline).

    Same call signature as the other fingerprinters: ``__call__(list[str]) -> (n, d)``.
    Defaults to ECFP4 (radius 2) folded to 2048 bits, matching the Shields/EDBO+ baseline.
    """

    def __init__(self, radius: int = 2, n_bits: int = 2048):
        self.radius = radius
        self.n_bits = n_bits

    def __call__(self, smiles_list):
        from rdkit import Chem
        from rdkit.Chem import AllChem
        import numpy as np
        gen = AllChem.GetMorganGenerator(radius=self.radius, fpSize=self.n_bits)
        out = np.zeros((len(smiles_list), self.n_bits), dtype=np.float32)
        for i, smi in enumerate(smiles_list):
            m = Chem.MolFromSmiles(smi)
            if m is None:
                continue
            out[i] = gen.GetFingerprintAsNumPy(m).astype(np.float32)
        return out


class MordredFingerprint:
    """Mordred 2D molecular-descriptor fingerprinter (the EDBO+/QSAR baseline).

    Computes the ~1600 2D Mordred descriptors per molecule. Non-finite values
    (failed descriptors / divide-by-zero) are imputed column-wise with the median
    over the supplied set, so the returned matrix is always finite. Heterogeneous
    descriptor scales are handled downstream by the pipeline (z-score before PCA,
    same path as the physical-descriptor representation).
    """

    def __init__(self, ignore_3D: bool = True):
        self.ignore_3D = ignore_3D
        self._calc = None

    def _calculator(self):
        if self._calc is None:
            from mordred import Calculator, descriptors
            self._calc = Calculator(descriptors, ignore_3D=self.ignore_3D)
        return self._calc

    def __call__(self, smiles_list):
        from rdkit import Chem
        import numpy as np
        mols = [Chem.MolFromSmiles(s) for s in smiles_list]
        calc = self._calculator()
        df = calc.pandas([m for m in mols], nproc=1, quiet=True)
        # Mordred returns Missing/error objects for failures -> coerce to float.
        X = df.apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
        X[~np.isfinite(X)] = np.nan
        # rows where the molecule failed to parse -> all-NaN; column-median impute.
        col_med = np.nanmedian(np.where(np.isfinite(X), X, np.nan), axis=0)
        col_med = np.where(np.isfinite(col_med), col_med, 0.0)
        inds = np.where(np.isnan(X))
        X[inds] = np.take(col_med, inds[1])
        return X.astype(np.float32)


def make_fingerprinter(name: str, **kwargs):
    """Return a CheMeleon-style fingerprinter object for representation ``name``."""
    name = name.lower()
    if name == "morgan":
        return MorganFingerprint(radius=kwargs.get("radius", 2),
                                 n_bits=kwargs.get("n_bits", 2048))
    if name == "mordred":
        return MordredFingerprint(ignore_3D=kwargs.get("ignore_3D", True))
    if name == "mace_mp0_rxn":   # reactive-site (local-atom) pooled MACE-MP-0
        from reactive_repr import ReactiveMACEMP0Fingerprint
        return ReactiveMACEMP0Fingerprint(device=kwargs.get("device", "cpu"),
                                          rule=kwargs.get("rule", "hetero_nbr"))
    if name == "mace_off23_rxn":
        from reactive_repr import ReactiveMACEOFF23Fingerprint
        return ReactiveMACEOFF23Fingerprint(device=kwargs.get("device", "cpu"),
                                            rule=kwargs.get("rule", "hetero_nbr"))
    if name == "mace_mp0_2scale":   # whole-molecule pooling concatenated with reactive-site pooling
        from reactive_repr import TwoScaleMACEMP0Fingerprint
        return TwoScaleMACEMP0Fingerprint(device=kwargs.get("device", "cpu"),
                                          rule=kwargs.get("rule", "hetero_nbr"))
    if name == "aimnet2":
        from aimnet2_repr import AIMNet2Embedder
        return AIMNet2Embedder(model_name=kwargs.get("model_name", "aimnet2"),
                               device=kwargs.get("device", "cpu"),
                               layers=kwargs.get("layers", "last"))
    if name == "aimnet2_all":   # all 3 MLP passes concatenated (layer-ablation winner)
        from aimnet2_repr import AIMNet2Embedder
        return AIMNet2Embedder(model_name=kwargs.get("model_name", "aimnet2"),
                               device=kwargs.get("device", "cpu"), layers="all")
    if name == "phys":          # MH-1/POLAR physical descriptor (stats pooling, single conf)
        from phys_descriptor import make_gollum_phys_fp
        return make_gollum_phys_fp(device=kwargs.get("device", "cpu"))
    if name == "mace_off23":
        from mace_repr import MACEOFF23Fingerprint
        return MACEOFF23Fingerprint(model_size=kwargs.get("model_size", "medium"),
                                    device=kwargs.get("device", "cpu"))
    if name == "mace_mp0":
        from mace_repr import MACEMP0Fingerprint
        return MACEMP0Fingerprint(model_size=kwargs.get("model_size", "medium"),
                                  device=kwargs.get("device", "cpu"))
    if name == "mace_mh1":
        from mace_repr import MACEMH1Fingerprint
        fk = {k: kwargs[k] for k in ("n_features", "lengthscale", "standardize") if k in kwargs}
        return MACEMH1Fingerprint(head=kwargs.get("head", "spice_wB97M"),
                                  device=kwargs.get("device", "cpu"),
                                  pooling=kwargs.get("pooling", "mean"), **fk)
    if name == "chemeleon":
        # tutorial baseline -- unchanged
        from base.pretrained_repr import PretrainedWrapper, CheMeleonFingerprint
        return PretrainedWrapper(CheMeleonFingerprint)
    if name == "chemberta":
        from base.pretrained_repr import PretrainedWrapper, ChemBERTa_Fingerprint
        return PretrainedWrapper(ChemBERTa_Fingerprint,
                                 variant=kwargs.get("variant", "zinc-base-v1"))
    if name == "t5":
        # The tutorial's featured / best-performing HSF: chemistry-T5 encoder,
        # average-pooled last hidden state (LLM_Fingerprint in base/).
        # Default model = GT4SD chemistry-T5; pass model_name="t5-base" for plain T5.
        from base.pretrained_repr import PretrainedWrapper, LLM_Fingerprint
        return PretrainedWrapper(
            LLM_Fingerprint,
            model_name=kwargs.get(
                "model_name", "GT4SD/multitask-text-and-chemistry-t5-base-augm"),
            pooling_method=kwargs.get("pooling_method", "average"),
            normalize_embeddings=kwargs.get("normalize_embeddings", False),
        )
    raise ValueError(f"Unknown representation {name!r}; choose from {REPRESENTATIONS}")


# ---------------------------------------------------------------------------
# Descriptor table  (index = molecule label, columns = feature dims)
# ---------------------------------------------------------------------------


def descriptor_frame(name: str, smiles_dict: dict[str, str],
                     fingerprinter=None, *, normalize: str | None = None,
                     drop_constant: bool = True, **kwargs) -> pd.DataFrame:
    """Build the per-molecule descriptor DataFrame for representation ``name``.

    Same layout as the tutorial's ``custom_fingerprinter`` (``base/utils.py``),
    so any representation produces a table consumable by both BayBE and EDBO+.

    NOTE on column dropping: constant columns are removed on the **raw** features
    *before* normalization. Doing it after global normalization can wrongly drop
    every low-variance column (a component whose few molecules barely differ
    relative to the global min-max collapses to 0 columns, which then crashes
    BayBE's decorrelation). ``normalize`` defaults to ``None`` to match the
    paper's ``custom_fingerprinter`` decorrelate path.
    """
    labels = list(smiles_dict.keys())

    # OHE baseline: identity one-hot, no fingerprinter (the "no chemistry" floor).
    if name == "ohe":
        eye = np.eye(len(labels), dtype=float)
        return pd.DataFrame(eye, index=labels,
                            columns=[f"ohe_{i}" for i in range(len(labels))])

    if fingerprinter is None:
        fingerprinter = make_fingerprinter(name, **kwargs)

    smiles = [smiles_dict[k] for k in labels]
    feats = np.asarray(fingerprinter(smiles), dtype=float)

    cols = [f"{name}_{i}" for i in range(feats.shape[1])]
    df = pd.DataFrame(feats, index=labels, columns=cols)

    # Drop genuinely-constant columns on the RAW features (range-based).
    if drop_constant:
        keep = (df.max() - df.min()) > 1e-8
        df = df.loc[:, keep]

    # Mordred / phys descriptors span many orders of magnitude (descriptor units;
    # eV site energies vs |q|~1) -> z-score so the GP kernel / decorrelation are
    # not dominated by a few large-scale columns. (Latent NN embeddings are
    # homogeneous and need no standardization.)
    if name in ("mordred", "phys") and df.shape[1] > 0:
        mu = df.mean(axis=0); sd = df.std(axis=0).replace(0, 1.0)
        df = (df - mu) / sd

    if normalize == "global":
        lo, hi = df.values.min(), df.values.max()
        df = (df - lo) / (hi - lo if hi > lo else 1e-12)
    elif normalize == "local":
        lo = df.min(axis=0)
        rng = (df.max(axis=0) - lo).replace(0, 1e-12)
        df = (df - lo) / rng
    return df


# ---------------------------------------------------------------------------
# BayBE CustomDiscreteParameter factory
# ---------------------------------------------------------------------------


def baybe_parameter(name: str, param_name: str, smiles_dict: dict[str, str],
                    fingerprinter=None, *, decorrelate=0.7,
                    normalize: str | None = None, **kwargs):
    """Return a BayBE ``CustomDiscreteParameter`` for representation ``name``.

    Drop-in for the tutorial pattern::

        CustomDiscreteParameter(name=param_name,
                                data=custom_fingerprinter(smiles_dict, fp))
    """
    from baybe.parameters import CustomDiscreteParameter

    df = descriptor_frame(name, smiles_dict, fingerprinter=fingerprinter,
                          normalize=normalize, **kwargs)
    return CustomDiscreteParameter(name=param_name, data=df, decorrelate=decorrelate)
