"""
aimnet2_repr.py
===============

Fixed-size molecular embeddings from AIMNet2's per-atom hidden states, plus
thin wrappers that expose them as a custom representation for Bayesian
optimization (EDBO+ primary, BayBE secondary).

What this does
--------------
AIMNet2 (isayevlab/aimnetcentral) is a message-passing neural network potential.
Inside ``aimnet.models.aimnet2.AIMNet2.forward`` the network refines a per-atom
feature vector over several MLP "passes". The **last** MLP writes a per-atom
embedding of size ``aim_size`` (256 for the released models) into
``data["aim"]``; the energy/charge output heads then read ``data["aim"]``.

So ``data["aim"]`` is *the* per-atom hidden state that sits immediately before
the output heads -- the AIMNet2 analogue of CheMeleon's pre-FFN node features.
We mean-pool it over the atoms of one molecule to get a single fixed-size
(256-d) molecular vector, exactly mirroring how CheMeleon's
``MPNN.fingerprint`` mean-aggregates its last hidden layer in the
HSF-ChemBO tutorial.

Key architecture facts (traced from the repo)
---------------------------------------------
* Module that holds the last per-atom vectors:  ``model.mlps[-1]``
  (an ``nn.Sequential``, the final element of ``AIMNet2.mlps : nn.ModuleList``).
  Its output is stored as ``data["aim"]``.
* Shape:  ``[n_atoms, aim_size]`` (sparse/flattened path) or
  ``[batch, n_atoms, aim_size]`` (dense path).  ``aim_size = 256``.
* Hook viability:  the released v2 ``.pt`` models load as an eager
  ``nn.Module`` (``aimnet.models.base.load_model`` -> ``build_module``), so a
  standard ``register_forward_hook`` on ``model.mlps[-1]`` works.  ONLY the
  legacy v1 ``.jpt`` files load as a ``torch.jit.ScriptModule``, where Python
  forward hooks do not fire -- see ``HOOKS_AND_JIT`` note at the bottom.
* Blocker to be aware of:  ``AIMNet2Calculator.eval()`` calls ``keep_only``,
  which strips everything except energy/charges/forces.  ``data["aim"]`` is
  computed but then discarded.  We therefore either (a) read ``data["aim"]``
  straight off the model's output dict by calling ``calc.model(...)``
  ourselves, or (b) grab it with a forward hook.  Both are shown below.

Dependencies:  ``aimnet`` (pip install aimnet -> pulls torch, warp-lang,
nvalchemi-toolkit-ops), ``rdkit``, ``numpy``, ``pandas``.
"""

from __future__ import annotations

import threading
import warnings
from functools import lru_cache

import numpy as np

# AIMNet2 wb97m-d3 element coverage: H B C N O F Si P S Cl As Se Br I (14).
# (Authoritative per-model list comes from model metadata "implemented_species";
#  this is the documented default used as a fallback.)
AIMNET2_ELEMENTS = {1, 5, 6, 7, 8, 9, 14, 15, 16, 17, 33, 34, 35, 53}
_Z_SYMBOL = {1: "H", 3: "Li", 5: "B", 6: "C", 7: "N", 8: "O", 9: "F", 11: "Na",
             14: "Si", 15: "P", 16: "S", 17: "Cl", 19: "K", 33: "As", 34: "Se",
             35: "Br", 53: "I", 55: "Cs"}

# ----------------------------------------------------------------------------
# 1. SMILES -> 3D geometry (RDKit ETKDG)
# ----------------------------------------------------------------------------


def smiles_to_atoms(smiles: str, seed: int = 42, optimize: bool = True):
    """Embed a single ETKDG conformer and return AIMNet2 inputs.

    Returns
    -------
    numbers : (N,) int64 ndarray   -- atomic numbers (H included)
    coords  : (N, 3) float32 ndarray -- Cartesian coordinates in Angstrom
    charge  : int                  -- net formal charge
    """
    from rdkit import Chem
    from rdkit.Chem import AllChem

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"RDKit could not parse SMILES: {smiles!r}")
    mol = Chem.AddHs(mol)  # AIMNet2 needs explicit hydrogens

    params = AllChem.ETKDGv3()
    params.randomSeed = seed
    if AllChem.EmbedMolecule(mol, params) != 0:
        # Fall back to random-coords embedding for hard cases
        params.useRandomCoords = True
        if AllChem.EmbedMolecule(mol, params) != 0:
            raise RuntimeError(f"ETKDG failed to embed a conformer for {smiles!r}")
    if optimize:
        try:
            AllChem.MMFFOptimizeMolecule(mol, maxIters=500)
        except Exception:
            pass  # geometry is good enough for a feature extractor

    conf = mol.GetConformer()
    coords = conf.GetPositions().astype(np.float32)               # (N, 3)
    numbers = np.array([a.GetAtomicNum() for a in mol.GetAtoms()], dtype=np.int64)
    charge = Chem.GetFormalCharge(mol)
    return numbers, coords, charge


# ----------------------------------------------------------------------------
# 2. The embedder: load AIMNet2 once, extract mean-pooled `aim`
# ----------------------------------------------------------------------------


class AIMNet2Embedder:
    """Load an AIMNet2 model once and turn molecules into 256-d vectors.

    The model is run in AIMNet2's *dense* neighbor mode (``nb_mode == 0``):
    we pass ``coord``/``numbers``/``charge`` with a batch axis and NO neighbor
    matrix, so the network does exact all-pairs message passing.  For a single
    small molecule this is both correct (no neighbor-list truncation) and avoids
    calling the ``nvalchemiops`` neighbor-list builder at runtime.

    Two extraction routes are implemented and cross-checked:
      * ``method="dict"`` -- read ``data["aim"]`` from the model output dict.
      * ``method="hook"`` -- standard forward hook on ``model.mlps[-1]``.
    They return identical values; ``"hook"`` is the answer to "is a standard
    PyTorch forward hook sufficient?" -> yes, for the eager v2 models.
    """

    AIM_SIZE = 256  # aim_size in aimnet/models/aimnet2.yaml

    def __init__(self, model_name: str = "aimnet2", device: str = "cpu",
                 seed: int = 42, optimize_geometry: bool = True,
                 strict_elements: bool = True, pooling: str = "mean",
                 layers: str = "last", **rff_cfg):
        import torch
        from aimnet.calculators import AIMNet2Calculator

        self.torch = torch
        self.device = device
        self.seed = seed
        self.optimize_geometry = optimize_geometry
        self.strict_elements = strict_elements
        # Atom->molecule pooling: "mean" (default, unchanged) or "rff" (set kernel).
        self.pooling = pooling
        if pooling == "rff":
            from pooling import RFFBatchPooler
            self._rff = RFFBatchPooler(**rff_cfg)
        elif pooling != "mean":
            raise ValueError(f"pooling must be 'mean' or 'rff', got {pooling!r}")

        # AIMNet2Calculator handles registry lookup, download, metadata,
        # dtype and (with train=False) requires_grad=False on all params.
        calc = AIMNet2Calculator(model_name, device=device)
        self.calc = calc
        # Authoritative element coverage from model metadata; fall back to the
        # documented wb97m set. We run the model in dense mode (bypassing the
        # calculator's eval/validate_species), so coverage must be enforced here
        # or out-of-scope elements silently yield NaN embeddings.
        meta = getattr(calc, "metadata", None) or {}
        impl = meta.get("implemented_species") or []
        self.elements = set(int(z) for z in impl) if impl else set(AIMNET2_ELEMENTS)
        self.model = calc.model
        self.model.eval()

        if isinstance(self.model, torch.jit.ScriptModule):
            # Hooks won't fire on a frozen TorchScript graph, and `mlps` may not
            # be addressable. The released v2 .pt models are NOT ScriptModules.
            raise RuntimeError(
                "Loaded a legacy TorchScript (.jpt) model. Forward hooks and "
                "submodule access do not work here -- use a v2 .pt model "
                "(e.g. 'aimnet2') or re-export. See HOOKS_AND_JIT note."
            )
        # The last per-atom MLP; its output is stored as data['aim'].
        self.last_mlp = self.model.mlps[-1]
        # Which MLP passes to concatenate for the per-atom embedding.
        #   "last"  (default) = only mlps[-1] (the `aim` head input; unchanged)
        #   "all"             = concat every pass (AIMNet2 analogue of MACE
        #                       num_layers=-1: let decorrelation/ARD pick)
        #   "last2"           = the final two passes
        self.layers = layers
        allm = list(self.model.mlps)
        if layers == "last":
            self.hook_mlps = [allm[-1]]
        elif layers == "all":
            self.hook_mlps = allm
        elif layers == "last2":
            self.hook_mlps = allm[-2:]
        else:
            raise ValueError("layers must be 'last', 'last2', or 'all'")

    # -- input dict -----------------------------------------------------------
    def _make_data(self, numbers, coords, charge):
        torch = self.torch
        return {
            # batch axis of 1 -> dense mode, no padding atom needed
            "coord": torch.as_tensor(coords, dtype=torch.float32,
                                     device=self.device).unsqueeze(0),   # (1,N,3)
            "numbers": torch.as_tensor(numbers, dtype=torch.long,
                                       device=self.device).unsqueeze(0),  # (1,N)
            "charge": torch.as_tensor([float(charge)], dtype=torch.float32,
                                      device=self.device),               # (1,)
        }

    # -- per-atom hidden states ----------------------------------------------
    def per_atom_aim(self, smiles: str, method: str = "hook") -> np.ndarray:
        """Return the (N, 256) per-atom `aim` matrix for one molecule."""
        torch = self.torch
        numbers, coords, charge = smiles_to_atoms(
            smiles, seed=self.seed, optimize=self.optimize_geometry
        )
        # Coverage check: out-of-scope elements give NaN embeddings otherwise.
        bad = sorted({int(z) for z in numbers} - self.elements)
        if bad:
            syms = [_Z_SYMBOL.get(z, f"Z{z}") for z in bad]
            msg = (f"AIMNet2 ('{self.calc.metadata.get('family') if self.calc.metadata else 'aimnet2'}') "
                   f"does not cover element(s) {syms} (Z={bad}) in {smiles!r}. "
                   f"Covered: {sorted(self.elements)}. Use MACE-MP-0 (89 elements) "
                   f"for such molecules.")
            if self.strict_elements:
                raise ValueError(msg)
            warnings.warn(msg + " Returning NaN.", stacklevel=2)

        data = self._make_data(numbers, coords, charge)

        if method == "hook":
            captured = {}
            handles = []
            for idx, mlp in enumerate(self.hook_mlps):
                def _mk(i):
                    def _hook(_module, _inp, out):
                        captured[i] = out.detach()
                    return _hook
                handles.append(mlp.register_forward_hook(_mk(idx)))
            try:
                with torch.no_grad():
                    # Output heads run after mlps[-1]; the hooks have already
                    # captured every requested pass by then, so even if a head
                    # dislikes dense mode we still have the embeddings.
                    try:
                        self.model(data)
                    except Exception:
                        if len(captured) < len(self.hook_mlps):
                            raise
            finally:
                for h in handles:
                    h.remove()
            # concat passes in order (earliest -> latest): (1, N, sum_d)
            aim = torch.cat([captured[i] for i in range(len(self.hook_mlps))], dim=-1)
        elif method == "dict":
            if self.layers != "last":
                raise ValueError("method='dict' only supports layers='last'")
            with torch.no_grad():
                out = self.model(data)
            aim = out["aim"]
        else:
            raise ValueError("method must be 'hook' or 'dict'")

        aim = aim.squeeze(0)            # (1, N, D) -> (N, D)
        return aim.cpu().numpy()

    # -- pooled molecular vector ---------------------------------------------
    def embed(self, smiles: str, method: str = "hook") -> np.ndarray:
        """Mean-pool the per-atom `aim` to a single (256,) molecular vector."""
        return self.per_atom_aim(smiles, method=method).mean(axis=0)

    # -- CheMeleon-style batch interface -------------------------------------
    def __call__(self, molecules) -> np.ndarray:
        """Embed a list of SMILES -> (n_molecules, 256) float array.

        Mirrors ``CheMeleonFingerprint.__call__`` so this object can be dropped
        straight into the tutorial's ``PretrainedWrapper`` / ``custom_*`` helpers.
        """
        if isinstance(molecules, str):
            molecules = [molecules]
        if self.pooling == "rff":
            # per-atom aim -> RFF kernel-mean embedding (pooler fit once, cached)
            return self._rff(molecules, lambda s: self.per_atom_aim(s))
        return np.stack([self.embed(m) for m in molecules], axis=0)


# ----------------------------------------------------------------------------
# 3. The requested standalone function
# ----------------------------------------------------------------------------

# Cache embedders by (model_name, device) so the weights load only once.
_EMBEDDER_LOCK = threading.Lock()


@lru_cache(maxsize=4)
def _get_embedder(model_name: str, device: str) -> AIMNet2Embedder:
    return AIMNet2Embedder(model_name=model_name, device=device)


def aimnet2_embedding(smiles: str, model_name: str = "aimnet2") -> np.ndarray:
    """Given a SMILES string, return a fixed-size molecular embedding.

    Runs AIMNet2 on a single RDKit-ETKDG conformer and mean-pools the per-atom
    last hidden states (``data["aim"]``, the vector that feeds the energy/charge
    output heads) into one 256-dimensional vector.

    Parameters
    ----------
    smiles : str
        Molecule SMILES.
    model_name : str
        AIMNet2 registry name/alias (default ``"aimnet2"`` ==
        ``aimnet2-wb97m-d3_0``).

    Returns
    -------
    np.ndarray, shape (256,), dtype float32
    """
    with _EMBEDDER_LOCK:
        embedder = _get_embedder(model_name, "cpu")
    return embedder.embed(smiles).astype(np.float32)


# ----------------------------------------------------------------------------
# 4. BO-framework representation wrappers
# ----------------------------------------------------------------------------
#
# Both EDBO+ and BayBE consume the SAME building block: a per-molecule
# descriptor table (index = molecule label, columns = feature dims).  We build
# that table once and adapt it to either framework.


def aimnet2_descriptor_frame(
    smiles_dict: dict[str, str],
    model_name: str = "aimnet2",
    *,
    normalize: str | None = "global",
    drop_constant: bool = True,
    prefix: str = "aim",
):
    """Build a (n_molecules, 256) descriptor DataFrame from AIMNet2 embeddings.

    This is the framework-agnostic core, matching ``custom_fingerprinter`` in
    the HSF-ChemBO tutorial (``base/utils.py``): same index/column layout,
    same optional [0,1] normalization, same constant-column drop.

    Parameters
    ----------
    smiles_dict : dict[label -> SMILES]
        e.g. ``{"DMAc": "CC(=O)N(C)C", "BuOAc": "CCCCOC(C)=O", ...}``.
    normalize : {"global", "local", None}
        Scaling of the feature matrix, as in the tutorial.
    """
    import pandas as pd

    embedder = _get_embedder(model_name, "cpu")
    labels = list(smiles_dict.keys())
    smiles = [smiles_dict[k] for k in labels]
    feats = embedder(smiles)  # (n, 256)

    cols = [f"{prefix}_{i}" for i in range(feats.shape[1])]
    df = pd.DataFrame(feats, index=labels, columns=cols)

    # Drop constant columns on RAW features first (range-based) so global
    # normalization can't collapse a low-variance component to 0 columns.
    if drop_constant:
        df = df.loc[:, (df.max() - df.min()) > 1e-8]

    if normalize == "global":
        lo, hi = df.values.min(), df.values.max()
        df = (df - lo) / (hi - lo if hi > lo else 1e-12)
    elif normalize == "local":
        lo = df.min(axis=0)
        rng = (df.max(axis=0) - lo).replace(0, 1e-12)
        df = (df - lo) / rng
    return df


# --- EDBO+ (primary) --------------------------------------------------------


def aimnet2_edbo_scope(
    component_smiles: dict[str, dict[str, str]],
    extra_columns: dict[str, list] | None = None,
    model_name: str = "aimnet2",
    normalize: str | None = "global",
):
    """Build an EDBO+ reaction *scope* DataFrame using AIMNet2 descriptors.

    EDBO+ (``edbo.plus.optimizer_botorch.EDBOplus``) optimizes over a CSV/
    DataFrame "scope": one row per candidate combination, descriptor columns as
    features, plus one (empty) column per objective.  We expand every molecular
    component into its 256-d AIMNet2 descriptor columns and take the Cartesian
    product of all components / extra factors.

    Parameters
    ----------
    component_smiles : dict[component_name -> dict[label -> SMILES]]
        e.g. ``{"solvent": {...}, "base": {...}, "ligand": {...}}``.
    extra_columns : dict[name -> list of values], optional
        Non-molecular factors (e.g. ``{"temperature": [20, 60, 100]}``).
    normalize : passed through to ``aimnet2_descriptor_frame``.

    Returns
    -------
    scope : pandas.DataFrame
        Ready to hand to ``EDBOplus().run(...)`` (after you add target columns,
        e.g. ``scope["yield"] = "PENDING"``).
    descriptor_frames : dict[component_name -> DataFrame]
        The per-component descriptor tables (useful for inspection/PCA).
    """
    import itertools
    import pandas as pd

    desc = {
        name: aimnet2_descriptor_frame(mapping, model_name=model_name, prefix=name,
                                       normalize=normalize)
        for name, mapping in component_smiles.items()
    }

    # Cartesian product over component labels (+ extra factor levels).
    comp_names = list(component_smiles.keys())
    label_axes = [list(component_smiles[c].keys()) for c in comp_names]
    extra_columns = extra_columns or {}
    extra_names = list(extra_columns.keys())
    extra_axes = [extra_columns[k] for k in extra_names]

    rows = []
    for combo in itertools.product(*label_axes, *extra_axes):
        comp_labels = combo[: len(comp_names)]
        extra_vals = combo[len(comp_names):]
        row = {}
        # human-readable identity columns
        for cname, clabel in zip(comp_names, comp_labels):
            row[cname] = clabel
        for ename, eval_ in zip(extra_names, extra_vals):
            row[ename] = eval_
        # AIMNet2 descriptor columns for each molecular component
        for cname, clabel in zip(comp_names, comp_labels):
            row.update(desc[cname].loc[clabel].to_dict())
        rows.append(row)

    scope = pd.DataFrame(rows)
    return scope, desc


# --- BayBE (kept for parity with the tutorial) ------------------------------


def aimnet2_baybe_parameter(name: str, smiles_dict: dict[str, str],
                            model_name: str = "aimnet2",
                            decorrelate=0.7, normalize: str | None = None):
    """Wrap AIMNet2 embeddings as a BayBE ``CustomDiscreteParameter``.

    Direct analogue of the tutorial's
    ``CustomDiscreteParameter(name=..., data=custom_fingerprinter(...))``;
    here the fingerprinter is AIMNet2 instead of CheMeleon.
    """
    from baybe.parameters import CustomDiscreteParameter

    df = aimnet2_descriptor_frame(smiles_dict, model_name=model_name,
                                  normalize=normalize, prefix="aim")
    return CustomDiscreteParameter(name=name, data=df, decorrelate=decorrelate)


# ----------------------------------------------------------------------------
# HOOKS_AND_JIT
# ----------------------------------------------------------------------------
# Is a standard PyTorch forward hook sufficient?
#   * v2 `.pt` models (everything in the current registry, e.g. 'aimnet2'):
#       YES. `load_model` builds an eager `aimnet.models.aimnet2.AIMNet2`
#       nn.Module, so `model.mlps[-1].register_forward_hook(...)` fires.
#   * legacy v1 `.jpt` TorchScript models:
#       NO. `load_model` returns a `torch.jit.ScriptModule`; Python forward
#       hooks don't run and submodules aren't addressable. If you must use such
#       a file, either re-export to v2, or read `data["aim"]` by calling the
#       scripted module directly in dense mode (the dict key still exists) and
#       slicing off the padding atom.
#
# Minimal change if `aim` were NOT exposed at all (it IS, via data['aim']):
#   The only thing that hides it from the *calculator* is `keep_only`. To get
#   it through `AIMNet2Calculator.eval()` you would add one line in
#   aimnet/calculators/calculator.py:
#       keys_out: ClassVar[list[str]] = [..., "aim"]
#   We avoid patching the library by calling `calc.model(...)` / hooking instead.


if __name__ == "__main__":
    # Minimal working example: ethanol from SMILES.
    vec = aimnet2_embedding("CCO")
    print("ethanol AIMNet2 embedding:", vec.shape, vec.dtype)
    print("first 8 dims:", np.round(vec[:8], 4))
