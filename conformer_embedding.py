"""
conformer_embedding.py
======================

A lightweight, CREST-style conformer-ensemble embedder that uses the **MLIP
itself** (AIMNet2 / MACE-OFF23 / MACE-MP-0) for the energies -- instead of xTB
as in CREST (https://github.com/crest-lab/crest) -- so the *same* network gives
both (a) the conformer energy used for ranking / Boltzmann weighting and
(b) the per-atom hidden-state embedding.

Pipeline (a practical MLIP analogue of CREST's metadynamics search):
    1. RDKit ETKDG generates an ensemble of conformers (multiple seeds).
    2. (optional) MMFF pre-optimization + RMSD pruning of duplicates.
    3. (optional) MLIP geometry relaxation of each survivor (ASE BFGS) -- this
       is the "explore conformers with the MLIP" step; turn off for speed.
    4. The MLIP scores every conformer's energy.
    5. The MLIP embeds every conformer (mean-pooled per-atom hidden state).
    6. Conformer embeddings are aggregated to ONE molecular vector by:
         * "boltzmann" -- energy-weighted average, w_i ~ exp(-(E_i-E_min)/kT)
         * "lowest"    -- embedding of the single lowest-energy conformer
         * "mean"      -- plain average over conformers
         * "max"       -- element-wise max over conformers (a coarse pooling)

Why this is nice: an ETKDG single conformer (what ``aimnet2_embedding`` uses)
is geometry-noisy; a Boltzmann/lowest-energy ensemble embedding is a smoother,
more physically meaningful molecular descriptor for a GP kernel -- and it costs
only extra forward passes of a model you are already running.

Backends share the loaders in ``aimnet2_repr`` / ``mace_repr`` so a model loads
once.  Energies are in eV for all three backends (consistent kT).
"""

from __future__ import annotations

import os
import warnings
from dataclasses import dataclass, field

import numpy as np

os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
# Geometry relaxation computes forces through a path that invokes torch's inductor,
# which needs a C++ compiler (cl.exe on Windows). Force eager so the default relax=True
# runs without MSVC installed; harmless where the model would not benefit from compile.
# Override by exporting TORCHDYNAMO_DISABLE=0 before importing this module.
os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")

KB_EV_PER_K = 8.617333262e-5  # Boltzmann constant in eV/K


_UNRELAXED_WARNED = False


def _warn_unrelaxed_once():
    """Emit, once per process, the caveat that unrelaxed single-point energies
    make Boltzmann weights untrustworthy."""
    global _UNRELAXED_WARNED
    if _UNRELAXED_WARNED:
        return
    _UNRELAXED_WARNED = True
    warnings.warn(
        "relax=False: Boltzmann weights are derived from MLIP single-point energies "
        "on unrelaxed MMFF/ETKDG geometries, which are noisy and not trustworthy. "
        "Prefer relax=True; in unrelaxed mode use aggregate='lowest' or 'mean', which "
        "do not depend on the energy ranking.",
        stacklevel=3,
    )


# ---------------------------------------------------------------------------
# Conformer generation (RDKit ETKDG ensemble)
# ---------------------------------------------------------------------------


def generate_conformers(smiles: str, n_confs: int = 20, seed: int = 42,
                        prune_rms: float = 0.3, mmff: bool = True):
    """Return ``(numbers, [coords_0, coords_1, ...], charge)`` for an ETKDG ensemble."""
    from rdkit import Chem
    from rdkit.Chem import AllChem

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"RDKit could not parse SMILES: {smiles!r}")
    mol = Chem.AddHs(mol)

    params = AllChem.ETKDGv3()
    params.randomSeed = seed
    params.pruneRmsThresh = prune_rms  # drop near-duplicate conformers
    cids = list(AllChem.EmbedMultipleConfs(mol, numConfs=n_confs, params=params))
    if not cids:
        params.useRandomCoords = True
        cids = list(AllChem.EmbedMultipleConfs(mol, numConfs=n_confs, params=params))
    if mmff:
        try:
            AllChem.MMFFOptimizeMoleculeConfs(mol, maxIters=500)
        except Exception:
            pass

    numbers = np.array([a.GetAtomicNum() for a in mol.GetAtoms()], dtype=np.int64)
    charge = Chem.GetFormalCharge(mol)
    coords = [mol.GetConformer(c).GetPositions().astype(np.float64) for c in cids]
    return numbers, coords, charge


# ---------------------------------------------------------------------------
# MLIP backends: one loaded model gives BOTH energy and embedding
# ---------------------------------------------------------------------------


class _AIMNet2Backend:
    name = "aimnet2"

    def __init__(self, model_name="aimnet2", device="cpu"):
        import torch
        from aimnet.calculators import AIMNet2Calculator
        self.torch = torch
        self.calc = AIMNet2Calculator(model_name, device=device)
        self.model = self.calc.model
        self.model.eval()
        self.device = device

    def _dense(self, numbers, coords, charge):
        torch = self.torch
        return {
            "coord": torch.tensor(coords, dtype=torch.float32, device=self.device).unsqueeze(0),
            "numbers": torch.tensor(numbers, dtype=torch.long, device=self.device).unsqueeze(0),
            "charge": torch.tensor([float(charge)], dtype=torch.float32, device=self.device),
        }

    def energy_and_per_atom(self, numbers, coords, charge):
        """Return (energy_eV, per_atom_aim (N, 256)) from ONE forward pass."""
        torch = self.torch
        data = self._dense(numbers, coords, charge)
        with torch.no_grad():
            out = self.model(data)
        aim = out["aim"].squeeze(0).cpu().numpy()          # (N, 256)
        energy = float(out["energy"].reshape(-1)[0].item())  # eV
        return energy, aim

    def energy_and_embedding(self, numbers, coords, charge):
        """Return (energy_eV, mean-pooled embedding)."""
        e, aim = self.energy_and_per_atom(numbers, coords, charge)
        return e, aim.mean(axis=0)

    def relax(self, numbers, coords, charge, fmax=0.05, steps=50):
        from ase import Atoms
        from ase.optimize import BFGS
        from aimnet.calculators import AIMNet2ASE
        atoms = Atoms(numbers=numbers, positions=coords)
        atoms.calc = AIMNet2ASE(self.calc, charge=int(charge))
        BFGS(atoms, logfile=None).run(fmax=fmax, steps=steps)
        return atoms.get_positions()


class _MACEBackend:
    def __init__(self, kind, model_size, device="cpu"):
        from mace_repr import _load_mace_off, _load_mace_mp
        self.name = kind
        self.calc = (_load_mace_off if kind == "mace_off23" else _load_mace_mp)(
            model_size, device, "float64")

    def energy_and_per_atom(self, numbers, coords, charge):
        from ase import Atoms
        atoms = Atoms(numbers=numbers, positions=coords)
        atoms.calc = self.calc
        energy = float(atoms.get_potential_energy())  # eV
        desc = self.calc.get_descriptors(atoms, invariants_only=True)  # (N, D)
        return energy, np.asarray(desc, dtype=np.float64)

    def energy_and_embedding(self, numbers, coords, charge):
        e, desc = self.energy_and_per_atom(numbers, coords, charge)
        return e, desc.mean(axis=0)

    def relax(self, numbers, coords, charge, fmax=0.05, steps=50):
        from ase import Atoms
        from ase.optimize import BFGS
        atoms = Atoms(numbers=numbers, positions=coords)
        atoms.calc = self.calc
        BFGS(atoms, logfile=None).run(fmax=fmax, steps=steps)
        return atoms.get_positions()


class _MACEMHBackend:
    """MACE-MH-1 multi-head foundation model (1024-d invariant descriptor).

    Loaded via mace_mp with an explicit head (default spice_wB97M, the organic
    wB97M head — matches AIMNet2's functional family). The shared message-passing
    layers feed all heads, so the descriptor (node_feats) is head-independent;
    only the energy (used for conformer ranking) depends on the head.
    """
    name = "mace_mh1"

    def __init__(self, head="spice_wB97M", device="cpu", model_path=None):
        import os
        from mace.calculators import mace_mp
        model_path = model_path or os.path.expanduser("~/.cache/mace/mace-mh-1.model")
        self.calc = mace_mp(model=model_path, device=device,
                            default_dtype="float64", head=head)

    def energy_and_per_atom(self, numbers, coords, charge):
        from ase import Atoms
        atoms = Atoms(numbers=numbers, positions=coords)
        atoms.calc = self.calc
        energy = float(atoms.get_potential_energy())          # eV
        desc = self.calc.get_descriptors(atoms, invariants_only=True)  # (N, 1024)
        return energy, np.asarray(desc, dtype=np.float64)

    def energy_and_embedding(self, numbers, coords, charge):
        e, d = self.energy_and_per_atom(numbers, coords, charge)
        return e, d.mean(axis=0)

    def relax(self, numbers, coords, charge, fmax=0.05, steps=50):
        from ase import Atoms
        from ase.optimize import BFGS
        atoms = Atoms(numbers=numbers, positions=coords); atoms.calc = self.calc
        BFGS(atoms, logfile=None).run(fmax=fmax, steps=steps)
        return atoms.get_positions()


def _polar_electronic_per_atom(numbers, coords, polar_calc):
    """POLAR-1 per-atom electronic features: [charge, spin, density_coeffs(4)] -> (N, 6).

    POLAR's MACECalculator does not implement get_descriptors, so we read the
    physically-meaningful per-atom electronic predictions from calc.results
    (the learned atomic charge/spin density multipole expansion).
    """
    from ase import Atoms
    atoms = Atoms(numbers=numbers, positions=coords)
    atoms.calc = polar_calc
    atoms.get_potential_energy()                 # populate results
    r = polar_calc.results
    q = np.asarray(r["charges"], dtype=np.float64).reshape(-1, 1)
    s = np.asarray(r["spins"], dtype=np.float64).reshape(-1, 1)
    dc = np.asarray(r["density_coefficients"], dtype=np.float64)   # (N, 4)
    return np.concatenate([q, s, dc], axis=1)                      # (N, 6)


class _MACEComboBackend:
    """Scenario B: MACE-MH-1 (geometry/energy node_feats) + MACE-POLAR-1
    (electronic charge/spin/density features), concatenated PER ATOM.

    Per-atom vector = [ MH-1 node_feats (1024) | POLAR electronic (6) ] -> (N, 1030).
    Energy (for conformer ranking) comes from MH-1 (POLAR's energy is unreliable,
    esp. for ionic systems). The pooling (mean / RFF) then runs over this combined
    per-atom set -- honouring the "concatenate per-atom then pool once" choice.
    """
    name = "mh1_polar"

    def __init__(self, head="spice_wB97M", polar="polar-1-m", device="cpu"):
        import os
        from mace.calculators import mace_mp
        self.mh = mace_mp(model=os.path.expanduser("~/.cache/mace/mace-mh-1.model"),
                          device=device, default_dtype="float64", head=head)
        try:
            from mace.calculators import mace_polar
        except Exception:
            from mace.calculators.foundations_models import mace_polar
        self.pol = mace_polar(model=polar, device=device, default_dtype="float64")

    def energy_and_per_atom(self, numbers, coords, charge):
        from ase import Atoms
        atoms = Atoms(numbers=numbers, positions=coords)
        atoms.calc = self.mh
        energy = float(atoms.get_potential_energy())                   # MH-1 energy
        mh_desc = np.asarray(self.mh.get_descriptors(atoms, invariants_only=True),
                             dtype=np.float64)                          # (N, 1024)
        pol_feat = _polar_electronic_per_atom(numbers, coords, self.pol)  # (N, 6)
        return energy, np.concatenate([mh_desc, pol_feat], axis=1)     # (N, 1030)

    def energy_and_embedding(self, numbers, coords, charge):
        e, d = self.energy_and_per_atom(numbers, coords, charge)
        return e, d.mean(axis=0)

    def relax(self, numbers, coords, charge, fmax=0.05, steps=50):
        from ase import Atoms
        from ase.optimize import BFGS
        atoms = Atoms(numbers=numbers, positions=coords); atoms.calc = self.mh
        BFGS(atoms, logfile=None).run(fmax=fmax, steps=steps)
        return atoms.get_positions()


def _make_backend(backend: str, **kwargs):
    if backend == "mh1_polar":
        return _MACEComboBackend(head=kwargs.get("head", "spice_wB97M"),
                                 polar=kwargs.get("polar", "polar-1-m"),
                                 device=kwargs.get("device", "cpu"))
    if backend == "aimnet2":
        return _AIMNet2Backend(model_name=kwargs.get("model_name", "aimnet2"),
                               device=kwargs.get("device", "cpu"))
    if backend == "mace_off23":
        return _MACEBackend("mace_off23", kwargs.get("model_size", "medium"),
                            kwargs.get("device", "cpu"))
    if backend == "mace_mp0":
        return _MACEBackend("mace_mp0", kwargs.get("model_size", "medium"),
                            kwargs.get("device", "cpu"))
    if backend == "mace_mh1":
        return _MACEMHBackend(head=kwargs.get("head", "spice_wB97M"),
                              device=kwargs.get("device", "cpu"))
    raise ValueError(f"Unknown backend {backend!r}")


# ---------------------------------------------------------------------------
# Ensemble aggregation
# ---------------------------------------------------------------------------


@dataclass
class EnsembleResult:
    embedding: np.ndarray              # aggregated molecular vector
    energies: np.ndarray               # per-conformer energies (eV)
    per_conformer: np.ndarray          # (n_conf, D) conformer embeddings
    weights: np.ndarray                # Boltzmann weights
    aggregate: str
    meta: dict = field(default_factory=dict)


def aggregate_embeddings(per_conf, energies, aggregate="boltzmann",
                         temperature=298.15):
    energies = np.asarray(energies, dtype=np.float64)
    per_conf = np.asarray(per_conf, dtype=np.float64)
    e_rel = energies - energies.min()
    weights = np.exp(-e_rel / (KB_EV_PER_K * temperature))
    weights /= weights.sum()

    if aggregate == "boltzmann":
        emb = (weights[:, None] * per_conf).sum(axis=0)
    elif aggregate in ("lowest", "min"):
        emb = per_conf[int(np.argmin(energies))]
    elif aggregate == "mean":
        emb = per_conf.mean(axis=0)
    elif aggregate == "max":
        emb = per_conf.max(axis=0)
    else:
        raise ValueError(f"Unknown aggregate {aggregate!r}")
    return emb, weights


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def conformer_ensemble_embedding(
    smiles: str,
    backend: str = "aimnet2",
    *,
    n_confs: int = 20,
    aggregate: str = "boltzmann",
    temperature: float = 298.15,
    relax: bool = True,
    relax_steps: int = 50,
    prune_rms: float = 0.3,
    seed: int = 42,
    return_full: bool = False,
    backend_obj=None,
    pooler=None,
    **backend_kwargs,
):
    """MLIP conformer-ensemble embedding (CREST-style, MLIP energies).

    Parameters
    ----------
    backend : {"aimnet2", "mace_off23", "mace_mp0"}
    aggregate : {"boltzmann", "lowest", "mean", "max"}
    relax : if True (default), MLIP-relax each conformer (ASE BFGS) before
        scoring -- closer to CREST but slower. With relax=False the energies
        are MLIP single-points on MMFF/ETKDG geometries; those are noisy, so the
        Boltzmann weights are unreliable (a one-time warning is emitted) and
        aggregate='lowest'/'mean' is safer.
    return_full : if True, return an ``EnsembleResult`` (energies, weights, all
        conformer embeddings); otherwise just the aggregated vector.

    Returns
    -------
    np.ndarray (float32) or EnsembleResult
    """
    if not relax:
        _warn_unrelaxed_once()
    be = backend_obj if backend_obj is not None else _make_backend(backend, **backend_kwargs)
    numbers, conformers, charge = generate_conformers(
        smiles, n_confs=n_confs, seed=seed, prune_rms=prune_rms)

    energies, per_atoms = [], []
    for coords in conformers:
        if relax:
            coords = be.relax(numbers, coords, charge, steps=relax_steps)
        e, pa = be.energy_and_per_atom(numbers, coords, charge)
        energies.append(e)
        per_atoms.append(pa)
    energies = np.array(energies)
    # per-conformer pooling: "mean" (default) or a fitted set-kernel pooler.
    pool = pooler if pooler is not None else (lambda a: np.asarray(a).mean(axis=0))
    per_conf = np.stack([pool(pa) for pa in per_atoms], axis=0)

    emb, weights = aggregate_embeddings(per_conf, energies, aggregate, temperature)
    if return_full:
        return EnsembleResult(
            embedding=emb.astype(np.float32), energies=energies,
            per_conformer=per_conf, weights=weights, aggregate=aggregate,
            meta={"smiles": smiles, "backend": be.name, "n_conf": len(conformers),
                  "charge": charge, "relaxed": relax})
    return emb.astype(np.float32)


def make_ensemble_fingerprinter(backend="aimnet2", *, n_confs=20,
                                aggregate="boltzmann", relax=True, relax_steps=50,
                                pooling="mean", temperature=298.15,
                                prune_rms=0.3, seed=42, rff_cfg=None, **kwargs):
    """CheMeleon-style batch fingerprinter using ensemble embeddings.

    Drop-in for ``representations.make_fingerprinter`` / the tutorial's
    ``custom_fingerprinter``: loads the MLIP once, embeds each SMILES via its
    conformer ensemble.

    pooling : "mean" (per-conformer atom mean, default) or "rff" (per-conformer
        RFF set-kernel embedding). For "rff" the pooler is fit ONCE on the atoms
        of all conformers across the first batch (warm up by calling on the full
        unique-molecule set), then reused -- so the projection is shared.
    """
    if not relax:
        _warn_unrelaxed_once()
    be = _make_backend(backend, **kwargs)
    cache: dict = {}            # smiles -> (energies, [per_atom, ...])
    state = {"pooler": None}

    def _collect(m):
        if m not in cache:
            numbers, confs, charge = generate_conformers(
                m, n_confs=n_confs, seed=seed, prune_rms=prune_rms)
            energies, per_atoms = [], []
            for coords in confs:
                if relax:
                    coords = be.relax(numbers, coords, charge, steps=relax_steps)
                e, pa = be.energy_and_per_atom(numbers, coords, charge)
                energies.append(e)
                per_atoms.append(np.asarray(pa, dtype=np.float64))
            cache[m] = (np.array(energies), per_atoms)
        return cache[m]

    def _fp(molecules):
        if isinstance(molecules, str):
            molecules = [molecules]
        data = [_collect(m) for m in molecules]
        if pooling == "rff":
            if state["pooler"] is None:
                from pooling import RFFMeanPooler
                allatoms = np.vstack([pa for (_, pas) in data for pa in pas])
                state["pooler"] = RFFMeanPooler(**(rff_cfg or {})).fit(allatoms)
            pool = state["pooler"]
        elif pooling == "mean":
            pool = (lambda a: np.asarray(a).mean(axis=0))            # intensive (centroid)
        elif pooling == "sum":
            pool = (lambda a: np.asarray(a).sum(axis=0))             # extensive (size-dependent)
        elif pooling == "stats":                                    # mean/std/min/max (distribution)
            def pool(a):
                a = np.asarray(a)
                return np.concatenate([a.mean(0), a.std(0), a.min(0), a.max(0)])
        else:
            raise ValueError(f"pooling must be 'mean'|'sum'|'stats'|'rff', got {pooling!r}")
        out = []
        for energies, per_atoms in data:
            per_conf = np.stack([pool(pa) for pa in per_atoms], axis=0)
            emb, _ = aggregate_embeddings(per_conf, energies, aggregate, temperature)
            out.append(emb.astype(np.float32))
        return np.stack(out, axis=0)

    return _fp


def _cosine(a, b):
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-12 or nb < 1e-12:
        return float("nan")
    return float(a @ b / (na * nb))


def diagnose_weight_collapse(smiles_list, backend="aimnet2", n_confs=20,
                             temperature=298.15, **kwargs):
    """Check whether Boltzmann weighting does real work or just collapses onto the
    single lowest-energy conformer (which would make it identical to
    aggregate='lowest', i.e. the Boltzmann machinery is decorative).

    For each SMILES, runs the full ensemble pipeline (``return_full=True``) with
    ``relax=False`` AND ``relax=True``, and -- from each single MLIP pass -- reuses
    the cached per-conformer embeddings to compute BOTH the boltzmann and lowest
    aggregates (no extra model calls). Reports per (SMILES, relax):
        * ``w_max``       -- largest single Boltzmann weight (1.0 = full collapse)
        * ``eff_conf``    -- effective # conformers = 1 / sum(w**2) (participation ratio)
        * ``cos(bz,low)`` -- cosine similarity of the boltzmann vs lowest embedding
        * ``dE_eV``       -- per-conformer energy spread (peak-to-peak)

    Interpretation: kT at 298 K ~ 0.0257 eV (~0.59 kcal/mol). If, across molecules,
    ``w_max`` ~ 1.0, ``eff_conf`` ~ 1, and ``cos(bz,low)`` ~ 1.0, then Boltzmann
    weighting adds nothing in this setup -- prefer ``aggregate='lowest'`` for clarity.
    If they diverge meaningfully (especially after relaxation, which should reduce
    energy noise and de-collapse the softmax), the ensemble weighting is doing real
    work. Running both relax modes makes it visible whether relaxing reduces collapse.
    """
    header = (f"{'SMILES':<16}{'relax':>7}{'w_max':>8}{'eff_conf':>10}"
              f"{'cos(bz,low)':>13}{'dE_eV':>9}")
    print(header)
    print("-" * len(header))
    rows = []
    for smi in smiles_list:
        for relax in (False, True):
            try:
                r = conformer_ensemble_embedding(
                    smi, backend=backend, n_confs=n_confs, aggregate="boltzmann",
                    temperature=temperature, relax=relax, return_full=True, **kwargs)
            except Exception as e:
                print(f"{smi:<16}{str(relax):>7}   FAIL: {type(e).__name__}: {e}")
                continue
            w = np.asarray(r.weights, dtype=np.float64)
            w_max = float(w.max())
            eff_conf = float(1.0 / np.sum(w ** 2))
            lowest = r.per_conformer[int(np.argmin(r.energies))]   # reuse cached embeddings
            cos = _cosine(r.embedding, lowest)                     # r.embedding is the boltzmann aggregate
            dE = float(np.ptp(r.energies)) if len(r.energies) > 1 else 0.0
            print(f"{smi:<16}{str(relax):>7}{w_max:>8.3f}{eff_conf:>10.2f}"
                  f"{cos:>13.4f}{dE:>9.4f}")
            rows.append(dict(smiles=smi, relax=relax, w_max=w_max, eff_conf=eff_conf,
                             cos_bz_low=cos, dE_eV=dE))
    return rows


if __name__ == "__main__":
    for agg in ("boltzmann", "lowest", "mean"):
        r = conformer_ensemble_embedding("CCO", backend="aimnet2", n_confs=8,
                                         aggregate=agg, return_full=True)
        print(f"ethanol aimnet2 {agg:9s}: dim={r.embedding.shape} "
              f"n_conf={r.meta['n_conf']} dE(eV)={np.ptp(r.energies):.4f} "
              f"w_max={r.weights.max():.3f} first3={np.round(r.embedding[:3],4)}")

    # Does Boltzmann weighting actually differ from 'lowest'? Rigid / flexible /
    # ionizable molecules, each at relax=False vs relax=True.
    print("\n=== Boltzmann weight-collapse diagnostic (relax=False vs relax=True) ===")
    diagnose_weight_collapse(["c1ccccc1", "CCCCCCCO", "CC(=O)O"],
                             backend="aimnet2", n_confs=20)
