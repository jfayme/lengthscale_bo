"""
lsab/featurize.py
=================

Module 2 of the lengthscale-A/B rewrite: SMILES -> embedding matrix for every
representation the A/B compares, behind ONE disk cache. It is what the
collaborator runs, so it needs only the model weights: nothing here imports the
old tree or, at module scope, any model stack. The registry holds thunks that
import a featurizer module (`lsab/featurizers/*.py`) on first use.

Division of labour:
  * a featurizer maps SMILES -> (n, d) float32 and RAISES on any molecule it
    cannot embed. It is dumb on purpose.
  * this module decides element coverage (here and nowhere else), owns the cache,
    and turns a raised or non-finite molecule into a recorded failure and a NaN row.
  * the caller (module 3) imputes NaN rows with the column median, so the pool is
    identical for every representation, and raises if more than 1% of a
    component's molecules failed. One rule, written down here.

Cache: CACHE_DIR/<rep>/{vectors.npz, failures.json}, written atomically every 100
new molecules and at the end. A cached failure is not retried unless
`--retry-failures` clears it. The path says nothing about model size or device:
change a registry entry and you must delete that rep's directory.

    python -m lsab.featurize --rep mace_mp0 aimnet2_all --dataset bh_reaction_1 photoswitches
    python -m lsab.featurize --status
"""
from __future__ import annotations

import argparse
import functools
import importlib
import json
import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Protocol

import numpy as np

log = logging.getLogger("lsab.featurize")   # not __name__: that is "__main__" under python -m

CACHE_DIR = Path(os.environ.get("LSAB_CACHE") or Path(__file__).resolve().parents[1] / "embeddings")
FLUSH_EVERY = 100   # new molecules between cache writes: an interrupted run keeps its work


# =============================================================================
# 1. THE REGISTRY
# =============================================================================
class Featurizer(Protocol):
    name: str                                               # registry key; also the cache key
    def __call__(self, smiles: list[str]) -> np.ndarray: ...  # (n, d) float32; raises on ANY failure


@dataclass(frozen=True)
class RepSpec:
    name: str
    build: Callable[..., Featurizer]     # thunk: build(device) imports the stack, returns a fresh featurizer
    elements: frozenset[int] | None      # STATIC coverage (checkable without loading a model); None = any
    fallback: str | None                 # rep used for a component this one cannot cover
    cached: bool                         # False for morgan (ms per molecule; a cache is dead weight)


def _thunk(module: str, cls: str, **fixed) -> Callable[[str], Featurizer]:
    """build(device): import lsab.featurizers.<module> only when called, then construct.
    Everything but `device` is fixed per entry: a variant is a new entry (and a new
    cache directory), never a kwarg."""
    def build(device: str) -> Featurizer:
        return getattr(importlib.import_module(f"lsab.featurizers.{module}"), cls)(device=device, **fixed)
    return build


def _morgan(device: str) -> Featurizer:   # a fingerprint has no device
    from lsab.featurizers.morgan import MorganFeaturizer
    return MorganFeaturizer()


_MACE_OFF23 = frozenset({1, 6, 7, 8, 9, 15, 16, 17, 35, 53})                 # H C N O F P S Cl Br I
_AIMNET2 = frozenset({1, 5, 6, 7, 8, 9, 14, 15, 16, 17, 33, 34, 35, 53})     # + B Si As Se (wB97M-D3)

REPS: dict[str, RepSpec] = {spec.name: spec for spec in [
    RepSpec("morgan", _morgan, None, None, cached=False),
    RepSpec("mace_mp0", _thunk("mace", "MaceFeaturizer", model="mp0"),
            frozenset(range(1, 90)), None, cached=True),
    RepSpec("mace_off23", _thunk("mace", "MaceFeaturizer", model="off23"),
            _MACE_OFF23, "mace_mp0", cached=True),
    RepSpec("aimnet2", _thunk("aimnet2", "AIMNet2Featurizer", layers="last"),
            _AIMNET2, "mace_mp0", cached=True),
    RepSpec("aimnet2_all", _thunk("aimnet2", "AIMNet2Featurizer", layers="all"),
            _AIMNET2, "mace_mp0", cached=True),
    RepSpec("t5", _thunk("lm", "T5Featurizer"), None, None, cached=True),
    RepSpec("chemberta", _thunk("lm", "ChemBERTaFeaturizer"), None, None, cached=True),
]}


# =============================================================================
# 2. COVERAGE  -- the ONLY place element coverage is decided
# =============================================================================
def uncovered_elements(rep: str, smiles: list[str]) -> set[int]:
    """Atomic numbers present in `smiles` but not in REPS[rep].elements; set()
    when the rep takes any element. An unparsable SMILES is skipped here: it
    fails later, in its featurizer, and is recorded like any other failure."""
    allowed = REPS[rep].elements
    if allowed is None:
        return set()
    from rdkit import Chem
    present = set()
    for s in dict.fromkeys(smiles):   # parse each SMILES once
        mol = Chem.MolFromSmiles(s)
        if mol is not None:
            present |= {atom.GetAtomicNum() for atom in mol.GetAtoms()}
    return present - allowed


# =============================================================================
# 3. THE CACHE
# =============================================================================
def _read_cache(folder: Path) -> tuple[dict[str, np.ndarray], dict[str, str]]:
    vectors, failures = {}, {}
    if (folder / "vectors.npz").exists():
        with np.load(folder / "vectors.npz") as saved:
            vectors = dict(zip(saved["smiles"].tolist(), saved["X"]))
    if (folder / "failures.json").exists():
        failures = json.loads((folder / "failures.json").read_text(encoding="utf-8"))
    return vectors, failures


def _write_atomic(path: Path, write: Callable) -> None:
    """Write `path.tmp`, then os.replace it: an interrupted write never leaves a torn file."""
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as handle:
        write(handle)
    os.replace(tmp, path)


def _write_cache(folder: Path, vectors: dict[str, np.ndarray], failures: dict[str, str]) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    X = np.stack(list(vectors.values())) if vectors else np.zeros((0, 0), dtype=np.float32)
    # SMILES as a unicode array, not an object array: np.load then never needs
    # allow_pickle, so a cache someone shares with you cannot run code.
    smiles = np.array(list(vectors), dtype=str)
    _write_atomic(folder / "vectors.npz", lambda handle: np.savez(handle, smiles=smiles, X=X))
    _write_atomic(folder / "failures.json",
                  lambda handle: handle.write(json.dumps(failures, indent=1).encode("utf-8")))


@functools.lru_cache(maxsize=2)   # two slots: a rep and its fallback
def _featurizer(spec: RepSpec, device: str) -> Featurizer:
    """Load a model once per process, not once per component."""
    return spec.build(device)


# =============================================================================
# 4. THE TWO ENTRY POINTS
# =============================================================================
def embed(rep: str, smiles: list[str], *, device: str = "cpu",
          cache_dir: Path = CACHE_DIR) -> np.ndarray:
    """(len(smiles), d) float32 in input order; NaN rows for molecules that failed.

    Duplicates are embedded once. The model loads only if there is a cache miss,
    and misses are embedded one molecule at a time so a failure cannot lose a
    batch. A rep with `cached=False` computes every molecule and persists nothing.
    """
    spec = REPS[rep]
    folder = Path(cache_dir) / rep
    vectors, failures = _read_cache(folder) if spec.cached else ({}, {})
    misses = [s for s in dict.fromkeys(smiles) if s not in vectors and s not in failures]
    if misses:
        featurizer = _featurizer(spec, device)
        for count, s in enumerate(misses, 1):
            try:
                vector = np.asarray(featurizer([s]), dtype=np.float32)[0]
                if not np.isfinite(vector).all():
                    raise ValueError("non-finite embedding")
                vectors[s] = vector
            except Exception as error:   # KeyboardInterrupt is not an Exception: it stops the run
                failures[s] = f"{type(error).__name__}: {error}"[:200]
            if spec.cached and count % FLUSH_EVERY == 0:
                _write_cache(folder, vectors, failures)
        if spec.cached:
            _write_cache(folder, vectors, failures)
    if not vectors:
        raise RuntimeError(f"{rep}: no molecule has ever embedded, so d is unknown; "
                           f"e.g. {list(failures.items())[:3]}")
    X = np.full((len(smiles), len(next(iter(vectors.values())))), np.nan, dtype=np.float32)
    for row, s in enumerate(smiles):
        if s in vectors:
            X[row] = vectors[s]
    return X


def embed_component(rep: str, smiles: list[str], **kw) -> tuple[np.ndarray, str]:
    """Coverage-aware `embed` for ONE component. Returns (X, used_rep).

    If `rep` cannot cover an element in the component, the WHOLE component is
    embedded with the rep's fallback (a component's columns must come from one
    representation) and one warning names the elements. No fallback -> raise.
    This is how the Shields Cs/K bases reach mace_mp0 under mace_off23/aimnet2.
    """
    missing = uncovered_elements(rep, smiles)
    if not missing:
        return embed(rep, smiles, **kw), rep
    fallback = REPS[rep].fallback
    if fallback is None:
        raise ValueError(f"{rep} cannot embed elements Z={sorted(missing)} and has no fallback")
    log.warning("%s cannot embed elements Z=%s; embedding the whole component with %s",
                rep, sorted(missing), fallback)
    return embed_component(fallback, smiles, **kw)


def uncached(rep: str, smiles: list[str], *, cache_dir: Path = CACHE_DIR) -> list[str]:
    """The molecules `embed_component(rep, smiles)` would have to COMPUTE, after the
    coverage fallback: [] for a rep that is never cached (morgan is computed on demand).
    Lets a dry run say what to precompute instead of silently loading models."""
    while uncovered_elements(rep, smiles) and REPS[rep].fallback:
        rep = REPS[rep].fallback
    if not REPS[rep].cached:
        return []
    vectors, failures = _read_cache(Path(cache_dir) / rep)
    return [s for s in dict.fromkeys(smiles) if s not in vectors and s not in failures]


# =============================================================================
# 5. CLI  -- what the collaborator runs
# =============================================================================
def _status(cache_dir: Path) -> None:
    for rep, spec in REPS.items():
        if not spec.cached:
            print(f"{rep:12s} not cached (computed on demand)")
            continue
        vectors, failures = _read_cache(Path(cache_dir) / rep)
        d = len(next(iter(vectors.values()))) if vectors else "-"
        print(f"{rep:12s} cached={len(vectors):<6d} failed={len(failures):<5d} d={d}")


def main(argv: list[str] | None = None) -> None:
    from lsab import datasets

    parser = argparse.ArgumentParser(prog="python -m lsab.featurize",
                                     description="Compute and cache embeddings for the lengthscale A/B.")
    parser.add_argument("--rep", nargs="+", default=[], choices=list(REPS))
    parser.add_argument("--all-reps", action="store_true")
    parser.add_argument("--dataset", nargs="+", default=[], choices=datasets.names())
    parser.add_argument("--all-datasets", action="store_true", help="lsab.datasets.DEFAULT_DATASETS")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--cache-dir", type=Path, default=CACHE_DIR)
    parser.add_argument("--status", action="store_true", help="per rep: cached / failed counts and d")
    parser.add_argument("--retry-failures", action="store_true",
                        help="forget the reps' recorded failures before embedding")
    args = parser.parse_args(argv)
    if args.status:
        return _status(args.cache_dir)
    reps = list(REPS) if args.all_reps else args.rep
    names = list(datasets.DEFAULT_DATASETS) if args.all_datasets else args.dataset
    if not reps or not names:
        parser.error("give --rep or --all-reps, and --dataset or --all-datasets (or --status)")
    if args.retry_failures:
        for rep in reps:
            (args.cache_dir / rep / "failures.json").unlink(missing_ok=True)

    pools = {name: datasets.load(name) for name in names}
    failed = []
    for rep in reps:          # rep-outer, so each model loads once
        for name, pool in pools.items():
            for component, smiles in pool.components.items():
                start = time.time()
                X, used = embed_component(rep, smiles, device=args.device, cache_dir=args.cache_dir)
                bad = [s for s, row in zip(smiles, X) if np.isnan(row).any()]
                failed += [(used, name, component, s) for s in bad]
                print(f"{rep:12s} {name:18s} {component:20s} n={len(smiles):<6d} failed={len(bad):<4d} "
                      f"used={used:12s} {time.time() - start:8.1f} s", flush=True)
    reasons = {rep: _read_cache(args.cache_dir / rep)[1] for rep in {f[0] for f in failed}}
    for used, name, component, s in failed:
        print(f"FAILED {used} {name}/{component}: {s}  {reasons[used].get(s, '')}")


if __name__ == "__main__":
    main()
