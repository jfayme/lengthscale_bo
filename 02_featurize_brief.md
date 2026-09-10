# Rewrite — module 2: `lsab/featurize.py` + `lsab/featurizers/`

Input contract: `lsab.datasets.Pool` as merged (`pool.components: dict[str,
list[str]]` of unique SMILES per component; no null SMILES ever reach this
module — Suzuki rows without a ligand/reagent are dropped in module 1).

This is the module the collaborator will actually run to produce embeddings,
so it must work from a clean environment with the model weights and nothing
from the old tree. Read the old `representations.py`, `aimnet2_repr.py`,
`mace_repr.py`, `gollum_pipeline.embed_smiles` and
`HSF-ChemBO-tutorial/base/pretrained_repr.py` for the exact model-loading
calls, then write it fresh.

---

## Layout

```
lsab/featurize.py            registry, coverage, cache, embed(), embed_component(), CLI
lsab/featurizers/__init__.py  empty
lsab/featurizers/conformer.py SMILES -> one ETKDG conformer (shared by mace + aimnet2)
lsab/featurizers/morgan.py
lsab/featurizers/mace.py      one class, model="mp0" | "off23"
lsab/featurizers/aimnet2.py   one class, layers="last" | "all"
lsab/featurizers/lm.py        T5 and ChemBERTa, one shared masked-mean pooler
```

`lsab.featurize` must import none of torch / mace / aimnet / transformers /
rdkit at module scope. Each featurizer module imports its own stack at ITS
module scope (so `import lsab.featurizers.mace` is what costs the time), and
the registry holds thunks that import them.

---

## The featurizer protocol

```python
class Featurizer(Protocol):
    name: str                                  # registry key; also the cache key
    def __call__(self, smiles: list[str]) -> np.ndarray   # (n, d) float32; raises on ANY failure
```

Rules for every featurizer:
- Raise (any exception) for a molecule it cannot embed. Never return NaN, a
  zero vector, or a warning-and-continue. The cache layer turns the exception
  into a recorded failure; the featurizer stays dumb.
- Same `d` for every molecule; float32 output.
- No element check inside the featurizer beyond what the model itself raises;
  coverage is checked statically by `featurize.py` before any model loads.
- Deterministic on CPU: fixed conformer seed (42), `torch.no_grad()`,
  `model.eval()`. GPU results may differ in the last bits; say so in the
  docstring, do not try to fix it.

### `featurizers/conformer.py` — use this code as-is (already verified)

```python
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
```

### `featurizers/morgan.py`
`MorganFeaturizer(radius=2, n_bits=2048)`; `rdFingerprintGenerator.GetMorganGenerator`,
`GetFingerprintAsNumPy`, float32. ~20 lines. Not cached (see cache section).

### `featurizers/mace.py`
```python
class MaceFeaturizer:
    """Mean-pooled per-atom INVARIANT (l=0) descriptor from a MACE foundation model.
    model="mp0" (Materials Project, Z=1..89) or "off23" (organic, 10 elements)."""
    def __init__(self, model: Literal["mp0", "off23"], size: str = "medium",
                 device: str = "cpu", dtype: str = "float32", seed: int = 42)
```
- Set `os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")` at the
  top of this module, before `import mace`, with the old comment about
  torch >= 2.6 and the foundation-model checkpoints. Nothing else from the old
  env-var block.
- Load via `mace.calculators.mace_mp(model=size, ...)` / `mace_off(...)`.
- Per molecule: `smiles_to_atoms` → `ase.Atoms` →
  `calc.get_descriptors(atoms, invariants_only=True, num_layers=-1)` → mean
  over atoms → float32. That is the whole `__call__`. No `last_layer_only`,
  no RFF, no reactive-site masks.
- One dtype for both models. The old code used float64 for OFF23 and float32
  for MP-0 for no recorded reason; use float32 for both (the cache is float32
  anyway) and note it in the README as a difference from the old runs.

### `featurizers/aimnet2.py`
```python
class AIMNet2Featurizer:
    """Mean-pooled per-atom hidden state of AIMNet2 (dense, all-pairs mode).
    layers="last": the 256-d `aim` vector that feeds the output heads.
    layers="all" : every MLP pass concatenated (the layer-ablation winner in the A/B)."""
    def __init__(self, layers: Literal["last", "all"] = "all", model_name: str = "aimnet2",
                 device: str = "cpu", seed: int = 42)
```
- `aimnet.calculators.AIMNet2Calculator(model_name, device=device)`, `.model.eval()`.
- Reject a TorchScript model at construction (hooks cannot fire), as the old
  code did, with a one-line error.
- Input dict for dense mode, batch axis 1: `coord` (1,N,3) float32, `numbers`
  (1,N) long, `charge` (1,) float32 — copy from the old `_make_data`.
- Per-atom states via forward hooks on `model.mlps[-1]` (last) or all of
  `model.mlps` (all); concatenate in order; mean over atoms; float32. Keep the
  old defensive pattern of swallowing an exception from the output heads only
  if every hook already fired. Drop the "dict" method and `strict_elements`.
- `d` is 256 for "last"; for "all" assert at construction time, once, from a
  probe molecule ("C"), and expose it as `self.d`.

### `featurizers/lm.py`
```python
def masked_mean(hidden, attention_mask):           # (B,L,D),(B,L) -> (B,D); the one pooler
class T5Featurizer:        # GT4SD/multitask-text-and-chemistry-t5-base-augm, T5EncoderModel
class ChemBERTaFeaturizer: # seyonec/ChemBERTa-zinc-base-v1, AutoModel, last_hidden_state
```
- Both: `AutoTokenizer`, padding+truncation (max_length 512), batches of 16,
  `torch.no_grad()`, eval, float32, masked mean over the LAST hidden state,
  no L2 normalisation. One constructor arg `model_name` with the default
  above, one `device`.
- DECISION RECORDED HERE: the HSF-ChemBO baseline's ChemBERTa fingerprint sums
  `hidden_states[0]`, i.e. the token-embedding layer BEFORE any transformer
  block, so the model's weights never touch the vector. The rewrite pools the
  last hidden state like T5. State this in the class docstring and in the
  README's "differences from the old runs" list. Likewise T5: the published
  GOLLuM runs used `t5-base` (a generic English model) because the chemistry
  checkpoint would not download on that machine; the rewrite defaults to the
  chemistry checkpoint everywhere. README line.
- `cache_dir` for HF weights: leave to `HF_HOME`; do not hard-code
  `./from_pretrained`.

---

## `lsab/featurize.py`

### Registry

```python
@dataclass(frozen=True)
class RepSpec:
    name: str
    build: Callable[..., Featurizer]     # thunk: imports the stack, returns a fresh featurizer
    elements: frozenset[int] | None      # STATIC coverage (checkable without loading a model); None = any
    fallback: str | None                 # rep used for a component this one cannot cover
    cached: bool                         # False for morgan (ms per molecule; a cache is dead weight)

REPS: dict[str, RepSpec] = {
    "morgan":      RepSpec(..., elements=None, fallback=None, cached=False),
    "mace_mp0":    RepSpec(..., elements=frozenset(range(1, 90)), fallback=None, cached=True),
    "mace_off23":  RepSpec(..., elements=frozenset({1,6,7,8,9,15,16,17,35,53}), fallback="mace_mp0", cached=True),
    "aimnet2":     RepSpec(..., elements=frozenset({1,5,6,7,8,9,14,15,16,17,33,34,35,53}), fallback="mace_mp0", cached=True),
    "aimnet2_all": RepSpec(..., elements=<same>, fallback="mace_mp0", cached=True),
    "t5":          RepSpec(..., elements=None, fallback=None, cached=True),
    "chemberta":   RepSpec(..., elements=None, fallback=None, cached=True),
}
```
The thunks take `device` only. No other kwargs plumbing: model size, layers,
model names are fixed per registry entry; a variant is a new entry.

### Coverage

```python
def uncovered_elements(rep: str, smiles: list[str]) -> set[int]   # atomic numbers present but not in REPS[rep].elements; {} if elements is None
```
RDKit, lazy import, parse each SMILES once. This is the ONLY place element
coverage is decided.

### Cache

Per rep, one directory: `CACHE_DIR/<rep>/` with
```
vectors.npz     smiles: object array (n,), X: float32 (n, d)     -- successes
failures.json   {smiles: "error message[:200]"}                    -- so a run does not retry a molecule that cannot embed
```
- Written atomically (write `*.tmp`, `os.replace`), after every 100 new
  molecules and at the end, so an interrupted run keeps its work.
- `CACHE_DIR` = `<repo>/embeddings/` by default, overridable by argument and by
  `LSAB_CACHE` env var. Add it to `.gitignore`.
- A cached failure is a failure: `embed` does not retry it. `--retry-failures`
  on the CLI clears `failures.json` for that rep first.
- Do not store anything about model size / device in the cache path. If
  someone changes the registry entry they must delete the directory; write
  that in the README. (The old 99 MB cache is incompatible and is not
  migrated: the whole point of "re-run all results".)

### The two entry points

```python
def embed(rep: str, smiles: list[str], *, device: str = "cpu", cache_dir: Path = CACHE_DIR) -> np.ndarray
```
Returns `(len(smiles), d)` float32. Rows for failed molecules are NaN.
Loads the model ONCE (only if there is at least one cache miss), embeds
misses one molecule at a time (a failure must not lose the batch), updates
the cache, returns in input order. Duplicates in `smiles` are embedded once.
For a rep with `cached=False` it just calls the featurizer.

```python
def embed_component(rep: str, smiles: list[str], **kw) -> tuple[np.ndarray, str]
```
Coverage-aware: if `uncovered_elements(rep, smiles)` is non-empty and the rep
has a fallback, embed the WHOLE component with the fallback (a component's
columns must be one representation) and log one warning naming the elements;
if there is no fallback, raise. Returns `(X, used_rep)`. Module 3 records
`used_rep` per component in the run metadata. Shields' Cs/K bases go through
this path exactly as before.

Failure policy for the caller (module 3, written down here so it is one
rule): NaN rows are imputed with the column median so the candidate pool is
the same for every representation; if more than 1% of a component's
molecules failed, raise. The old code both dropped and imputed depending on
the failure type; that distinction goes.

### CLI — this is what the collaborator runs

```
python -m lsab.featurize --rep mace_mp0 aimnet2_all --dataset bh_reaction_1 photoswitches
python -m lsab.featurize --all-reps --all-datasets --device cuda
python -m lsab.featurize --status            # per rep: cached / failed counts, d
python -m lsab.featurize --retry-failures --rep t5 --dataset shields
```
`--dataset` pulls the SMILES from `lsab.datasets.load(name).components` and
calls `embed_component` per component. Progress: one line per (rep, dataset,
component) with counts and elapsed seconds; failures printed at the end.
Uses `DEFAULT_DATASETS` for `--all-datasets`.

---

## Dropped, on purpose (the old zoo)
phys / MH-1 / POLAR, reactive-site and two-scale pooling, RFF and every
`pooling=` argument, conformer ensembles / Boltzmann weighting, mordred, ohe,
chemeleon (and with it the tutorial's `base/` import path and `sys.path`
hacks), `normalize=` global/local, `last_layer_only`, `--quick`, the
`PretrainedWrapper` dtype juggling, `_SYMBOL` tables (the error just prints
atomic numbers), the AIMNet2 "dict" method, `aimnet2_descriptor_frame`,
`descriptor_frame` / `baybe_parameter` (module 3 builds matrices, not BayBE
parameters).

---

## Tests (`tests/test_featurize.py`) — no model weights needed except the `slow` set

1. `test_featurize_import_is_light`: after `import lsab.featurize`, none of
   `torch`, `mace`, `aimnet`, `transformers`, `rdkit` is in `sys.modules`.
2. `test_morgan`: shape `(n, 2048)`, float32, values in {0,1}, deterministic,
   two different molecules differ, an unparsable SMILES raises.
3. `test_conformer_helper`: hydrogens added; numbers int64, coords float32
   `(N,3)`; deterministic across two calls; `optimize=False` differs from
   `True` for a flexible molecule; unparsable SMILES raises ValueError; charge
   correct for `[NH4+]`.
4. `test_cache_roundtrip` with a fake featurizer registered in the test
   (monkeypatch `REPS`): first `embed` builds the model and computes; second
   call does not build (count constructor calls) and returns identical
   values; input order and duplicates handled; a molecule the fake raises on
   comes back as a NaN row, lands in `failures.json`, and is NOT retried on
   the next call; the cache directory contains no `*.tmp` afterwards.
5. `test_cache_survives_interruption`: fake featurizer raises `KeyboardInterrupt`
   on the 150th molecule; the cache holds the first 100 (the periodic flush).
6. `test_coverage`: `uncovered_elements("mace_off23", ["[Cs+].[O-]C(=O)C"]) == {55}`;
   `uncovered_elements("t5", ...) == set()`; `embed_component("mace_off23",
   shields_bases)` (fakes for both reps) returns `used_rep == "mace_mp0"` and
   warns once; a rep with no fallback and an uncovered element raises.
7. `test_real_featurizers` marked `slow` and skipped unless the stack imports
   and weights are present: for each of mace_mp0, mace_off23, aimnet2,
   aimnet2_all, t5, chemberta, embed `["CCO", "c1ccccc1"]` → finite, float32,
   `(2, d)` with `d` consistent, the two rows differ, and embedding `"CCO"`
   twice is `np.array_equal` on CPU.

## Definition of done
Tests 1–6 pass in CI without weights; test 7 passes in `aimnet-bo` on CPU for
all six; `python -m lsab.featurize --rep morgan --dataset bh_reaction_1` runs
in seconds; `--status` works on an empty cache; `featurize.py` under ~250
lines, each featurizer module under ~120; no import from the old tree; the
README gains a "computing the embeddings" section (env, weights, the CLI,
GPU non-determinism note) and a "differences from the old runs" list with the
three entries above (MACE dtype, ChemBERTa pooling, T5 checkpoint).

## When it's reviewed
Send me: the merged `RepSpec`/`embed`/`embed_component` signatures, the `d`
of each real featurizer from test 7, and the elapsed time per molecule for
the two MACE models and AIMNet2 on CPU. Module 3 (matrices + reduction) is
written against those.
