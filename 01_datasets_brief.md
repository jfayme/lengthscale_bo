# Rewrite — module 1: `lsab/datasets.py`   (supersedes the earlier brief)

Decisions in force: free redesign, results re-run; representations = morgan,
mace_mp0, mace_off23, aimnet2, t5, chemberta with working embedding code;
pure BoTorch, no BayBE, no dynamic acquisition policy.

Package name below is `lsab`; substitute as you like. New code goes in
`lsab/`; the old tree stays next to it, read-only, until module 7 lands.

---

## Step 0 — a sanity snapshot from the OLD tree (20 lines, once)

Not a bit-identity gate. It exists so a loader bug that silently drops rows or
flips a sign is caught. In the old tree, write `tools/snapshot_pools.py`:
for every dataset in `gollum_pipeline.DATASETS` plus `"shields"`, replicate the
first half of `botorch_bo.load_pool` (build the campaign with `"morgan"`, take
`searchspace.discrete.exp_rep` and the merged objective, keep measured rows)
and write `tests/old_pool_snapshot.json`:

```json
{"bh_reaction_1": {"n": 3955, "obj_min": 0.0, "obj_max": 99.4, "obj_mean": 31.72,
                   "components": {"Aryl_halide": 15, "Additive": 22, "Base": 3, "Ligand": 4}}}
```

(`components` = number of unique SMILES per component.) That is all.

---

## Module 1 — `lsab/datasets.py`

### Job
Turn a dataset name into a `Pool`: the measured candidates, their per-component
SMILES (or numeric settings), and a maximise-oriented objective. No BO, no
features, no torch. Imports: `pandas`, `numpy`, `dataclasses`, `pathlib`;
`rdkit` only inside `validate`.

### Replaces (read these first)
`gollum_pipeline.DATASETS` / `load_dataset` / `SKIPPED` and its target
standardisation; `benchmark_representations.load_shields`; the yield-matching
and `measured` filter inside `botorch_bo.load_pool`; `hyperprior_general._load`;
`FAMILY` and `DEFAULT_DATASETS` in `lengthscale_ab.py`; `_is_yield_dataset`.

The CSV column names, each dataset's MIN/MAX direction, the Shields
component / numeric columns, and the element filter applied to the molecule
pools come from the old code, not from memory.

### Contract

```python
@dataclass(frozen=True)
class DatasetSpec:
    name: str
    family: str                        # "bh", "additives", "suzuki", "photoswitches", ... — for family-clustered stats
    path: str                          # relative to DATA_ROOT
    target: str                        # source column
    direction: Literal["max", "min"]   # source direction; "min" is negated on load
    components: tuple[str, ...]        # SMILES columns; a molecule dataset has exactly one
    numeric: tuple[str, ...] = ()      # numeric setting columns (Shields: Temp_C, Concentration)
    element_filter: frozenset[int] | None = None   # keep only candidates whose SMILES use these elements (molecule pools; see note)

@dataclass(frozen=True)
class Pool:
    spec: DatasetSpec
    frame: pd.DataFrame                # one row per MEASURED candidate: columns = components + numeric + ["objective"]
    components: dict[str, list[str]]   # component -> unique SMILES, first-seen order
    numeric: dict[str, list[float]]    # numeric column -> sorted unique values

    @property
    def name(self) -> str
    @property
    def n(self) -> int
    @property
    def objective(self) -> np.ndarray  # float64, maximise

DATASETS: dict[str, DatasetSpec]       # the registry, in one place, no per-name branches
def names() -> list[str]
def load(name: str, root: Path = DATA_ROOT) -> Pool
def validate(pool: Pool) -> None       # raises on NaN objective, duplicate candidate rows, unparsable SMILES, empty component
```

Rules:
- `objective` is ALWAYS maximise; a `"min"` target is negated. The column is
  named `objective`, never `yield`.
- Rows = measured candidates only. The full product space is never
  materialised.
- Candidate order = source-file order after filtering. Deterministic; nothing
  downstream may rely on it.
- Special cases become `DatasetSpec` fields with a one-line WHY comment, not
  `if name == ...` branches. In particular the old `gollum_pipeline` filters
  molecule pools to the AIMNet2 ∩ MACE-OFF23 element set so every
  representation embeds the same pool; that is a property of the pool and
  belongs here as `element_filter`. Shields is NOT filtered (its Cs/K bases
  stay in the pool; how a representation that cannot embed them is handled is
  module 2's problem, not a dataset property).
- Port the old `FAMILY` mapping verbatim into `family`.

### Dropped, on purpose
`SKIPPED` (a README line instead), the `kind` field (a molecule pool is a
one-component reaction pool), every BayBE object, the `lookup` dict return
shape, `impute_mode`, `_is_yield_dataset` / the rule-book ceiling.

### Tests (`tests/test_datasets.py`)
1. `test_every_dataset_loads`: `load(n)` then `validate` for every `names()`.
2. `test_matches_old_snapshot`: `n`, `obj_min/max/mean` (to 6 places) and
   per-component unique counts equal `tests/old_pool_snapshot.json`.
3. `test_min_target_is_negated`: for every spec with `direction == "min"`,
   `pool.objective == -source_column` (read the CSV directly in the test).
4. `test_element_filter`: for a filtered molecule dataset, every SMILES in the
   pool uses only allowed elements; and the unfiltered count from the raw CSV
   is larger, so the filter is doing something.
5. `test_registry_is_consistent`: every spec's `family` is non-empty, `path`
   exists, `components` non-empty, `numeric` disjoint from `components`.

### CLI
`python -m lsab.datasets` prints one line per dataset: name, family, n,
component sizes, numeric, direction. Nothing else.

### Definition of done
Tests 1–5 pass; the CLI runs in under 5 s; the module imports nothing from the
old tree; the file is under ~200 lines including the registry.

### When it's reviewed
Send me the final `DatasetSpec` / `Pool` as merged (they are the input contract
for module 2) and I'll write the module 2 brief against them.
