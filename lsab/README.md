# lsab

The lengthscale A/B, rewritten as a package. The old tree at the repo root stays
next to it, read-only, until module 7 lands.

| module | job |
|---|---|
| `datasets.py` | dataset name -> `Pool`: measured candidates, per-component SMILES / numeric settings, a maximise objective |
| `featurize.py` + `featurizers/` | SMILES -> embedding matrix per representation, behind one disk cache |
| `reduce.py` | pool + representation + reduction -> the float64 matrix the GP sees, with per-block metadata |
| `lengthscale.py` | the chen and geom rules, the model space, the Gamma prior, and the preflight |
| `bo.py` | one pool-BO campaign: the GP, analytic LogEI, the fitted-lengthscale record, the metrics |

```bash
python -m lsab.datasets                     # one line per registered dataset
python -m lsab.featurize --status           # what is in the embedding cache
python -m lsab.reduce --rep morgan --reduction decorr0.7   # D and its blocks, every dataset
python -m lsab.lengthscale --rank           # the preflight: geometry and both prior centres per cell
python -m lsab.bo --dataset bh_reaction_1 --rep morgan --rule geom --iter 20   # one campaign, eyeballed
python -m unittest tests.test_datasets tests.test_featurize tests.test_reduce tests.test_lengthscale tests.test_bo -v
python tools/snapshot_pools.py              # (old tree, once) regenerate tests/old_pool_snapshot.json
```

**The A/B is chen against geom.** chen centres the lengthscale prior at
0.4*sqrt(d) + 4; geom centres it at the pool's mean pairwise distance divided by
the Matern-5/2 constant u*. diamgate, the old per-iteration house prior, is not
ported.

**Not registered:** `gollum/data/` also ships c2-yield (catalyst composition as
elements and molar ratios), hplc (six continuous process parameters), oer (six
elemental loadings) and vapdiff (organics given by name, mostly process variables).
None has a SMILES per variable, so none can be a molecular pool.

## Computing the embeddings

**Environment.** The `aimnet-bo` conda env is the reference: torch 2.12 (CPU),
mace-torch 0.3.16, aimnet (aimnetcentral), transformers 4.57, ase 3.28,
rdkit 2025.09. `morgan` needs only rdkit; each other representation needs only
its own stack, because `lsab.featurize` imports a stack when that representation
is first used.

**Weights.**

| rep | weights | where they come from |
|---|---|---|
| `mace_mp0`, `mace_off23` | MACE-MP-0 / MACE-OFF23 "medium" | auto-download to `~/.cache/mace/` |
| `aimnet2`, `aimnet2_all` | AIMNet2 wB97M-D3 | auto-download to `~/.cache/aimnet/` |
| `t5` | `GT4SD/multitask-text-and-chemistry-t5-base-augm` | HuggingFace cache (`HF_HOME`) |
| `chemberta` | `seyonec/ChemBERTa-zinc-base-v1` | HuggingFace cache (`HF_HOME`) |

Set `HF_HUB_OFFLINE=1` to forbid downloads and use only what is cached.

**The CLI.**

```bash
python -m lsab.featurize --rep mace_mp0 aimnet2_all --dataset bh_reaction_1 photoswitches
python -m lsab.featurize --all-reps --all-datasets --device cuda   # DEFAULT_DATASETS
python -m lsab.featurize --status                                  # cached / failed counts and d
python -m lsab.featurize --retry-failures --rep t5 --dataset shields
```

It prints one line per (rep, dataset, component) and lists failed molecules at
the end. A component containing an element the rep cannot embed (the Shields
Cs/K bases under `mace_off23` or `aimnet2`) is embedded whole with the rep's
fallback, `mace_mp0`, and says so.

**The cache** lives in `embeddings/<rep>/` (override with `--cache-dir` or the
`LSAB_CACHE` env var; it is git-ignored). It holds `vectors.npz` (the SMILES and a
float32 matrix) and `failures.json` (molecules that could not be embedded, with
the error, so a run does not retry them). It is written every 100 new molecules,
so an interrupted run keeps its work. Three rules:
- **Nothing about model size or device is in the path.** If you change a
  representation's registry entry, delete its directory.
- **Run one process per representation at a time.** Two processes writing the
  same directory overwrite each other's work.
- **The old 99 MB `gollum_out/_emb_cache/` is incompatible** and is not migrated:
  the rewrite re-runs every result.

**CPU cost**, measured on the 32 molecules of bh_reaction_1 (median 19 atoms,
largest 107), including conformer generation:

| rep | d | mean per molecule | median |
|---|---|---|---|
| mace_mp0 | 256 | 0.28 s | 0.13 s |
| mace_off23 | 256 | 0.25 s | 0.13 s |
| aimnet2 | 256 | 0.07 s | 0.02 s |
| aimnet2_all | 772 | 0.07 s | 0.02 s |
| t5 | 768 | 0.05 s | 0.05 s |
| chemberta | 768 | 0.03 s | 0.02 s |

**Determinism.** On CPU every representation is bit-for-bit repeatable: the
conformer is seeded (42) and the models run in eval mode. On GPU the last bits
can differ between runs. That is not worth fixing for a GP feature, but do not
expect a GPU cache to match a CPU cache exactly.

## The GP has no outputscale

The kernel is a bare Matern-5/2 with one lengthscale per dimension, not wrapped in
a `ScaleKernel`. Outcomes go through `Standardize`, so they have unit variance and
the signal variance is about 1 by construction. An outputscale, and the prior it
would need, would put back an amplitude factor that the A/B holds fixed, and it
would interact with the lengthscale prior the A/B is varying. The old `build_gp`
did the same.

## Failed molecules: impute or raise

A molecule a representation cannot embed keeps its place in the pool. Its row
is filled with the per-column median of the molecules in the same component that
did embed. The candidate pool must be identical for every representation;
dropping a candidate would make the A/B compare different pools across
representations. If more than 1% of a component's molecules fail, `lsab.reduce`
raises and lists them instead. At that point the representation is not being
compared, it is being patched. For a component of fewer than 100 molecules this
means a single failure raises. The old code dropped some failures and imputed
others depending on the kind of failure; this one rule replaces both paths.

## Differences from the old runs

1. **MACE dtype.** Both MACE models run in float32. The old code ran MACE-OFF23
   in float64 and MACE-MP-0 in float32 for no recorded reason.
2. **ChemBERTa pooling.** The HSF-ChemBO baseline summed `hidden_states[0]`, the
   token-embedding layer before any transformer block, so no transformer block
   ever touched the vector. The rewrite takes the masked mean of the last hidden
   state, exactly like T5.
3. **T5 checkpoint.** The published GOLLuM runs used `t5-base`, a generic English
   model, because the chemistry checkpoint would not download on that machine.
   The rewrite uses `GT4SD/multitask-text-and-chemistry-t5-base-augm` everywhere.
4. **Variance-ordered decorrelation.** `decorr0.7` visits columns in descending
   variance and keeps one unless it correlates above 0.7 with a column already
   kept. BayBE visits them in index order, so the survivor of a correlated
   cluster was whichever column came first; embedding columns have no meaning to
   their index. With index order the rewrite reproduces the old morgan `d` exactly on
   all 18 datasets; variance order leaves `d` equal or slightly larger (morgan:
   +26 on the additive plates, +11 on pce10k, +6 on photoswitches).
5. **No tie-breaking noise.** The old code added 1e-7 noise whenever two
   molecules had identical features, because BayBE rejects duplicate rows.
   Identical features are a fact about the representation; the GP's jitter
   handles them, so no noise is added.
6. **One geometry probe.** The pool's distances come from a single probe: every
   row up to 2,000, else a 2,000-row subsample drawn with seed 0 from a fresh
   generator. The old code had two probes: a 1,500-point mean for geom, and a
   2,000-point median and p95 for diamgate drawn from module-level random
   generators, so diamgate's subsample depended on call order.
7. **Analytic LogEI.** Each candidate is scored with BoTorch's analytic
   `LogExpectedImprovement`, one candidate at a time. The old runs used `qLogEI`
   with one candidate, which is the Monte Carlo approximation of the same
   quantity, so the rewrite removes sampling noise from the choice of experiment.
