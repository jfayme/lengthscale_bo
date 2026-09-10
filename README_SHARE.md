# Lengthscale A/B — shareable bundle

A paired comparison of two rules for where the GP lengthscale prior is centred,
with everything else in the Bayesian-optimisation pipeline held fixed.

| rule | centre | |
|---|---|---|
| `chen` | `0.4*sqrt(d) + 4` | the published dimension-aware prior (`base/kernels.py::AdaptiveKernelFactory`); a function of the feature dimension only |
| `geom` | `D_bar / 1.221810` | the pool's mean pairwise distance / `u* = (1+sqrt3)/sqrt5`, the argmax of `u*|k'(u)|` for **Matern-5/2**. Change the kernel and `u*` must change with it |

crossed with how the prior's *width* is set around that centre, because Chen's
`Gamma(2*l0, 2)` has CV `1/sqrt(2*l0)` — re-centring it silently changes how hard
it pulls:

* `match_parameterisation` — reuse `Gamma(2*l0, 2)` verbatim (centre and strength move together)
* `match_concentration` — fix the CV in both arms, so **only** the centre differs (the clean comparison)

`D_bar` is computed in **the space the kernel sees** (post-reduction, post
`Normalize`). Every arm of a given (dataset, representation, reduction, seed)
starts from the **identical** initial design, drawn from the seed before any rule
is applied.

## Quick start

```bash
python -m unittest test_lengthscale_ab -v     # 14 acceptance tests, ~50 s
python lengthscale_ab.py --dry-run            # the matrix + a cost estimate
python lengthscale_ab.py --datasets bh_reaction_1 --reps morgan --seeds 3 --iter 20
python analyze_lengthscale_ab.py --csv run_output/lengthscale_ab*.csv
```

The sweep is **resumable** (completed rows are skipped) and **shardable** — run
`python lengthscale_ab.py --shard 0/4` … `3/4` in four shells, each with its own
`--csv`.

## What is in here

### The A/B
| file | role |
|---|---|
| `lengthscale_rules.py` | both rules, the model-space distance helper, the degeneracy check, `USTAR` per kernel, and the two prior parameterisations |
| `lengthscale_ab.py` | the sweep: dataset x representation x reduction x rule x prior mode x seed; one row per campaign |
| `analyze_lengthscale_ab.py` | family-clustered paired analysis (Wilcoxon on family means), per-representation split, two diagnostic scatters, markdown summary |
| `test_lengthscale_ab.py` | acceptance tests, stdlib `unittest` |

### The BO pipeline it imports
| file | role |
|---|---|
| `botorch_bo.py` | the workhorse: pool BO, the GP, and `LengthscaleSpec` — the single seam the A/B varies |
| `af_selection.py` | acquisition-function rule book (top-level import; the A/B itself uses static qLogEI) |
| `hyperprior_general.py` | lengthscale-prior strategies + dataset/campaign loading helpers |
| `gollum_pipeline.py` | dataset registry, dimensionality reduction, **the embedding disk cache** |
| `benchmark_representations.py` | the BayBE-side driver (Shields) and the A/B kernel-factory pass-through |
| `representations.py` | `make_fingerprinter(name)` — the dispatcher every representation goes through |
| `HSF-ChemBO-tutorial/base/kernels.py` | the published Chen prior — the specification for arm A |
| `HSF-ChemBO-tutorial/base/utils.py` | Shields column normalisation |

### Embedding generation (only needed to featurise NEW molecules)
These are imported **lazily, on a cache miss only** (`gollum_pipeline.embed_smiles`).
With the embedding cache in place they are never touched.

| file | representations it produces | needs |
|---|---|---|
| `representations.py` | `morgan`, `ohe`, `mordred` directly | rdkit, mordred |
| `aimnet2_repr.py` | `aimnet2`, `aimnet2_all` | `aimnet` (aimnetcentral), torch, rdkit |
| `mace_repr.py` | `mace_mp0`, `mace_off23`, `mace_mh1` | `mace-torch`, ase, rdkit |
| `reactive_repr.py` | `mace_mp0_rxn`, `mace_off23_rxn`, `mace_mp0_2scale` | as `mace_repr` |
| `phys_descriptor.py` | `phys` (MH-1/POLAR physical descriptor) | `mace-torch` + MH-1/POLAR weights |
| `conformer_embedding.py`, `conformer_ab.py` | conformer generation/pooling used by the above | rdkit ETKDG, torch |
| `pooling.py` | pooling utilities shared by the MLIP embedders | numpy |
| `base/pretrained_repr.py`, `base/llm_utils.py` | `t5`, `chemberta`, `chemeleon` | transformers, chemprop |

### Data
| path | what |
|---|---|
| `gollum/data/` | the dataset CSVs (additives, buchwald-hartwig, molecules, suzuki-miyaura, reasoning) |
| `HSF-ChemBO-tutorial/shields_dataset.xlsx` | the Shields lookup table |
| `tests/reference_default_run.json` | the stored reference for `test_default_unchanged` |

## Embeddings: copy the cache, or recompute

**Recommended — copy the cache.** Drop the sender's `gollum_out/_emb_cache/`
(~99 MB, `emb__<rep>.pkl`, keyed by SMILES) into `gollum_out/_emb_cache/` here.
Nothing in the MLIP stack is then needed: no `mace`, no `aimnet`, no model
weights, no HuggingFace downloads. This is how the published results were produced.

**To recompute instead**, you need, beyond the env below:

* `pip install mace-torch==0.3.16` (or the in-repo `mace/` source tree, 282 MB)
* `aimnetcentral` from https://github.com/isayevlab/aimnetcentral (29 MB, editable install)
* MACE weights in `~/.cache/mace/`: the foundation models auto-download, but
  `mace-mh-1.model` (59 MB) and `MACEPOLAR1M/1S` (68/33 MB) — used by `phys` —
  do not, and must be fetched separately
* `transformers` + `chemprop` for `t5` / `chemberta` / `chemeleon`

The cache fills incrementally and is written after every molecule, so an
interrupted embedding run resumes without losing work. Caveat: on torch 2.12 the
MACE model load can trip `weights_only` unpickling (`WeightsUnpickler error:
Unsupported global: GLOBAL slice`); either allowlist it with
`torch.serialization.add_safe_globals([slice])` or use the cache.

`morgan`, `ohe` and `mordred` need none of this — they recompute from rdkit in
seconds. (Running the test suite in a fresh copy of this bundle regenerates
`emb__morgan.pkl` from scratch; that is the expected behaviour.)

## Environment

```
torch 2.12.0+cpu   botorch 0.18.0   gpytorch 1.15.2   baybe 0.12.2
scikit-learn 1.9.0   pandas 2.3.3   scipy 1.14.1   numpy 1.26.4
rdkit 2025.09.3   openpyxl 3.1.5   matplotlib 3.10.9 (figures only)
```

Pin these if you need bit-for-bit agreement: the initial designs and the torch
seed per campaign are deterministic, but BoTorch/GPyTorch version drift can shift
the marginal-likelihood fit slightly, so `ell_fitted_*` may differ in the last
digits otherwise.

## Output

One row per campaign (not pre-aggregated) with the prior centre, the pool
geometry, the fitted lengthscale, and the metrics:

```
dataset, family, representation, reduction, lengthscale_rule, prior_mode, seed,
d, n_pool, D_bar, ell_star, ell_0, ell_0_over_ell_star,
ell_fitted_mean, ell_fitted_log_sd, ell_fitted_over_ell_star,
auc, simple_regret, top5_coverage, auc_random, lift
```

`lift = (auc - auc_random) / (1 - auc_random)` is the headline: raw AUC is
dominated by how easy a dataset is. Cells whose pairwise distances are all equal
(a one-hot single-component pool — every candidate ties and the campaign picks in
file order) are flagged `degenerate` and excluded from aggregates.
