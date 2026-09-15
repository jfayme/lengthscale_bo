# 00 — The rewrite-by-hand plan (start here)

## What this repo is

A paired A/B on **where to centre a GP's lengthscale prior** in pool-based Bayesian
optimisation over molecular datasets, with everything else held fixed:

- **chen** — `ell_0 = 0.4*sqrt(d) + 4`, the published prior. Blind to the pool.
- **geom** — `ell_0 = D_bar / u*`, the pool's mean pairwise distance over a
  kernel-matched constant (`u* = 1.221810` for Matérn-5/2).

crossed with how the Gamma prior's *width* is set around that centre
(`match_parameterisation` = Chen's tied `Gamma(2*ell_0, 2)`, whose width follows the
centre; `match_concentration` = fixed CV, so only the centre differs). The full matrix
is 6 datasets × 6 representations × 2 reductions × 2 rules × 2 prior modes × 10 seeds
= **2,880 campaigns** of 50 experiments each.

The working code is the **`lsab/` package: six modules, ~2,200 lines.**

| module | file | job |
|---|---|---|
| 1 | `lsab/datasets.py` | dataset name → `Pool`: measured candidates, per-component SMILES, a maximise objective |
| 2 | `lsab/featurize.py` + `featurizers/` | SMILES → embedding matrix per representation, behind one disk cache |
| 3 | `lsab/reduce.py` | pool + rep + reduction → the float64 matrix the GP sees, with per-block metadata |
| 4 | `lsab/lengthscale.py` | the chen and geom rules, model space, the Gamma prior, the preflight |
| 5 | `lsab/bo.py` | one pool-BO campaign: the GP, analytic LogEI, the fitted-lengthscale record, the metrics |
| 6 | `lsab/sweep.py` | the A/B matrix as a resumable, shardable task list, one CSV row per campaign |

**Module 7 (paired analysis, family clustering, figures) does not exist yet.** See
`08_after_module6_fixes_and_run_plan.md` §5 for what it needs.

## The goal of this session

Rewrite `lsab/` **by hand, line by line**, as a learning exercise — so the author
understands every decision in the code, and knows exactly what is being sent to a
collaborator. The existing `lsab/` is correct and reviewed; this is not a redesign.

## The setup that makes this work

Three assets are already in the repo:

| asset | role |
|---|---|
| `01`–`06_*_brief.md` (1,298 lines) | **the spec.** These were written *before* the code — job, contract, algorithm, dropped-on-purpose list, and the tests to write, per module. |
| `08_after_module6_fixes_and_run_plan.md` | the post-review fix list and the road to the real run. |
| `lsab/` on branch `main` | **the answer key.** |
| `tests/` (53 tests, 6 modules) | **the checker.** |

Recommended mechanics — rewrite *in place* on a branch, so `tests/` runs unmodified
(it imports `from lsab import ...`) and the reference is always one command away:

```bash
git switch -c rewrite
git rm lsab/*.py lsab/featurizers/*.py

git show main:lsab/datasets.py       # the answer key, on demand
git diff main -- lsab/               # everything you diverged on, at the end
```

**One wrinkle:** each brief opens with a "Replaces (read these first)" section naming
modules of the old flat tree — `gollum_pipeline.load_dataset`, `botorch_bo.load_pool`,
`representations.make_fingerprinter`, and so on. Those files were deleted. They are in
git history and can be read without checking anything out:

```bash
git show 16134a1:gollum_pipeline.py
git log --diff-filter=D --name-only --oneline -1    # everything that was removed
```

Reading them is optional — each brief *specifies* the behaviour it wants, so the spec
stands alone. They are useful only when a brief's "why" is unclear.

## Install — staged to match the rewrite order

**The `aimnet-bo` conda env is currently bare** (Python 3.11.15 + pip/setuptools only).
Nothing runs until this is fixed. It can be staged, so the early modules need no torch:

```bash
PY="D:/Conda/envs/aimnet-bo/python.exe"
$PY -m pip install numpy scipy "pandas>=2.1" openpyxl rdkit   # modules 1-4 + the test suite
$PY -m pip install torch==2.12.0 botorch==0.18.0              # module 5 (gpytorch comes with botorch)
```

Modules 1–4 and **the entire default test suite** run on the first line alone: every
test that would need model weights either monkeypatches a fake featurizer or is gated
behind `LSAB_SLOW=1`. `transformers`/`sentencepiece` and `mace-torch`/`aimnetcentral`
are only needed to actually compute embeddings.

`openpyxl` is not an import anywhere — `pandas.read_excel` needs it for
`shields_dataset.xlsx`, and two test modules load Shields.

## Order, and where to start

Dependency order, which is also brief order:

**datasets → featurize → reduce → lengthscale → bo → sweep**

**Start with `lsab/datasets.py`.** Zero dependencies on other `lsab` modules, the
lightest install, 5 tests, and it is the module a collaborator will scrutinise hardest,
because dataset provenance is where silent errors hide.

Write the **contract first** — two frozen dataclasses everything downstream depends on:

- `DatasetSpec` — one registry entry: `name`, `family`, `path`, `target`, `direction`,
  `components`, `numeric`, `element_filter`. The point is that *every dataset quirk is
  a field here*, so `load()` has no `if name == "shields"` branches.
- `Pool` — what `load()` returns: `spec`, `frame`, `components`, `numeric`, plus an
  `objective` property that is always maximise-oriented.

Then the registry (18 datasets), then `load()`'s four row rules, then `validate()`.

## The working loop

For each section: **the author writes it from the brief**, then Claude reviews it
against the reference and explains *why* the reference diverges where it does — the
interesting parts are always the non-obvious decisions (the regex element filter
instead of RDKit; `eq=False` on the dataclasses; first-measurement-wins on duplicates;
variance-ordered decorrelation). Then run that module's tests until green, and move on.

```bash
$PY -m unittest tests.test_datasets -v
```

## Do not break

- `gollum/data/` and `HSF-ChemBO-tutorial/shields_dataset.xlsx` — the datasets.
  `lsab/datasets.py` sets `DATA_ROOT` to the repo root and reads 16 files from them.
- **`tests/old_pool_snapshot.json`** — the old tree's ground truth (row counts,
  objective min/max/mean, unique SMILES per component). `tests.test_datasets` checks the
  loader against it. The old tree and the script that wrote it were deleted; this file
  **cannot be regenerated**.
- `tests/*.py` and the briefs.

## State as of this session

- The flat old tree at the repo root (17 modules, ~6,900 lines: `botorch_bo.py`,
  `lengthscale_ab.py`, `representations.py`, the dynamic-acquisition rule book, the
  conformer/phys/pooling experiments) was **deleted**. It is in git history at
  commit `16134a1` and earlier. `lsab/` never imported any of it — only the data files.
- `auc_random` and `lift` were added to module 5 and the sweep's row schema
  (`lift = (auc - auc_random)/(1 - auc_random)`, the headline metric). `ROW_FIELDS` is
  now 43 fields. No run CSVs exist, so nothing was invalidated.
- **The test suite has never been executed on this machine** — the env is bare. Running
  it green for the first time is a genuine open item, not a formality.

## Machine

AMD Ryzen AI 9 HX 370 (12 cores / 24 threads), 63 GB RAM, RTX 5090 Laptop (24 GB).

**The GPU is nearly irrelevant here.** `lsab/bo.py` has no device parameter at all —
the GP, the fit and the LogEI argmax are CPU-only by construction, so the ~10-hour
sweep never touches it. The only GPU-capable stage is embedding, and the default matrix
is ~1,950 unique molecules ≈ 22 minutes on CPU, cached once. What actually helps is the
core count: the README's 8-shard recipe becomes `--shard i/12`.

A CUDA torch build for Blackwell (`sm_120`) must be cu128 or newer, or it will not see
the card. Given the above, **CPU torch is the sane default.**
