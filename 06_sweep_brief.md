# Rewrite — module 6: `lsab/sweep.py`

Inputs as merged: `datasets.load`, `reduce.build`, `lengthscale.pool_geometry`
/ `make_prior`, `bo.Campaign` / `run_campaign` / `metrics` / `CampaignError`.

## Job
Run the A/B matrix — cells × arms × seeds — as a flat task list that can be
interrupted, resumed, and split across processes, writing one CSV row per
campaign with a schema that is written down once and checked on every append.
Nothing here computes anything; it schedules and records.

Read the old `lengthscale_ab.py` (`Cell`, `sweep`, `ROW_COLUMNS`,
`KEY_COLUMNS`, the resume path, `--shard`) before writing. Keep what it got
right: the seed-major loop, resume by key, sharding. Fix what it got wrong: the
schema was only frozen by a comment, and a header mismatch on resume silently
misaligned rows.

## Vocabulary
- **Cell** = `(dataset, reduction, rep)`: one `FeaturePool` and one `Geometry`;
  everything that depends only on the pool.
- **Arm** = `(rule, prior_mode, cv)`: one `Prior` per cell.
- **Task** = `(seed, cell, arm)`: one campaign, one row.

## Decisions in force
- **Task order is seed-major: `for seed: for cell: for arm`.** An interrupted or
  partially sharded run then has every cell at seeds `0..k` with both arms, so
  the paired analysis never sees an arm with more seeds than its partner. The
  cell is rebuilt per seed (a cache read plus a reduction — negligible next to
  a campaign; do not cache built cells across seeds, the 10k pools would cost
  gigabytes).
- **Sharding is round-robin over the task list in that order**, `--shard i/k`,
  each shard writing its own file `<out>.shard<i>of<k>.csv`. Consecutive tasks
  are the two arms of the same (seed, cell), so round-robin keeps every shard
  balanced within one task. No locking, no shared file; the analysis
  concatenates shards.
- **One thread per process by default.** `torch.set_num_threads(args.threads)`
  with `--threads 1`, and print the value at start. Running `k` shards on `k`
  cores is the intended parallelism; `--threads` exists for a single unsharded
  run.
- **Resume by key.** Before running, read every existing row of the target
  file (and, for a sharded run, only that shard's file), skip tasks whose key
  is present. Append one row per campaign, flush after each.
- **Schema guard.** `ROW_FIELDS` is a tuple constant. A new file gets the
  header; an existing file must have exactly that header or the run refuses
  with a message naming the file and the first differing column. Never append
  `header=False` to an unchecked file.
- **A failed campaign is a row**, not a crash: `failed = True`,
  `fail_iteration`, metric columns NaN, `error` = first 200 chars of the cause.
  The sweep continues. The old analysis found rule-specific failure counts
  interesting; this keeps them.
- **Degenerate cells are skipped** (no row) with one printed line, unless
  `--include-degenerate`.
- **Default matrix** (sized from the module-5 timings; the 10k pools and
  `none` are opt-in):
  `--datasets DEFAULT_DATASETS`, `--reps morgan mace_mp0 mace_off23 aimnet2_all t5 chemberta`,
  `--reductions decorr0.7 pca64`, `--rules chen geom`,
  `--prior-modes match_concentration match_parameterisation`, `--cv 0.3`,
  `--seeds 10`, `--seed-offset 0`, `--iter 50`, `--init 5`, `--seed-base 1337`.
  That is 6 × 6 × 2 × 2 × 2 × 10 = 2,880 campaigns; at 20–60 s each, roughly
  a day single-threaded, a few hours on eight shards.

## Row schema — `ROW_FIELDS`, in this order

```
# key (uniquely identifies a task; KEY_FIELDS is this prefix)
dataset, reduction, rep, rule, prior_mode, cv, seed, n_init, n_iter, seed_base
# provenance
family, timestamp, lsab_commit           # git rev-parse --short HEAD if available else ""
# cell
n, d, used_reps, n_failed_embed, D_bar, geom_cv, degenerate, ell_geom, ell_chen
# arm
ell_0, concentration, rate
# design
init_indices                              # "12;45;301;..." — the n_init indices; lets the analysis verify pairing
# outcome (from bo.metrics)
auc, coverage_top5, simple_regret, best_found, first_top5_hit,
ell_fitted_final, ell_fitted_traj, ell_fitted_log_sd_final, n_fit_warnings, seconds
# status
failed, fail_iteration, error
```
`used_reps` is one string, `"Aryl_halide=morgan;Base=mace_mp0;..."`, in block
order, so the schema does not vary by dataset. `family` comes from
`pool.spec.family`. Booleans written as `True`/`False`, NaN as empty.

## Contract

```python
ROW_FIELDS: tuple[str, ...]
KEY_FIELDS: tuple[str, ...]              # the first 10

@dataclass(frozen=True)
class Arm:   rule: str; prior_mode: str; cv: float
@dataclass(frozen=True)
class CellKey: dataset: str; reduction: str; rep: str
@dataclass(frozen=True)
class Task:  seed: int; cell: CellKey; arm: Arm

def task_list(datasets, reductions, reps, rules, prior_modes, cv, seeds, seed_offset) -> list[Task]   # seed-major order
def shard(tasks: list[Task], i: int, k: int) -> list[Task]                                              # tasks[i::k]
def row_key(task, n_init, n_iter, seed_base) -> tuple
def read_done_keys(path) -> set[tuple]                                                                   # {} if no file; raises on header mismatch
def run_task(task, *, n_init, n_iter, seed_base, device, cache_dir) -> dict                              # one full row (builds the cell, the prior, runs, or records the failure)
def sweep(tasks, out_path, *, n_init, n_iter, seed_base, include_degenerate, device, cache_dir, dry_run) -> int   # rows written
```

`run_task` builds the cell fresh: `load` → `build` → `pool_geometry` →
`make_prior(fp, arm.rule, arm.prior_mode, arm.cv, geometry=g)` → `Campaign`
→ `run_campaign` → `metrics`. Both arms of the same (seed, cell) therefore see
the identical `FeaturePool` bytes and the identical `init_indices` — the
pairing the analysis relies on.

## CLI

```
python -m lsab.sweep --out runs/ab_v1.csv --dry-run
python -m lsab.sweep --out runs/ab_v1.csv --datasets bh_reaction_1 shields --reps morgan --seeds 2   # pilot
python -m lsab.sweep --out runs/ab_v1.csv --shard 0/8      # ... one process per shard
python -m lsab.sweep --out runs/ab_v1.csv --status          # rows done / tasks total, per cell, across shard files
```

`--dry-run` writes nothing and prints: the task count after resume
subtraction, the cells with their `d` and both prior centres, and an estimated
wall time. The estimate uses the median `seconds` of existing rows for the
same `(dataset, reduction, rep)` across all shard files of `--out`; cells with
no rows print `?` and are excluded from the total. Say that in the output.
Nothing cleverer: a `d`-based guess would be wrong by an order of magnitude on
the pools we have not timed.

Progress: one line per task: key, `ell_0`, then `auc`/`seconds` or
`FAILED@<it>`. Start-up prints thread count, output file, tasks done / total.

A shell one-liner in the README for eight shards:
`for i in $(seq 0 7); do python -m lsab.sweep --out runs/ab_v1.csv --shard $i/8 > runs/ab_v1.shard$i.log 2>&1 & done; wait`

## Dropped, on purpose
The old `ROW_COLUMNS`/`KEY_COLUMNS` and its `run_output/` (new schema, new
directory; the ~2,300 old rows are not migrated — a different acquisition,
decorrelation and T5/ChemBERTa featurisation make them a different
experiment). `Cell.initial_indices` living on a class with mutable state;
`FAMILY` regexes (family is data now); `--skip-degenerate` (skip is the
default); the `chunked` seed loop; `--rank` in the sweep (it is a preflight
diagnostic, and an SVD per task is waste).

## Tests (`tests/test_sweep.py`) — monkeypatch `run_campaign` with a fake that returns a deterministic `CampaignResult` in milliseconds
1. `test_task_order_is_seed_major`: for 2 seeds × 2 cells × 2 arms, the list
   is `[(0,c0,a0),(0,c0,a1),(0,c1,a0),(0,c1,a1),(1,...)]`.
2. `test_shard_partitions_and_balances`: shards `0..k-1` are disjoint, their
   union is the list, and within each shard every `(rule, prior_mode)` count
   differs by at most one.
3. `test_resume_by_key`: sweep 3 tasks into a temp file; sweep again → 0 rows
   written; add a seed → only the new tasks run; the header appears once;
   every row has exactly `len(ROW_FIELDS)` fields.
4. `test_schema_guard`: a file whose header lacks one column → `read_done_keys`
   raises naming the column; the sweep does not append.
5. `test_failed_campaign_is_a_row`: fake raises `CampaignError(iteration=7)`
   for one arm → that row has `failed == True`, `fail_iteration == 7`, NaN
   `auc`, and the other arm's row is normal.
6. `test_degenerate_skipped`: monkeypatch `pool_geometry` to return
   `degenerate=True` → no row, one printed line; `include_degenerate=True`
   → the row is written.
7. `test_pairing_is_real`: on `bh_reaction_1` with the module-3 fake rep, one
   seed, both rules: the two rows have identical `init_indices`, identical
   `n`/`d`/`D_bar`, and different `ell_0`.
8. `test_dry_run_writes_nothing`.

## Definition of done
Tests 1–8 pass; the pilot command above runs to completion on the real morgan
cache (2 seeds × 2 datasets × 2 reductions × 2 rules × 2 modes = 32
campaigns) and `--status` reports 32/32; `--dry-run` on the default matrix
prints 2,880 tasks; `sweep.py` under ~250 lines; README gains a "running the
A/B" section with the pilot, the shard one-liner, the thread note, and the
default matrix with its cost.

## When it's reviewed
Send me the pilot's 32 rows (the CSV) and the `--dry-run` estimate for the
default matrix. Module 7 (paired analysis + figures) is written against the
schema above and I'd rather write it looking at real rows.
