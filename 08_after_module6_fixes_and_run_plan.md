# After module 6: review fixes (modules 1–6) and the road to the real run

Order: §1 → §2 → §3 → §4. One commit per section. Nothing in §1–§2 changes any
number; §3 is what makes `lsab_commit` mean something; §4 is the last cheap
step before ten hours of compute.

---

## §1 Fixes to modules 1–5 (unchanged from the earlier review)

1. **`featurize.embed`: width check against the cache.** Before
   `vectors[s] = vector`: if the cache is non-empty and `len(vector)` differs
   from the cached width, raise `RuntimeError` naming `folder` and both widths
   ("the registry entry changed; delete that directory"). A `RuntimeError`, not
   a recorded failure — it must stop the run. Test: two fakes of different
   widths on the same cache directory → the error names the directory, no
   `*.tmp` remains, `failures.json` is untouched.
2. **`featurize._write_atomic`: never leave a `.tmp`.** Wrap the write in
   `try / except BaseException: tmp.unlink(missing_ok=True); raise`.
   Test: monkeypatch `np.savez` to raise on the second call; assert no `.tmp`.
3. **`--retry-failures` clears the fallback chain.** Walk `REPS[rep].fallback`
   to `None` and unlink each `failures.json`. One README line.
4. **README environment line: pandas ≥ 2.1** (`DataFrame.map` in the element
   filter). The regex filter stays, with `test_element_filter` as its
   permanent guard.
5. **`bo.run_campaign` docstring:** one sentence saying every iteration builds a
   fresh GP whose ARD lengthscales start at `ell_0`, i.e. each fit is
   warm-started from the prior centre, not from the previous fit — intended,
   and what makes the centre matter.

---

## §2 Fixes to module 6

### 2.1 The sweep must not embed — close the race the dry run only reports
`uncached()` is called in `--dry-run`. The sweep itself still calls
`reduce.build` → `embed_component` → `embed`, which loads a model and writes
`vectors.npz` on a miss. Eight shards hitting the same miss overwrite each
other's cache (the README forbids two writers per rep). So, in `sweep()` and
not only in the dry run:

- Before the first task, for every cell in **this process's** task list, run
  the same `uncached()` check the dry run uses. If anything is missing: print
  one line per (rep, dataset, component, n_missing), then the exact
  `python -m lsab.featurize --rep … --dataset …` command, and `sys.exit(2)`.
- `--allow-embedding` overrides; it is rejected together with `--shard`
  (`parser.error`). It exists for a single unsharded run on a machine that
  has the model stacks, nothing else.
- Test: monkeypatch `uncached` to report one miss → `sweep` exits 2 and writes
  no row; with `allow_embedding=True` and no shard it proceeds; with a shard
  it refuses.

### 2.2 A cell that cannot be built skips, it does not stop the shard
Today a build error (e.g. one component over the 1% embedding-failure limit
under one rep) ends the shard, and every later cell in that shard is lost until
someone notices. Instead: print `SKIP <cell>: <error>` (first 200 chars), skip
all of that cell's tasks in this run, continue, and exit non-zero at the end
if any cell was skipped, listing them. No row is written for a skipped cell
(the schema has no "cell failed" state, and inventing NaN rows there would look
like campaigns to the analysis). Test: monkeypatch `reduce.build` to raise for
one cell of two → the other cell's rows exist, the exit code is 1, the skipped
cell is named on stdout.

### 2.3 Thread pinning covers BLAS, not only torch
`torch.set_num_threads(1)` covers the fit and LogEI. The cell build
(`np.corrcoef`, the SVD, `pdist`) runs in numpy's BLAS, which reads
`OMP_NUM_THREADS` / `MKL_NUM_THREADS` / `OPENBLAS_NUM_THREADS` **at import
time**. So:
- `sweep.main` reads `--threads` from `sys.argv` first and does
  `os.environ.setdefault(var, str(threads))` for the three variables **before**
  any numpy/torch import — keep those imports inside `main` (or below that
  block) with a comment saying why.
- The README shard loop sets it explicitly too, so it works whatever imports
  first:
  `for i in $(seq 0 7); do OMP_NUM_THREADS=1 python -m lsab.sweep --out runs/ab_v1.csv --shard $i/8 > runs/ab_v1.shard$i.log 2>&1 & done; wait`
- Start-up prints both `torch.get_num_threads()` and `OMP_NUM_THREADS`.
- Add `torch_threads` and `omp_threads` to `ROW_FIELDS`, after `lsab_commit`.
  (Yes, the schema changes: the pilot file is deleted and re-run in §3
  anyway. This is the last schema change; after §3 the header is frozen.)
- Test: `python -m lsab.sweep --dry-run --threads 1` in a subprocess prints
  thread count 1.
- README: your measurement that one thread beats torch's default ten on these
  GPs (41 s → 24 s for the same campaign) belongs next to the loop, as the
  reason.

### 2.4 Column comment on `n_fit_warnings`
It counts every warning raised inside `fit_gpytorch_mll` — gpytorch
`NumericalWarning`s for jitter included — not only optimisation failures. Say
so where `ROW_FIELDS` is defined, so the analysis does not treat it as a
failure count.

---

## §3 Commit, then re-run the pilot

`lsab/` is untracked, so every pilot row's `lsab_commit` is a commit that
contains none of this code. After §1–§2 pass the full suite:

1. `git add lsab tests tools .gitignore lsab/README.md` and commit
   ("lsab: the rewrite, modules 1–6"). Check `git status` shows no
   `embeddings/`, `gollum_out/`, `runs/`, `__pycache__/`.
2. Delete `runs/ab_v1.csv` and `runs/pilot.log`; re-run the pilot (7 min on one
   thread). Every row now carries the commit that produced it and the new
   schema.
3. `--status` reports 32/32; `git status` still clean of run outputs
   (`runs/` is git-ignored — confirm, or add it).

---

## §4 A one-seed timing pass before the full matrix

The dry run can only estimate the four pilot cells; the other 68 cells of the
default matrix are untimed, and the higher-`d` ones (photoswitches, redox_mer
under `decorr0.7`; the LM reps everywhere) will not run at the pilot's 13 s.
Before the 2,880-task run:

1. `python -m lsab.featurize --all-reps --all-datasets` (one process, one rep at
   a time is what the CLI already does). Report the failures list.
2. `python -m lsab.sweep --out runs/ab_v1.csv --seeds 1 --shard i/8` for
   `i = 0..7` with the README loop. That is 72 cells × 4 arms = 288 campaigns,
   seed 0 only — every cell gets a timing, every cell that cannot be built is
   found now, and none of it is wasted: the rows are seed 0 of the real run.
3. `python -m lsab.sweep --out runs/ab_v1.csv --dry-run` — now every cell has a
   median, so the estimate covers the whole matrix. Send me that output, the
   `--status` table, and the per-cell `seconds` medians (a 72-line table).

Only after that: the full matrix, same command without `--seeds 1`.

---

## §5 For module 7 — what I need in the project

Add `lsab/sweep.py`, `tests/test_sweep.py` and `runs/ab_v1.csv` (the §3 re-run
is fine; the §4 seed-0 rows are better) to the project. Module 7's brief
(paired analysis, family clustering, the fitted-vs-prior figure) will be written
against the real header and real rows, and its first section will be the
sanity checks on exactly those rows:

- `init_indices` identical across the four arms of every (seed, cell);
- `n`, `d`, `D_bar`, `ell_geom`, `ell_chen` identical across arms of a cell;
- `ell_0` equals `ell_chen` for chen rows and `ell_geom` for geom rows;
- `concentration / rate == ell_0` on every row;
- the sign and size of `auc(geom) − auc(chen)` per pair, and how far
  `ell_fitted_final` moved from `ell_0` in each arm — the first look at whether
  the prior did anything.

Do not start module 7 before those files are in the project.
