"""
lsab/sweep.py
=============

Module 6 of the lengthscale-A/B rewrite: the A/B matrix as a flat task list that can
be interrupted, resumed and split across processes, one CSV row per campaign. It
schedules and records; it computes nothing. A cell is (dataset, reduction, rep), one
FeaturePool and Geometry; an arm is (rule, prior_mode, cv), one Prior; a task is
(seed, cell, arm), one campaign and one row.

Seed-major order (for seed: for cell: for arm) leaves an interrupted run with every
cell at seeds 0..k in every arm, so the paired analysis never sees an unpaired arm.
Shards take whole (seed, cell) groups round-robin: dealing out single tasks would give
each shard ONE arm whenever the shard count is a multiple of the arm count (8 shards,
4 arms), and chen and geom campaigns do not take equally long. Each shard writes its
own file; resume reads every file of the run, so a pilot and a later sharded run never
repeat a task. ROW_FIELDS is the schema, written once: an existing file must carry
exactly that header or the run refuses before writing a byte.

The sweep never embeds (unless --allow-embedding, which a shard may not use): shards
writing one representation's cache at once would overwrite each other, so a run that
finds a molecule missing from the cache names the featurize command and stops before
its first task. A failed campaign is a row, not a crash. A cell that cannot be built,
or is degenerate, is skipped with one printed line; a skipped build exits 1 at the end.

    python -m lsab.sweep --out runs/ab_v1.csv --dry-run
    python -m lsab.sweep --out runs/ab_v1.csv --datasets bh_reaction_1 shields --reps morgan --seeds 2
    python -m lsab.sweep --out runs/ab_v1.csv --shard 0/8        (then --status)
"""

from __future__ import annotations

import os
import sys


def _threads_from_argv(argv: list[str]) -> str:
    """The --threads value, read before argparse -- and before numpy or torch exist."""
    for i, arg in enumerate(argv):
        if arg == "--threads" and i + 1 < len(argv):
            return argv[i + 1]
        if arg.startswith("--threads="):
            return arg.split("=", 1)[1]
    return "1"


if __name__ == "__main__":
    # numpy's BLAS (np.corrcoef, the SVD, pdist in every cell build) reads these at
    # IMPORT time, so they are set HERE, before the imports below load numpy and torch.
    # setdefault: a value already in the environment (the README shard loop) wins.
    for _var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ.setdefault(_var, _threads_from_argv(sys.argv))

import csv  # noqa: E402  -- every import below the thread block, on purpose
import functools  # noqa: E402
import math  # noqa: E402
import statistics  # noqa: E402
import subprocess  # noqa: E402
from collections import Counter, defaultdict  # noqa: E402
from dataclasses import dataclass  # noqa: E402
from datetime import datetime, timezone  # noqa: E402
from pathlib import Path  # noqa: E402

import torch  # noqa: E402

from lsab import featurize  # noqa: E402
from lsab.bo import Campaign, CampaignError, initial_design, metrics, run_campaign  # noqa: E402
from lsab.datasets import DEFAULT_DATASETS, load  # noqa: E402
from lsab.lengthscale import make_prior, pool_geometry, rule_chen, rule_geom  # noqa: E402
from lsab.reduce import build  # noqa: E402

ROW_FIELDS = (
    # key: identifies a task (KEY_FIELDS is this prefix)
    "dataset",
    "reduction",
    "rep",
    "rule",
    "prior_mode",
    "cv",
    "seed",
    "n_init",
    "n_iter",
    "seed_base",
    # provenance: torch_threads / omp_threads are what the process actually ran with
    "family",
    "timestamp",
    "lsab_commit",
    "torch_threads",
    "omp_threads",
    # cell
    "n",
    "d",
    "used_reps",
    "n_failed_embed",
    "D_bar",
    "geom_cv",
    "degenerate",
    "ell_geom",
    "ell_chen",
    # arm
    "ell_0",
    "concentration",
    "rate",
    # design: the n_init starting indices, so the analysis can verify the pairing
    "init_indices",
    # outcome (bo.metrics). n_fit_warnings counts EVERY warning raised inside
    # fit_gpytorch_mll, gpytorch NumericalWarnings for added jitter included, not only
    # optimisation failures: it is not a failure count.
    "auc",
    "coverage_top5",
    "simple_regret",
    "best_found",
    "first_top5_hit",
    "ell_fitted_final",
    "ell_fitted_traj",
    "ell_fitted_log_sd_final",
    "n_fit_warnings",
    "seconds",
    # status
    "failed",
    "fail_iteration",
    "error",
)
KEY_FIELDS = ROW_FIELDS[:10]
OUTCOME_FIELDS = ROW_FIELDS[ROW_FIELDS.index("auc") : ROW_FIELDS.index("failed")]
_KEY_TYPES = (str, str, str, str, str, float, int, int, int, int)
DEFAULT_REPS = ("morgan", "mace_mp0", "mace_off23", "aimnet2_all", "t5", "chemberta")


@dataclass(frozen=True)
class Arm:
    rule: str
    prior_mode: str
    cv: float


@dataclass(frozen=True)
class CellKey:
    dataset: str
    reduction: str
    rep: str


@dataclass(frozen=True)
class Task:
    seed: int
    cell: CellKey
    arm: Arm


# =============================================================================
# 1. THE TASK LIST
# =============================================================================
def task_list(
    datasets, reductions, reps, rules, prior_modes, cv, seeds, seed_offset
) -> list[Task]:
    """Seed-major: for seed, for cell (dataset, reduction, rep), for arm (rule, prior_mode)."""
    cells = [
        CellKey(d, red, rep) for d in datasets for red in reductions for rep in reps
    ]
    arms = [Arm(rule, mode, float(cv)) for rule in rules for mode in prior_modes]
    return [
        Task(seed, cell, arm)
        for seed in range(seed_offset, seed_offset + seeds)
        for cell in cells
        for arm in arms
    ]


def shard(tasks: list[Task], i: int, k: int) -> list[Task]:
    """The tasks of (seed, cell) groups i, i+k, i+2k, ...: whole groups, so every arm
    of a pair runs in the same shard and each shard holds every arm equally often."""
    if not 0 <= i < k:
        raise ValueError(f"shard {i}/{k}: need 0 <= i < k")
    mine = set(list(dict.fromkeys((t.seed, t.cell) for t in tasks))[i::k])
    return [t for t in tasks if (t.seed, t.cell) in mine]


def row_key(task: Task, n_init: int, n_iter: int, seed_base: int) -> tuple:
    c, a = task.cell, task.arm
    return (
        c.dataset,
        c.reduction,
        c.rep,
        a.rule,
        a.prior_mode,
        float(a.cv),
        int(task.seed),
        int(n_init),
        int(n_iter),
        int(seed_base),
    )


# =============================================================================
# 2. THE FILES  -- the schema guard lives here
# =============================================================================
def result_files(out) -> list[Path]:
    """`out` and every shard file of it (`<stem>.shard<i>of<k><suffix>`) that exists."""
    out = Path(out)
    return [out] + sorted(out.parent.glob(f"{out.stem}.shard*of*{out.suffix}"))


def shard_path(out, i: int, k: int) -> Path:
    out = Path(out)
    return out.with_name(f"{out.stem}.shard{i}of{k}{out.suffix}")


def _records(path) -> list[dict]:
    """The complete rows of `path` as dicts; raises unless its header is exactly ROW_FIELDS."""
    path = Path(path)
    if not path.exists() or path.stat().st_size == 0:
        return []
    with open(path, newline="", encoding="utf-8") as handle:
        rows = csv.reader(handle)
        header = tuple(next(rows))
        if header != ROW_FIELDS:
            at = next(
                (i for i, (a, b) in enumerate(zip(header, ROW_FIELDS)) if a != b),
                min(len(header), len(ROW_FIELDS)),
            )
            expected = ROW_FIELDS[at] if at < len(ROW_FIELDS) else "(end)"
            found = header[at] if at < len(header) else "(end)"
            raise ValueError(
                f"{path}: header differs from ROW_FIELDS at column {at}: expected "
                f"{expected!r}, found {found!r}. Refusing to read or append to it."
            )
        return [
            dict(zip(ROW_FIELDS, r)) for r in rows if len(r) == len(ROW_FIELDS)
        ]  # a torn line is not done


def read_done_keys(path) -> set[tuple]:
    """Keys of the complete rows in `path`; set() if it does not exist; raises on a header mismatch."""
    return {
        tuple(kind(r[f]) for kind, f in zip(_KEY_TYPES, KEY_FIELDS))
        for r in _records(path)
    }


def _text(value) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    return str(value)  # bools as True/False, floats in full round-trip precision


def _append(path: Path, row: dict) -> None:
    if set(row) != set(ROW_FIELDS):
        raise ValueError(
            f"row fields differ from ROW_FIELDS: {sorted(set(row) ^ set(ROW_FIELDS))}"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    fresh, torn = not path.exists() or path.stat().st_size == 0, False
    if not fresh:  # a write killed mid-row must not glue two rows together
        with open(path, "rb") as handle:
            handle.seek(-1, 2)
            torn = handle.read(1) != b"\n"
    with open(path, "a", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        if torn:
            handle.write("\n")
        if fresh:
            writer.writerow(ROW_FIELDS)
        writer.writerow([_text(row[f]) for f in ROW_FIELDS])


# =============================================================================
# 3. ONE TASK
# =============================================================================
@functools.lru_cache(maxsize=1)
def _commit() -> str:
    try:
        run = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            cwd=Path(__file__).resolve().parent,
            timeout=10,
        )
        return run.stdout.strip() if run.returncode == 0 else ""
    except (OSError, subprocess.SubprocessError):
        return ""


def missing_embeddings(cells, cache_dir=None) -> list[tuple[str, str, str, int]]:
    """(rep, dataset, component, n_missing) wherever building `cells` would have to embed."""
    kw = {"cache_dir": cache_dir} if cache_dir else {}
    pools, missing = {}, []
    for dataset, rep in dict.fromkeys((c.dataset, c.rep) for c in cells):
        if dataset not in pools:
            pools[dataset] = load(dataset)
        for component, smiles in pools[dataset].components.items():
            if n := len(featurize.uncached(rep, smiles, **kw)):
                missing.append((rep, dataset, component, n))
    return missing


def build_cell(cell: CellKey, *, device: str = "cpu", cache_dir=None):
    """(FeaturePool, Geometry) for one cell, built fresh."""
    fp = build(
        load(cell.dataset), cell.rep, cell.reduction, device=device, cache_dir=cache_dir
    )
    return fp, pool_geometry(fp)


def run_task(
    task: Task,
    *,
    n_init: int,
    n_iter: int,
    seed_base: int,
    device: str = "cpu",
    cache_dir=None,
    built=None,
) -> dict:
    """One full row. Builds the cell unless `built` = (FeaturePool, Geometry) is passed,
    as the sweep does for the arms of one (seed, cell). A failed campaign is recorded."""
    fp, g = (
        built
        if built is not None
        else build_cell(task.cell, device=device, cache_dir=cache_dir)
    )
    prior = make_prior(fp, task.arm.rule, task.arm.prior_mode, task.arm.cv, geometry=g)
    components = [b for b in fp.blocks if b.kind == "component"]
    row = dict(zip(KEY_FIELDS, row_key(task, n_init, n_iter, seed_base)))
    row.update(
        family=fp.pool.spec.family,
        lsab_commit=_commit(),
        torch_threads=torch.get_num_threads(),
        omp_threads=os.environ.get("OMP_NUM_THREADS", ""),
        n=fp.pool.n,
        d=fp.d,
        used_reps=";".join(f"{b.name}={b.used_rep}" for b in components),
        n_failed_embed=sum(b.n_failed for b in components),
        D_bar=g.mean,
        geom_cv=g.cv,
        degenerate=g.degenerate,
        ell_geom=rule_geom(g),
        ell_chen=rule_chen(g),
        ell_0=prior.ell_0,
        concentration=prior.concentration,
        rate=prior.rate,
    )
    try:
        result = run_campaign(
            Campaign(
                fp,
                prior,
                n_init=n_init,
                n_iter=n_iter,
                seed=task.seed,
                seed_base=seed_base,
            )
        )
    except CampaignError as error:
        cause = error.__cause__ or error
        design = initial_design(fp.pool.n, n_init, task.seed, seed_base)
        row.update(
            dict.fromkeys(OUTCOME_FIELDS, math.nan),
            failed=True,
            fail_iteration=error.iteration,
            error=f"{type(cause).__name__}: {cause}"[:200],
        )
    else:
        design = result.sampled_indices[:n_init]
        row.update(
            metrics(result, fp.pool.objective, n_init),
            failed=False,
            fail_iteration=math.nan,
            error="",
        )
    row.update(
        init_indices=";".join(str(int(i)) for i in design),
        timestamp=datetime.now(timezone.utc).isoformat(timespec="seconds"),
    )
    return row


# =============================================================================
# 4. THE SWEEP, THE DRY RUN, THE STATUS
# =============================================================================
def sweep(
    tasks,
    out_path,
    *,
    n_init,
    n_iter,
    seed_base,
    include_degenerate=False,
    device="cpu",
    cache_dir=None,
    dry_run=False,
    done_from=None,
    allow_embedding=False,
) -> int:
    """Run the tasks not yet in `done_from` (default: `out_path`), appending to `out_path`.
    Returns the rows written. Exits 2 before the first task if a cell would have to embed
    (unless allow_embedding), and 1 at the end if any cell could not be built."""
    out_path = Path(out_path)
    done = set().union(*(read_done_keys(p) for p in (done_from or [out_path])))
    todo = [t for t in tasks if row_key(t, n_init, n_iter, seed_base) not in done]
    print(
        f"tasks: {len(tasks)} in this run, {len(tasks) - len(todo)} done, {len(todo)} to run",
        flush=True,
    )
    missing = missing_embeddings([t.cell for t in todo], cache_dir)
    if dry_run:
        _dry_run(todo, done_from or [out_path], missing, device, cache_dir)
        return 0
    if missing and not allow_embedding:
        for rep, dataset, component, n in missing:
            print(f"not embedded: {rep} {dataset}/{component}: {n} molecules")
        print(
            "embed them first (one process), then rerun:\n  python -m lsab.featurize --rep "
            f"{' '.join(dict.fromkeys(m[0] for m in missing))} --dataset {' '.join(dict.fromkeys(m[1] for m in missing))}"
        )
        sys.exit(2)
    written, built, degenerate, broken = 0, (None, None), set(), {}
    for task in todo:
        if task.cell in broken:
            continue
        if built[0] != (
            task.seed,
            task.cell,
        ):  # one cell in memory: the 10k pools are large
            try:
                built = (
                    (task.seed, task.cell),
                    build_cell(task.cell, device=device, cache_dir=cache_dir),
                )
            except (
                Exception
            ) as error:  # skip the cell; do not lose the rest of the shard
                broken[task.cell] = f"{type(error).__name__}: {error}"[:200]
                print(f"SKIP {_label(task.cell)}: {broken[task.cell]}", flush=True)
                continue
        if built[1][1].degenerate and not include_degenerate:
            if task.cell not in degenerate:
                print(
                    f"skip {_label(task.cell)}: degenerate pool (every distance equal)",
                    flush=True,
                )
                degenerate.add(task.cell)
            continue
        row = run_task(
            task, n_init=n_init, n_iter=n_iter, seed_base=seed_base, built=built[1]
        )
        _append(out_path, row)
        written += 1
        outcome = (
            f"FAILED@{row['fail_iteration']}"
            if row["failed"]
            else f"auc={row['auc']:.3f}  {row['seconds']:.1f}s"
        )
        print(
            f"seed={task.seed:<3} {_label(task.cell):40s} {task.arm.rule:5s} {task.arm.prior_mode:23s} "
            f"cv={task.arm.cv:g}  ell_0={row['ell_0']:<8.3g} {outcome}",
            flush=True,
        )
    if broken:
        print(
            f"{written} rows written; {len(broken)} cell(s) could not be built and were skipped:"
        )
        for cell, error in broken.items():
            print(f"  {_label(cell)}: {error}")
        sys.exit(1)
    return written


def _label(cell: CellKey) -> str:
    return f"{cell.dataset}/{cell.reduction}/{cell.rep}"


def _dry_run(todo, files, missing, device, cache_dir) -> None:
    """Per cell: tasks left, d and both centres (unless its embeddings are not cached
    yet), and a wall-time estimate from the median seconds of rows already written."""
    timings, unembedded = defaultdict(list), Counter()
    for path in files:
        for r in _records(path):
            if r["failed"] == "False" and r["seconds"]:
                timings[(r["dataset"], r["reduction"], r["rep"])].append(
                    float(r["seconds"])
                )
    for rep, dataset, _component, n in missing:
        unembedded[(dataset, rep)] += n
    remaining, total, untimed = Counter(t.cell for t in todo), 0.0, 0
    for cell in remaining:
        if unembedded[(cell.dataset, cell.rep)]:
            shape = (
                f"{unembedded[(cell.dataset, cell.rep)]} molecules not embedded yet "
                f"(python -m lsab.featurize --rep {cell.rep} --dataset {cell.dataset})"
            )
        else:
            try:
                fp, g = build_cell(cell, device=device, cache_dir=cache_dir)
                shape = f"d={fp.d:<5} chen={rule_chen(g):<7.3g} geom={rule_geom(g):<7.3g}{'  DEGENERATE' if g.degenerate else ''}"
            except Exception as error:  # the real run would SKIP this cell
                shape = f"CANNOT BUILD: {type(error).__name__}: {error}"[:160]
        seen = timings[(cell.dataset, cell.reduction, cell.rep)]
        median = statistics.median(seen) if seen else None
        if median is None:
            untimed += remaining[cell]
        else:
            total += median * remaining[cell]
        estimate = (
            "?"
            if median is None
            else f"{median:.0f} s/task -> {median * remaining[cell] / 3600:.2f} h"
        )
        print(
            f"  {_label(cell):40s} {remaining[cell]:>5} tasks  {estimate:22s} {shape}",
            flush=True,
        )
    print(
        f"estimated wall time: {total / 3600:.1f} h in one process, for cells with timed rows only; "
        f"{untimed} tasks in cells with no rows yet ('?') are not in that total. Divide by the shard count."
    )


def status(tasks, files, *, n_init, n_iter, seed_base) -> None:
    done = set().union(*(read_done_keys(p) for p in files))
    per_cell = defaultdict(lambda: [0, 0])
    for t in tasks:
        per_cell[t.cell][1] += 1
        per_cell[t.cell][0] += row_key(t, n_init, n_iter, seed_base) in done
    for cell, (n_done, n_total) in per_cell.items():
        print(f"  {_label(cell):40s} {n_done:>5} / {n_total}")
    print(
        f"done: {sum(v[0] for v in per_cell.values())} / {len(tasks)}  (files: {', '.join(str(f) for f in files if f.exists())})"
    )


# =============================================================================
# 5. CLI
# =============================================================================
def main(argv: list[str] | None = None) -> None:
    import argparse

    from lsab.datasets import names
    from lsab.lengthscale import PRIOR_MODES, RULES
    from lsab.reduce import REDUCTIONS

    parser = argparse.ArgumentParser(
        prog="python -m lsab.sweep", description="Run the lengthscale A/B matrix."
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--datasets", nargs="+", default=list(DEFAULT_DATASETS), choices=names()
    )
    parser.add_argument(
        "--reps", nargs="+", default=list(DEFAULT_REPS), choices=list(featurize.REPS)
    )
    parser.add_argument(
        "--reductions",
        nargs="+",
        default=["decorr0.7", "pca64"],
        choices=list(REDUCTIONS),
    )
    parser.add_argument(
        "--rules", nargs="+", default=["chen", "geom"], choices=list(RULES)
    )
    parser.add_argument(
        "--prior-modes",
        nargs="+",
        default=["match_concentration", "match_parameterisation"],
        choices=PRIOR_MODES,
    )
    parser.add_argument("--cv", type=float, default=0.3)
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--seed-offset", type=int, default=0)
    parser.add_argument(
        "--iter",
        type=int,
        default=50,
        help="total experiments per campaign, initial design included",
    )
    parser.add_argument("--init", type=int, default=5)
    parser.add_argument("--seed-base", type=int, default=1337)
    parser.add_argument(
        "--shard", default=None, help="i/k: run the i-th of k shards, into its own file"
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=1,
        help="torch AND BLAS threads; run k shards on k cores instead",
    )
    parser.add_argument("--include-degenerate", action="store_true")
    parser.add_argument(
        "--allow-embedding",
        action="store_true",
        help="embed missing molecules during the run (one unsharded process only)",
    )
    parser.add_argument("--device", default="cpu", help="for --allow-embedding")
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--status", action="store_true")
    args = parser.parse_args(argv)
    if args.allow_embedding and args.shard:
        parser.error(
            "--allow-embedding cannot be combined with --shard: shards must never write the embedding cache"
        )
    tasks = task_list(
        args.datasets,
        args.reductions,
        args.reps,
        args.rules,
        args.prior_modes,
        args.cv,
        args.seeds,
        args.seed_offset,
    )
    design = dict(n_init=args.init, n_iter=args.iter, seed_base=args.seed_base)
    files = result_files(args.out)
    if args.status:
        return status(tasks, files, **design)
    out = args.out
    if args.shard:
        i, k = (int(part) for part in args.shard.split("/"))
        tasks, out = shard(tasks, i, k), shard_path(args.out, i, k)
    torch.set_num_threads(args.threads)
    print(
        f"threads: torch {torch.get_num_threads()}, OMP_NUM_THREADS={os.environ.get('OMP_NUM_THREADS', 'unset')}  "
        f"output: {out}",
        flush=True,
    )
    sweep(
        tasks,
        out,
        include_degenerate=args.include_degenerate,
        device=args.device,
        cache_dir=args.cache_dir,
        dry_run=args.dry_run,
        done_from=files + ([out] if out not in files else []),
        allow_embedding=args.allow_embedding,
        **design,
    )


if __name__ == "__main__":
    main()
