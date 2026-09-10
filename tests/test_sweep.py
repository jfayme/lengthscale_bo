"""
tests/test_sweep.py
===================

Acceptance tests for module 6, `lsab/sweep.py` (stdlib unittest):

    python -m unittest tests.test_sweep -v

`run_campaign` is replaced by a fake that returns a deterministic CampaignResult in
milliseconds, and the cells use module 3's fake representation, so no model runs.
The pairing test is the exception: it runs two short REAL campaigns.
"""

import contextlib
import csv
import dataclasses
import io
import os
import subprocess
import sys
import tempfile
import unittest
from collections import Counter
from pathlib import Path
from unittest import mock

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from lsab import featurize, sweep as S  # noqa: E402
from lsab.bo import CampaignError, CampaignResult, initial_design  # noqa: E402
from lsab.lengthscale import pool_geometry  # noqa: E402
from tests.test_reduce import _fake_spec  # noqa: E402

DESIGN = dict(n_init=5, n_iter=8, seed_base=1337)


def _fake_campaign(fail_rule=None):
    """Stands in for bo.run_campaign: the real initial design, then the next unsampled rows."""
    def run(c):
        if c.prior.rule == fail_rule:
            raise CampaignError(7) from RuntimeError("synthetic fit failure")
        start = initial_design(c.fp.pool.n, c.n_init, c.seed, c.seed_base)
        rest = [i for i in range(c.fp.pool.n) if i not in set(start.tolist())][: c.n_iter - c.n_init]
        indices = np.concatenate([start, rest]).astype(np.int64)
        fits = np.full(c.n_iter - c.n_init, 1.5)
        return CampaignResult(indices, c.fp.pool.objective[indices], fits, 0.1 * fits, 0, 0.01)
    return run


def _tasks(seeds=1, modes=("match_concentration",)):
    return S.task_list(["bh_reaction_1"], ["decorr0.7"], ["fake"], ["chen", "geom"], list(modes), 0.3, seeds, 0)


def _rows(path):
    with open(path, newline="", encoding="utf-8") as handle:
        return list(csv.reader(handle))


class TestTaskList(unittest.TestCase):
    def test_task_order_is_seed_major(self):
        tasks = S.task_list(["d0", "d1"], ["none"], ["morgan"], ["chen", "geom"], ["match_concentration"], 0.3, 2, 0)
        c0, c1 = S.CellKey("d0", "none", "morgan"), S.CellKey("d1", "none", "morgan")
        a0, a1 = S.Arm("chen", "match_concentration", 0.3), S.Arm("geom", "match_concentration", 0.3)
        expected = [S.Task(s, c, a) for s in (0, 1) for c in (c0, c1) for a in (a0, a1)]
        self.assertEqual(tasks, expected)

    def test_shard_partitions_and_balances(self):
        tasks = S.task_list(["d0", "d1"], ["none"], ["r0", "r1"], ["chen", "geom"],
                            ["match_concentration", "match_parameterisation"], 0.3, 5, 0)
        for k in (3, 8):    # 8 shards over 4 arms is where dealing out single tasks goes wrong
            shards = [S.shard(tasks, i, k) for i in range(k)]
            self.assertEqual(sum(len(s) for s in shards), len(tasks))
            self.assertEqual(set().union(*map(set, shards)), set(tasks))
            for part in shards:
                arms = Counter((t.arm.rule, t.arm.prior_mode) for t in part)
                self.assertLessEqual(max(arms.values()) - min(arms.values()), 1, (k, arms))
                self.assertEqual(len(arms), 4)


class TestSweep(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.out = self.tmp / "ab.csv"
        patches = [mock.patch.dict(featurize.REPS, {"fake": _fake_spec()}),
                   mock.patch.object(S, "run_campaign", _fake_campaign())]
        for patch in patches:
            patch.start()
            self.addCleanup(patch.stop)

    def _sweep(self, tasks, **kw):
        with contextlib.redirect_stdout(io.StringIO()) as printed:
            written = S.sweep(tasks, self.out, **DESIGN, **kw)
        return written, printed.getvalue()

    def test_resume_by_key(self):
        four = _tasks(modes=("match_concentration", "match_parameterisation"))
        self.assertEqual(self._sweep(four[:3])[0], 3)
        self.assertEqual(self._sweep(four[:3])[0], 0)
        self.assertEqual(self._sweep(_tasks(seeds=2, modes=("match_concentration", "match_parameterisation")))[0], 5)
        rows = _rows(self.out)
        self.assertEqual(rows[0], list(S.ROW_FIELDS))
        self.assertEqual(sum(r == list(S.ROW_FIELDS) for r in rows), 1)   # the header appears once
        self.assertEqual({len(r) for r in rows}, {len(S.ROW_FIELDS)})
        self.assertEqual(len(rows), 1 + 8)

    def test_schema_guard(self):
        short = [f for f in S.ROW_FIELDS if f != "geom_cv"]
        self.out.write_text(",".join(short) + "\n" + ",".join("x" * len(short)) + "\n", encoding="utf-8")
        before = self.out.read_bytes()
        with self.assertRaisesRegex(ValueError, "geom_cv"):
            S.read_done_keys(self.out)
        with self.assertRaisesRegex(ValueError, "geom_cv"):
            self._sweep(_tasks())
        self.assertEqual(self.out.read_bytes(), before)                    # not one byte appended

    def test_failed_campaign_is_a_row(self):
        with mock.patch.object(S, "run_campaign", _fake_campaign(fail_rule="geom")):
            self.assertEqual(self._sweep(_tasks())[0], 2)
        with open(self.out, newline="", encoding="utf-8") as handle:
            rows = {r["rule"]: r for r in csv.DictReader(handle)}
        self.assertEqual((rows["geom"]["failed"], rows["geom"]["fail_iteration"], rows["geom"]["auc"]), ("True", "7", ""))
        self.assertIn("synthetic fit failure", rows["geom"]["error"])
        self.assertEqual(rows["chen"]["failed"], "False")
        self.assertTrue(float(rows["chen"]["auc"]) >= 0)

    def test_degenerate_skipped(self):
        degenerate = lambda fp: dataclasses.replace(pool_geometry(fp), degenerate=True)  # noqa: E731
        with mock.patch.object(S, "pool_geometry", degenerate):
            written, printed = self._sweep(_tasks())
            self.assertEqual(written, 0)
            self.assertFalse(self.out.exists())
            self.assertEqual(sum("degenerate" in line for line in printed.splitlines()), 1)
            self.assertEqual(self._sweep(_tasks(), include_degenerate=True)[0], 2)

    def test_pairing_is_real(self):
        mock.patch.stopall()                                              # the REAL run_campaign
        with mock.patch.dict(featurize.REPS, {"fake": _fake_spec()}):
            self._sweep(_tasks())
        with open(self.out, newline="", encoding="utf-8") as handle:
            chen, geom = sorted(csv.DictReader(handle), key=lambda r: r["rule"])
        for field in ("init_indices", "n", "d", "D_bar", "used_reps"):
            self.assertEqual(chen[field], geom[field], field)
        self.assertNotEqual(chen["ell_0"], geom["ell_0"])
        self.assertEqual((chen["failed"], geom["failed"]), ("False", "False"))

    def test_sweep_refuses_to_embed(self):
        one_miss = lambda rep, smiles, **kw: ["C1=CC=CC=C1"]   # noqa: E731
        with mock.patch.object(featurize, "uncached", one_miss):
            with self.assertRaises(SystemExit) as raised:
                _, printed = self._sweep(_tasks())
            self.assertEqual(raised.exception.code, 2)
            self.assertFalse(self.out.exists())
            self.assertEqual(self._sweep(_tasks(), allow_embedding=True)[0], 2)   # one unsharded process may
        args = ["--out", str(self.out), "--datasets", "bh_reaction_1", "--reps", "morgan",
                "--shard", "0/2", "--allow-embedding"]
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as raised:
            S.main(args)                                               # a shard may never write the cache
        self.assertEqual(raised.exception.code, 2)

    def test_unbuildable_cell_is_skipped(self):
        real_build = S.build

        def build_fails_without_reduction(pool, rep, reduction, **kw):
            if reduction == "none":
                raise ValueError("synthetic: 3 of 22 molecules failed to embed")
            return real_build(pool, rep, reduction, **kw)

        tasks = S.task_list(["bh_reaction_1"], ["none", "decorr0.7"], ["fake"], ["chen", "geom"],
                            ["match_concentration"], 0.3, 1, 0)
        with mock.patch.object(S, "build", build_fails_without_reduction):
            with contextlib.redirect_stdout(io.StringIO()) as printed, self.assertRaises(SystemExit) as raised:
                S.sweep(tasks, self.out, **DESIGN)
        self.assertEqual(raised.exception.code, 1)
        self.assertIn("SKIP bh_reaction_1/none/fake", printed.getvalue())
        with open(self.out, newline="", encoding="utf-8") as handle:
            self.assertEqual({r["reduction"] for r in csv.DictReader(handle)}, {"decorr0.7"})   # the rest ran

    def test_threads_reach_blas(self):
        env = {k: v for k, v in os.environ.items() if k not in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")}
        run = subprocess.run([sys.executable, "-m", "lsab.sweep", "--out", str(self.tmp / "t.csv"),
                              "--datasets", "bh_reaction_1", "--reps", "morgan", "--reductions", "none",
                              "--seeds", "1", "--dry-run", "--threads", "2"],     # 2, not the default: argv was read
                             cwd=ROOT, env=env, capture_output=True, text=True, check=True)
        self.assertIn("threads: torch 2, OMP_NUM_THREADS=2", run.stdout)

    def test_dry_run_writes_nothing(self):
        written, printed = self._sweep(_tasks(seeds=2), dry_run=True)
        self.assertEqual(written, 0)
        self.assertFalse(self.out.exists())
        self.assertIn("4 to run", printed)
        self.assertIn("estimated wall time", printed)


if __name__ == "__main__":
    unittest.main()
