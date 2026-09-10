"""
lsab/reduce.py
==============

Module 3 of the lengthscale-A/B rewrite: a `Pool` + a representation + a reduction
-> the float64 matrix the GP sees (one row per candidate, float64 because BoTorch
runs in double) and the metadata the sweep records. Pure numpy: no torch, no
BayBE, no sklearn. This is the seam that decides `d`, the whole argument of the
chen rule, so every step is deterministic and written down.

Per component, in order: (1) embed the unique molecules, coverage fallback
included; (2) impute a failed molecule with the per-column median of the others,
or raise if more than 1% failed -- every representation must see the SAME
candidate pool; (3) drop constant columns (max - min <= 1e-8) on the raw
features, because a reduction fit on them is ill-posed; (4) reduce; (5) expand to
candidates by SMILES lookup. Numeric settings (Shields Temp_C, Concentration)
enter raw, one column each: BoTorch's Normalize (module 5) is the only scaling.
Blocks go in `spec.components` order, then `spec.numeric` order.

    python -m lsab.reduce --dataset bh_reaction_1 shields --rep morgan mace_mp0 --reduction decorr0.7 pca64 none
"""
from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Protocol

import numpy as np

from lsab.datasets import Pool
from lsab.featurize import embed_component

log = logging.getLogger("lsab.reduce")

MAX_FAILED_FRACTION = 0.01   # of a component's molecules; above this, build raises
CONSTANT_RANGE = 1e-8        # a column whose max - min is at most this is constant


# =============================================================================
# 1. THE REDUCTIONS  (the three names are an axis of the A/B: keep them)
# =============================================================================
class Reduction(Protocol):
    def fit_transform(self, X: np.ndarray) -> np.ndarray: ...   # (n_unique, d) -> (n_unique, d')


@dataclass(frozen=True)
class Decorrelate:
    """Greedy: visit columns in DESCENDING VARIANCE (stable sort), keep a column
    unless |corr| > threshold with an already-kept column. Output keeps the
    original column order.

    Why variance order: embedding columns have no meaning to their index, so
    'keep the first' (what BayBE does) lets the survivor of a correlated cluster
    be whichever came first. Descending variance keeps the most informative one.
    This is a deliberate difference from the old runs.

    With two molecules every pair of columns has |corr| = 1, so exactly one
    column survives. That is the method's behaviour, not an error.
    """
    threshold: float

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        if X.shape[1] < 2:
            return X
        if X.shape[0] == 2:
            log.info("two molecules: every column pair has |corr| = 1, so one column survives")
        corr = np.abs(np.corrcoef(X, rowvar=False))
        kept = []
        for column in np.argsort(-X.var(axis=0), kind="stable"):   # ties keep index order
            if not kept or corr[column, kept].max() <= self.threshold:
                kept.append(column)
        return X[:, np.sort(kept)]


@dataclass(frozen=True)
class PCA:
    """Centred SVD (numpy, full_matrices=False). Keep the smallest k with
    cumulative explained variance >= `variance`, capped at `cap` and at
    rank = min(n_unique - 1, d). Deterministic sign: flip each component so its
    largest-|loading| entry is positive (BLAS-independent output). No whitening."""
    variance: float
    cap: int

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        centred = X - X.mean(axis=0)
        _, singular, components = np.linalg.svd(centred, full_matrices=False)
        cumulative = np.cumsum(singular ** 2) / np.sum(singular ** 2)
        k = int(np.searchsorted(cumulative, self.variance) + 1)
        k = min(k, self.cap, X.shape[0] - 1, X.shape[1])
        components = components[:k]
        largest = components[np.arange(k), np.abs(components).argmax(axis=1)]
        return centred @ (components * np.sign(largest)[:, None]).T


@dataclass(frozen=True)
class Identity:
    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return np.asarray(X, dtype=np.float64)


REDUCTIONS: dict[str, Reduction] = {
    "decorr0.7": Decorrelate(threshold=0.7),   # house default
    "pca64": PCA(variance=0.98, cap=64),       # the paper's
    "none": Identity(),
}


# =============================================================================
# 2. THE CONTRACT
# =============================================================================
@dataclass(frozen=True)
class Block:                      # one column block of the pool matrix
    name: str                     # component or numeric column name
    kind: Literal["component", "numeric"]
    columns: slice                # its columns in X
    rep: str | None               # requested rep (component blocks)
    used_rep: str | None          # after coverage fallback
    d_raw: int                    # embedding width before anything
    d_constant_dropped: int       # width once constant columns are dropped (step 3)
    d_reduced: int                # == columns.stop - columns.start
    n_unique: int                 # unique molecules in the component (or unique values, numeric)
    n_failed: int                 # NaN rows imputed


@dataclass(frozen=True, eq=False)
class FeaturePool:
    pool: Pool
    rep: str
    reduction: str
    X: np.ndarray                 # (pool.n, D) float64, row i <-> pool.frame.iloc[i]
    blocks: tuple[Block, ...]     # in column order

    @property
    def d(self) -> int:
        """X.shape[1]: THIS is the d the chen rule sees."""
        return self.X.shape[1]

    def meta(self) -> dict:
        """Flat and CSV-friendly: d, and per block its used_rep, d_reduced, n_failed."""
        meta = {"d": self.d}
        for block in self.blocks:
            meta[f"{block.name}.used_rep"] = block.used_rep or block.kind   # "numeric" for a setting
            meta[f"{block.name}.d_reduced"] = block.d_reduced
            meta[f"{block.name}.n_failed"] = block.n_failed
        return meta


# =============================================================================
# 3. BUILD
# =============================================================================
def _component(pool: Pool, name: str, rep: str, reducer: Reduction, embed_kw: dict):
    """Steps 1-5 for one component -> (candidate rows, the Block fields they imply)."""
    smiles = pool.components[name]
    raw, used_rep = embed_component(rep, smiles, **embed_kw)
    raw = raw.astype(np.float64)
    finite = np.isfinite(raw).all(axis=1)
    failed = [s for s, ok in zip(smiles, finite) if not ok]
    if len(failed) / len(smiles) > MAX_FAILED_FRACTION:
        raise ValueError(f"{pool.name}/{name}: {len(failed)} of {len(smiles)} molecules failed to "
                         f"embed with {used_rep} (the limit is 1%): {failed}")
    if failed:   # at least 99% of rows are finite here, so no column's median is NaN
        raw[~finite] = np.median(raw[finite], axis=0)
    varying = raw[:, raw.max(axis=0) - raw.min(axis=0) > CONSTANT_RANGE]
    if varying.shape[1] == 0:
        raise ValueError(f"{pool.name}/{name}: no feature varies across its {len(smiles)} "
                         "molecule(s); a one-molecule component is not a search dimension")
    reduced = reducer.fit_transform(varying)
    rows = pool.frame[name].map({s: i for i, s in enumerate(smiles)}).to_numpy()
    return reduced[rows], dict(rep=rep, used_rep=used_rep, d_raw=raw.shape[1],
                               d_constant_dropped=varying.shape[1], n_unique=len(smiles),
                               n_failed=len(failed))


def build(pool: Pool, rep: str, reduction: str = "decorr0.7", *,
          device: str = "cpu", cache_dir=None) -> FeaturePool:
    """The matrix the GP sees for `pool` under `rep` and `reduction`, with its blocks."""
    if reduction not in REDUCTIONS:
        raise ValueError(f"unknown reduction {reduction!r}; expected one of {', '.join(REDUCTIONS)}")
    embed_kw = {"device": device} if cache_dir is None else {"device": device, "cache_dir": cache_dir}
    parts, blocks = [], []
    for name in pool.spec.components:
        part, fields = _component(pool, name, rep, REDUCTIONS[reduction], embed_kw)
        start = sum(p.shape[1] for p in parts)
        blocks.append(Block(name, "component", slice(start, start + part.shape[1]),
                            d_reduced=part.shape[1], **fields))
        parts.append(part)
    for name in pool.spec.numeric:
        start = sum(p.shape[1] for p in parts)
        blocks.append(Block(name, "numeric", slice(start, start + 1), rep=None, used_rep=None,
                            d_raw=1, d_constant_dropped=1, d_reduced=1,
                            n_unique=len(pool.numeric[name]), n_failed=0))
        parts.append(pool.frame[name].to_numpy(dtype=np.float64)[:, None])
    return FeaturePool(pool, rep, reduction, np.hstack(parts), tuple(blocks))


# =============================================================================
# 4. CLI  -- the n, d half of the old preflight.py
# =============================================================================
def main(argv: list[str] | None = None) -> None:
    from lsab import datasets
    from lsab.featurize import REPS

    parser = argparse.ArgumentParser(prog="python -m lsab.reduce",
                                     description="Print D and its blocks per (dataset, rep, reduction).")
    parser.add_argument("--dataset", nargs="+", default=datasets.names(), choices=datasets.names())
    parser.add_argument("--rep", nargs="+", default=["morgan"], choices=list(REPS))
    parser.add_argument("--reduction", nargs="+", default=list(REDUCTIONS), choices=list(REDUCTIONS))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--cache-dir", type=Path, default=None)
    args = parser.parse_args(argv)
    for name in args.dataset:
        pool = datasets.load(name)
        for rep in args.rep:
            for reduction in args.reduction:
                features = build(pool, rep, reduction, device=args.device, cache_dir=args.cache_dir)
                blocks = "  ".join(f"{b.name}: {b.used_rep or b.kind} {b.d_raw}->{b.d_reduced} "
                                   f"({b.n_unique}, {b.n_failed})" for b in features.blocks)
                print(f"{name:18s} {rep:12s} {reduction:10s} D={features.d:<5d} {blocks}", flush=True)


if __name__ == "__main__":
    main()
