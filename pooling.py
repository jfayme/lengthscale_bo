"""
pooling.py
==========

Atom -> molecule pooling for the MLIP embeddings (AIMNet2 / MACE / conformer
ensemble). Two options:

  * "mean"  : the current plain average over atoms  ->  the *centroid* of the
              per-atom embedding set (first moment only).
  * "rff"   : a Random-Fourier-Features approximation of the RBF set-kernel
              MEAN EMBEDDING (kernel mean embedding of the atom distribution).
              A linear kernel on this vector ~ an RBF-MMD set kernel between two
              molecules' atom-embedding distributions, so it captures spread /
              multi-modality that the plain mean discards. Mean pooling is the
              exact linear special case (z(x)=x), so RFF is a strict superset.

Key implementation detail: the random projection (W, b) and the standardization
/ bandwidth must be **fixed across all molecules** in a run, or the per-molecule
vectors are not comparable. So an ``RFFMeanPooler`` is *fit once* on a
representative sample of per-atom embeddings (which also sets the bandwidth by
the median-distance heuristic) and then applied to every molecule.

This module is additive: callers default to "mean", so existing results are
unchanged.
"""
from __future__ import annotations
import numpy as np


def mean_pool(per_atom) -> np.ndarray:
    """Plain centroid of the per-atom embeddings: (N, d) -> (d,)."""
    return np.asarray(per_atom, dtype=np.float64).mean(axis=0)


class RFFMeanPooler:
    """RFF kernel-mean-embedding pooler (set kernel). Fit once, apply to all.

    Parameters
    ----------
    n_features : int      number of random Fourier features (output dim).
    lengthscale : float | "median"
        RBF bandwidth on the *atom-embedding* space. "median" uses the
        median-distance heuristic on the (standardized) fit sample — strongly
        recommended, since a fixed value rarely matches AIMNet/MACE scales.
    standardize : bool    per-dim zero-mean/unit-var the atom features first
        (makes the bandwidth meaningful across heterogeneous feature scales).
    seed : int            fixes W, b (and the median subsample).
    """

    def __init__(self, n_features: int = 1024, lengthscale="median",
                 standardize: bool = True, seed: int = 0):
        self.n_features = int(n_features)
        self.lengthscale = lengthscale
        self.standardize = standardize
        self.seed = seed
        self._fitted = False

    def fit(self, atom_embeddings) -> "RFFMeanPooler":
        X = np.asarray(atom_embeddings, dtype=np.float64)
        if X.ndim != 2:
            raise ValueError("fit expects a (n_atoms_total, d_in) array")
        self.d_in_ = X.shape[1]
        rng = np.random.default_rng(self.seed)
        if self.standardize:
            self.mu_ = X.mean(axis=0)
            self.sd_ = X.std(axis=0) + 1e-8
        else:
            self.mu_ = np.zeros(self.d_in_)
            self.sd_ = np.ones(self.d_in_)
        Xs = (X - self.mu_) / self.sd_

        if isinstance(self.lengthscale, str) and self.lengthscale == "median":
            n = len(Xs); m = min(n, 800)            # subsample for the O(m^2) heuristic
            sub = Xs[rng.choice(n, size=m, replace=False)] if n > m else Xs
            d2 = ((sub[:, None, :] - sub[None, :, :]) ** 2).sum(-1)
            iu = np.triu_indices(len(sub), k=1)
            med = float(np.sqrt(np.median(d2[iu]))) if iu[0].size else 1.0
            self.ell_ = med if med > 1e-8 else 1.0
        else:
            self.ell_ = float(self.lengthscale)

        self.W_ = rng.normal(0.0, 1.0 / self.ell_, size=(self.d_in_, self.n_features))
        self.b_ = rng.uniform(0.0, 2 * np.pi, size=self.n_features)
        self.scale_ = np.sqrt(2.0 / self.n_features)
        self._fitted = True
        return self

    def __call__(self, per_atom) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("RFFMeanPooler must be .fit() before use")
        X = np.asarray(per_atom, dtype=np.float64)
        Xs = (X - self.mu_) / self.sd_
        z = self.scale_ * np.cos(Xs @ self.W_ + self.b_)   # (N, n_features)
        return z.mean(axis=0)                               # (n_features,)


class RFFBatchPooler:
    """Stateful helper for the fingerprinters: caches per-atom embeddings by
    SMILES, fits ONE RFFMeanPooler on the first batch it sees, and reuses it.

    Usage inside a fingerprinter::

        self._rff = RFFBatchPooler(n_features=..., lengthscale=...)
        return self._rff(smiles_list, self._per_atom_one)   # -> (M, n_features)

    where ``per_atom_one(smiles) -> (N, d)``. Warm it up by calling once on the
    full set of unique molecules so the bandwidth is fit on a representative
    atom distribution.
    """

    def __init__(self, **rff_cfg):
        self.cfg = rff_cfg
        self.cache: dict[str, np.ndarray] = {}
        self.pooler: RFFMeanPooler | None = None

    def __call__(self, molecules, per_atom_fn) -> np.ndarray:
        per = []
        for m in molecules:
            if m not in self.cache:
                self.cache[m] = np.asarray(per_atom_fn(m), dtype=np.float64)
            per.append(self.cache[m])
        if self.pooler is None:
            self.pooler = RFFMeanPooler(**self.cfg).fit(np.vstack(per))
        return np.stack([self.pooler(p) for p in per], axis=0).astype(np.float32)
