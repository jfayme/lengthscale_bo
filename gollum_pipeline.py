"""
gollum_pipeline.py
==================

Run our 6-representation molecular-embedding BO benchmark (AIMNet2, MACE-OFF23,
MACE-MP-0, CheMeleon, ChemBERTa, T5) on the GOLLuM datasets, mirroring exactly
the Shields pipeline (paper Adaptive_ours hyperprior, qLogEI, Matern 5/2,
5 random init, 50 iterations, 20 Monte-Carlo runs).

This file is self-contained orchestration that **reuses, without modifying**,
our existing modules:
  * representations.py        (the 6 fingerprinters + coverage tables)
  * benchmark_representations  (per_run_metrics, pairwise_wilcoxon, _cumbest_column)
  * base.kernels.AdaptiveKernelFactory  (the paper's adaptive hyperprior)

Two dataset shapes are handled:
  * "reaction"  -- multi molecular-component search space (BH, Suzuki, additives),
                   one CustomDiscreteParameter per varying component (Shields-style).
  * "molecule"  -- single-molecule property optimization (photoswitches, redox, ...),
                   one CustomDiscreteParameter whose candidates are the molecule pool.

Targets are standardized to a single maximize-me column named "yield" (MIN
objectives are negated) so all downstream metric code is reused verbatim.
"""

from __future__ import annotations

import os, pickle, time
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
GROOT = os.path.join(HERE, "gollum", "data")
OUT = os.path.join(HERE, "gollum_out")
CACHE = os.path.join(OUT, "_emb_cache")
os.makedirs(CACHE, exist_ok=True)

REPS = ["aimnet2", "aimnet2_all", "mace_off23", "mace_mp0",
        "chemeleon", "chemberta", "t5", "phys"]
#   aimnet2_all : AIMNet2 all-3-pass concat (layer-ablation winner, +0.034 on Shields)
#   phys        : MH-1/POLAR physical descriptor (stats pooling, interpretable)

# Common-coverage element set = AIMNet2 ∩ MACE-OFF23 = {H,C,N,O,F,P,S,Cl,Br,I}.
# Molecule pools are filtered to this so all 6 reps embed an identical pool.
COMMON_ELEMENTS = {1, 6, 7, 8, 9, 15, 16, 17, 35, 53}


# ---------------------------------------------------------------------------
# Dataset registry
# ---------------------------------------------------------------------------
# kind: "reaction" | "molecule"
# reaction: components = candidate molecular columns (constants auto-dropped)
# molecule: smiles col + target col + direction ("max"/"min")
DATASETS = [
    # --- fast molecular property optimization ---
    dict(name="photoswitches", kind="molecule", priority=0,
         file="molecules/photoswitches.csv.gz",
         smiles="SMILES", target="Pi-Pi* Transition Wavelength", direction="max"),
    # --- reaction optimization (Shields-like) ---
    dict(name="bh_reaction_1", kind="reaction", priority=1,
         file="buchwald-hartwig/bh_reaction_1.csv",
         components=["ligand", "additive", "base", "aryl halide"],
         target="objective", direction="max"),
    dict(name="additives_plate_1", kind="reaction", priority=2,
         file="additives/additive_rxn_screening_plate_1.csv",
         components=["additives"], target="objective", direction="max"),
    dict(name="redox_mer", kind="molecule", priority=3,
         file="molecules/redox_mer_with_iupac.csv.gz",
         smiles="SMILES", target="Ered", direction="min"),
    dict(name="bh_reaction_2", kind="reaction", priority=4,
         file="buchwald-hartwig/bh_reaction_2.csv",
         components=["ligand", "additive", "base", "aryl halide"],
         target="objective", direction="max"),
    dict(name="bh_reaction_3", kind="reaction", priority=5,
         file="buchwald-hartwig/bh_reaction_3.csv",
         components=["ligand", "additive", "base", "aryl halide"],
         target="objective", direction="max"),
    dict(name="bh_reaction_4", kind="reaction", priority=6,
         file="buchwald-hartwig/bh_reaction_4.csv",
         components=["ligand", "additive", "base", "aryl halide"],
         target="objective", direction="max"),
    dict(name="bh_reaction_5", kind="reaction", priority=7,
         file="buchwald-hartwig/bh_reaction_5.csv",
         components=["ligand", "additive", "base", "aryl halide"],
         target="objective", direction="max"),
    dict(name="additives_plate_2", kind="reaction", priority=8,
         file="additives/additive_rxn_screening_plate_2.csv",
         components=["additives"], target="objective", direction="max"),
    dict(name="additives_plate_3", kind="reaction", priority=9,
         file="additives/additive_rxn_screening_plate_3.csv",
         components=["additives"], target="objective", direction="max"),
    dict(name="additives_plate_4", kind="reaction", priority=10,
         file="additives/additive_rxn_screening_plate_4.csv",
         components=["additives"], target="objective", direction="max"),
    dict(name="suzuki_miyaura", kind="reaction", priority=11,
         file="suzuki-miyaura/suzuki_miyaura_data.csv",
         components=["reactant_1_smiles", "reactant_2_smiles", "catalyst_smiles",
                     "ligand_smiles", "reagent_1_smiles", "solvent_1_smiles"],
         target="objective", direction="max"),
    # --- Reasoning-BO / Ax complete combinatorial grids (added 2026-06-08) ---
    # Canonical full-coverage HTE grids from the Reasoning-BO repo. CPA is new;
    # bh_full is the complete 3955-pt Ahneman/Doyle set (vs our 5 sub-screens);
    # suzuki_perera is the complete 3696-pt Perera grid (vs our 67%-covered one).
    # (reasoning-BO direct_arylation == our existing Shields set, so not re-added.)
    dict(name="cpa_thiol_imine", kind="reaction", priority=12,
         file="reasoning/CPA.csv",
         components=["Catalyst", "Imine", "Thiol"],
         target="yield", direction="max"),
    dict(name="bh_full", kind="reaction", priority=13,
         file="reasoning/Buchwald_Hartwig.csv",
         components=["Ligand", "Additive", "Base", "Aryl halide"],
         target="yield", direction="max"),
    dict(name="suzuki_perera", kind="reaction", priority=14,
         file="reasoning/suzuki.csv",
         components=["Electrophile_SMILES", "Nucleophile_SMILES",
                     "Ligand_SMILES", "Base_SMILES", "Solvent_SMILES"],
         target="yield", direction="max"),
    # --- large molecular pools (deferred unless time allows) ---
    dict(name="pce10k", kind="molecule", priority=20, big=True,
         file="molecules/photovoltaics_pce10k.csv.gz",
         smiles="SMILES", target="pce", direction="max"),
    dict(name="enamine10k", kind="molecule", priority=21, big=True,
         file="molecules/enamine10k.csv.gz",
         smiles="SMILES", target="score", direction="min"),
]

# Incompatible (no per-molecule SMILES): documented, never run.
SKIPPED = {
    "c2-yield": "catalyst composition (elements + molar ratios), no per-variable SMILES",
    "hplc": "6 continuous process parameters, no molecules",
    "oer": "6 elemental loadings (Ni/Fe/Co/Mn/Ce/La), no molecules",
    "vapdiff": "organic given as name not SMILES; mostly process/molarity variables",
}


# ---------------------------------------------------------------------------
# Embedding disk cache (per representation, keyed by SMILES)
# ---------------------------------------------------------------------------
_FP = {}          # live fingerprinter objects
_MEM = {}         # in-memory {rep: {smiles: vec}}


def _cache_path(rep):
    return os.path.join(CACHE, f"emb__{rep}.pkl")


def _load_cache(rep):
    if rep not in _MEM:
        p = _cache_path(rep)
        if os.path.exists(p):
            with open(p, "rb") as f:
                _MEM[rep] = pickle.load(f)
        else:
            _MEM[rep] = {}
    return _MEM[rep]


def _save_cache(rep):
    with open(_cache_path(rep), "wb") as f:
        pickle.dump(_MEM[rep], f)


def _fingerprinter(rep):
    if rep not in _FP:
        import representations as R
        # Use base T5 for the GOLLuM benchmark: it is the T5 variant GOLLuM
        # itself evaluates, and the chemistry-T5 (GT4SD) weights would not finish
        # downloading reliably on this machine. (Our Shields run used GT4SD.)
        kw = {"model_name": "t5-base"} if rep == "t5" else {}
        _FP[rep] = R.make_fingerprinter(rep, **kw)
    return _FP[rep]


def embed_smiles(rep, smiles_list):
    """Return (n, d) embeddings for smiles_list, using/filling the disk cache.

    Raises through the underlying rep (e.g. AIMNet2 on out-of-coverage element);
    callers handle coverage upstream.
    """
    cache = _load_cache(rep)
    missing = [s for s in dict.fromkeys(smiles_list) if s not in cache]
    if missing:
        fp = _fingerprinter(rep)
        # embed one at a time so a single failure doesn't lose a batch, and so
        # the cache grows incrementally (robust to interruption).
        for s in missing:
            try:
                v = np.asarray(fp([s]), dtype=np.float32)[0]
            except Exception as e:
                v = ("ERR", str(e)[:200])
            cache[s] = v
        _save_cache(rep)
    return [cache[s] for s in smiles_list]


def _ok(vec):
    return not (isinstance(vec, tuple) and vec and vec[0] == "ERR")


# ---------------------------------------------------------------------------
# Descriptor frame builder (cached, with coverage fallback)
# ---------------------------------------------------------------------------


def descriptor_frame(rep, label_to_smiles, *, fallback="mace_mp0",
                     normalize=None, drop_constant=True, prefix=None, pca_cap=64):
    """Build a per-label descriptor DataFrame using the cache + coverage fallback.

    label_to_smiles : dict label -> SMILES (labels become the DataFrame index /
        BayBE parameter values). For element-limited reps that can't cover some
        SMILES, the whole frame falls back to `fallback` (flagged via attr).
    """
    import representations as R
    labels = list(label_to_smiles)
    smis = [label_to_smiles[k] for k in labels]

    # --- OHE baseline: identity one-hot, no embedding / PCA / fallback. ---
    # This is the "no chemistry" reference (every candidate is an unordered
    # category); the GP cannot generalize between candidates, so it lower-bounds
    # what structural features must beat. Kept verbatim (no dim control) so the
    # one-hot meaning is preserved.
    if rep == "ohe":
        eye = np.eye(len(labels), dtype=float)
        cols = [f"{prefix or 'ohe'}_{i}" for i in range(len(labels))]
        df = pd.DataFrame(eye, index=labels, columns=cols)
        df.attrs["used_rep"] = "ohe"
        df.attrs["n_failed"] = 0
        return df

    used = rep
    bad = R.uncovered_elements(rep, smis)
    if bad and fallback:
        used = fallback
    vecs = embed_smiles(used, smis)
    # if element-limited rep still failed on some molecules, try the fallback
    if not all(_ok(v) for v in vecs) and used != fallback and fallback:
        used = fallback
        vecs = embed_smiles(used, smis)
    # drop labels whose embedding failed (e.g. conformer generation)
    keep = [i for i, v in enumerate(vecs) if _ok(v)]
    labels = [labels[i] for i in keep]
    feats = np.vstack([vecs[i] for i in keep]).astype(float)

    # Some embeddings yield non-finite values for degenerate inputs (e.g. an MLIP
    # on a disconnected ionic salt like [Na+].[OH-], whose single ETKDG conformer
    # is ill-defined). Impute per-column (median; all-NaN -> 0) so a few bad rows
    # don't poison PCA / the whole component. Keeps every candidate in the pool.
    bad_mask = ~np.isfinite(feats)
    if bad_mask.any():
        col_med = np.nanmedian(np.where(bad_mask, np.nan, feats), axis=0)
        col_med = np.where(np.isfinite(col_med), col_med, 0.0)
        bi = np.where(bad_mask)
        feats[bi] = np.take(col_med, bi[1])

    cols = [f"{prefix or rep}_{i}" for i in range(feats.shape[1])]
    df = pd.DataFrame(feats, index=labels, columns=cols)
    if drop_constant:
        df = df.loc[:, (df.max() - df.min()) > 1e-8]
    if rep in ("phys", "mordred") and df.shape[1] > 1:
        # Physical / Mordred features have heterogeneous units (eV site energies vs
        # |q|~1; descriptor magnitudes span orders of magnitude); PCA is scale-sensitive,
        # so z-score each column BEFORE PCA. (Latent reps are homogeneous, no standardization.)
        mu = df.mean(axis=0); sd = df.std(axis=0).replace(0, 1.0)
        df = (df - mu) / sd
    if normalize == "global":
        lo, hi = df.values.min(), df.values.max()
        df = (df - lo) / (hi - lo if hi > lo else 1e-12)
    # PCA dimensionality control (the paper's strategy): retain 98% variance,
    # hard-capped at `pca_cap`. This keeps high-dim text reps (CheMeleon 2048-d,
    # ChemBERTa/T5 768-d) tractable for the GP and applies the SAME dim-control
    # policy to every representation, so the comparison stays fair. Replaces the
    # BayBE decorrelation step (decorrelate=False downstream).
    n = df.shape[0]
    # Always cap dimensionality: even small components (e.g. 3 molecules) must be
    # reduced (3 points span <=2 dims), else they keep ~200 raw embedding columns
    # and inflate the search-space dimensionality (the high-dim BO failure mode).
    if pca_cap and df.shape[1] > 1 and n >= 2:
        from sklearn.decomposition import PCA
        k = int(min(df.shape[1], n - 1, pca_cap))
        pca = PCA(n_components=k).fit(df.values)
        cum = np.cumsum(pca.explained_variance_ratio_)
        npc = max(1, int(min(np.searchsorted(cum, 0.98) + 1, pca_cap, k)))
        scores = pca.transform(df.values)[:, :npc]
        df = pd.DataFrame(scores, index=df.index,
                          columns=[f"{prefix or rep}_pca{i}" for i in range(npc)])
    # BayBE requires a unique computational row per label; break exact ties
    # (rare, from near-identical embeddings) with negligible deterministic noise.
    if df.duplicated().any():
        rng = np.random.default_rng(0)
        df = df + rng.normal(0, 1e-7, size=df.shape)
    df.attrs["used_rep"] = used
    df.attrs["n_failed"] = len(vecs) - len(keep)
    return df


# ---------------------------------------------------------------------------
# Dataset loading -> (lookup, build_inputs)
# ---------------------------------------------------------------------------


def _read(cfg):
    return pd.read_csv(os.path.join(GROOT, cfg["file"]))


def _elements_ok(smiles):
    from rdkit import Chem
    m = Chem.MolFromSmiles(smiles)
    if m is None:
        return False
    return {a.GetAtomicNum() for a in m.GetAtoms()} <= COMMON_ELEMENTS


def load_dataset(cfg):
    """Return a dict with: lookup (df incl 'yield'), kind, and per-kind inputs.

    For reaction: 'components' = {comp: {label->smiles}}, lookup has one column
        per component (values = labels) + 'yield'.
    For molecule: 'pool' = {label->smiles}, lookup has 'mol' column + 'yield'.
    """
    df = _read(cfg).copy()
    tgt = cfg["target"]
    df = df[pd.to_numeric(df[tgt], errors="coerce").notna()].copy()
    df[tgt] = df[tgt].astype(float)
    sign = 1.0 if cfg["direction"] == "max" else -1.0
    df["yield"] = sign * df[tgt]

    if cfg["kind"] == "reaction":
        comps = {}
        keep_cols = []
        for c in cfg["components"]:
            if df[c].nunique() <= 1:
                continue  # constant component -> drop
            uniq = sorted(df[c].dropna().unique().tolist())
            comps[c] = {s: s for s in uniq}   # label == SMILES
            keep_cols.append(c)
        lookup = df[keep_cols + ["yield"]].copy()
        return dict(kind="reaction", components=comps, comp_cols=keep_cols,
                    lookup=lookup, n_raw=len(df))

    # molecule
    smi_col = cfg["smiles"]
    df = df.dropna(subset=[smi_col]).drop_duplicates(subset=[smi_col])
    mask = df[smi_col].map(_elements_ok)
    n_excl = int((~mask).sum())
    df = df[mask].copy()
    pool = {s: s for s in df[smi_col].tolist()}
    lookup = pd.DataFrame({"mol": df[smi_col].values, "yield": df["yield"].values})
    return dict(kind="molecule", pool=pool, lookup=lookup,
                n_raw=len(mask), n_excluded=n_excl)


# ---------------------------------------------------------------------------
# Campaign construction + run (one representation)
# ---------------------------------------------------------------------------


def build_campaign(rep, loaded, *, decorrelate=False, switch_after=5,
                   coverage_fallback="mace_mp0", pca_cap=64):
    from baybe import Campaign
    from baybe.objectives import SingleTargetObjective
    from baybe.targets import NumericalTarget
    from baybe.parameters import CustomDiscreteParameter
    from baybe.searchspace import SearchSpace
    from baybe.surrogates import GaussianProcessSurrogate
    from baybe.recommenders import (
        BotorchRecommender, RandomRecommender, TwoPhaseMetaRecommender)
    from base.kernels import AdaptiveKernelFactory

    objective = SingleTargetObjective(target=NumericalTarget(name="yield", mode="MAX"))
    params, notes = [], []

    if loaded["kind"] == "reaction":
        for comp in loaded["comp_cols"]:
            frame = descriptor_frame(rep, loaded["components"][comp],
                                     fallback=coverage_fallback,
                                     normalize=None, prefix=comp, pca_cap=pca_cap)
            if frame.attrs.get("used_rep") != rep:
                notes.append(f"{comp}->{frame.attrs['used_rep']}")
            params.append(CustomDiscreteParameter(name=comp, data=frame,
                                                  decorrelate=decorrelate))
    else:
        frame = descriptor_frame(rep, loaded["pool"], fallback=None,
                                 normalize=None, prefix="mol", pca_cap=pca_cap)
        params.append(CustomDiscreteParameter(name="mol", data=frame,
                                               decorrelate=decorrelate))

    searchspace = SearchSpace.from_product(parameters=params)
    feat_dim = len(searchspace.comp_rep_columns)
    surrogate = GaussianProcessSurrogate(kernel_or_factory=AdaptiveKernelFactory())
    recommender = TwoPhaseMetaRecommender(
        initial_recommender=RandomRecommender(),
        recommender=BotorchRecommender(surrogate_model=surrogate,
                                       acquisition_function="qLogEI"),
        switch_after=switch_after)
    campaign = Campaign(searchspace=searchspace, objective=objective,
                        recommender=recommender)
    return campaign, feat_dim, ";".join(notes)


def run_rep(rep, loaded, dataset_name, *, n_iter=50, mc_runs=20, seed=1337,
            outdir=OUT, resume=True, decorrelate=False, pca_cap=64):
    """Run one representation on one dataset; persist curves__<dataset>__<rep>.csv.

    Dimension control: by default PCA (pca_cap=64, decorrelate=False). For a
    decorrelation-based variant pass pca_cap=None, decorrelate=0.7 (or 0.9).
    """
    from baybe.simulation import simulate_scenarios
    from baybe.utils.random import set_random_seed

    dsdir = os.path.join(outdir, dataset_name)
    os.makedirs(dsdir, exist_ok=True)
    rep_csv = os.path.join(dsdir, f"curves__{rep}.csv")
    if resume and os.path.exists(rep_csv):
        return pd.read_csv(rep_csv), "resumed"

    t0 = time.time()
    campaign, feat_dim, notes = build_campaign(
        rep, loaded, decorrelate=decorrelate, pca_cap=pca_cap)
    build_t = time.time() - t0
    set_random_seed(seed)
    t0 = time.time()
    result = simulate_scenarios({rep: campaign}, loaded["lookup"],
                                batch_size=1, n_doe_iterations=n_iter,
                                n_mc_iterations=mc_runs, impute_mode="ignore")
    sim_t = time.time() - t0
    result["representation"] = rep
    result["dim"] = feat_dim
    result.to_csv(rep_csv, index=False)
    return result, f"dim={feat_dim} build={build_t:.0f}s sim={sim_t:.0f}s notes={notes}"
