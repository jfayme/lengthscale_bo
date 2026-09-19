"""Measure how well force-field energy ranks conformers against MACE.

Throwaway experiment, not part of the pipeline: delete it once the numbers
are in. It answers the questions that decide how `pre_filter_conf_max` and
the cluster-selection rule should be set.

The expensive part is done once. For each molecule every Butina cluster
representative is relaxed with MACE, not just the ones the production cap
would keep, and the results are cached to JSON. Every selection rule is then
evaluated offline against that cache, so adding a rule costs no GPU time.

What it reports
---------------
1. How many clusters a real ligand actually produces at the configured RMSD
   threshold. If that number is routinely below the cap, the selection rule
   is irrelevant and nothing else here matters.
2. Where the MACE minimum sits in the force-field ordering. If it is almost
   always rank 1-4, the current configuration is already sufficient.
3. Spearman correlation between force-field and MACE energies, per molecule.
   This is the number that says whether force-field ranking carries signal
   at all on this chemistry.
4. Recall of the MACE minimum at every budget, for five selection rules:
   energy, population, seeded max-min (Kennard-Stone), a tiered split of the
   two, and a random baseline.

Usage
-----
    python exp_ff_vs_mace_ranking.py --dataset bh_full --limit 10
    python exp_ff_vs_mace_ranking.py --dataset bh_full --resume

"""

import argparse
import json
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from ase import Atoms
from ase.optimize import BFGS
from rdkit import Chem
from rdkit.Chem import rdMolAlign
from rdkit.ML.Cluster import Butina
from rdkit.rdBase import BlockLogs
from scipy.stats import spearmanr

from conformer import EV_TO_KCAL, ConformerGenerator
from dataset import DATASETS
from logging_config import get_logger, setup_logging
from settings import PROJECT_ROOT

logger = get_logger(__name__)

RULES = ("energy", "population", "maxmin", "tiered", "random")


def unique_smiles(dataset: str) -> list[str]:
    """Collect the distinct molecules a dataset is built from.

    Parameters
    ----------
    dataset
        Name of a spec registered in the dataset module.

    Returns
    -------
    smiles
        Every distinct SMILES appearing in the spec's compound columns, in
        first-seen order.

    """
    # DatasetSpec.path is documented as relative to the data root, but the
    # registered values already carry the `data/` prefix, so they resolve
    # against the project root instead. Worth reconciling in dataset.py.
    spec = DATASETS[dataset]
    frame = pd.read_csv(PROJECT_ROOT / spec.path)

    seen: dict[str, None] = {}
    for column in spec.compounds:
        for value in frame[column].dropna().unique():
            seen.setdefault(str(value), None)
    return list(seen)


def cluster_with_membership(
    generator: ConformerGenerator,
    mol: Chem.Mol,
    conf_ids: list[int],
    results: list[tuple[int, float]],
) -> tuple[list[tuple[int, ...]], list[int], dict[int, float]]:
    """Cluster the force-field conformers, keeping full cluster membership.

    This mirrors ConformerGenerator._cluster_conf_select up to the point
    where that method truncates to `pre_filter_conf_max`. It is duplicated
    rather than called because the experiment needs the cluster sizes to
    evaluate the population rule, and the production method returns only the
    representatives.

    Parameters
    ----------
    generator
        Supplies the energy window and the clustering threshold.
    mol
        The molecule carrying the embedded conformers.
    conf_ids
        Identifiers of the embedded conformers.
    results
        One status and energy pair per conformer, from the embedding step.

    Returns
    -------
    clusters
        Butina clusters, in the order the algorithm returned them, holding
        positions into the in-window conformer list.
    representatives
        Conformer identifier of the lowest-energy member of each cluster, in
        cluster order.
    energy_by_id
        Force-field energy of every converged conformer.

    Raises
    ------
    RuntimeError
        If no conformer survived the force-field optimisation.

    """
    energy_by_id = {
        conf_id: energy
        for conf_id, (status, energy) in zip(conf_ids, results, strict=True)
        if status == 0
    }
    if not energy_by_id:
        raise RuntimeError("no conformer survived force-field optimisation")

    ranked = sorted(energy_by_id, key=energy_by_id.__getitem__)
    cutoff = energy_by_id[ranked[0]] + generator.mmff_window_kcal
    in_window = [cid for cid in ranked if energy_by_id[cid] <= cutoff]

    mol_no_h = Chem.RemoveHs(mol)
    cluster_mol = Chem.Mol(mol_no_h)
    cluster_mol.RemoveAllConformers()
    for conf_id in in_window:
        cluster_mol.AddConformer(
            Chem.Conformer(mol_no_h.GetConformer(conf_id)), assignId=True
        )

    if len(in_window) == 1:
        clusters: list[tuple[int, ...]] = [(0,)]
    else:
        matrix = rdMolAlign.GetAllConformerBestRMS(
            cluster_mol, numThreads=generator.n_threads
        )
        clusters = Butina.ClusterData(
            data=matrix,
            nPts=len(in_window),
            distThresh=generator.rmsd_cluster_thresh,
            isDistData=True,
            reordering=True,
        )

    representatives = [
        in_window[min(c, key=lambda pos: energy_by_id[in_window[pos]])]
        for c in clusters
    ]
    return clusters, representatives, energy_by_id


def representative_rmsd(
    mol: Chem.Mol, representatives: list[int]
) -> np.ndarray:
    """Heavy-atom RMSD between every pair of cluster representatives.

    Parameters
    ----------
    mol
        The molecule carrying the conformers.
    representatives
        Conformer identifiers to compare.

    Returns
    -------
    distances
        Square matrix of best-alignment RMSD values, in Angstrom.

    """
    probe = Chem.RemoveHs(mol)
    size = len(representatives)
    distances = np.zeros((size, size), dtype=np.float64)
    for i in range(size):
        for j in range(i + 1, size):
            value = rdMolAlign.GetBestRMS(
                Chem.Mol(probe),
                probe,
                representatives[i],
                representatives[j],
            )
            distances[i, j] = distances[j, i] = value
    return distances


def relax_with_mace(
    generator: ConformerGenerator, mol: Chem.Mol, conf_id: int
) -> tuple[float, bool]:
    """Relax one conformer with the MACE potential.

    Parameters
    ----------
    generator
        Supplies the calculator and the convergence settings.
    mol
        The molecule carrying the conformer.
    conf_id
        Identifier of the conformer to relax.

    Returns
    -------
    energy
        Potential energy after relaxation, in eV.
    converged
        Whether BFGS met the force criterion within the step budget.

    """
    atoms = Atoms(
        numbers=[atom.GetAtomicNum() for atom in mol.GetAtoms()],
        positions=mol.GetConformer(conf_id).GetPositions(),
    )
    atoms.calc = generator.calc
    optimiser = BFGS(atoms, logfile=None)
    converged = optimiser.run(fmax=generator.fmax, steps=generator.opt_steps)
    return float(atoms.get_potential_energy()), bool(converged)


def profile_molecule(
    generator: ConformerGenerator, smiles: str, max_relax: int
) -> dict[str, Any]:
    """Relax every cluster representative of one molecule.

    Parameters
    ----------
    generator
        The configured conformer generator.
    smiles
        SMILES string of the molecule to profile.
    max_relax
        Safety cap on how many representatives are relaxed. The lowest
        force-field energies are kept when the cap bites.

    Returns
    -------
    record
        Per-molecule measurements, JSON-serialisable.

    """
    started = time.perf_counter()
    mol, conf_ids, results = generator.generate_rdkit_conformers(smiles)
    clusters, representatives, energy_by_id = cluster_with_membership(
        generator, mol, conf_ids, results
    )

    sizes = [len(c) for c in clusters]
    n_clusters = len(representatives)

    # The cap keeps the experiment affordable on very flexible molecules.
    # Keeping the force-field-lowest biases against the diversity rules, so
    # molecules that hit the cap are flagged and excluded from recall.
    capped = n_clusters > max_relax
    if capped:
        order = sorted(
            range(n_clusters), key=lambda i: energy_by_id[representatives[i]]
        )
        keep = sorted(order[:max_relax])
        representatives = [representatives[i] for i in keep]
        sizes = [sizes[i] for i in keep]

    distances = representative_rmsd(mol, representatives)

    ff_energies, mace_energies, converged_flags = [], [], []
    for conf_id in representatives:
        energy, converged = relax_with_mace(generator, mol, conf_id)
        ff_energies.append(float(energy_by_id[conf_id]))
        mace_energies.append(energy)
        converged_flags.append(converged)

    mace_kcal = [(e - min(mace_energies)) * EV_TO_KCAL for e in mace_energies]
    ff_kcal = [e - min(ff_energies) for e in ff_energies]

    rho = float("nan")
    if len(ff_kcal) > 2:
        rho = float(spearmanr(ff_kcal, mace_kcal).statistic)

    return {
        "smiles": smiles,
        "n_atoms": mol.GetNumAtoms(),
        "n_embedded": len(conf_ids),
        "n_clusters_found": n_clusters,
        "n_relaxed": len(representatives),
        "capped": capped,
        "cluster_sizes": sizes,
        "ff_kcal": ff_kcal,
        "mace_kcal": mace_kcal,
        "converged": converged_flags,
        "rmsd": distances.tolist(),
        "spearman": rho,
        "seconds": round(time.perf_counter() - started, 1),
    }


def select(
    rule: str,
    budget: int,
    ff_kcal: list[float],
    sizes: list[int],
    distances: np.ndarray,
    rng: random.Random,
) -> list[int]:
    """Choose which representatives a rule would send to the MLIP.

    Parameters
    ----------
    rule
        One of the names in RULES.
    budget
        Number of representatives the rule may keep.
    ff_kcal
        Force-field energy of each representative, relative to the lowest.
    sizes
        Population of the cluster each representative stands for.
    distances
        Pairwise heavy-atom RMSD between representatives.
    rng
        Source of randomness for the baseline rule.

    Returns
    -------
    chosen
        Indices of the selected representatives.

    Raises
    ------
    ValueError
        If the rule name is not recognised.

    """
    count = len(ff_kcal)
    by_energy = sorted(range(count), key=lambda i: ff_kcal[i])

    if rule == "energy":
        return by_energy[:budget]
    if rule == "population":
        # Butina returns clusters largest-first, so this is the order the
        # representatives already arrive in.
        return list(range(count))[:budget]
    if rule == "random":
        return rng.sample(range(count), min(budget, count))
    if rule == "maxmin":
        return _farthest_point(by_energy[:1], budget, count, distances)
    if rule == "tiered":
        seed = by_energy[: max(1, budget // 2)]
        return _farthest_point(seed, budget, count, distances)
    raise ValueError(f"unknown rule: {rule!r}")


def _farthest_point(
    seed: list[int], budget: int, count: int, distances: np.ndarray
) -> list[int]:
    """Extend a seed set by repeatedly taking the most dissimilar point.

    Parameters
    ----------
    seed
        Indices the selection starts from.
    budget
        Total number of indices to return.
    count
        Number of candidates available.
    distances
        Pairwise heavy-atom RMSD between candidates.

    Returns
    -------
    chosen
        The seed followed by the farthest-point additions.

    """
    chosen = list(seed)
    remaining = [i for i in range(count) if i not in set(chosen)]
    while len(chosen) < budget and remaining:
        picked = max(
            remaining, key=lambda i: min(distances[i][j] for j in chosen)
        )
        chosen.append(picked)
        remaining.remove(picked)
    return chosen


def analyse(records: list[dict[str, Any]], budgets: list[int]) -> None:
    """Print the summary tables the design decision needs.

    Parameters
    ----------
    records
        Per-molecule measurements from profile_molecule.
    budgets
        Selection budgets to evaluate the rules at.

    """
    usable = [r for r in records if not r["capped"] and r["n_relaxed"] > 1]
    print(f"\nmolecules profiled: {len(records)}  usable: {len(usable)}")
    if not usable:
        print("nothing to analyse")
        return

    clusters = [r["n_clusters_found"] for r in records]
    print("\n=== 1. clusters found per molecule ===")
    print(
        f"  min {min(clusters)}  median {int(np.median(clusters))}  "
        f"mean {np.mean(clusters):.1f}  max {max(clusters)}"
    )
    for cap in (4, 8, 12):
        share = sum(1 for c in clusters if c <= cap) / len(clusters)
        print(f"  molecules with <= {cap:2} clusters: {100 * share:.0f}%")

    print("\n=== 2. force-field rank of the MACE minimum ===")
    ranks = []
    for record in usable:
        target = int(np.argmin(record["mace_kcal"]))
        order = sorted(
            range(record["n_relaxed"]), key=lambda i: record["ff_kcal"][i]
        )
        ranks.append(order.index(target) + 1)
    print(
        f"  rank 1 (force field got it right): "
        f"{100 * sum(r == 1 for r in ranks) / len(ranks):.0f}%"
    )
    for cap in (4, 8, 12):
        share = sum(1 for r in ranks if r <= cap) / len(ranks)
        print(f"  within the top {cap:2}: {100 * share:.0f}%")
    print(f"  worst rank observed: {max(ranks)}")

    print("\n=== 3. Spearman rho, force field vs MACE ===")
    rhos = [r["spearman"] for r in usable if not np.isnan(r["spearman"])]
    if rhos:
        print(
            f"  median {np.median(rhos):.2f}  mean {np.mean(rhos):.2f}  "
            f"min {min(rhos):.2f}  max {max(rhos):.2f}"
        )
        negative = sum(1 for v in rhos if v < 0)
        print(f"  molecules with negative correlation: {negative}")

    print("\n=== 4. recall of the MACE minimum, by rule and budget ===")
    header = "  budget " + "".join(f"{rule:>12}" for rule in RULES)
    print(header)
    rng = random.Random(0xF00D)
    for budget in budgets:
        cells = []
        for rule in RULES:
            hits = 0
            for record in usable:
                chosen = select(
                    rule,
                    budget,
                    record["ff_kcal"],
                    record["cluster_sizes"],
                    np.asarray(record["rmsd"]),
                    rng,
                )
                if int(np.argmin(record["mace_kcal"])) in chosen:
                    hits += 1
            cells.append(f"{100 * hits / len(usable):>11.0f}%")
        print(f"  {budget:>6} " + "".join(cells))


def main() -> None:
    """Run the experiment from the command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="bh_full")
    parser.add_argument(
        "--limit", type=int, default=0, help="0 profiles every molecule"
    )
    parser.add_argument("--max-relax", type=int, default=20)
    parser.add_argument(
        "--out", type=Path, default=Path("exp_ff_vs_mace_ranking.json")
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--analyse-only", action="store_true")
    args = parser.parse_args()

    setup_logging()

    budgets = [1, 2, 4, 6, 8, 10, 12]
    records: list[dict[str, Any]] = []
    if (args.resume or args.analyse_only) and args.out.is_file():
        with args.out.open(encoding="utf-8") as handle:
            records = json.load(handle)
        logger.info("resuming with %d cached molecules", len(records))

    if not args.analyse_only:
        done = {r["smiles"] for r in records}
        targets = [s for s in unique_smiles(args.dataset) if s not in done]
        if args.limit:
            targets = targets[: args.limit]

        generator = ConformerGenerator()
        logger.info("profiling %d molecules", len(targets))

        with BlockLogs():
            for index, smiles in enumerate(targets, 1):
                try:
                    record = profile_molecule(
                        generator, smiles, args.max_relax
                    )
                except (ValueError, RuntimeError):
                    logger.exception("skipping %r", smiles)
                    continue
                records.append(record)
                logger.info(
                    "%d/%d  %d clusters, %d relaxed, rho %.2f, %.0f s",
                    index,
                    len(targets),
                    record["n_clusters_found"],
                    record["n_relaxed"],
                    record["spearman"],
                    record["seconds"],
                )
                with args.out.open("w", encoding="utf-8") as handle:
                    json.dump(records, handle, indent=2)

    analyse(records, budgets)


if __name__ == "__main__":
    main()
