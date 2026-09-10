"""
phys_descriptor.py
==================

The "physical-descriptor-only" representation (the interpretability experiment).

Instead of MACE/AIMNet2 *latent* embeddings, we build the molecular feature
vector from the foundation models' **predicted physical properties** + a couple
of tabulated atomic constants, every dimension chemically named so the surrogate
is SHAP-interpretable:

  per-atom (statistical pooling: mean / std / min / max across atoms)
    - MH-1   site energy            (node_energy, eV)  -- local stability
    - MH-1   |force|                (eV/A)             -- local strain/reactivity
    - POLAR  partial charge q                          -- electronics
    - POLAR  spin density
    - POLAR  density coefficients dc0..dc3             -- atomic multipole/polarizability
    - electronegativity (Pauling, tabulated)
    - covalent radius   (Cordero,  tabulated)
  molecular scalars (one value / molecule)
    - POLAR  |dipole|
    - POLAR  electron / electrostatic / interaction energy  (per atom -> intensive)
    - net charge  (sum q)
    - MH-1   energy per atom
    - n_atoms

POLAR's electronic predictions blow up for alkali-metal ions, so components that
contain one fall back to an MH-1-only physical descriptor (fewer, MH-1-derived
features). Each component is an independent BayBE parameter, so differing dims
across components are fine.

Conformers handled exactly like the latent pipeline (random / lowest / boltzmann
via generate_conformers + aggregate_embeddings on the conformer feature vectors).

Out -> bench_out_phys/<ds>_<case>/curves__phys.csv  (+ report, + SHAP).
Resumable. Nothing else is overwritten.

Usage:
    python phys_descriptor.py --selftest
    python phys_descriptor.py --datasets shields --cases boltzmann
    python phys_descriptor.py --datasets shields --cases boltzmann --shap
"""
from __future__ import annotations
import os, sys, argparse, base64
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
import warnings; warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, "HSF-ChemBO-tutorial"))

import representations as R
import benchmark_representations as B
from conformer_ab import load_target, CB, DISP
from conformer_embedding import generate_conformers, aggregate_embeddings

OUT = os.path.join(HERE, "bench_out_phys")
CASES = ["random", "lowest", "boltzmann"]
CASE_AGG = {"random": "mean", "lowest": "lowest", "boltzmann": "boltzmann"}
CASE_NCONF = {"random": 1, "lowest": 20, "boltzmann": 20}
POLAR_UNSAFE = {3, 11, 19, 37, 55}   # Li, Na, K, Rb, Cs

# old MACE-MP-0 latent baselines + MH-1+POLAR latent combo for comparison
MP0_DIR = {"random": os.path.join(HERE, "bench_out"),
           "lowest": os.path.join(HERE, "bench_out_conformer", "shields_lowest"),
           "boltzmann": os.path.join(HERE, "bench_out_conformer", "shields_ensemble")}
COMBO_DIR = {c: os.path.join(HERE, "bench_out_mh", f"shields_{c}_mean") for c in CASES}

# tabulated atomic constants (Pauling EN; Cordero covalent radius, A)
PAULING_EN = {1: 2.20, 5: 2.04, 6: 2.55, 7: 3.04, 8: 3.44, 9: 3.98, 14: 1.90,
              15: 2.19, 16: 2.58, 17: 3.16, 33: 2.18, 34: 2.55, 35: 2.96, 53: 2.66,
              46: 2.20, 26: 1.83, 29: 1.90, 30: 1.65, 3: 0.98, 11: 0.93, 19: 0.82,
              37: 0.82, 55: 0.79}
COV_R = {1: 0.31, 5: 0.84, 6: 0.76, 7: 0.71, 8: 0.66, 9: 0.57, 14: 1.11, 15: 1.07,
         16: 1.05, 17: 1.02, 33: 1.21, 34: 1.20, 35: 1.20, 53: 1.39, 46: 1.39,
         26: 1.32, 29: 1.32, 30: 1.22, 3: 1.28, 11: 1.66, 19: 2.03, 37: 2.20, 55: 2.44}

STATS = ("mean", "std", "min", "max")


def _polar_safe(sd):
    from rdkit import Chem
    for s in sd.values():
        m = Chem.MolFromSmiles(s)
        if m and ({a.GetAtomicNum() for a in m.GetAtoms()} & POLAR_UNSAFE):
            return False
    return True


# ---------------------------------------------------------------------------
# physical-feature backend
# ---------------------------------------------------------------------------
class _PhysBackend:
    """Per-atom physical features + molecular scalars from MH-1 (+ optional POLAR)."""

    def __init__(self, use_polar=True, head="spice_wB97M", polar="polar-1-m", device="cpu"):
        from mace.calculators import mace_mp
        self.use_polar = use_polar
        self.mh = mace_mp(model=os.path.expanduser("~/.cache/mace/mace-mh-1.model"),
                          device=device, default_dtype="float64", head=head)
        self.pol = None
        if use_polar:
            try:
                from mace.calculators import mace_polar
            except Exception:
                from mace.calculators.foundations_models import mace_polar
            self.pol = mace_polar(model=polar, device=device, default_dtype="float64")

    # feature naming -------------------------------------------------------
    def atom_feature_names(self):
        if self.use_polar:
            return ["site_energy", "force_mag", "charge", "spin",
                    "dens0", "dens1", "dens2", "dens3", "electronegativity", "cov_radius"]
        return ["site_energy", "force_mag", "electronegativity", "cov_radius"]

    def mol_feature_names(self):
        # mh_E_total = SUM of site energies = total MLIP energy (the extensive
        # "sum-pooled" energy advised for atom->molecule pooling); kept alongside
        # the intensive per-atom version + explicit n_atoms.
        if self.use_polar:
            return ["dipole_mag", "electron_E_per_atom", "electrostatic_E_per_atom",
                    "interaction_E_per_atom", "net_charge", "mh_E_per_atom", "mh_E_total", "n_atoms"]
        return ["mh_E_per_atom", "mh_E_total", "n_atoms"]

    def feature_names(self, pooling="stats"):
        if pooling == "sum":      # extensive sum aggregator (DeepSets/MACE-style)
            names = [f"{a}_sum" for a in self.atom_feature_names()]
        else:                     # statistical pooling (distribution, intensive)
            names = [f"{a}_{s}" for a in self.atom_feature_names() for s in STATS]
        names += self.mol_feature_names()
        return names

    # per-conformer computation -------------------------------------------
    def conformer_features(self, numbers, coords, charge):
        """Return (energy_for_ranking, per_atom (N,Fa), mol_scalars (Fm,))."""
        from ase import Atoms
        numbers = np.asarray(numbers)
        N = len(numbers)
        en = np.array([PAULING_EN.get(int(z), 2.2) for z in numbers], dtype=np.float64)
        rc = np.array([COV_R.get(int(z), 1.2) for z in numbers], dtype=np.float64)

        atoms = Atoms(numbers=numbers, positions=coords); atoms.calc = self.mh
        e_mh = float(atoms.get_potential_energy())
        fmag = np.linalg.norm(atoms.get_forces(), axis=1)              # (N,)
        site_e = np.asarray(self.mh.results["node_energy"], dtype=np.float64).reshape(-1)

        if self.use_polar:
            a2 = Atoms(numbers=numbers, positions=coords); a2.calc = self.pol
            a2.get_potential_energy()
            r = self.pol.results
            q = np.asarray(r["charges"], dtype=np.float64).reshape(-1)
            sp = np.asarray(r["spins"], dtype=np.float64).reshape(-1)
            dc = np.asarray(r["density_coefficients"], dtype=np.float64)   # (N,4)
            per_atom = np.column_stack([site_e, fmag, q, sp, dc, en, rc])  # (N,10)
            dip = float(np.linalg.norm(np.asarray(r["dipole"], dtype=np.float64)))
            mol = np.array([dip, float(r["electron_energy"]) / N,
                            float(r["electrostatic_energy"]) / N,
                            float(r["interaction_energy"]) / N,
                            float(q.sum()), e_mh / N, e_mh, float(N)], dtype=np.float64)
        else:
            per_atom = np.column_stack([site_e, fmag, en, rc])            # (N,4)
            mol = np.array([e_mh / N, e_mh, float(N)], dtype=np.float64)
        return e_mh, per_atom, mol


def _stats_pool(per_atom):
    """(N,Fa) -> (4*Fa,) : [mean, std, min, max] per atom-feature, interleaved by feature."""
    m = per_atom.mean(axis=0); s = per_atom.std(axis=0)
    lo = per_atom.min(axis=0); hi = per_atom.max(axis=0)
    return np.concatenate([np.stack([m[i], s[i], lo[i], hi[i]]) for i in range(per_atom.shape[1])])


def _sum_pool(per_atom):
    """(N,Fa) -> (Fa,) : extensive sum over atoms (the conference-advised aggregator)."""
    return per_atom.sum(axis=0)


def make_phys_fingerprinter(use_polar=True, *, pooling="stats", n_confs=20, aggregate="boltzmann",
                            temperature=298.15, prune_rms=0.3, seed=42, backend_obj=None):
    be = backend_obj if backend_obj is not None else _PhysBackend(use_polar=use_polar)
    poolfn = _sum_pool if pooling == "sum" else _stats_pool
    cache: dict = {}

    def _collect(m):
        if m not in cache:
            numbers, confs, charge = generate_conformers(m, n_confs=n_confs, seed=seed,
                                                         prune_rms=prune_rms)
            energies, vecs = [], []
            for coords in confs:
                e, pa, mol = be.conformer_features(numbers, coords, charge)
                energies.append(e)
                vecs.append(np.concatenate([poolfn(pa), mol]))
            cache[m] = (np.array(energies), np.stack(vecs, axis=0))
        return cache[m]

    def _fp(molecules):
        if isinstance(molecules, str):
            molecules = [molecules]
        out = []
        for m in molecules:
            energies, per_conf = _collect(m)
            emb, _ = aggregate_embeddings(per_conf, energies, aggregate, temperature)
            out.append(emb.astype(np.float64))
        return np.stack(out, axis=0)

    _fp.feature_names = be.feature_names(pooling)
    _fp.backend = be
    return _fp


def make_gollum_phys_fp(head="spice_wB97M", polar="polar-1-m", device="cpu", seed=42):
    """Robust single-conformer physical-descriptor fingerprinter for the GOLLuM
    benchmark: callable(list[smiles]) -> (n, 48) float (stats pooling).

    Always returns the full 48-dim vector. POLAR-derived dims are ZERO-FILLED for
    alkali-metal molecules (POLAR blows up) or on any POLAR error/non-finite, so the
    dimension is uniform across molecules (required by the per-SMILES embedding cache).
    MH-1 features (site energy, |force|) + tabulated EN/radius are always present.
    Single ETKDG conformer (matches the MLIP-latent reps in the stack). Raises only if
    MH-1 / conformer generation fails for a molecule (caught upstream by embed_smiles).
    """
    from ase import Atoms
    from mace_repr import smiles_to_atoms
    be = _PhysBackend(use_polar=True, head=head, polar=polar, device=device)

    def _one(smi):
        numbers, coords, charge = smiles_to_atoms(smi, seed=seed)   # single conformer
        numbers = np.asarray(numbers); N = len(numbers)
        en = np.array([PAULING_EN.get(int(z), 2.2) for z in numbers], dtype=np.float64)
        rc = np.array([COV_R.get(int(z), 1.2) for z in numbers], dtype=np.float64)
        atoms = Atoms(numbers=numbers, positions=coords); atoms.calc = be.mh
        e_mh = float(atoms.get_potential_energy())
        fmag = np.linalg.norm(atoms.get_forces(), axis=1)
        site_e = np.asarray(be.mh.results["node_energy"], dtype=np.float64).reshape(-1)
        # POLAR block: zero-fill for alkali metals or any failure/non-finite.
        q = np.zeros(N); sp = np.zeros(N); dc = np.zeros((N, 4))
        dip = eE = esE = iE = 0.0
        if not ({int(z) for z in numbers} & POLAR_UNSAFE):
            try:
                a2 = Atoms(numbers=numbers, positions=coords); a2.calc = be.pol
                a2.get_potential_energy(); r = be.pol.results
                q_ = np.asarray(r["charges"], np.float64).reshape(-1)
                sp_ = np.asarray(r["spins"], np.float64).reshape(-1)
                dc_ = np.asarray(r["density_coefficients"], np.float64)
                dip_ = float(np.linalg.norm(np.asarray(r["dipole"], np.float64)))
                eE_, esE_, iE_ = (float(r["electron_energy"]), float(r["electrostatic_energy"]),
                                  float(r["interaction_energy"]))
                blob = np.concatenate([q_, sp_, dc_.ravel(), [dip_, eE_, esE_, iE_]])
                if np.all(np.isfinite(blob)) and np.abs(blob).max() < 1e6:
                    q, sp, dc, dip, eE, esE, iE = q_, sp_, dc_, dip_, eE_, esE_, iE_
            except Exception:
                pass
        per_atom = np.column_stack([site_e, fmag, q, sp, dc, en, rc])      # (N,10)
        mol = np.array([dip, eE / N, esE / N, iE / N, float(q.sum()),
                        e_mh / N, e_mh, float(N)], dtype=np.float64)        # (8,)
        return np.concatenate([_stats_pool(per_atom), mol])                # (48,)

    def _fp(molecules):
        if isinstance(molecules, str):
            molecules = [molecules]
        return np.stack([_one(m) for m in molecules], axis=0)

    _fp.feature_names = be.feature_names("stats")
    return _fp


# ---------------------------------------------------------------------------
# BO campaign
# ---------------------------------------------------------------------------
def build_campaign(case, pooling, comps, numeric, n_confs):
    from baybe import Campaign
    from baybe.objectives import SingleTargetObjective
    from baybe.targets import NumericalTarget
    from baybe.parameters import NumericalDiscreteParameter
    from baybe.searchspace import SearchSpace
    from baybe.surrogates import GaussianProcessSurrogate
    from baybe.recommenders import BotorchRecommender, RandomRecommender, TwoPhaseMetaRecommender
    from base.kernels import AdaptiveKernelFactory

    agg = CASE_AGG[case]
    comp_kind = {comp: ("phys_full" if _polar_safe(sd) else "phys_mh") for comp, sd in comps.items()}
    fps = {}
    for kind in set(comp_kind.values()):
        fps[kind] = make_phys_fingerprinter(use_polar=(kind == "phys_full"), pooling=pooling,
                                            n_confs=n_confs, aggregate=agg)
    # No explicit scaling: BayBE's GP applies botorch Normalize (min-max over the
    # candidate pool) to every comp_rep column, and min-max is invariant to any
    # affine pre-scaling -- so it would wash out. Identical preprocessing to the
    # latent baselines => clean comparison.
    params = [R.baybe_parameter(comp_kind[comp], comp, sd, fingerprinter=fps[comp_kind[comp]],
                                decorrelate=0.7) for comp, sd in comps.items()]
    for nm, vals in numeric:
        params.append(NumericalDiscreteParameter(name=nm, values=set(vals)))
    ss = SearchSpace.from_product(parameters=params)
    surrogate = GaussianProcessSurrogate(kernel_or_factory=AdaptiveKernelFactory())
    rec = TwoPhaseMetaRecommender(
        initial_recommender=RandomRecommender(),
        recommender=BotorchRecommender(surrogate_model=surrogate, acquisition_function="qLogEI"),
        switch_after=5)
    obj = SingleTargetObjective(target=NumericalTarget(name="yield", mode="MAX"))
    return Campaign(searchspace=ss, objective=obj, recommender=rec), len(ss.comp_rep_columns)


def run_cell(ds, case, pooling, comps, numeric, lookup, n_confs, mc, niter):
    from baybe.simulation import simulate_scenarios
    from baybe.utils.random import set_random_seed
    d = os.path.join(OUT, f"{ds}_{case}_{pooling}"); os.makedirs(d, exist_ok=True)
    csv = os.path.join(d, "curves__phys.csv")
    if os.path.exists(csv):
        return pd.read_csv(csv), "resumed"
    camp, dim = build_campaign(case, pooling, comps, numeric, n_confs)
    set_random_seed(1337)
    res = simulate_scenarios({"phys": camp}, lookup, batch_size=1, n_doe_iterations=niter,
                             n_mc_iterations=mc, impute_mode="ignore")
    res["representation"] = "phys"; res["dim"] = dim
    res.to_csv(csv, index=False)
    return res, f"dim={dim}"


def _auc(curves, lookup):
    return B.per_run_metrics(curves, lookup).AUC.mean()


def report(datasets, cases, poolings):
    secs = ""
    fmt = lambda v: "" if v is None else f"{v:.3f}"
    for ds in datasets:
        lookup, _, _ = load_target(ds)
        rows = ""
        for case in cases:
            mp = os.path.join(MP0_DIR.get(case, ""), "curves__mace_mp0.csv")
            mp_auc = _auc(pd.read_csv(mp), lookup) if os.path.exists(mp) else None
            cb = os.path.join(COMBO_DIR.get(case, ""), "curves__mh1_polar.csv")
            cb_auc = _auc(pd.read_csv(cb), lookup) if os.path.exists(cb) else None
            pool_cells = ""
            for pooling in poolings:
                ph = os.path.join(OUT, f"{ds}_{case}_{pooling}", "curves__phys.csv")
                ph_auc = _auc(pd.read_csv(ph), lookup) if os.path.exists(ph) else None
                dM = (ph_auc - mp_auc) if (ph_auc and mp_auc) else None
                pool_cells += (f"<td><b>{fmt(ph_auc)}</b></td>"
                               f"<td class='{'g' if (dM or 0)>0 else 'r'}'>"
                               f"{'' if dM is None else f'{dM:+.3f}'}</td>")
            rows += (f"<tr><td>{case}</td><td>{fmt(mp_auc)}</td>"
                     f"<td>{fmt(cb_auc)}</td>{pool_cells}</tr>")
        phead = "".join(f"<th>phys-{p} AUC</th><th>Δ vs MP-0</th>" for p in poolings)
        secs += (f"<h2>{ds}</h2><table><tr><th>conformer case</th>"
                 f"<th>MACE-MP-0 latent</th><th>MH-1+POLAR latent combo</th>{phead}</tr>"
                 f"{rows}</table>")
    html = f"""<!doctype html><html><head><meta charset="utf-8"><title>Physical descriptor</title>
<style>body{{font-family:Segoe UI,Arial;max-width:1050px;margin:22px auto;padding:0 16px}}
table{{border-collapse:collapse;width:100%;margin:10px 0}}th,td{{border:1px solid #ddd;padding:6px 8px;text-align:center}}
th{{background:#f5f7fa}}.g{{color:#1a7f37;font-weight:bold}}.r{{color:#b00}}
h2{{border-bottom:2px solid #eee;padding-bottom:5px;margin-top:26px}}
.key{{background:#fbfbe8;border-left:4px solid #d9b400;padding:10px 14px;margin:14px 0}}</style></head><body>
<h1>Physical-descriptor-only representation vs latent embeddings</h1>
<p>Molecular features = MH-1/POLAR <b>predicted physical properties</b> (charges, spins, density
coefficients, site energies, forces, dipole, electronic energies) + tabulated atomic constants.
Two atom→molecule poolings compared: <b>stats</b> (mean/std/min/max — intensive, distribution) and
<b>sum</b> (extensive DeepSets/MACE-style aggregator). Decorr 0.7, 20 MC × 50 iter. Every dimension
is chemically named → SHAP-interpretable. Δ &gt; 0 ⇒ the physical descriptor matches/beats MACE-MP-0 latent.</p>
<div class="key">If a physical descriptor stays close on AUC, it is the preferable embedding for a
chemist: SHAP attributions map to real chemistry, and the dimension is far lower. The stats-vs-sum
columns test directly whether the conference-advised <i>sum</i> pooling helps or hurts for a
cross-molecule GP descriptor.</div>
{secs}</body></html>"""
    fn = os.path.join(OUT, "phys_report.html")
    with open(fn, "w", encoding="utf-8") as f:
        f.write(html)
    print("wrote", fn)


def selftest():
    be = _PhysBackend(use_polar=True)
    for pooling in ("stats", "sum"):
        fp = make_phys_fingerprinter(pooling=pooling, n_confs=3, aggregate="boltzmann",
                                     backend_obj=be)
        X = fp(["CCO", "c1ccccc1N"])
        assert X.shape[1] == len(fp.feature_names)
        print(f"full {pooling}: dim={X.shape[1]}  e.g. {fp.feature_names[:3]}...{fp.feature_names[-3:]}")
    be2 = _PhysBackend(use_polar=False)
    for pooling in ("stats", "sum"):
        fp = make_phys_fingerprinter(use_polar=False, pooling=pooling, n_confs=2,
                                     aggregate="lowest", backend_obj=be2)
        print(f"mh-only {pooling}: dim={fp(['CCO']).shape[1]}")
    print("SELFTEST OK")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["shields"])
    ap.add_argument("--cases", nargs="+", default=["boltzmann"], choices=CASES)
    ap.add_argument("--poolings", nargs="+", default=["stats", "sum"], choices=["stats", "sum"])
    ap.add_argument("--mc-runs", type=int, default=20)
    ap.add_argument("--n-iter", type=int, default=50)
    ap.add_argument("--shap", action="store_true")
    ap.add_argument("--reports-only", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        selftest(); return
    os.makedirs(OUT, exist_ok=True)
    if not a.reports_only:
        for ds in a.datasets:
            lookup, comps, numeric = load_target(ds)
            for case in a.cases:
                for pooling in a.poolings:
                    print(f"\n##### phys {ds} / {case} / {pooling} #####", flush=True)
                    _, info = run_cell(ds, case, pooling, comps, numeric, lookup,
                                       CASE_NCONF[case], a.mc_runs, a.n_iter)
                    print(f"  {info}", flush=True)
    report(a.datasets, a.cases, a.poolings)
    if a.shap:
        from phys_shap import run_shap
        for ds in a.datasets:
            for case in a.cases:
                run_shap(ds, case)   # SHAP on the interpretable stats descriptor
    print("DONE")


if __name__ == "__main__":
    main()
