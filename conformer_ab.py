"""
conformer_ab.py
===============

A/B test: does a CONFORMER-ENSEMBLE MLIP embedding (conformer_embedding.py,
Boltzmann over an ETKDG/MMFF ensemble scored by the MLIP) beat the SINGLE
MMFF-relaxed ETKDG conformer used everywhere else?

Datasets : Shields, bh_reaction_1
MLIPs    : aimnet2, mace_off23, mace_mp0
Dim ctrl : decorrelation 0.7 (both arms — so only the conformer treatment differs)
BO       : Adaptive_ours, qLogEI, Matérn 5/2, 5 init, 50 iter, 20 MC

Arms:
  * "single"   -> representations.make_fingerprinter        (1 conformer)
  * "ensemble" -> conformer_embedding.make_ensemble_fingerprinter (N confs, Boltzmann)

Shields single arm is reused from bench_out/ (already decorr 0.7); everything
else is computed here. Out -> bench_out_conformer/<dataset>_<arm>/ + comparison
report per dataset + overall conformer_ab_report.html. Resumable per (dataset,arm,rep).

Usage:
    python conformer_ab.py [--n-confs 20] [--aggregate boltzmann] [--mc-runs 20] [--n-iter 50]
"""
from __future__ import annotations
import os, sys, argparse, base64, time
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon

os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
import warnings; warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, "HSF-ChemBO-tutorial"))

import representations as R
import benchmark_representations as B
import gollum_pipeline as GP
from conformer_embedding import make_ensemble_fingerprinter

MLIPS = ["aimnet2", "mace_off23", "mace_mp0"]
DISP = {"aimnet2": "AIMNet2", "mace_off23": "MACE-OFF23", "mace_mp0": "MACE-MP-0"}
OUT = os.path.join(HERE, "bench_out_conformer")
CB = "yield_CumBest"
_FPCACHE = {}


def b64(p):
    return "data:image/png;base64," + base64.b64encode(open(p, "rb").read()).decode()


def get_fp(arm, backend, n_confs, aggregate):
    key = (arm, backend)
    if key not in _FPCACHE:
        if arm == "single":
            _FPCACHE[key] = R.make_fingerprinter(backend)
        else:
            _FPCACHE[key] = make_ensemble_fingerprinter(
                backend=backend, n_confs=n_confs, aggregate=aggregate, relax=False)
    return _FPCACHE[key]


def load_target(name):
    if name == "shields":
        lookup, comps = B.load_shields()
        numeric = [("Temp_C", sorted(set(lookup.Temp_C))),
                   ("Concentration", sorted(set(lookup.Concentration)))]
        return lookup, comps, numeric
    cfg = next(d for d in GP.DATASETS if d["name"] == name)
    L = GP.load_dataset(cfg)
    return L["lookup"], {c: L["components"][c] for c in L["comp_cols"]}, []


def build_campaign(rep, arm, comps, numeric, n_confs, aggregate):
    from baybe import Campaign
    from baybe.objectives import SingleTargetObjective
    from baybe.targets import NumericalTarget
    from baybe.parameters import NumericalDiscreteParameter
    from baybe.searchspace import SearchSpace
    from baybe.surrogates import GaussianProcessSurrogate
    from baybe.recommenders import BotorchRecommender, RandomRecommender, TwoPhaseMetaRecommender
    from base.kernels import AdaptiveKernelFactory

    params, notes = [], []
    for comp, smiles_dict in comps.items():
        smis = list(smiles_dict.values())
        use = rep
        if R.uncovered_elements(rep, smis):       # Cs/K bases etc.
            use = "mace_mp0"; notes.append(f"{comp}->mace_mp0")
        fp = get_fp(arm, use, n_confs, aggregate)
        params.append(R.baybe_parameter(use, comp, smiles_dict, fingerprinter=fp,
                                        decorrelate=0.7))
    for nm, vals in numeric:
        params.append(NumericalDiscreteParameter(name=nm, values=set(vals)))

    ss = SearchSpace.from_product(parameters=params)
    surrogate = GaussianProcessSurrogate(kernel_or_factory=AdaptiveKernelFactory())
    rec = TwoPhaseMetaRecommender(
        initial_recommender=RandomRecommender(),
        recommender=BotorchRecommender(surrogate_model=surrogate, acquisition_function="qLogEI"),
        switch_after=5)
    objective = SingleTargetObjective(target=NumericalTarget(name="yield", mode="MAX"))
    camp = Campaign(searchspace=ss, objective=objective, recommender=rec)
    return camp, len(ss.comp_rep_columns), ";".join(notes)


def run_rep(dataset, arm, rep, comps, numeric, lookup, n_confs, aggregate, mc, niter):
    from baybe.simulation import simulate_scenarios
    from baybe.utils.random import set_random_seed
    d = os.path.join(OUT, f"{dataset}_{arm}"); os.makedirs(d, exist_ok=True)
    csv = os.path.join(d, f"curves__{rep}.csv")
    if os.path.exists(csv):
        return pd.read_csv(csv), "resumed"
    t0 = time.time()
    camp, dim, notes = build_campaign(rep, arm, comps, numeric, n_confs, aggregate)
    set_random_seed(1337)
    res = simulate_scenarios({rep: camp}, lookup, batch_size=1, n_doe_iterations=niter,
                             n_mc_iterations=mc, impute_mode="ignore")
    res["representation"] = rep; res["dim"] = dim
    res.to_csv(csv, index=False)
    return res, f"dim={dim} t={time.time()-t0:.0f}s notes={notes}"


def curves_for(dataset, arm):
    """Return {rep: curves_df}. Shields single arm is reused from bench_out/."""
    if dataset == "shields" and arm == "single":
        out = {}
        for rep in MLIPS:
            p = os.path.join(HERE, "bench_out", f"curves__{rep}.csv")
            if os.path.exists(p):
                out[rep] = pd.read_csv(p)
        return out
    d = os.path.join(OUT, f"{dataset}_{arm}")
    out = {}
    for rep in MLIPS:
        p = os.path.join(d, f"curves__{rep}.csv")
        if os.path.exists(p):
            out[rep] = pd.read_csv(p)
    return out


def metrics(curves, lookup):
    allc = pd.concat(curves.values(), ignore_index=True)
    pr = B.per_run_metrics(allc, lookup)
    return pr


def dataset_report(dataset, lookup, n_confs, aggregate):
    cs = curves_for(dataset, "single"); ce = curves_for(dataset, "ensemble")
    reps = [r for r in MLIPS if r in cs and r in ce]
    if not reps:
        return None
    prS = metrics(cs, lookup); prE = metrics(ce, lookup)
    aggS = prS.groupby("representation").agg(AUC=("AUC", "mean"), cov=("Top5pct_coverage", "mean"))
    aggE = prE.groupby("representation").agg(AUC=("AUC", "mean"), cov=("Top5pct_coverage", "mean"))

    # per-rep overlay figure
    d = os.path.join(OUT, f"{dataset}_compare"); os.makedirs(d, exist_ok=True)
    fig, axes = plt.subplots(1, len(reps), figsize=(5.2 * len(reps), 4.4), squeeze=False)
    for k, rep in enumerate(reps):
        ax = axes[0][k]
        ax.axhline(lookup["yield"].max(), ls="--", color="k", lw=1, alpha=.5)
        for cur, lab, c in [(cs[rep], "single", "#888"), (ce[rep], "ensemble", "#1a7f37")]:
            piv = cur.pivot_table(index="Num_Experiments", columns="Monte_Carlo_Run", values=CB)
            ax.plot(piv.index, piv.median(axis=1), lw=2, color=c, label=lab)
            ax.fill_between(piv.index, piv.quantile(.25, axis=1), piv.quantile(.75, axis=1), alpha=.12, color=c)
        ax.set_title(f"{DISP[rep]}  ΔAUC={aggE.loc[rep,'AUC']-aggS.loc[rep,'AUC']:+.3f}")
        ax.set_xlabel("experiments"); ax.legend(fontsize=8, loc="lower right"); ax.grid(alpha=.25)
    axes[0][0].set_ylabel("best objective (median±IQR)")
    fig.suptitle(f"{dataset}: single vs conformer-ensemble ({aggregate}, {n_confs} confs), decorr 0.7")
    fig.tight_layout(); png = os.path.join(d, "compare.png"); fig.savefig(png, dpi=130); plt.close(fig)

    rows = ""
    for rep in reps:
        # paired Wilcoxon on AUC (single vs ensemble), by mc_run
        a = prS[prS.representation == rep].set_index("mc_run").AUC
        b = prE[prE.representation == rep].set_index("mc_run").AUC
        j = a.index.intersection(b.index); a, b = a.loc[j].values, b.loc[j].values
        W, p = (np.nan, 1.0) if np.allclose(a, b) else wilcoxon(b, a)
        dA = aggE.loc[rep, "AUC"] - aggS.loc[rep, "AUC"]
        dC = aggE.loc[rep, "cov"] - aggS.loc[rep, "cov"]
        win = "ensemble" if dA > 0 else "single"
        rows += (f"<tr><td>{DISP[rep]}</td><td>{aggS.loc[rep,'AUC']:.3f}</td><td>{aggE.loc[rep,'AUC']:.3f}</td>"
                 f"<td class='{'g' if dA>0 else 'r'}'>{dA:+.3f}</td>"
                 f"<td>{aggS.loc[rep,'cov']:.3f}</td><td>{aggE.loc[rep,'cov']:.3f}</td><td>{dC:+.3f}</td>"
                 f"<td>{p:.3f}{' ★' if p<0.05 else ''}</td><td>{win}</td></tr>")
    return dict(dataset=dataset, reps=reps, rows=rows, png=png, aggS=aggS, aggE=aggE)


def write_overall(results, n_confs, aggregate):
    sec = ""
    for r in results:
        if r is None:
            continue
        sec += f"""<h2>{r['dataset']}</h2>
<table><tr><th>MLIP</th><th>AUC single</th><th>AUC ensemble</th><th>ΔAUC</th>
<th>cov single</th><th>cov ensemble</th><th>Δcov</th><th>p (Wilcoxon)</th><th>winner</th></tr>{r['rows']}</table>
<img src="{b64(r['png'])}">"""
    html = f"""<!doctype html><html><head><meta charset="utf-8"><title>Conformer A/B</title>
<style>body{{font-family:Segoe UI,Arial;max-width:1100px;margin:22px auto;padding:0 16px}}
table{{border-collapse:collapse;width:100%;margin:10px 0;font-size:14px}}th,td{{border:1px solid #ddd;padding:6px 8px;text-align:center}}
th{{background:#f5f7fa}}img{{width:100%;border:1px solid #e3e3e3;border-radius:6px;margin:8px 0}}
.g{{color:#1a7f37;font-weight:bold}}.r{{color:#b00}}h2{{border-bottom:2px solid #eee;padding-bottom:5px;margin-top:30px}}
.key{{background:#fbfbe8;border-left:4px solid #d9b400;padding:10px 14px;margin:14px 0}}</style></head><body>
<h1>Conformer treatment A/B — single vs ensemble MLIP embeddings</h1>
<p>Single MMFF-relaxed ETKDG conformer vs a {n_confs}-conformer ensemble ({aggregate}-weighted by MLIP energy),
for AIMNet2 / MACE-OFF23 / MACE-MP-0. Decorrelation 0.7 in both arms, so the only difference is the conformer
treatment. 20 MC × 50 iter, Adaptive_ours, qLogEI. Δ = ensemble − single (positive ⇒ ensemble helps).</p>
<div class="key">Use this to decide whether the conformer-ensemble upgrade is worth the extra forward passes,
and whether it specifically helps the high-variance / floppy-molecule cases (AIMNet2 on Shields, MACE on
bh_reaction_1's bulky organophosphorus reagents).</div>
{sec}
</body></html>"""
    with open(os.path.join(OUT, "conformer_ab_report.html"), "w", encoding="utf-8") as f:
        f.write(html)
    print("wrote", os.path.join(OUT, "conformer_ab_report.html"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-confs", type=int, default=20)
    ap.add_argument("--aggregate", default="boltzmann",
                    choices=["boltzmann", "lowest", "mean", "max"])
    ap.add_argument("--mc-runs", type=int, default=20)
    ap.add_argument("--n-iter", type=int, default=50)
    ap.add_argument("--datasets", nargs="+", default=["shields", "bh_reaction_1"])
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)

    # arms to compute: ensemble for both; single only for bh_reaction_1 (Shields single reused)
    plan = []
    for ds in a.datasets:
        plan.append((ds, "ensemble"))
        if ds != "shields":
            plan.append((ds, "single"))

    for ds, arm in plan:
        lookup, comps, numeric = load_target(ds)
        print(f"\n##### {ds} / {arm} #####", flush=True)
        for rep in MLIPS:
            _, info = run_rep(ds, arm, rep, comps, numeric, lookup,
                              a.n_confs, a.aggregate, a.mc_runs, a.n_iter)
            print(f"  {rep}: {info}", flush=True)

    results = []
    for ds in a.datasets:
        lookup, _, _ = load_target(ds)
        results.append(dataset_report(ds, lookup, a.n_confs, a.aggregate))
    write_overall(results, a.n_confs, a.aggregate)
    print("DONE")


if __name__ == "__main__":
    main()
