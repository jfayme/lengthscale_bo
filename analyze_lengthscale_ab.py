"""
analyze_lengthscale_ab.py
=========================
Analysis of the paired lengthscale A/B (`run_output/lengthscale_ab.csv`).

Produces, per prior mode and under BOTH metrics (lift and simple regret):

  1. the paired difference geom - chen, at run level and clustered by family,
     with a Wilcoxon signed-rank test on the FAMILY means -- that is the honest
     unit of replication, because the Buchwald-Hartwig sub-grids are correlated
     slices of one chemistry, as are the four additive plates;
  2. the same split by representation;
  3. a scatter of ell_0/ell_star against lift, coloured by representation;
  4. a scatter of ell_0/ell_star (before fitting) against ell_fitted/ell_star
     (after), with the identity line -- points on the diagonal mean the marginal
     -likelihood fit did not move the lengthscale, which is the single most
     informative plot in the study;
  5. a markdown summary -> run_output/lengthscale_ab_analysis.md.

Headline metric is LIFT, not raw AUC: raw AUC is dominated by how easy a dataset
is (dataset identity explains ~76% of AUC variance but only ~9% of lift variance).
Raw AUC stays in the CSV.

Degenerate cells (all pairwise distances equal -- e.g. a one-hot single-component
pool, where every candidate ties and the campaign picks in file order) carry no
information about either rule and are EXCLUDED from every aggregate, then reported
separately.

    python analyze_lengthscale_ab.py
    python analyze_lengthscale_ab.py --csv run_output/lengthscale_ab.csv --no-figures
"""
from __future__ import annotations

import os
import sys
import argparse

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, "run_output")
DEFAULT_CSV = os.path.join(OUT_DIR, "lengthscale_ab*.csv")  # glob: the sweep shards
SUMMARY_PATH = os.path.join(OUT_DIR, "lengthscale_ab_analysis.md")

# Pairing keys: one (cell, seed) is one paired observation; the arms differ only
# in `lengthscale_rule`.
PAIR_KEYS = ["dataset", "family", "representation", "reduction", "prior_mode", "seed"]
METRICS = {
    "lift": "higher is better",
    "simple_regret": "LOWER is better",
    "top5_coverage": "higher is better",
}


def load(csv_path):
    """Read one CSV, or every CSV matching a glob (the sweep's per-shard files)."""
    import glob as globlib

    paths = sorted(globlib.glob(csv_path)) if any(c in csv_path for c in "*?[") else [csv_path]
    if not paths:
        raise SystemExit(f"no results matching {csv_path}")
    frame = pd.concat([pd.read_csv(path) for path in paths], ignore_index=True)
    frame = frame.drop_duplicates(subset=PAIR_KEYS + ["lengthscale_rule"], keep="last")
    if "degenerate" not in frame:
        frame["degenerate"] = False
    frame["degenerate"] = frame["degenerate"].astype(bool)
    return frame


def paired_differences(frame, metric):
    """One row per (cell, seed): the geom - chen difference in `metric`."""
    wide = frame.pivot_table(index=PAIR_KEYS, columns="lengthscale_rule",
                             values=metric, aggfunc="first")
    wide = wide.dropna(subset=[c for c in ("chen", "geom") if c in wide])
    if not {"chen", "geom"} <= set(wide.columns):
        return pd.DataFrame()
    wide = wide.reset_index()
    wide["diff"] = wide["geom"] - wide["chen"]
    return wide


def wilcoxon_on(values):
    """Wilcoxon signed-rank against zero; returns (n, statistic, p) or (n, nan, nan)."""
    values = np.asarray([v for v in values if np.isfinite(v)], dtype=float)
    if len(values) < 3 or np.allclose(values, 0):
        return len(values), np.nan, np.nan
    statistic, p = wilcoxon(values)
    return len(values), float(statistic), float(p)


def family_table(differences, metric):
    """Family means of the paired difference + the family-clustered test."""
    by_family = (differences.groupby("family")["diff"]
                 .agg(mean_diff="mean", sd="std", n_runs="size").reset_index())
    n, statistic, p = wilcoxon_on(by_family["mean_diff"].values)
    return by_family, dict(metric=metric, n_families=n, statistic=statistic, p=p,
                           mean_of_family_means=float(by_family["mean_diff"].mean()),
                           families_favouring_geom=int((by_family["mean_diff"] > 0).sum()))


def representation_table(differences):
    """Per representation: family-clustered mean difference and its test."""
    rows = []
    for representation, block in differences.groupby("representation"):
        by_family = block.groupby("family")["diff"].mean()
        n, _, p = wilcoxon_on(by_family.values)
        rows.append(dict(representation=representation,
                         n_families=n, n_runs=len(block),
                         mean_diff_runs=block["diff"].mean(),
                         mean_diff_families=by_family.mean(),
                         p_family_clustered=p))
    return pd.DataFrame(rows).sort_values("mean_diff_families", ascending=False)


def fmt(value, places=4):
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return "n/a"
    return f"{value:.{places}f}"


def markdown_table(frame, places=4):
    frame = frame.copy()
    for column in frame.columns:
        if pd.api.types.is_float_dtype(frame[column]):
            frame[column] = frame[column].map(lambda v: fmt(v, places))
    header = "| " + " | ".join(str(c) for c in frame.columns) + " |"
    rule = "|" + "|".join("---" for _ in frame.columns) + "|"
    body = ["| " + " | ".join(str(v) for v in row) + " |"
            for row in frame.itertuples(index=False)]
    return "\n".join([header, rule] + body)


def figures(frame, out_dir):
    """The two diagnostic scatters. Returns the paths written (empty if no matplotlib)."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"[figures] matplotlib unavailable ({exc}); skipping plots", flush=True)
        return []

    written = []
    representations = sorted(frame["representation"].unique())
    colours = {rep: plt.cm.tab10(i % 10) for i, rep in enumerate(representations)}

    # --- 3. ell_0/ell_star vs lift, coloured by representation ---------------
    modes = sorted(frame["prior_mode"].unique())
    fig, axes = plt.subplots(1, len(modes), figsize=(6.2 * len(modes), 5.0), squeeze=False)
    for axis, mode in zip(axes[0], modes):
        block = frame[frame["prior_mode"] == mode]
        for rep in representations:
            sub = block[block["representation"] == rep]
            axis.scatter(sub["ell_0_over_ell_star"], sub["lift"], s=18, alpha=0.55,
                         color=colours[rep], label=rep,
                         marker=("o" if rep != "ohe" else "x"))
        axis.axvline(1.0, color="0.4", lw=1, ls="--")
        axis.set_xscale("log")
        axis.set_xlabel(r"$\ell_0/\ell^*$  (1.0 = kernel-matched geometry)")
        axis.set_ylabel("lift over random")
        axis.set_title(mode)
    axes[0][-1].legend(fontsize=8, loc="best")
    fig.suptitle("Prior centre relative to pool geometry vs BO performance")
    fig.tight_layout()
    path = os.path.join(out_dir, "lengthscale_ab_ratio_vs_lift.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    written.append(path)

    # --- 4. before vs after fitting, with the identity line ------------------
    fig, axes = plt.subplots(1, len(modes), figsize=(6.2 * len(modes), 5.0), squeeze=False)
    for axis, mode in zip(axes[0], modes):
        block = frame[frame["prior_mode"] == mode]
        for rep in representations:
            sub = block[block["representation"] == rep]
            for rule, marker in (("chen", "o"), ("geom", "^")):
                arm = sub[sub["lengthscale_rule"] == rule]
                axis.scatter(arm["ell_0_over_ell_star"], arm["ell_fitted_over_ell_star"],
                             s=18, alpha=0.55, color=colours[rep], marker=marker,
                             label=f"{rep} / {rule}")
        finite = block[["ell_0_over_ell_star", "ell_fitted_over_ell_star"]].replace(
            [np.inf, -np.inf], np.nan).dropna()
        if len(finite):
            lo = max(min(finite.min()) * 0.7, 1e-3)
            hi = max(finite.max()) * 1.4
            axis.plot([lo, hi], [lo, hi], color="0.3", lw=1, ls="--")
            axis.set_xlim(lo, hi)
            axis.set_ylim(lo, hi)
        axis.set_xscale("log")
        axis.set_yscale("log")
        axis.set_xlabel(r"$\ell_0/\ell^*$  (prior centre, before fitting)")
        axis.set_ylabel(r"$\hat{\ell}/\ell^*$  (after marginal-likelihood fitting)")
        axis.set_title(mode)
    handles, labels = axes[0][0].get_legend_handles_labels()
    if handles:
        axes[0][-1].legend(handles, labels, fontsize=7, loc="best", ncol=2)
    fig.suptitle("Did the fit move the lengthscale? (points on the diagonal = no)")
    fig.tight_layout()
    path = os.path.join(out_dir, "lengthscale_ab_prior_vs_fitted.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    written.append(path)
    return written


def analyse(csv_path, out_dir=OUT_DIR, make_figures=True):
    raw = load(csv_path)
    degenerate = raw[raw["degenerate"]]
    frame = raw[~raw["degenerate"]].copy()

    lines = ["# Lengthscale A/B: geom vs chen", ""]
    lines.append(f"Source: `{os.path.relpath(csv_path, HERE)}` -- {len(raw)} campaigns "
                 f"({raw['dataset'].nunique()} datasets, {raw['representation'].nunique()} "
                 f"representations, {raw['seed'].nunique()} seeds, "
                 f"reduction(s): {', '.join(sorted(raw['reduction'].unique()))}).")
    if len(degenerate):
        cells = (degenerate[["dataset", "representation"]].drop_duplicates()
                 .apply(lambda r: f"{r.dataset}/{r.representation}", axis=1).tolist())
        lines.append("")
        lines.append(f"**Excluded as degenerate** ({len(degenerate)} campaigns, "
                     f"all pairwise distances equal, so every candidate ties): "
                     f"{', '.join(cells)}. Reported separately at the end.")
    lines.append("")

    # --- where the two rules put the prior ---------------------------------
    centres = (frame.groupby(["dataset", "representation", "lengthscale_rule"])
               .agg(d=("d", "first"), ell_star=("ell_star", "first"),
                    ell_0=("ell_0", "first"), ratio=("ell_0_over_ell_star", "first"))
               .reset_index())
    chen_ratio = centres[centres["lengthscale_rule"] == "chen"]["ratio"]
    lines += ["## Where the rules put the prior", "",
              f"`chen` sits at ell_0/ell* = {chen_ratio.median():.2f} "
              f"(range {chen_ratio.min():.2f}-{chen_ratio.max():.2f}) across the cells; "
              f"`geom` sits at 1.00 by construction.", ""]

    summary_rows = []
    for metric in ("lift", "simple_regret"):
        lines += [f"## Paired difference (geom - chen) in {metric}",
                  f"_{METRICS[metric]}_", ""]
        for mode in sorted(frame["prior_mode"].unique()):
            block = frame[frame["prior_mode"] == mode]
            differences = paired_differences(block, metric)
            if differences.empty:
                lines.append(f"### {mode}\n\nNo paired data.\n")
                continue
            by_family, test = family_table(differences, metric)
            n_runs, _, p_runs = wilcoxon_on(differences["diff"].values)
            lines += [f"### {mode}", "",
                      f"- run level: mean {fmt(differences['diff'].mean())} "
                      f"(median {fmt(differences['diff'].median())}, n={n_runs} pairs, "
                      f"Wilcoxon p={fmt(p_runs, 4)} -- runs are NOT independent, shown "
                      f"only for completeness)",
                      f"- **family-clustered: mean of family means "
                      f"{fmt(test['mean_of_family_means'])}, "
                      f"{test['families_favouring_geom']}/{test['n_families']} families "
                      f"favour geom, Wilcoxon signed-rank p={fmt(test['p'], 4)}**", "",
                      markdown_table(by_family), ""]
            per_rep = representation_table(differences)
            lines += [f"By representation ({mode}, {metric}):", "",
                      markdown_table(per_rep), ""]
            summary_rows.append(dict(metric=metric, prior_mode=mode,
                                     mean_family_diff=test["mean_of_family_means"],
                                     n_families=test["n_families"], p=test["p"]))

    # --- did the fit move the lengthscale? ---------------------------------
    moved = (frame.assign(log_move=np.log(frame["ell_fitted_over_ell_star"] /
                                          frame["ell_0_over_ell_star"]))
             .groupby(["prior_mode", "lengthscale_rule"])
             .agg(prior_ratio=("ell_0_over_ell_star", "median"),
                  fitted_ratio=("ell_fitted_over_ell_star", "median"),
                  median_log_move=("log_move", "median"),
                  fitted_log_sd=("ell_fitted_log_sd", "median")).reset_index())
    lines += ["## Did marginal-likelihood fitting undo the prior?", "",
              "If the fit lands in the same place from both starting points, the arms "
              "should tie -- that is a finding about the pipeline, not a null result. "
              "`median_log_move` is the median of log(fitted/prior centre): 0 means the "
              "fit did not move at all.", "", markdown_table(moved), ""]

    lines += ["## Headline", "", markdown_table(pd.DataFrame(summary_rows)), ""]
    modes = {r["prior_mode"] for r in summary_rows}
    if {"match_concentration", "match_parameterisation"} <= modes:
        for metric in ("lift", "simple_regret"):
            values = {r["prior_mode"]: r["mean_family_diff"]
                      for r in summary_rows if r["metric"] == metric}
            if len(values) == 2 and np.sign(values["match_concentration"]) != np.sign(
                    values["match_parameterisation"]):
                lines.append(f"**The two prior modes DISAGREE under {metric}** "
                             f"({fmt(values['match_concentration'])} vs "
                             f"{fmt(values['match_parameterisation'])}): the difference "
                             f"is a prior-STRENGTH effect, not a centre effect.")
        lines.append("")

    if len(degenerate):
        degenerate_summary = (degenerate.groupby(
            ["dataset", "representation", "prior_mode", "lengthscale_rule"])
            .agg(lift=("lift", "mean"), auc=("auc", "mean"), n=("seed", "size"))
            .reset_index())
        lines += ["## Degenerate cells (excluded above)", "",
                  markdown_table(degenerate_summary), ""]

    # --- campaigns that could not be fitted at all --------------------------
    import glob as globlib

    failure_files = globlib.glob(os.path.join(out_dir, "*_failures.csv"))
    if failure_files:
        failures = pd.concat([pd.read_csv(f) for f in failure_files], ignore_index=True)
        by_arm = (failures.groupby(["lengthscale_rule", "prior_mode"])
                  .size().reset_index(name="n_failed"))
        lines += ["## Campaigns that failed to fit", "",
                  f"{len(failures)} campaign(s) raised during GP fitting and were "
                  "dropped; the analysis discards their pair partners too. A rule "
                  "that fails more often has a real disadvantage -- check the split:",
                  "", markdown_table(by_arm), ""]

    written = figures(frame, out_dir) if make_figures else []
    if written:
        lines += ["## Figures", ""] + [f"- `{os.path.basename(p)}`" for p in written] + [""]

    text = "\n".join(lines)
    with open(SUMMARY_PATH, "w", encoding="utf-8") as handle:
        handle.write(text)
    print(text)
    print(f"\nwrote {SUMMARY_PATH}")
    for path in written:
        print(f"wrote {path}")
    return text


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--csv", default=DEFAULT_CSV)
    parser.add_argument("--no-figures", action="store_true")
    args = parser.parse_args()
    import glob as globlib

    if not (globlib.glob(args.csv) or os.path.exists(args.csv)):
        sys.exit(f"no results at {args.csv}; run lengthscale_ab.py first")
    analyse(args.csv, make_figures=not args.no_figures)


if __name__ == "__main__":
    main()
