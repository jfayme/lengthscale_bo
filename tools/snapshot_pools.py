"""Step 0 of the lsab rewrite: a sanity snapshot of every pool the OLD tree builds.

Not a bit-identity gate. It replicates the first half of `botorch_bo.load_pool`
(morgan campaign -> `searchspace.discrete.exp_rep` -> merge the objective -> keep
measured rows) so the new loader can be checked for dropped rows or flipped signs.
Components are keyed by the SOURCE SMILES column, the one thing both trees share.

    python tools/snapshot_pools.py      # writes tests/old_pool_snapshot.json
"""
import json, os, sys
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import botorch_bo as BO  # noqa: E402  (imports hyperprior_general, which patches the kernel factory)

H = BO.H
snapshot = {}
for ds in [d["name"] for d in H.GP.DATASETS] + ["shields"]:
    loaded = H._load(ds)
    lookup = loaded["lookup"]
    campaign, _ = H._campaign("morgan", ds, loaded, *BO.REDUCTIONS["decorr0.7"])
    labels = campaign.searchspace.discrete.exp_rep.reset_index(drop=True)
    names = [p.name for p in campaign.searchspace.parameters]
    join = [c for c in names if c in lookup.columns]
    obj = labels.merge(lookup[join + ["yield"]].drop_duplicates(join), on=join,
                       how="left")["yield"].values.astype(float)
    labels, obj = labels[~np.isnan(obj)], obj[~np.isnan(obj)]
    if ds == "shields":   # labels are names; count the SMILES they stand for
        smiles = H.B.load_shields()[1]
        comps = {f"{p}_SMILES": labels[p].map(smiles[p]).nunique() for p in smiles}
    elif loaded["kind"] == "molecule":
        comps = {H.CFG[ds]["smiles"]: labels["mol"].nunique()}
    else:
        comps = {p: labels[p].nunique() for p in names}
    snapshot[ds] = dict(n=int(len(obj)), obj_min=float(obj.min()), obj_max=float(obj.max()),
                        obj_mean=float(obj.mean()), components={k: int(v) for k, v in comps.items()})
    print(ds, snapshot[ds], flush=True)

with open(os.path.join(ROOT, "tests", "old_pool_snapshot.json"), "w") as f:
    json.dump(snapshot, f, indent=1)
