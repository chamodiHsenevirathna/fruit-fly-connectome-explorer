"""Tie-breaking sensitivity: Monte Carlo ESTIMATE of top-k Jaccard under random tie-breaking.

Reads the aggregated per-body arrays and the saved stability tables; writes NEW files only:
  tie_sensitivity_<population>_<metric>.csv     one row per threshold pair x k where a boundary tie exists in either list
Existing rankings, stability tables, tie tables and figures are neither read for modification nor overwritten.

Only comparisons with a boundary tie (the k-th and (k+1)-th largest values are equal in at least one of the two lists) are
sampled: otherwise the top-k set cannot depend on the tie-break. The deterministic value in each row is checked against
the saved stability table (headline "id" view) and the script stops if they disagree.
"""
import argparse, os

import numpy as np
import pandas as pd

from connectome.aggregate import M
from connectome.annotations import load_annotations, make_populations
from connectome.ranking import boundary_tie, jaccard_random_ties

KS = (10, 50, 100)
METRICS = {"out_strength_nn": "out_str_n", "out_degree_nn": "out_deg_n"}
COLS = ["population", "metric", "threshold_a", "threshold_b", "k", "boundary_tie_a", "boundary_tie_b", "jaccard_id_deterministic", "jaccard_inclusive",
        "mc_mean", "mc_sd", "mc_sem", "mc_p2_5", "mc_p97_5", "id_minus_mc_mean", "n_draws", "seed"]

ap = argparse.ArgumentParser()
ap.add_argument("--agg", default="data/interim/aggregation/aggregates.npz")
ap.add_argument("--annot", default="data/raw/body-annotations-male-cns-v1.0-minconf-0.5.feather")
ap.add_argument("--tables", default="outputs/tables", help="where the saved stability tables are read from and the new tables written")
ap.add_argument("--draws", type=int, default=10000)
ap.add_argument("--seed", type=int, default=20261014, help="base seed; each row gets its own seed derived from (base, threshold_a, threshold_b, k)")
a = ap.parse_args()
os.makedirs(a.tables, exist_ok=True)

z = np.load(a.agg); thr = [int(t) for t in z["thresholds"]]
ann = load_annotations(a.annot); pops = make_populations(ann)
assert np.array_equal(z["ann_ids"], ann.bodyId.to_numpy()), "aggregates were built from a different annotation table"
for p, member in pops.items():
    ids = ann.bodyId.to_numpy()[member]
    for mname, key in METRICS.items():
        vals = {T: z[f"per_body_{p}"][ti, M[key]][member] for ti, T in enumerate(thr)}
        stab = pd.read_csv(f"{a.tables}/stability_{p}_{mname}.csv")
        rows = []
        for i, t1 in enumerate(thr):
            for t2 in thr[i + 1:]:
                for k in KS:
                    if k > len(ids):
                        continue
                    bt = (boundary_tie(vals[t1], k), boundary_tie(vals[t2], k))
                    if not any(bt):
                        continue
                    seed = int(np.random.SeedSequence([a.seed, t1, t2, k]).generate_state(1)[0])
                    r = jaccard_random_ties(vals[t1], vals[t2], ids, k, n_draws=a.draws, seed=seed)
                    sel = stab[(stab.threshold_a == t1) & (stab.threshold_b == t2) & (stab.k == k)].set_index("tie_policy")
                    saved = float(sel.loc["id", "jaccard"])
                    assert abs(saved - r["deterministic"]) < 1e-12, f"{p}/{mname} T{t1}-{t2} k={k}: deterministic Jaccard {r['deterministic']} != saved {saved}"
                    assert bool(sel.loc["id", "boundary_tie_a"]) == bt[0] and bool(sel.loc["id", "boundary_tie_b"]) == bt[1]
                    rows.append(dict(population=p, metric=mname, threshold_a=t1, threshold_b=t2, k=k, boundary_tie_a=bt[0], boundary_tie_b=bt[1],
                                     jaccard_id_deterministic=saved, jaccard_inclusive=float(sel.loc["inclusive", "jaccard"]), mc_mean=r["mean"], mc_sd=r["sd"],
                                     mc_sem=r["sem"], mc_p2_5=r["p2_5"], mc_p97_5=r["p97_5"], id_minus_mc_mean=saved - r["mean"], n_draws=r["n_draws"], seed=seed))
        out = pd.DataFrame(rows, columns=COLS)
        out.to_csv(f"{a.tables}/tie_sensitivity_{p}_{mname}.csv", index=False)
        n_total = len(stab[stab.tie_policy == "id"])
        print(f"{p:11s} {mname:16s}: {len(out)} of {n_total} comparisons have a boundary tie"
              + (f"; largest |id - mc_mean| = {out.id_minus_mc_mean.abs().max():.4f}" if len(out) else ""))
