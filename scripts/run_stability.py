"""Rank neurons at each threshold and compare top-k lists across thresholds.

Reads only the aggregated per-body arrays (<agg>), never the 1 GB connection table.
For each population (primary, sensitivity) and each metric it writes to <out>:
  top100_<population>_<metric>.csv     the 100 highest-ranked neurons at each threshold (deterministic body-ID tie-break)
  stability_<population>_<metric>.csv  every threshold pair x k x tie view (see connectome.ranking for what each column uses)
  ties_<population>_<metric>.csv       how many neurons share the k-th value at each threshold (the size of the tie)
  ranking_support.csv                  how many neurons have a positive value at each threshold

Metrics (neuron-to-neuron connectivity only; partners must be members of the SAME population):
  out_strength_nn = summed weight of outgoing records to neuron partners        (aggregate metric out_str_n)
  out_degree_nn   = number of distinct downstream neuron partners               (aggregate metric out_deg_n)
The ranking population is the full fixed population at every threshold; neurons with no surviving record have value 0.
"""
import argparse, os

import numpy as np
import pandas as pd

from connectome.aggregate import M
from connectome.annotations import load_annotations, make_populations
from connectome.ranking import boundary_tie, order_ids, stability_table, top_k

KS = (10, 50, 100)
METRICS = {"out_strength_nn": "out_str_n", "out_degree_nn": "out_deg_n"}
TIE_COLS = ["population", "metric", "threshold", "k", "kth_value", "n_strictly_above", "n_tied_at_kth_value", "id_set_size", "inclusive_set_size", "boundary_tie"]

ap = argparse.ArgumentParser()
ap.add_argument("--agg", default="data/interim/aggregation/aggregates.npz")
ap.add_argument("--annot", default="data/raw/body-annotations-male-cns-v1.0-minconf-0.5.feather")
ap.add_argument("--out", default="outputs/tables")
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)

z = np.load(a.agg); thr = [int(t) for t in z["thresholds"]]
ann = load_annotations(a.annot); pops = make_populations(ann)
assert np.array_equal(z["ann_ids"], ann.bodyId.to_numpy()), "aggregates were built from a different annotation table"
support_rows = []
for p, member in pops.items():
    ids = ann.bodyId.to_numpy()[member]; meta = ann.loc[member, [c for c in ("type", "superclass", "somaSide") if c in ann.columns]].reset_index(drop=True)
    for mname, key in METRICS.items():
        vals = {T: z[f"per_body_{p}"][ti, M[key]][member] for ti, T in enumerate(thr)}
        stab = stability_table(vals, ids, ks=KS); stab.insert(0, "metric", mname); stab.insert(0, "population", p)
        stab.to_csv(f"{a.out}/stability_{p}_{mname}.csv", index=False)
        top_rows, tie_rows = [], []
        for T in thr:
            v = vals[T]; order = order_ids(v, ids)
            support_rows.append(dict(population=p, metric=mname, threshold=T, population_size=len(v), neurons_with_positive_value=int((v > 0).sum()),
                                     max_value=int(v.max()), median_positive_value=float(np.median(v[v > 0])) if (v > 0).any() else float("nan")))
            n_top = min(100, len(v))
            top = pd.DataFrame({"population": p, "metric": mname, "threshold": T, "rank": np.arange(1, n_top + 1), "bodyId": ids[order[:n_top]], "value": v[order[:n_top]]})
            top_rows.append(pd.concat([top, meta.iloc[order[:n_top]].reset_index(drop=True)], axis=1))
            vs = np.sort(v)[::-1]
            for k in KS:
                if k > len(v):
                    continue            # population smaller than k: no k-th place exists
                kth = int(vs[k - 1])
                tie_rows.append(dict(population=p, metric=mname, threshold=T, k=k, kth_value=kth, n_strictly_above=int((v > kth).sum()), n_tied_at_kth_value=int((v == kth).sum()),
                                     id_set_size=len(top_k(v, ids, k, "id")), inclusive_set_size=len(top_k(v, ids, k, "inclusive")), boundary_tie=boundary_tie(v, k)))
        pd.concat(top_rows).to_csv(f"{a.out}/top100_{p}_{mname}.csv", index=False)
        pd.DataFrame(tie_rows, columns=TIE_COLS).to_csv(f"{a.out}/ties_{p}_{mname}.csv", index=False)
        print(f"{p:11s} {mname:16s}: stability rows={len(stab)}, tie rows={len(tie_rows)}")
pd.DataFrame(support_rows).to_csv(f"{a.out}/ranking_support.csv", index=False)
