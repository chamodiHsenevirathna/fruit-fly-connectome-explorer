"""Slow, independent pandas reference for the streaming aggregator.

It shares NO code with connectome.aggregate. It is written directly from the definitions in that module's docstring
and is used only to verify the aggregator (tests, and scripts/compare_aggregation_to_pandas.py on real data slices).

For annotated body b and population M (partner type n = partner in M, o = anything else):
    out_deg_n[b] = number of records with pre == b and post in M;  out_str_n[b] = sum of their weights
    out_deg_o[b] = records with pre == b and post not in M          in_deg_n[b]  = records with post == b and pre in M
    in_deg_o[b]  = records with post == b and pre not in M          (and the matching *_str_* sums)
b itself need not be in M. A record is kept at threshold T if weight >= T.
"""
import numpy as np
import pandas as pd

from connectome.aggregate import CATEGORIES, METRICS  # names only (labels), no computation is shared


def compare(agg_per_body, agg_cat, agg_glob, df, ann_ids, members, thresholds):
    """Return a list of human-readable mismatch descriptions (empty list = everything agrees).

    df has columns pre, post, w. agg_* are one population's arrays from Aggregator (per_body, cat) and the global table.
    """
    ann_ids = np.asarray(ann_ids); members = set(members); annset = set(ann_ids.tolist())
    bad = []
    for ti, T in enumerate(thresholds):
        d = df[df.w >= T]
        pre_m, post_m = d.pre.isin(members), d.post.isin(members)

        def per_body(sel, key):
            g = d[sel].groupby(key).w.agg(["count", "sum"]).reindex(ann_ids, fill_value=0)
            return g["count"].to_numpy(), g["sum"].to_numpy()

        ref = {}
        ref["out_deg_n"], ref["out_str_n"] = per_body(post_m, "pre"); ref["out_deg_o"], ref["out_str_o"] = per_body(~post_m, "pre")
        ref["in_deg_n"], ref["in_str_n"] = per_body(pre_m, "post"); ref["in_deg_o"], ref["in_str_o"] = per_body(~pre_m, "post")
        for mi, m in enumerate(METRICS):
            n_bad = int((agg_per_body[ti, mi] != ref[m]).sum())
            if n_bad:
                bad.append(f"T>={T} {m}: {n_bad} bodies differ")
        for c, (s, t_) in enumerate([(True, True), (True, False), (False, True), (False, False)]):
            sel = (pre_m == s) & (post_m == t_)
            if agg_cat[ti, c].tolist() != [int(sel.sum()), int(d.w[sel].sum())]:
                bad.append(f"T>={T} category {CATEGORIES[c]} totals differ")
        un_pre, un_post = ~d.pre.isin(annset), ~d.post.isin(annset)
        glob = [len(d), int(d.w.sum()), int(un_pre.sum()), int(d.w[un_pre].sum()), int(un_post.sum()), int(d.w[un_post].sum())]
        if agg_glob[ti].tolist() != glob:
            bad.append(f"T>={T} global totals differ")
    return bad
