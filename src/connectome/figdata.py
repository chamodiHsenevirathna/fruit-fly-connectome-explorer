"""Data preparation for the figures (no plotting here, so every number can be unit-tested).

Everything is computed from the saved result tables in outputs/tables/ (and, for the rank-trajectory table and
the optional distribution figure, from the per-body aggregates). Nothing is typed in by hand.

Population naming used in public-facing material
------------------------------------------------
"primary population"   bodies with a `superclass` annotation (internal key `primary`)
"fully traced subset"  bodies with a `superclass` annotation AND status == "Traced" (internal key `sensitivity`)
The fully traced subset is a sensitivity analysis, not a better biological definition: it also drops many sensory
bodies labelled "Out of scope" (about 1,984 `ol_sensory` bodies) that the data providers count as neurons.
"""
import numpy as np
import pandas as pd

from connectome.ranking import order_ids

THRESHOLDS = (1, 2, 5, 10, 20, 50)
CATEGORIES = ("NN", "NO", "ON", "OO")
POPULATION_LABELS = {"primary": "primary population", "sensitivity": "fully traced subset"}


# ----------------------------------------------------------------------------- Figures 1-2: retention
def retention(totals):
    """Share of records and of summed weight kept at each threshold, relative to the lowest threshold (must be 1).

    `totals` has columns threshold, records, strength (threshold_totals.csv). Shares are computed here, not typed in.
    """
    t = totals.sort_values("threshold").reset_index(drop=True)
    if t.empty:
        raise ValueError("threshold totals are empty")
    if int(t.threshold.iloc[0]) != 1:
        raise ValueError("the baseline (threshold 1, nothing removed) is required to compute retained shares")
    base = t.iloc[0]
    return pd.DataFrame({"threshold": t.threshold, "records": t.records, "strength": t.strength,
                         "records_share": t.records / base.records, "weight_share": t.strength / base.strength})


def histogram_retention(hist, thresholds=THRESHOLDS):
    """Records and summed weight with weight >= T computed from the weight histogram (weight, n_records).

    This is an independent route to the same numbers as threshold_totals.csv, used to cross-check the figures.
    """
    h = hist.sort_values("weight")
    rows = []
    for T in thresholds:
        kept = h[h.weight >= T]
        rows.append(dict(threshold=T, records=int(kept.n_records.sum()), strength=int((kept.weight * kept.n_records).sum())))
    return pd.DataFrame(rows)


def population_shares(pop_df, population):
    """Fractions of the FIXED population with outgoing / incoming / any / no neuron-to-neuron connection, per threshold."""
    d = pop_df[pop_df.population == population].sort_values("threshold")
    if d.empty:
        raise ValueError(f"no rows for population {population!r}")
    if d.population_size.nunique() != 1:
        raise ValueError("the population size must be identical at every threshold (fixed population)")
    n = d.population_size.iloc[0]
    out = pd.DataFrame({"threshold": d.threshold.to_numpy(), "population_size": n,
                        "outgoing": d.neurons_with_any_NN_out.to_numpy() / n, "incoming": d.neurons_with_any_NN_in.to_numpy() / n,
                        "any": d.neurons_with_any_NN_connection.to_numpy() / n})
    out["none"] = 1.0 - out["any"]            # no neuron-to-neuron connection at all (they may still connect to non-neurons)
    return out


# ----------------------------------------------------------------------------- Figure 3: edge-category composition
def category_shares(edge_df, population):
    """Share of records and of summed weight in each edge category (NN, NO, ON, OO) per threshold."""
    d = edge_df[edge_df.population == population]
    if d.empty:
        raise ValueError(f"no rows for population {population!r}")
    rows = []
    for T, g in d.groupby("threshold"):
        g = g.set_index("category").reindex(CATEGORIES)
        if g.records.isna().any():
            raise ValueError(f"threshold {T}: missing edge categories")
        for c in CATEGORIES:
            rows.append(dict(threshold=int(T), category=c, records=int(g.loc[c, "records"]), strength=int(g.loc[c, "strength"]),
                             records_share=g.loc[c, "records"] / g.records.sum() if g.records.sum() else np.nan,
                             weight_share=g.loc[c, "strength"] / g.strength.sum() if g.strength.sum() else np.nan))
    return pd.DataFrame(rows).sort_values(["threshold", "category"]).reset_index(drop=True)


# ----------------------------------------------------------------------------- Figures 4-5: stability
def _pair_rows(stab, a, b):
    return stab[((stab.threshold_a == a) & (stab.threshold_b == b)) | ((stab.threshold_a == b) & (stab.threshold_b == a))]


def jaccard_vs_reference(stab, reference, ks=(10, 50, 100), thresholds=THRESHOLDS):
    """Jaccard of the top-k list at each threshold against the list at `reference`, for BOTH tie views.

    Returns tidy rows: threshold, k, tie_policy ("id" = headline, "inclusive" = companion), jaccard, boundary_tie.
    The reference compared with itself is not returned (it is 1 by definition). Missing comparisons raise an error.
    """
    rows = []
    for T in thresholds:
        if T == reference:
            continue
        pair = _pair_rows(stab, reference, T)
        for k in ks:
            for pol in ("id", "inclusive"):
                r = pair[(pair.k == k) & (pair.tie_policy == pol)]
                if len(r) != 1:
                    raise ValueError(f"expected exactly one stability row for thresholds ({reference},{T}), k={k}, view={pol}; found {len(r)}")
                r = r.iloc[0]
                rows.append(dict(threshold=T, k=k, tie_policy=pol, jaccard=float(r.jaccard),
                                 boundary_tie=bool(r.boundary_tie_a or r.boundary_tie_b), size_a=int(r.size_a), size_b=int(r.size_b)))
    return pd.DataFrame(rows)


def pairwise_matrix(stab, value, k=None, policy="id", thresholds=THRESHOLDS):
    """Symmetric matrix of `value` ("jaccard" for a given k and view, or "spearman_full_population") with ones on the diagonal."""
    m = pd.DataFrame(np.nan, index=list(thresholds), columns=list(thresholds))
    for i in thresholds:
        m.loc[i, i] = 1.0
    for a in thresholds:
        for b in thresholds:
            if a >= b:
                continue
            pair = _pair_rows(stab, a, b)
            if value == "jaccard":
                pair = pair[(pair.k == k) & (pair.tie_policy == policy)]
            if len(pair) == 0:
                raise ValueError(f"no stability row for thresholds ({a},{b})")
            vals = pair[value].dropna().unique() if value != "jaccard" else pair[value].to_numpy()
            if value != "jaccard" and len(vals) != 1:
                raise ValueError(f"{value} is not constant across k/views for thresholds ({a},{b}); it should not depend on top-k")
            m.loc[a, b] = m.loc[b, a] = float(vals[0])
    return m


def tie_summary(ties_df):
    """Counts of boundary ties and of the largest tied group; an empty table (population smaller than k) gives zeros."""
    if ties_df is None or len(ties_df) == 0:
        return dict(rows=0, boundary_tie_rows=0, max_tied_at_kth_value=0)
    return dict(rows=int(len(ties_df)), boundary_tie_rows=int(ties_df.boundary_tie.sum()), max_tied_at_kth_value=int(ties_df.n_tied_at_kth_value.max()))


def zero_valued_counts(support, population, metric):
    """Neurons whose value is zero at each threshold (they tie at the bottom of the ranking): size - neurons_with_positive_value."""
    d = support[(support.population == population) & (support.metric == metric)].sort_values("threshold")
    return pd.DataFrame({"threshold": d.threshold.to_numpy(), "zero_valued": (d.population_size - d.neurons_with_positive_value).to_numpy(),
                         "zero_share": ((d.population_size - d.neurons_with_positive_value) / d.population_size).to_numpy()})


# ----------------------------------------------------------------------------- Figure 6: rank trajectories
def select_trajectory_neurons(top_df, thresholds=(1, 10, 50), n=10):
    """Union of neurons in the top `n` at any of `thresholds`; `selected_at` lists which thresholds selected each neuron.

    `top_df` is a top100_* table (columns threshold, rank, bodyId, ...). A neuron selected at several thresholds appears once.
    """
    sel = top_df[(top_df.threshold.isin(thresholds)) & (top_df["rank"] <= n)]
    if sel.empty:
        return pd.DataFrame(columns=["bodyId", "selected_at"])
    g = sel.groupby("bodyId").threshold.apply(lambda s: ",".join(str(int(t)) for t in sorted(s))).rename("selected_at").reset_index()
    return g.sort_values(["selected_at", "bodyId"]).reset_index(drop=True)


def trajectory_table(values_by_threshold, ids, selection, meta):
    """One row per selected neuron x threshold: value, rank, and whether it has any surviving outgoing connection.

    values_by_threshold: {threshold: value array aligned with `ids`}, the FULL fixed population (so ranks are comparable).
    Rank = position among all neurons by value descending, ties broken by bodyId ascending (same rule as the stability
    tables). A neuron whose value is 0 has NO rank (NaN): ranking a neuron with no connections would be arbitrary.
    meta: DataFrame aligned with `ids` with columns type and superclass.
    """
    ids = np.asarray(ids)
    pos = {int(b): i for i, b in enumerate(ids)}
    rank_by_t = {}
    for T, v in values_by_threshold.items():
        v = np.asarray(v)
        order = order_ids(v, ids)
        r = np.empty(len(v), dtype=np.int64)
        r[order] = np.arange(1, len(v) + 1)
        rank_by_t[T] = (v, r)
    rows = []
    for _, s in selection.iterrows():
        i = pos[int(s.bodyId)]
        for T in sorted(values_by_threshold):
            v, r = rank_by_t[T]
            rows.append(dict(bodyId=int(s.bodyId), type=meta["type"].iloc[i], superclass=meta["superclass"].iloc[i], selected_at=s.selected_at,
                             threshold=int(T), value=int(v[i]), rank=(float(r[i]) if v[i] > 0 else np.nan), has_surviving_outgoing=bool(v[i] > 0)))
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- Figure 7: distributions
def ccdf_counts(values):
    """Number of neurons with value >= x, for each distinct positive value x (descending count as x increases)."""
    v = np.asarray(values)
    v = v[v > 0]
    if len(v) == 0:
        return np.array([]), np.array([])
    x, c = np.unique(v, return_counts=True)
    n_ge = c[::-1].cumsum()[::-1]
    return x, n_ge


# ----------------------------------------------------------------------------- cross-checks of the inputs
def verify_sources(totals, hist, edge_df, pop_df, support):
    """Return a list of problems found by comparing the saved tables with each other (empty list = consistent)."""
    problems = []
    h = histogram_retention(hist, THRESHOLDS)
    t = totals.set_index("threshold")
    for _, r in h.iterrows():
        T = int(r.threshold)
        if T in t.index:
            if int(t.loc[T, "records"]) != r.records:
                problems.append(f"T>={T}: records differ between the weight histogram ({r.records}) and threshold totals ({int(t.loc[T, 'records'])})")
            if int(t.loc[T, "strength"]) != r.strength:
                problems.append(f"T>={T}: summed weight differs between the weight histogram and threshold totals")
    for pop, d in edge_df.groupby("population"):
        for T, g in d.groupby("threshold"):
            if T in t.index and int(g.records.sum()) != int(t.loc[T, "records"]):
                problems.append(f"{pop} T>={T}: edge categories do not sum to all retained records")
            if T in t.index and int(g.strength.sum()) != int(t.loc[T, "strength"]):
                problems.append(f"{pop} T>={T}: edge categories do not sum to the retained weight")
    for pop in pop_df.population.unique():
        p = pop_df[pop_df.population == pop].set_index("threshold")
        for metric in ("out_strength_nn", "out_degree_nn"):
            s = support[(support.population == pop) & (support.metric == metric)].set_index("threshold")
            if not (s.neurons_with_positive_value.reindex(p.index) == p.neurons_with_any_NN_out).all():
                problems.append(f"{pop}/{metric}: neurons with a positive value differ from neurons with any outgoing neuron-to-neuron connection")
    return problems
