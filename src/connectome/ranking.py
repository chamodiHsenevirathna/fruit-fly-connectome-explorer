"""Ranking neurons by a metric and comparing top-k lists across thresholds.

Jaccard similarity J(A, B) = |A intersect B| / |A union B|: how much two lists overlap, from 0 (nothing shared)
to 1 (identical sets).

Tie policy (approved)
---------------------
Many neurons share the same value (degree is a small integer, so ties are common). Two views of "the top k" are
therefore always computed side by side and labelled in every output row by `tie_policy`:

  "id"         HEADLINE view. Sort by value descending, then by bodyId ascending, take the first k.
               Exactly k members; which tied neuron makes the cut is arbitrary but reproducible.
  "inclusive"  COMPANION view. Take every neuron whose value is >= the k-th largest value, so tied neurons are
               never split. May have more than k members. If the k-th value is 0, zero-valued neurons are NOT
               admitted (neurons with no connections are not a ranking), so the set can be smaller than k.

Which view each reported quantity uses:
  size_a, size_b, intersection, union, jaccard   follow the row's `tie_policy` (so each appears once per view).
  boundary_tie_a / boundary_tie_b                 do not depend on the view: True if the k-th and (k+1)-th largest values
                                                  are equal, i.e. the "id" top-k set depended on the arbitrary tie-break.
                                                  Where a boundary tie is True, treat the "id" Jaccard with caution and
                                                  read it together with the "inclusive" row.
  spearman_full_population                        does not use top-k at all: rank correlation over the whole fixed
                                                  population (average ranks for ties), repeated on each row of a pair.
"""
import numpy as np
import pandas as pd


def order_ids(values, ids):
    """Indices that sort by value descending, then bodyId ascending (deterministic)."""
    values = np.asarray(values); ids = np.asarray(ids)
    return np.lexsort((ids, -values))


def top_k(values, ids, k, policy="id"):
    """Set of bodyIds in the top k under `policy` ("id" or "inclusive")."""
    values = np.asarray(values); ids = np.asarray(ids)
    if k <= 0 or len(values) == 0:
        return set()
    order = order_ids(values, ids)
    if policy == "id":
        return set(ids[order[:k]].tolist())
    if policy == "inclusive":
        kth = values[order[min(k, len(values)) - 1]]
        if kth <= 0:
            return set(ids[values > 0].tolist())
        return set(ids[values >= kth].tolist())
    raise ValueError(f"unknown tie policy {policy!r}")


def boundary_tie(values, k):
    """True if the k-th and (k+1)-th largest values are equal (so the top-k set depends on the tie-break)."""
    v = np.sort(np.asarray(values))[::-1]
    return bool(len(v) > k and v[k - 1] == v[k])


def jaccard(a, b):
    """Jaccard similarity of two sets; NaN if both are empty (undefined)."""
    a, b = set(a), set(b)
    if not a and not b:
        return float("nan")
    return len(a & b) / len(a | b)


def spearman(x, y):
    """Spearman rank correlation over the SAME fixed population (ties get average ranks); NaN if either is constant."""
    rx = pd.Series(x).rank(method="average").to_numpy(); ry = pd.Series(y).rank(method="average").to_numpy()
    if np.ptp(rx) == 0 or np.ptp(ry) == 0:
        return float("nan")
    return float(np.corrcoef(rx, ry)[0, 1])


def stability_table(values_by_threshold, ids, ks=(10, 50, 100), policies=("id", "inclusive")):
    """All threshold pairs x k x tie policy. `values_by_threshold` maps threshold -> value array aligned with `ids`.

    The ranking population (`ids`) is identical for every threshold. Spearman is computed once per pair over that
    whole population (zeros tie at the bottom), not over top-k lists.
    """
    thr = sorted(values_by_threshold)
    rows = []
    for i, t1 in enumerate(thr):
        for t2 in thr[i + 1:]:
            v1, v2 = values_by_threshold[t1], values_by_threshold[t2]
            rho = spearman(v1, v2)
            for k in ks:
                for pol in policies:
                    a, b = top_k(v1, ids, k, pol), top_k(v2, ids, k, pol)
                    rows.append(dict(threshold_a=t1, threshold_b=t2, k=k, tie_policy=pol, headline=(pol == "id"), size_a=len(a), size_b=len(b),
                                     intersection=len(a & b), union=len(a | b), jaccard=jaccard(a, b),
                                     boundary_tie_a=boundary_tie(v1, k), boundary_tie_b=boundary_tie(v2, k),
                                     spearman_full_population=rho))
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- tie-breaking sensitivity (Monte Carlo)
def _boundary_group(values, ids, k):
    """For the top-k under the "id" view: (ids strictly above the k-th value, ids tied AT the k-th value, free slots m).

    The k-th value is fixed, so the strictly-above neurons are in every possible top-k; only the m = k - (#above) slots
    are open, and any m of the tied neurons could fill them.
    """
    order = order_ids(values, ids)
    kth = values[order[k - 1]]
    return ids[values > kth], ids[values == kth], k - int((values > kth).sum())


def jaccard_random_ties(values_a, values_b, ids, k, n_draws=10000, seed=0, max_tied=10000):
    """ESTIMATE of the top-k Jaccard when ties at the k-th place are broken at random instead of by bodyId.

    Procedure. In each list the neurons strictly above the k-th value are always in the top k. The remaining m slots are
    filled by a uniformly random subset of size m of the neurons tied at the k-th value, drawn independently for the two
    lists (random tie-breaking at each threshold is independent). The Jaccard of the two resulting sets is recorded; this
    is repeated `n_draws` times with a seeded generator.

    Returns a dict with the mean over draws, the standard deviation of single draws, the standard error of the mean
    (sd / sqrt(n)), the 2.5th and 97.5th percentiles of single draws, the draw count, and `deterministic` (the bodyId
    tie-break Jaccard, as in the headline view). The mean is a Monte Carlo ESTIMATE of the expected Jaccard, not an exact
    value; its error shrinks as 1/sqrt(n_draws). It is only a sensitivity check: random tie-breaking is one way to treat
    ties as "unknown order", and it says nothing about which tied neuron is more important.

    If neither list has a boundary tie, the top-k set does not depend on the tie-break, so nothing is sampled
    (n_draws = 0, sd = 0) and the value is exact. A tied group larger than `max_tied` raises an error rather than
    silently taking a very long time.
    """
    va, vb, ids = np.asarray(values_a), np.asarray(values_b), np.asarray(ids)
    if k <= 0:
        raise ValueError("k must be positive")
    k = min(k, len(ids))
    det = jaccard(top_k(va, ids, k, "id"), top_k(vb, ids, k, "id"))
    if len(ids) == 0 or not (boundary_tie(va, k) or boundary_tie(vb, k)):
        return dict(mean=det, sd=0.0, sem=0.0, p2_5=det, p97_5=det, n_draws=0, deterministic=det)
    if n_draws < 1:
        raise ValueError("n_draws must be at least 1 when a boundary tie is present")
    groups = [_boundary_group(v, ids, k) for v in (va, vb)]
    if max(len(g[1]) for g in groups) > max_tied:
        raise ValueError(f"tied group of {max(len(g[1]) for g in groups)} neurons exceeds max_tied={max_tied}")
    rng = np.random.default_rng(seed)
    above = [set(g[0].tolist()) for g in groups]
    draws = np.empty(n_draws)
    for i in range(n_draws):
        s = [above[j] | set(rng.choice(groups[j][1], size=groups[j][2], replace=False).tolist()) for j in (0, 1)]
        draws[i] = len(s[0] & s[1]) / len(s[0] | s[1])
    sd = float(draws.std(ddof=1)) if n_draws > 1 else 0.0
    return dict(mean=float(draws.mean()), sd=sd, sem=sd / np.sqrt(n_draws), p2_5=float(np.percentile(draws, 2.5)),
                p97_5=float(np.percentile(draws, 97.5)), n_draws=n_draws, deterministic=det)
