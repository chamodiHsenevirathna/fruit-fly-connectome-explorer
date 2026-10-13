"""Tests for connectome.figdata (data preparation for the figures). Expected values are worked out by hand."""
import numpy as np
import pandas as pd
import pytest

from connectome import figdata as fd


def totals():
    return pd.DataFrame({"threshold": [5, 1, 2], "records": [10, 100, 40], "strength": [90, 400, 200]})   # deliberately unsorted


def test_retention_shares_are_relative_to_threshold_1_and_sorted():
    r = fd.retention(totals())
    assert r.threshold.tolist() == [1, 2, 5]
    assert r.records_share.tolist() == [1.0, 0.4, 0.1] and r.weight_share.tolist() == [1.0, 0.5, 0.225]


def test_retention_requires_baseline_and_nonempty():
    with pytest.raises(ValueError):
        fd.retention(totals()[totals().threshold != 1])
    with pytest.raises(ValueError):
        fd.retention(totals().iloc[0:0])


def test_histogram_retention_matches_hand_count():
    hist = pd.DataFrame({"weight": [1, 2, 3, 10], "n_records": [5, 3, 2, 1]})       # 11 records, total weight 5+6+6+10 = 27
    h = fd.histogram_retention(hist, thresholds=(1, 2, 3, 10, 11)).set_index("threshold")
    assert h.loc[1].tolist() == [11, 27] and h.loc[2].tolist() == [6, 22] and h.loc[3].tolist() == [3, 16]
    assert h.loc[10].tolist() == [1, 10]                 # boundary: weight == threshold is kept
    assert h.loc[11].tolist() == [0, 0]                  # nothing survives


def pop_table():
    return pd.DataFrame({"population": ["primary"] * 2 + ["sensitivity"] * 2, "threshold": [1, 10, 1, 10], "population_size": [100, 100, 80, 80],
                         "neurons_with_any_NN_out": [90, 50, 70, 40], "neurons_with_any_NN_in": [95, 60, 75, 45],
                         "neurons_with_any_NN_connection": [98, 70, 78, 55], "neurons_with_zero_surviving_connections": [2, 25, 2, 20]})


def test_population_shares_use_fixed_denominator_and_none_means_no_neuron_connection():
    s = fd.population_shares(pop_table(), "primary")
    assert s.population_size.tolist() == [100, 100]
    assert s.outgoing.tolist() == [0.9, 0.5] and s.incoming.tolist() == [0.95, 0.6] and s["any"].tolist() == [0.98, 0.7]
    assert np.allclose(s.none, [0.02, 0.3])              # 1 - any: neurons with no neuron-to-neuron connection
    assert np.allclose(fd.population_shares(pop_table(), "sensitivity").none, [0.025, 0.3125])


def test_population_shares_reject_changing_population_and_unknown_population():
    bad = pop_table(); bad.loc[1, "population_size"] = 99
    with pytest.raises(ValueError):
        fd.population_shares(bad, "primary")
    with pytest.raises(ValueError):
        fd.population_shares(pop_table(), "nope")


def edge_table():
    rows = []
    for T, cats in {1: [5, 2, 2, 1], 10: [3, 1, 0, 0]}.items():            # records per category NN, NO, ON, OO
        for c, n in zip(fd.CATEGORIES, cats):
            rows.append(dict(population="primary", threshold=T, category=c, records=n, strength=n * 10))
    return pd.DataFrame(rows)


def test_category_shares_sum_to_one_and_zero_categories_are_kept():
    s = fd.category_shares(edge_table(), "primary")
    assert len(s) == 8
    for T, g in s.groupby("threshold"):
        assert np.isclose(g.records_share.sum(), 1) and np.isclose(g.weight_share.sum(), 1)
    nn1 = s[(s.threshold == 1) & (s.category == "NN")].iloc[0]
    assert nn1.records_share == 0.5 and nn1.weight_share == 0.5                # 5 of 10 records
    assert s[(s.threshold == 10) & (s.category == "ON")].iloc[0].records_share == 0.0


def test_category_shares_report_missing_category():
    with pytest.raises(ValueError):
        fd.category_shares(edge_table().query("category != 'OO'"), "primary")


def stab_table():
    rows = []
    for (a, b, j_id, j_inc) in [(1, 10, 0.5, 0.6), (1, 50, 0.2, 0.2), (10, 50, 0.4, 0.4)]:
        for k in (10, 50, 100):
            for pol, j in (("id", j_id), ("inclusive", j_inc)):
                rows.append(dict(threshold_a=a, threshold_b=b, k=k, tie_policy=pol, jaccard=j, boundary_tie_a=(k == 100 and pol == "id" and b == 50),
                                 boundary_tie_b=False, size_a=k, size_b=k, spearman_full_population=0.77))
    return pd.DataFrame(rows)


def test_jaccard_vs_reference_finds_pairs_in_either_orientation_and_flags_ties():
    r = fd.jaccard_vs_reference(stab_table(), reference=10, ks=(10, 100), thresholds=(1, 10, 50))
    assert sorted(r.threshold.unique()) == [1, 50]                              # the reference itself is not returned
    one = r[(r.threshold == 1) & (r.k == 10) & (r.tie_policy == "id")].iloc[0]
    assert one.jaccard == 0.5                                                     # stored as (1,10): found although reference is 10
    assert r[(r.threshold == 50) & (r.k == 100) & (r.tie_policy == "id")].iloc[0].boundary_tie
    assert not r[(r.threshold == 50) & (r.k == 10)].boundary_tie.any()


def test_jaccard_vs_reference_raises_on_missing_comparison():
    with pytest.raises(ValueError):
        fd.jaccard_vs_reference(stab_table(), reference=2, ks=(10,), thresholds=(1, 2, 10))


def test_pairwise_matrix_is_symmetric_with_unit_diagonal():
    m = fd.pairwise_matrix(stab_table(), "jaccard", k=50, policy="inclusive", thresholds=(1, 10, 50))
    assert m.loc[1, 10] == m.loc[10, 1] == 0.6 and m.loc[10, 50] == 0.4 and np.all(np.diag(m) == 1)
    s = fd.pairwise_matrix(stab_table(), "spearman_full_population", thresholds=(1, 10, 50))
    assert s.loc[1, 50] == 0.77


def test_pairwise_matrix_rejects_spearman_that_depends_on_k():
    bad = stab_table(); bad.loc[0, "spearman_full_population"] = 0.1
    with pytest.raises(ValueError):
        fd.pairwise_matrix(bad, "spearman_full_population", thresholds=(1, 10))


def test_tie_summary_handles_empty_and_missing_tables():
    assert fd.tie_summary(pd.DataFrame(columns=["boundary_tie", "n_tied_at_kth_value"])) == dict(rows=0, boundary_tie_rows=0, max_tied_at_kth_value=0)
    assert fd.tie_summary(None)["rows"] == 0
    t = pd.DataFrame({"boundary_tie": [True, False, True], "n_tied_at_kth_value": [3, 1, 8]})
    assert fd.tie_summary(t) == dict(rows=3, boundary_tie_rows=2, max_tied_at_kth_value=8)


def test_zero_valued_counts():
    sup = pd.DataFrame({"population": ["primary"] * 2, "metric": ["out_strength_nn"] * 2, "threshold": [1, 50], "population_size": [100, 100],
                        "neurons_with_positive_value": [90, 30]})
    z = fd.zero_valued_counts(sup, "primary", "out_strength_nn")
    assert z.zero_valued.tolist() == [10, 70] and z.zero_share.tolist() == [0.1, 0.7]


def top_table():
    rows = []
    for T, ranked in {1: [7, 8, 9], 10: [8, 7, 6], 50: [6, 5, 4], 20: [99, 98, 97]}.items():
        for r, b in enumerate(ranked, 1):
            rows.append(dict(threshold=T, rank=r, bodyId=b))
    return pd.DataFrame(rows)


def test_selection_is_union_over_requested_thresholds_only():
    s = fd.select_trajectory_neurons(top_table(), thresholds=(1, 10, 50), n=2)       # top 2 only; threshold 20 is ignored
    got = dict(zip(s.bodyId, s.selected_at))
    assert got == {7: "1,10", 8: "1,10", 6: "50", 5: "50"}
    assert len(fd.select_trajectory_neurons(top_table(), thresholds=(1, 10, 50), n=3)) == 6      # 7,8,9,6,5,4
    assert fd.select_trajectory_neurons(top_table(), thresholds=(2,), n=2).empty


def test_trajectory_table_ranks_ties_and_zero_values():
    ids = np.array([1, 2, 3, 4, 5])
    vals = {1: np.array([50, 50, 10, 0, 7]), 10: np.array([0, 30, 10, 0, 0])}
    sel = pd.DataFrame({"bodyId": [2, 1], "selected_at": ["1", "1"]})
    meta = pd.DataFrame({"type": list("abcde"), "superclass": list("vwxyz")})
    t = fd.trajectory_table(vals, ids, sel, meta)
    g = lambda b, T: t[(t.bodyId == b) & (t.threshold == T)].iloc[0]
    assert g(1, 1)["rank"] == 1 and g(2, 1)["rank"] == 2          # tie at 50 broken by body ID ascending
    assert g(2, 10)["rank"] == 1
    z = g(1, 10)                                                   # neuron 1 has no surviving connection at T>=10
    assert z.value == 0 and np.isnan(z["rank"]) and bool(z.has_surviving_outgoing) is False
    assert bool(g(1, 1).has_surviving_outgoing) is True and g(2, 1).type == "b" and g(2, 1).superclass == "w"
    assert len(t) == 4                                             # 2 neurons x 2 thresholds


def test_trajectory_ranks_are_computed_over_the_whole_population_not_the_selection():
    ids = np.arange(1, 101); vals = {1: np.arange(100, 0, -1)}      # neuron 1 has the highest value
    t = fd.trajectory_table(vals, ids, pd.DataFrame({"bodyId": [50], "selected_at": ["1"]}), pd.DataFrame({"type": ["t"] * 100, "superclass": ["s"] * 100}))
    assert t.iloc[0]["rank"] == 50


def test_ccdf_counts_hand_values_and_empty():
    x, n = fd.ccdf_counts([0, 0, 1, 1, 3, 5])
    assert x.tolist() == [1, 3, 5] and n.tolist() == [4, 2, 1]       # neurons with value >= 1, >= 3, >= 5
    x, n = fd.ccdf_counts([0, 0])
    assert len(x) == 0 and len(n) == 0


def test_verify_sources_consistent_and_detects_a_discrepancy():
    hist = pd.DataFrame({"weight": [1, 2], "n_records": [3, 1]})                   # at T=1: 4 records, weight 5; at T=2: 1 record, weight 2
    tot = pd.DataFrame({"threshold": [1, 2], "records": [4, 1], "strength": [5, 2]})
    edges = pd.DataFrame([dict(population="primary", threshold=T, category=c, records=r, strength=s) for T, rs in
                          {1: [(4, 5), (0, 0), (0, 0), (0, 0)], 2: [(1, 2), (0, 0), (0, 0), (0, 0)]}.items() for c, (r, s) in zip(fd.CATEGORIES, rs)])
    pop = pd.DataFrame({"population": ["primary"] * 2, "threshold": [1, 2], "neurons_with_any_NN_out": [9, 4]})
    sup = pd.DataFrame({"population": ["primary"] * 4, "metric": ["out_strength_nn"] * 2 + ["out_degree_nn"] * 2, "threshold": [1, 2, 1, 2],
                        "neurons_with_positive_value": [9, 4, 9, 4]})
    assert fd.verify_sources(tot, hist, edges, pop, sup) == []
    bad = tot.copy(); bad.loc[1, "records"] = 2
    assert any("records differ" in p for p in fd.verify_sources(bad, hist, edges, pop, sup))
