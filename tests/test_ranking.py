import numpy as np

from connectome.ranking import boundary_tie, jaccard, order_ids, spearman, stability_table, top_k

IDS = np.array([10, 11, 12, 13, 14, 15])
VAL = np.array([5, 5, 3, 3, 3, 1])


def test_order_breaks_ties_by_id_ascending():
    assert IDS[order_ids(VAL, IDS)].tolist() == [10, 11, 12, 13, 14, 15]
    shuffled = np.array([14, 10, 15, 12, 11, 13]); v = np.array([3, 5, 1, 3, 5, 3])
    assert shuffled[order_ids(v, shuffled)].tolist() == [10, 11, 12, 13, 14, 15]


def test_top_k_id_policy_exact_k_and_inclusive_policy_adds_ties():
    assert top_k(VAL, IDS, 3, "id") == {10, 11, 12}
    assert top_k(VAL, IDS, 3, "inclusive") == {10, 11, 12, 13, 14}      # 3rd value is 3 -> all three 3s are included
    assert boundary_tie(VAL, 3) is True and boundary_tie(VAL, 2) is False


def test_inclusive_never_admits_zero_valued_neurons():
    v = np.array([4, 0, 0, 0]); ids = np.array([1, 2, 3, 4])
    assert top_k(v, ids, 3, "inclusive") == {1}
    assert top_k(v, ids, 3, "id") == {1, 2, 3}            # the id policy is forced to pad with zeros: boundary_tie flags it
    assert boundary_tie(v, 3) is True


def test_k_larger_than_population():
    assert top_k(VAL, IDS, 100, "id") == set(IDS.tolist())
    assert top_k(np.array([]), np.array([]), 5) == set()


def test_jaccard_hand_values():
    assert jaccard({1, 2, 3}, {2, 3, 4}) == 2 / 4
    assert jaccard({1}, {1}) == 1.0 and jaccard({1}, {2}) == 0.0
    assert np.isnan(jaccard(set(), set()))


def test_spearman_known_values():
    assert spearman([1, 2, 3, 4], [10, 20, 30, 40]) == 1.0
    assert spearman([1, 2, 3, 4], [4, 3, 2, 1]) == -1.0
    assert np.isnan(spearman([1, 1, 1], [1, 2, 3]))
    # ties get average ranks: x ranks [1.5,1.5,3], y ranks [1,2,3] -> r = sqrt(3)/2
    assert abs(spearman([1, 1, 2], [1, 2, 3]) - np.sqrt(3) / 2) < 1e-12


def test_stability_table_shape_and_fixed_population():
    ids = np.arange(200)
    t = stability_table({1: np.arange(200), 5: np.arange(200)[::-1] * 0 + np.arange(200), 10: np.arange(200)[::-1]}, ids, ks=(10,))
    assert len(t) == 3 * 1 * 2                        # 3 threshold pairs x 1 k x 2 tie policies
    same = t[(t.threshold_a == 1) & (t.threshold_b == 5) & (t.tie_policy == "id")].iloc[0]
    assert same.jaccard == 1.0 and same.spearman_full_population == 1.0 and bool(same.headline) is True
    assert t[(t.tie_policy == "inclusive")].headline.eq(False).all()
    rev = t[(t.threshold_a == 1) & (t.threshold_b == 10) & (t.tie_policy == "id")].iloc[0]
    assert rev.jaccard == 0.0 and rev.spearman_full_population == -1.0
