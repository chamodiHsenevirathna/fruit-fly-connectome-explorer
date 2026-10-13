"""Monte Carlo Jaccard under random tie-breaking: hand-checked values, an independent exact enumeration, reproducibility, edge cases."""
import itertools

import numpy as np
import pytest

from connectome.ranking import jaccard, jaccard_random_ties, top_k


def test_hand_example_k1_shared_tie_expected_half_while_id_view_says_one():
    # Both lists have neurons 1 and 2 tied for first. By bodyId both pick neuron 1, so the headline Jaccard is 1.
    # With random tie-breaking each list picks 1 or 2 with probability 1/2, independently: J = 1 if same (p = 1/2), else 0.
    v = np.array([5, 5, 1]); ids = np.array([1, 2, 3])
    r = jaccard_random_ties(v, v, ids, k=1, n_draws=4000, seed=1)
    assert r["deterministic"] == 1.0
    assert r["mean"] == pytest.approx(0.5, abs=5 * r["sem"] + 1e-9) and abs(r["mean"] - 0.5) < 0.03
    assert r["p2_5"] == 0.0 and r["p97_5"] == 1.0                    # single draws are only ever 0 or 1


def test_hand_example_k2_expected_five_ninths():
    # Neuron 1 (value 9) is always in. One slot is filled from the three tied neurons {2,3,4}, independently per list.
    # Same pick (p = 1/3): J = 1. Different pick (p = 2/3): intersection 1, union 3 -> J = 1/3. E[J] = 1/3 + 2/3 * 1/3 = 5/9.
    v = np.array([9, 5, 5, 5]); ids = np.array([1, 2, 3, 4])
    r = jaccard_random_ties(v, v, ids, k=2, n_draws=6000, seed=2)
    assert abs(r["mean"] - 5 / 9) < 0.02 and abs(r["mean"] - 5 / 9) < 5 * r["sem"]
    assert r["deterministic"] == 1.0


def test_tie_in_only_one_list():
    # List a has a boundary tie (1 and 2), list b has none (neuron 1 is clearly first). a picks 1 with probability 1/2.
    a = np.array([5, 5, 1]); b = np.array([9, 1, 1]); ids = np.array([1, 2, 3])
    r = jaccard_random_ties(a, b, ids, k=1, n_draws=4000, seed=3)
    assert abs(r["mean"] - 0.5) < 0.03


def test_agrees_with_independent_exact_enumeration_on_random_small_cases():
    rng = np.random.default_rng(7)
    checked = 0
    for _ in range(40):
        n = 8; ids = np.arange(1, n + 1)
        a = rng.integers(0, 4, n); b = rng.integers(0, 4, n); k = int(rng.integers(2, 6))
        r = jaccard_random_ties(a, b, ids, k, n_draws=3000, seed=11)
        if r["n_draws"] == 0:
            continue
        # exact expectation: enumerate every way each list could fill its open slots (independently, all equally likely)
        def options(v):
            kth = np.sort(v)[::-1][k - 1]
            above = set(ids[v > kth].tolist()); tied = ids[v == kth].tolist()
            return [above | set(c) for c in itertools.combinations(tied, k - len(above))]
        oa, ob = options(a), options(b)
        exact = np.mean([jaccard(x, y) for x in oa for y in ob])
        assert abs(r["mean"] - exact) < 5 * r["sem"] + 1e-9, (a, b, k, r["mean"], exact)
        checked += 1
    assert checked >= 10                                              # the loop really exercised cases with ties


def test_fixed_seed_is_reproducible_and_seed_matters():
    v = np.array([9, 5, 5, 5, 5]); w = np.array([9, 5, 5, 5, 1]); ids = np.arange(1, 6)
    r1 = jaccard_random_ties(v, w, ids, 3, n_draws=500, seed=42)
    r2 = jaccard_random_ties(v, w, ids, 3, n_draws=500, seed=42)
    r3 = jaccard_random_ties(v, w, ids, 3, n_draws=500, seed=43)
    assert r1 == r2
    assert r1["mean"] != r3["mean"]


def test_no_ties_is_deterministic_and_exact_and_samples_nothing():
    v = np.array([9, 7, 5, 3, 1]); w = np.array([9, 8, 2, 6, 1]); ids = np.arange(1, 6)
    r = jaccard_random_ties(v, w, ids, 2, n_draws=100, seed=0)
    expected = jaccard(top_k(v, ids, 2), top_k(w, ids, 2))
    assert r["mean"] == r["deterministic"] == expected and r["sd"] == 0.0 and r["n_draws"] == 0


def test_ties_that_are_not_at_the_boundary_do_not_trigger_sampling():
    v = np.array([9, 9, 5, 3, 3]); ids = np.arange(1, 6)               # tie inside the top 2 and below it, but not across the cut
    assert jaccard_random_ties(v, v, ids, 2, n_draws=50)["n_draws"] == 0


def test_identical_lists_never_exceed_one_and_estimate_is_below_id_view_when_tie_is_shared():
    v = np.array([4, 4, 4, 4, 4, 4]); ids = np.arange(1, 7)             # everything tied: the id view says identical (1.0)
    r = jaccard_random_ties(v, v, ids, 3, n_draws=2000, seed=5)
    assert r["deterministic"] == 1.0 and r["mean"] < 1.0 and 0.0 <= r["p2_5"] <= r["p97_5"] <= 1.0


def test_edge_cases():
    ids = np.arange(1, 4); v = np.array([1, 1, 1])
    with pytest.raises(ValueError):
        jaccard_random_ties(v, v, ids, 0)
    with pytest.raises(ValueError):
        jaccard_random_ties(v, v, ids, 1, n_draws=0)
    with pytest.raises(ValueError):
        jaccard_random_ties(v, v, ids, 1, max_tied=2)                  # tied group of 3 exceeds the guard
    r = jaccard_random_ties(v, v, ids, 10, n_draws=10)                 # k larger than the population: the whole population, no ties matter
    assert r["mean"] == 1.0 and r["n_draws"] == 0
    e = jaccard_random_ties(np.array([]), np.array([]), np.array([]), 3)
    assert np.isnan(e["mean"]) and e["n_draws"] == 0                   # empty population: undefined, not a crash


def test_lists_with_different_tied_groups_use_their_own_groups():
    # k = 1. List a ties neurons {1, 2}; list b ties neurons {2, 3}. J = 1 only if both pick neuron 2: (1/2) * (1/2) = 1/4.
    # (The id view picks 1 and 2, so it says J = 0, an underestimate here: ties can push the headline value either way.)
    a = np.array([5, 5, 1]); b = np.array([1, 5, 5]); ids = np.array([1, 2, 3])
    r = jaccard_random_ties(a, b, ids, k=1, n_draws=6000, seed=9)
    assert r["deterministic"] == 0.0 and abs(r["mean"] - 0.25) < 0.02
