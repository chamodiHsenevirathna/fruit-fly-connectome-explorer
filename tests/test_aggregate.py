"""Tests for connectome.aggregate. The hand example below was computed on paper (see comments), not by the code."""
import numpy as np
import pandas as pd
import pytest

from connectome.aggregate import Aggregator, M, METRICS
from connectome.reference import compare

# Bodies: annotated = 1,2,3,4,5,6.  Neuron population P = {1,2,3,4}.  5 = annotated non-neuron, 6 = annotated, no edges.
# 100, 200 = unannotated bodies.  Records (pre, post, weight):
EDGES = [(1, 2, 10), (1, 3, 2), (1, 5, 5), (1, 100, 1), (2, 3, 20), (3, 1, 50), (5, 1, 7), (100, 2, 3), (100, 200, 4), (2, 2, 6)]
ANN_IDS = [1, 2, 3, 4, 5, 6]
POPS = {"primary": np.array([1, 1, 1, 1, 0, 0], bool), "sensitivity": np.array([1, 1, 0, 1, 0, 0], bool)}
THR = [1, 10, 50]


def run(edges=EDGES, batches=1, pops=POPS, thr=THR):
    agg = Aggregator(ANN_IDS, pops, thr)
    arr = np.array(edges, dtype=np.int64).reshape(-1, 3)
    for part in np.array_split(arr, batches):
        if len(part): agg.update(part[:, 0], part[:, 1], part[:, 2])
    return agg


def body(agg, pop, t, b):
    i = ANN_IDS.index(b); ti = agg.thresholds.index(t)
    return {m: int(agg.per_body[pop][ti, M[m], i]) for m in METRICS}


def test_hand_example_threshold_1():
    a = run()
    # body 1: out to neurons 2 (10), 3 (2) -> deg 2, str 12; out to others 5 (5), 100 (1) -> deg 2, str 6
    #         in from neuron 3 (50) -> 1/50; in from other 5 (7) -> 1/7
    assert body(a, "primary", 1, 1) == dict(out_deg_n=2, out_str_n=12, out_deg_o=2, out_str_o=6, in_deg_n=1, in_str_n=50, in_deg_o=1, in_str_o=7)
    # body 2: out to 3 (20) and to itself (6) -> deg 2, str 26; in from 1 (10) and itself (6) -> 2/16; in from unannotated 100 (3) -> 1/3
    assert body(a, "primary", 1, 2) == dict(out_deg_n=2, out_str_n=26, out_deg_o=0, out_str_o=0, in_deg_n=2, in_str_n=16, in_deg_o=1, in_str_o=3)
    # body 3: out to 1 (50); in from 1 (2) and 2 (20)
    assert body(a, "primary", 1, 3) == dict(out_deg_n=1, out_str_n=50, out_deg_o=0, out_str_o=0, in_deg_n=2, in_str_n=22, in_deg_o=0, in_str_o=0)
    # body 4 is a neuron with no records at all: kept in the population with zeros
    assert all(v == 0 for v in body(a, "primary", 1, 4).values())
    # body 5 is NOT a neuron: its edge to neuron 1 is "other -> neuron", seen from 5 as an outgoing edge to a neuron partner
    assert body(a, "primary", 1, 5) == dict(out_deg_n=1, out_str_n=7, out_deg_o=0, out_str_o=0, in_deg_n=1, in_str_n=5, in_deg_o=0, in_str_o=0)


def test_category_totals_hand_example():
    a = run(); c = a.cat["primary"][0]          # threshold 1; columns: records, strength
    assert c.tolist() == [[5, 88], [2, 6], [2, 10], [1, 4]]      # NN, NO, ON, OO
    g = a.glob[0].tolist()
    # records 10, strength 108; sources unannotated: edges 100->2 (3) and 100->200 (4) = 2 records, 7; targets unannotated: 1->100 (1), 100->200 (4) = 2, 5
    assert g == [10, 108, 2, 7, 2, 5]


def test_threshold_boundary_is_inclusive():
    a = run()
    # threshold 10 keeps weights 10, 20, 50 (10 itself IS kept) -> NN edges 1->2, 2->3, 3->1
    assert a.cat["primary"][1][0].tolist() == [3, 80]
    assert body(a, "primary", 10, 1)["out_deg_n"] == 1 and body(a, "primary", 10, 1)["out_str_n"] == 10
    # threshold 50 keeps only 3->1
    assert a.cat["primary"][2][0].tolist() == [1, 50]
    assert body(a, "primary", 50, 2)["out_deg_n"] == 0       # neuron 2 survives in the table with zeros


def test_sensitivity_population_uses_its_own_members():
    a = run()
    # sensitivity = {1,2,4}. Body 1: neuron partners are only 2 (10); body 3 is now "other", so 1->3 (2) joins 1->5 (5), 1->100 (1)
    assert body(a, "sensitivity", 1, 1) == dict(out_deg_n=1, out_str_n=10, out_deg_o=3, out_str_o=8, in_deg_n=0, in_str_n=0, in_deg_o=2, in_str_o=57)


def test_identities_hold_for_every_threshold_population():
    bad = [d for d, ok in run().check_identities() if not ok]
    assert bad == []


def test_batching_does_not_change_results():
    one, many = run(batches=1), run(batches=7)
    for p in POPS:
        assert np.array_equal(one.per_body[p], many.per_body[p]) and np.array_equal(one.cat[p], many.cat[p])
    assert np.array_equal(one.glob, many.glob)


def test_duplicate_records_are_counted_separately_by_contract():
    # The verified data contract says pairs never repeat, so the pipeline does NOT deduplicate. If a pair were repeated,
    # each record would count: out-degree would overstate distinct partners. validate_connections.py is the guard.
    a = run(edges=[(1, 2, 3), (1, 2, 3)], thr=[1])
    assert body(a, "primary", 1, 1)["out_deg_n"] == 2 and body(a, "primary", 1, 1)["out_str_n"] == 6


def test_empty_and_unannotated_only_input():
    a = run(edges=[], thr=[1])
    assert a.glob.sum() == 0 and all(ok for _, ok in a.check_identities())
    b = run(edges=[(100, 200, 4)], thr=[1])
    assert b.per_body["primary"].sum() == 0 and b.cat["primary"][0, 3].tolist() == [1, 4]


def test_unsorted_annotation_ids_rejected():
    with pytest.raises(ValueError):
        Aggregator([3, 1, 2], {"p": np.ones(3, bool)}, [1])


def _random_case(seed=0):
    rng = np.random.default_rng(seed)
    ann = np.arange(1000, 1060); other = np.arange(5000, 5040)
    pop = np.zeros(len(ann), bool); pop[:40] = True               # 40 neurons, 20 annotated non-neurons, 40 unannotated
    n = 4000
    pre = rng.choice(np.concatenate([ann, other]), n); post = rng.choice(np.concatenate([ann, other]), n)
    df = pd.DataFrame({"pre": pre, "post": post, "w": rng.integers(1, 60, n)}).drop_duplicates(["pre", "post"])
    return ann, pop, df


def _aggregate(ann, pop, df, thresholds, swap_direction=False):
    agg = Aggregator(ann, {"p": pop}, thresholds)
    for part in np.array_split(df.to_numpy(), 9):
        agg.update(part[:, 1], part[:, 0], part[:, 2]) if swap_direction else agg.update(part[:, 0], part[:, 1], part[:, 2])
    return agg


THR6 = [1, 2, 5, 10, 20, 50]


def test_streaming_matches_independent_reference_for_every_annotated_body():
    """The reference (connectome.reference) is written from the definitions and shares no code with the aggregator.
    It is compared for EVERY annotated body, not just members: a non-member annotated body still has connections to members."""
    ann, pop, df = _random_case()
    agg = _aggregate(ann, pop, df, THR6)
    assert compare(agg.per_body["p"], agg.cat["p"], agg.glob, df, ann, ann[pop].tolist(), THR6) == []
    assert all(ok for _, ok in agg.check_identities())
    nonmember = ~pop     # the case an earlier member-only test missed: non-members DO carry non-zero partner counts
    assert agg.per_body["p"][0, M["out_deg_n"]][nonmember].sum() > 0 and agg.per_body["p"][0, M["in_deg_n"]][nonmember].sum() > 0


def test_reference_comparison_detects_deliberate_bugs():
    """A check is only worth something if it can fail. Break the aggregator three ways; each must be reported."""
    ann, pop, df = _random_case()
    members = ann[pop].tolist()
    # (1) off-by-one threshold inclusivity: behaves like weight > T instead of weight >= T
    strict = _aggregate(ann, pop, df, [t + 1 for t in THR6])
    assert compare(strict.per_body["p"], strict.cat["p"], strict.glob, df, ann, members, THR6) != []
    # (2) incoming and outgoing swapped
    swapped = _aggregate(ann, pop, df, THR6, swap_direction=True)
    assert compare(swapped.per_body["p"], swapped.cat["p"], swapped.glob, df, ann, members, THR6) != []
    # (3) wrong population: first 30 bodies instead of first 40 are treated as neurons
    wrong_pop = pop.copy(); wrong_pop[30:40] = False
    wrong = _aggregate(ann, wrong_pop, df, THR6)
    assert compare(wrong.per_body["p"], wrong.cat["p"], wrong.glob, df, ann, members, THR6) != []
