"""Checks that the figure inputs agree with the saved result tables (skipped if the tables are absent, e.g. in a fresh clone)."""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from connectome import figdata as fd

T = Path(__file__).resolve().parents[1] / "outputs" / "tables"
needed = ["threshold_totals.csv", "weight_histogram.csv", "edge_category_totals.csv", "population_by_threshold.csv", "ranking_support.csv",
          "rank_trajectories_out_strength_nn.csv", "rank_trajectories_out_degree_nn.csv", "top100_primary_out_strength_nn.csv", "top100_primary_out_degree_nn.csv"]
pytestmark = pytest.mark.skipif(not all((T / f).exists() for f in needed), reason="saved result tables not present")


def test_saved_tables_are_mutually_consistent():
    p = fd.verify_sources(pd.read_csv(T / "threshold_totals.csv"), pd.read_csv(T / "weight_histogram.csv"), pd.read_csv(T / "edge_category_totals.csv"),
                          pd.read_csv(T / "population_by_threshold.csv"), pd.read_csv(T / "ranking_support.csv"))
    assert p == []


def test_retention_from_totals_matches_weight_histogram():
    tot = pd.read_csv(T / "threshold_totals.csv"); h = fd.histogram_retention(pd.read_csv(T / "weight_histogram.csv")).set_index("threshold")
    r = fd.retention(tot)
    for _, row in r.iterrows():
        assert row.records == h.loc[int(row.threshold), "records"]
        assert row.records_share == pytest.approx(h.loc[int(row.threshold), "records"] / h.loc[1, "records"])
        assert row.weight_share == pytest.approx(h.loc[int(row.threshold), "strength"] / h.loc[1, "strength"])


@pytest.mark.parametrize("metric", ["out_strength_nn", "out_degree_nn"])
def test_trajectory_table_obeys_its_rules(metric):
    t = pd.read_csv(T / f"rank_trajectories_{metric}.csv"); top = pd.read_csv(T / f"top100_primary_{metric}.csv")
    # 1. the neuron set is exactly the union of the top 10 at thresholds 1, 10, 50
    expected = set(top[(top.threshold.isin([1, 10, 50])) & (top["rank"] <= 10)].bodyId)
    assert set(t.bodyId) == expected
    # 2. every neuron appears once per threshold (6 thresholds)
    assert (t.groupby("bodyId").size() == 6).all()
    # 3. a zero-valued neuron has no rank and is flagged; a positive one has a rank
    assert (t["rank"].isna() == (t.value == 0)).all() and (t.has_surviving_outgoing == (t.value > 0)).all()
    # 4. ranks inside the top 100 equal the ranks in the saved top-100 tables
    m = t.dropna(subset=["rank"]).merge(top[["threshold", "bodyId", "rank"]].rename(columns={"rank": "saved"}), on=["threshold", "bodyId"])
    assert len(m) > 0 and (m["rank"] == m.saved).all()
    # 5. each neuron's selection source lists exactly the thresholds at which it is in the top 10
    for b, g in t.groupby("bodyId"):
        at = sorted(top[(top.bodyId == b) & (top.threshold.isin([1, 10, 50])) & (top["rank"] <= 10)].threshold)
        assert g.selected_at.iloc[0] == ",".join(map(str, at))
