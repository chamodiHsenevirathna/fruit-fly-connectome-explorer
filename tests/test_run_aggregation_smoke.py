"""End-to-end smoke test of scripts/run_aggregation.py on a tiny synthetic Feather file (no download needed)."""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.ipc as ipc

ROOT = Path(__file__).resolve().parents[1]
EDGES = [(1, 2, 50), (3, 1, 20), (1, 3, 10), (2, 3, 5), (5, 1, 2), (1, 100, 2), (100, 2, 1), (2, 2, 1)]   # sorted by weight, descending


def test_pipeline_smoke(tmp_path):
    schema = pa.schema([("body_pre", pa.int64()), ("body_post", pa.int64()), ("weight", pa.int64())])
    conn = tmp_path / "c.feather"
    with ipc.new_file(str(conn), schema) as w:
        for chunk in (EDGES[:3], EDGES[3:]):
            w.write_batch(pa.record_batch([pa.array([r[i] for r in chunk], pa.int64()) for i in range(3)], schema=schema))
    ann = pd.DataFrame({"bodyId": [1, 2, 3, 4, 5], "superclass": ["a", "a", "a", None, None],
                        "status": pd.Categorical(["Traced", "Traced", "Orphan", "Traced", "Traced"])})
    ann_path = tmp_path / "a.feather"; ann.to_feather(ann_path)
    r = subprocess.run([sys.executable, str(ROOT / "scripts/run_aggregation.py"), str(conn), str(ann_path),
                        "--out", str(tmp_path / "out"), "--tmp", str(tmp_path / "work"), "--every", "1"],
                       env={"PYTHONPATH": str(ROOT / "src")}, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    run = json.load(open(tmp_path / "out/aggregation_run.json"))
    assert run["rows_seen"] == 8 and run["identity_failures"] == 0
    cat = pd.read_csv(tmp_path / "out/edge_category_totals.csv")
    nn = cat[(cat.population == "primary") & (cat.threshold == 1) & (cat.category == "NN")].iloc[0]
    # primary neurons = {1,2,3}; neuron-to-neuron records: 1->2 (50), 3->1 (20), 1->3 (10), 2->3 (5), 2->2 (1) = 5 records, 86
    assert (nn.records, nn.strength) == (5, 86)
    pop = pd.read_csv(tmp_path / "out/population_by_threshold.csv")
    # At threshold 50 only record 1->2 (weight 50) survives.
    # sensitivity population = {1,2} (superclass present AND Traced): both ends survive -> 0 neurons with nothing left.
    sens = pop[(pop.population == "sensitivity") & (pop.threshold == 50)].iloc[0]
    assert sens.population_size == 2 and sens.neurons_with_zero_surviving_connections == 0
    # primary population = {1,2,3}: neuron 3 has no surviving record, but is KEPT in the population (fixed population).
    prim = pop[(pop.population == "primary") & (pop.threshold == 50)].iloc[0]
    assert prim.population_size == 3 and prim.neurons_with_zero_surviving_connections == 1
    assert (tmp_path / "work/aggregates.npz").exists()
