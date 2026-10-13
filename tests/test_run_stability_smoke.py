"""run_aggregation.py followed by run_stability.py on a tiny synthetic table (population smaller than k=10/50/100)."""
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.ipc as ipc

ROOT = Path(__file__).resolve().parents[1]
EDGES = [(1, 2, 50), (3, 1, 20), (1, 3, 10), (2, 3, 5), (5, 1, 2), (1, 100, 2), (100, 2, 1), (2, 2, 1)]


def _run(script, *args, cwd=None):
    r = subprocess.run([sys.executable, str(ROOT / "scripts" / script), *map(str, args)], env={"PYTHONPATH": str(ROOT / "src")}, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    return r


def test_stability_runs_on_tiny_population_and_ranks_by_hand(tmp_path):
    schema = pa.schema([("body_pre", pa.int64()), ("body_post", pa.int64()), ("weight", pa.int64())])
    conn = tmp_path / "c.feather"
    with ipc.new_file(str(conn), schema) as w:
        w.write_batch(pa.record_batch([pa.array([r[i] for r in EDGES], pa.int64()) for i in range(3)], schema=schema))
    ann = pd.DataFrame({"bodyId": [1, 2, 3, 4, 5], "superclass": ["a", "a", "a", None, None],
                        "status": pd.Categorical(["Traced", "Traced", "Orphan", "Traced", "Traced"]), "type": list("vwxyz")})
    ann_path = tmp_path / "a.feather"; ann.to_feather(ann_path)
    _run("run_aggregation.py", conn, ann_path, "--out", tmp_path / "out", "--tmp", tmp_path / "work")
    _run("run_stability.py", "--agg", tmp_path / "work/aggregates.npz", "--annot", ann_path, "--out", tmp_path / "out")
    top = pd.read_csv(tmp_path / "out/top100_primary_out_strength_nn.csv")
    # primary neurons = {1,2,3}. Neuron-to-neuron out-strength at T>=1: body 1 -> 2 (50) + 3 (10) = 60; body 2 -> 3 (5) + 2 (1) = 6; body 3 -> 1 (20) = 20
    t1 = top[top.threshold == 1].sort_values("rank")
    assert t1.bodyId.tolist() == [1, 3, 2] and t1.value.tolist() == [60, 20, 6]
    assert len(t1) == 3                                           # only 3 neurons exist, so the "top 100" has 3 rows, not a crash
    ties = pd.read_csv(tmp_path / "out/ties_primary_out_strength_nn.csv")
    assert ties.empty                                             # k = 10, 50, 100 all exceed the 3-neuron population: no k-th place
    stab = pd.read_csv(tmp_path / "out/stability_primary_out_strength_nn.csv")
    assert set(stab.columns) >= {"jaccard", "tie_policy", "headline", "boundary_tie_a", "spearman_full_population"}
