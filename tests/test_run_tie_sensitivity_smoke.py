"""run_aggregation -> run_stability -> run_tie_sensitivity on a 12-neuron ring where the expected answer can be worked out by hand."""
import hashlib
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.ipc as ipc

ROOT = Path(__file__).resolve().parents[1]
# Ring i -> i+1 (12 -> 1). Weights: neurons 1..9 have 12,11,...,4 ; neurons 10, 11, 12 have 3.
W = [12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 3, 3]
EDGES = [(i + 1, (i + 1) % 12 + 1, W[i]) for i in range(12)]


def _run(script, *args):
    r = subprocess.run([sys.executable, str(ROOT / "scripts" / script), *map(str, args)], env={"PYTHONPATH": str(ROOT / "src")}, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    return r


def _build(tmp_path):
    schema = pa.schema([("body_pre", pa.int64()), ("body_post", pa.int64()), ("weight", pa.int64())])
    conn = tmp_path / "c.feather"
    with ipc.new_file(str(conn), schema) as w:
        w.write_batch(pa.record_batch([pa.array([e[i] for e in EDGES], pa.int64()) for i in range(3)], schema=schema))
    ann = pd.DataFrame({"bodyId": list(range(1, 13)), "superclass": ["a"] * 12, "status": pd.Categorical(["Traced"] * 12), "type": ["t"] * 12})
    ann_path = tmp_path / "a.feather"; ann.to_feather(ann_path)
    _run("run_aggregation.py", conn, ann_path, "--out", tmp_path / "out", "--tmp", tmp_path / "work")
    _run("run_stability.py", "--agg", tmp_path / "work/aggregates.npz", "--annot", ann_path, "--out", tmp_path / "out")
    return ann_path


def _md5(p):
    return hashlib.md5(Path(p).read_bytes()).hexdigest()


def test_tie_sensitivity_matches_hand_calculation_is_reproducible_and_leaves_other_tables_alone(tmp_path):
    ann_path = _build(tmp_path)
    out = tmp_path / "out"
    protected = sorted(out.glob("stability_*.csv")) + sorted(out.glob("top100_*.csv")) + sorted(out.glob("ties_*.csv"))
    before = {p.name: _md5(p) for p in protected}
    args = ["--agg", tmp_path / "work/aggregates.npz", "--annot", ann_path, "--tables", out, "--draws", 4000, "--seed", 5]
    _run("run_tie_sensitivity.py", *args)
    t = pd.read_csv(out / "tie_sensitivity_primary_out_strength_nn.csv")
    r = t[(t.threshold_a == 1) & (t.threshold_b == 5) & (t.k == 10)].iloc[0]
    # T>=1 top 10: neurons 1..9 plus ONE of {10,11,12} (value 3, three tied). T>=5 top 10: neurons 1..8 plus TWO of the four zero-valued neurons {9..12}.
    # By bodyId: A = 1..10, B = 1..10, so the headline Jaccard is 1. Averaging over every possible random choice gives 163/198 (worked out by hand).
    assert r.jaccard_id_deterministic == 1.0 and r.boundary_tie_a and r.boundary_tie_b
    assert abs(r.mc_mean - 163 / 198) < 0.02 and abs(r.mc_mean - 163 / 198) < 5 * r.mc_sem
    assert r.mc_mean < r.jaccard_id_deterministic and r.n_draws == 4000 and 0 <= r.mc_p2_5 <= r.mc_p97_5 <= 1
    assert (t.k == 10).all()                                                    # k = 50 and 100 exceed the 12-neuron population: skipped, no crash
    first = (out / "tie_sensitivity_primary_out_strength_nn.csv").read_bytes()
    _run("run_tie_sensitivity.py", *args)
    assert (out / "tie_sensitivity_primary_out_strength_nn.csv").read_bytes() == first   # same seed -> byte-identical output
    assert {p.name: _md5(p) for p in protected} == before                       # existing tables untouched
