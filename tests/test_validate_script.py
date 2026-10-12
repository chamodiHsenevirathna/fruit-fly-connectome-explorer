"""Run scripts/validate_connections.py on a tiny synthetic Feather file whose answers are known by hand."""
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.ipc as ipc

ROOT = Path(__file__).resolve().parents[1]

# (body_pre, body_post, weight). Pair (1,2) repeats across the two batches; (2,2) is a self-loop;
# body 99 has no annotation. Sum of weights = 5+4+4+3+2+2+1+1 = 22. File is sorted by weight, descending.
ROWS = [(1, 2, 5), (1, 3, 4), (2, 3, 4), (3, 1, 3), (2, 2, 2), (99, 1, 2), (1, 2, 1), (4, 5, 1)]


def _run(tmp_path):
    schema = pa.schema([("body_pre", pa.int64()), ("body_post", pa.int64()), ("weight", pa.int64())])
    conn = tmp_path / "conn.feather"
    with ipc.new_file(str(conn), schema) as w:
        for chunk in (ROWS[:4], ROWS[4:]):
            w.write_batch(pa.record_batch([pa.array([r[i] for r in chunk], pa.int64()) for i in range(3)], schema=schema))
    ann = pd.DataFrame({"bodyId": [1, 2, 3, 4, 5], "superclass": ["a", "a", "a", None, None],
                        "status": pd.Categorical(["Traced", "Traced", "Traced", "Orphan", "Traced"]),
                        "type": ["t", None, "t", None, None]})
    ann_path = tmp_path / "ann.feather"
    ann.to_feather(ann_path)
    subprocess.run([sys.executable, str(ROOT / "scripts/validate_connections.py"), str(conn), str(ann_path), "--out", str(tmp_path), "--tmp", str(tmp_path / "work")],
                   check=True, env={"PYTHONPATH": str(ROOT / "src")}, capture_output=True)
    return json.load(open(tmp_path / "validation_summary.json"))


def test_validation_counts_match_hand_calculation(tmp_path):
    r = _run(tmp_path)
    assert r["rows"] == 8 and r["weight_sum"] == 22
    assert r["self_loops"] == 1 and r["nulls"] == 0
    assert r["unique_directed_pairs"] == 7 and r["extra_duplicate_records"] == 1
    assert r["distinct_pre"] == 5 and r["distinct_post"] == 4 and r["distinct_any"] == 6
    assert r["unmatched_ids_distinct"] == 1 and r["unmatched_ids_examples"] == [99]
    assert r["weight_nonincreasing_violations_batches"] == 0
    assert r["sum_out_strength"] == r["sum_in_strength"] == 22 and r["sum_out_degree"] == 8
    assert r["consistency_ok"] is True
    assert r["weight_ge"]["3"] == 4  # weights 5,4,4,3
