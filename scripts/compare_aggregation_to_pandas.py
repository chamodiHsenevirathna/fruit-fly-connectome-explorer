"""Check the streaming aggregator against the independent pandas reference (connectome.reference) on a slice of the real table.

Usage: compare_aggregation_to_pandas.py <aggregates.npz> <conn feather> <annotations feather> <start_batch> <stop_batch>
The aggregates must have been produced for exactly that slice (run_aggregation.py --start-batch/--max-batches).
Exit status 0 = zero mismatches.
"""
import sys

import numpy as np, pyarrow as pa, pyarrow.ipc as ipc

from connectome.annotations import load_annotations, make_populations
from connectome.reference import compare

npz, conn, annot, b0, b1 = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]), int(sys.argv[5])
z = np.load(npz); thr = [int(t) for t in z["thresholds"]]
ann = load_annotations(annot); pops = make_populations(ann); ids = ann.bodyId.to_numpy()
rd = ipc.open_file(pa.memory_map(conn))
df = pa.Table.from_batches([rd.get_batch(i) for i in range(b0, b1)]).to_pandas().rename(columns={"body_pre": "pre", "body_post": "post", "weight": "w"})
assert int(z["rows_seen"]) == len(df), (int(z["rows_seen"]), len(df))
total = 0
for p, mask in pops.items():
    bad = compare(z[f"per_body_{p}"], z[f"cat_{p}"], z["glob"], df, ids, ids[mask].tolist(), thr)
    for line in bad: print(f"MISMATCH {p}: {line}")
    total += len(bad)
print("TOTAL MISMATCHES:", total, "->", "ALL MATCH" if total == 0 else "FAIL")
sys.exit(1 if total else 0)
