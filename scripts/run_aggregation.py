"""One streaming pass over the connection table -> per-body degree/strength at several thresholds.

Usage: run_aggregation.py <connection feather> <annotations feather> [--max-batches N] [--out DIR] [--tmp DIR]
Checkpoints every --every batches to <tmp>/state.npz (gitignored) and resumes from it automatically.
Writes <tmp>/aggregates.npz and small tables to <out>. Reports elapsed time and peak memory.
"""
import argparse, json, os, resource, sys, time

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.ipc as ipc

from connectome.aggregate import CATEGORIES, GLOBAL_COLS, METRICS, Aggregator
from connectome.annotations import load_annotations, make_populations

THRESHOLDS = [1, 2, 5, 10, 20, 50]

ap = argparse.ArgumentParser()
ap.add_argument("conn"); ap.add_argument("annot")
ap.add_argument("--start-batch", type=int, default=0, help="first batch to read (slice checks only)")
ap.add_argument("--max-batches", type=int, default=None, help="stop before this batch index")
ap.add_argument("--out", default="outputs/tables")
ap.add_argument("--tmp", default="data/interim/aggregation")
ap.add_argument("--every", type=int, default=400)
a = ap.parse_args()
os.makedirs(a.tmp, exist_ok=True); os.makedirs(a.out, exist_ok=True)
t0 = time.time()

def peak_mb():
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return r / 1e6 if sys.platform == "darwin" else r / 1e3     # macOS reports bytes, Linux kilobytes

def log(msg): print(f"[{time.time()-t0:6.0f}s | peak {peak_mb():6.0f} MB] {msg}", flush=True)

ann = load_annotations(a.annot)
pops = make_populations(ann)
log("populations: " + ", ".join(f"{k}={int(v.sum())}" for k, v in pops.items()) + f"; annotated bodies={len(ann)}")
agg = Aggregator(ann.bodyId.to_numpy(), pops, THRESHOLDS)

rd = ipc.open_file(pa.memory_map(a.conn))
nb = rd.num_record_batches if a.max_batches is None else min(a.max_batches, rd.num_record_batches)
state_path = f"{a.tmp}/state.npz"; start = a.start_batch
if os.path.exists(state_path):
    st = np.load(state_path)
    if int(st["nb"]) == nb and a.start_batch == int(st["start0"]):
        agg.load_state(st); start = int(st["next_batch"]); log(f"resuming from checkpoint at batch {start}/{nb}")
    else:
        log("checkpoint is for a different batch count; ignoring it")

try:
    for i in range(start, nb):
        b = rd.get_batch(i)
        agg.update(b.column("body_pre").to_numpy(zero_copy_only=False), b.column("body_post").to_numpy(zero_copy_only=False),
                   b.column("weight").to_numpy(zero_copy_only=False))
        if (i + 1) % a.every == 0 or i + 1 == nb:
            np.savez(state_path, next_batch=i + 1, nb=nb, start0=a.start_batch, **agg.state()); log(f"batch {i+1}/{nb}  (checkpoint saved)")
except Exception:
    log(f"FAILED at batch {i}; last good checkpoint is in {state_path}; rerun the same command to resume"); raise

checks = agg.check_identities()
failed = [d for d, ok in checks if not ok]
log(f"identity checks: {len(checks) - len(failed)}/{len(checks)} passed" + (f"; FAILED: {failed[:3]}" if failed else ""))
np.savez(f"{a.tmp}/aggregates.npz", **agg.state())

# --- small summary tables ---
rows = []
for p in agg.pop_names:
    for t, T in enumerate(THRESHOLDS):
        for c, cname in enumerate(CATEGORIES):
            rows.append(dict(population=p, threshold=T, category=cname, records=int(agg.cat[p][t, c, 0]), strength=int(agg.cat[p][t, c, 1])))
pd.DataFrame(rows).to_csv(f"{a.out}/edge_category_totals.csv", index=False)
g = pd.DataFrame(agg.glob, columns=GLOBAL_COLS); g.insert(0, "threshold", THRESHOLDS)
g.to_csv(f"{a.out}/threshold_totals.csv", index=False)
pop_rows = []
for p in agg.pop_names:
    member = pops[p]
    for t, T in enumerate(THRESHOLDS):
        pb = agg.per_body[p][t][:, member]
        pop_rows.append(dict(population=p, threshold=T, population_size=int(member.sum()),
                             neurons_with_any_NN_out=int((pb[1] > 0).sum()), neurons_with_any_NN_in=int((pb[5] > 0).sum()),
                             neurons_with_any_NN_connection=int(((pb[1] > 0) | (pb[5] > 0)).sum()),
                             neurons_with_zero_surviving_connections=int((pb.sum(axis=0) == 0).sum())))
pd.DataFrame(pop_rows).to_csv(f"{a.out}/population_by_threshold.csv", index=False)
json.dump(dict(batches=nb, rows_seen=agg.rows_seen, seconds=round(time.time() - t0), peak_memory_mb=round(peak_mb()), identity_checks=len(checks), identity_failures=len(failed)),
          open(f"{a.out}/aggregation_run.json", "w"), indent=1)
log(f"done: {agg.rows_seen} rows")
