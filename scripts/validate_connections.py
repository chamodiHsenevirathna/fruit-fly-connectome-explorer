"""Phase 1b: low-memory, checkpointed streaming validation of the connection table.

Usage: validate_connections.py <feather path or URL> <annotations feather>
         [--max-batches N] [--out DIR] [--tmp DIR] [--stages scan,dup,final]

Stages (each saves its result to disk, so a later failure never forces an earlier stage to rerun):
  scan  : read every record batch once; keep scalars, a weight histogram, per-body partial sums,
          candidate-neuron-rule edge counts; route directed-pair keys into on-disk hash buckets.
          Saves <tmp>/checkpoint.pkl.
  dup   : sort each bucket and look for repeated directed pairs. Saves <tmp>/dup_result.json.
  final : combine the checkpoint and duplicate result into <out>/validation_summary.json (+ weight histogram).
Memory stays small: one record batch plus one bucket (about 1/16 of the keys) at a time.
"""
import argparse, json, os, pickle, shutil, time
from collections import Counter

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.ipc as ipc

from connectome.remote import HttpRangeFile

NB_BUCKETS = 16
THR = [1, 2, 3, 5, 10, 20]
T0 = time.time()


def log(msg):
    print(f"[{time.time()-T0:6.0f}s] {msg}", flush=True)


def member(x, sorted_ids):
    """Vectorised 'x in sorted_ids' by binary search (sorted_ids must be sorted, x need not be)."""
    pos = np.searchsorted(sorted_ids, x)
    pos[pos == len(sorted_ids)] = 0
    return sorted_ids[pos] == x


def sorted_union(a, b):
    """Sorted union of two integer arrays using plain sorting (no hash-based unique)."""
    c = np.concatenate([a, b]); c.sort()
    if len(c) == 0: return c
    keep = np.empty(len(c), dtype=bool); keep[0] = True; keep[1:] = c[1:] != c[:-1]
    return c[keep]


def reduce_partials(parts):
    """Merge (ids, weight_sum, count) partial tables into one table with unique sorted ids."""
    ids = np.concatenate([p[0] for p in parts]); s = np.concatenate([p[1] for p in parts]); d = np.concatenate([p[2] for p in parts])
    order = np.argsort(ids, kind="stable"); ids, s, d = ids[order], s[order], d[order]
    first = np.empty(len(ids), dtype=bool); first[0] = True; first[1:] = ids[1:] != ids[:-1]
    starts = np.flatnonzero(first)
    return ids[starts], np.add.reduceat(s, starts), np.add.reduceat(d, starts)


def open_conn(path):
    src = pa.PythonFile(HttpRangeFile(path), mode="r") if path.startswith("http") else pa.memory_map(path)
    return ipc.open_file(src)


def load_candidates(annot):
    ann = pd.read_feather(annot, columns=["bodyId", "superclass", "status", "type"])
    cands = {
        "superclass_notna": ann.superclass.notna(),
        "status_Traced": ann.status.eq("Traced"),
        "type_notna": ann.type.notna(),
        "superclass_notna_and_Traced": ann.superclass.notna() & ann.status.eq("Traced"),
        "superclass_notna_or_Traced": ann.superclass.notna() | ann.status.eq("Traced"),
        "annotated_any": pd.Series(True, index=ann.index),
    }
    return np.sort(ann.bodyId.to_numpy()), {k: np.sort(ann.bodyId[v].to_numpy()) for k, v in cands.items()}


def stage_scan(a):
    rd = open_conn(a.conn)
    nb = rd.num_record_batches if a.max_batches is None else min(a.max_batches, rd.num_record_batches)
    log(f"scan: schema {rd.schema.names}, {nb} batches")
    ids_all, cand_ids = load_candidates(a.annot)
    cand_edges = {k: 0 for k in cand_ids}; cand_edges_w = {k: Counter() for k in cand_ids}
    S = dict(rows=0, nulls=0, self_loops=0, nonpos_ids=0, nonpos_w=0, wsum=0, max_id=0, sort_violations=0,
             unm_pre=0, unm_post=0, unm_any=0, wmin=None, wmax=0)
    prev_last = None; wcount = Counter(); out_parts, in_parts = [], []
    os.makedirs(a.tmp, exist_ok=True)
    for f in os.listdir(a.tmp):
        if f.endswith(".u64"): os.remove(os.path.join(a.tmp, f))
    bfiles = [open(f"{a.tmp}/b{j:02d}.u64", "wb") for j in range(NB_BUCKETS)]
    for i in range(nb):
        b = rd.get_batch(i)
        S["nulls"] += sum(b.column(c).null_count for c in range(3))
        pre = b.column("body_pre").to_numpy(zero_copy_only=False); post = b.column("body_post").to_numpy(zero_copy_only=False)
        w = b.column("weight").to_numpy(zero_copy_only=False)
        S["rows"] += len(pre); S["self_loops"] += int((pre == post).sum())
        S["nonpos_ids"] += int(((pre <= 0) | (post <= 0)).sum()); S["nonpos_w"] += int((w <= 0).sum())
        S["max_id"] = max(S["max_id"], int(pre.max()), int(post.max()))
        S["wmin"] = int(w.min()) if S["wmin"] is None else min(S["wmin"], int(w.min())); S["wmax"] = max(S["wmax"], int(w.max()))
        S["wsum"] += int(w.sum())
        if np.any(np.diff(w) > 0) or (prev_last is not None and w[0] > prev_last): S["sort_violations"] += 1
        prev_last = int(w[-1])
        u, c = np.unique(w, return_counts=True); wcount.update(dict(zip(u.tolist(), c.tolist())))
        um_pre = ~member(pre, ids_all); um_post = ~member(post, ids_all)
        S["unm_pre"] += int(um_pre.sum()); S["unm_post"] += int(um_post.sum()); S["unm_any"] += int((um_pre | um_post).sum())
        for k, ids in cand_ids.items():
            both = member(pre, ids) & member(post, ids)
            cand_edges[k] += int(both.sum())
            for t in THR: cand_edges_w[k][t] += int((both & (w >= t)).sum())
        wi = w.astype(np.int64)
        for x, parts in ((pre, out_parts), (post, in_parts)):
            o = np.argsort(x, kind="stable"); xs = x[o]; first = np.empty(len(xs), dtype=bool); first[0] = True; first[1:] = xs[1:] != xs[:-1]
            st = np.flatnonzero(first)
            parts.append((xs[st], np.add.reduceat(wi[o], st), np.diff(np.append(st, len(xs))).astype(np.int64)))
        if len(out_parts) >= 50:
            out_parts = [reduce_partials(out_parts)]; in_parts = [reduce_partials(in_parts)]
        keys = (pre.astype(np.uint64) << np.uint64(32)) | post.astype(np.uint64)
        h = ((keys * np.uint64(0x9E3779B97F4A7C15)) >> np.uint64(60)).astype(np.int64)
        for j in range(NB_BUCKETS):
            m = h == j
            if m.any(): bfiles[j].write(keys[m].tobytes())
        if i % 200 == 0: log(f"scan: batch {i}/{nb}")
    for f in bfiles: f.close()
    out_t, in_t = reduce_partials(out_parts), reduce_partials(in_parts)
    assert S["max_id"] < 2**32, "IDs exceed 32 bits; key packing invalid"
    ck = dict(S=S, nb=nb, wcount=dict(wcount), out=out_t, inn=in_t, cand_edges=cand_edges,
              cand_edges_w={k: dict(v) for k, v in cand_edges_w.items()})
    pickle.dump(ck, open(f"{a.tmp}/checkpoint.pkl", "wb"))
    log(f"scan: done, {S['rows']} rows; {len(out_t[0])} distinct pre, {len(in_t[0])} distinct post; checkpoint saved")


def stage_dup(a):
    ck = pickle.load(open(f"{a.tmp}/checkpoint.pkl", "rb"))
    extra = 0; total = 0; dup_keys = []
    for j in range(NB_BUCKETS):
        k = np.fromfile(f"{a.tmp}/b{j:02d}.u64", dtype=np.uint64); total += len(k); k.sort()
        m = k[1:] == k[:-1]; extra += int(m.sum())
        if m.any(): dup_keys.append(np.unique(k[1:][m]))
        del k
    assert total == ck["S"]["rows"], f"bucket keys {total} != rows {ck['S']['rows']}"
    res = {"keys_checked": total, "extra_duplicate_records": extra, "unique_directed_pairs": total - extra,
           "duplicate_pairs_examples": [[int(x) >> 32, int(x) & 0xFFFFFFFF] for x in (np.concatenate(dup_keys)[:20] if dup_keys else [])]}
    json.dump(res, open(f"{a.tmp}/dup_result.json", "w"), indent=1)
    log(f"dup: {res['unique_directed_pairs']} unique pairs, {extra} extra duplicate records (saved)")


def stage_final(a):
    ck = pickle.load(open(f"{a.tmp}/checkpoint.pkl", "rb")); dup = json.load(open(f"{a.tmp}/dup_result.json"))
    S = ck["S"]; out_ids, out_s, out_d = ck["out"]; in_ids, in_s, in_d = ck["inn"]
    ids_all, cand_ids = load_candidates(a.annot)
    t = time.time(); all_ids = sorted_union(out_ids, in_ids); log(f"final: union {len(all_ids)} ids in {time.time()-t:.2f}s")
    unmatched = all_ids[~member(all_ids, ids_all)]
    cn = {}
    for k, ids in cand_ids.items():
        cn[k] = {"n_annotated": int(len(ids)), "n_in_connections": int(member(ids, all_ids).sum()),
                 "n_with_outgoing": int(member(ids, out_ids).sum()), "n_with_incoming": int(member(ids, in_ids).sum()),
                 "edges_both_ends": ck["cand_edges"][k], "edges_both_ends_by_min_weight": {str(t_): v for t_, v in ck["cand_edges_w"][k].items()}}
    wc = np.array(sorted(ck["wcount"].items()), dtype=np.int64)
    res = {
        "batches_read": ck["nb"], "rows": S["rows"], "nulls": S["nulls"], "self_loops": S["self_loops"],
        "nonpositive_ids": S["nonpos_ids"], "nonpositive_weights": S["nonpos_w"], "max_id": S["max_id"],
        "weight_min": S["wmin"], "weight_max": S["wmax"], "weight_sum": S["wsum"], "distinct_weight_values": int(len(wc)),
        "weight_nonincreasing_violations_batches": S["sort_violations"],
        "unique_directed_pairs": dup["unique_directed_pairs"], "extra_duplicate_records": dup["extra_duplicate_records"],
        "distinct_pre": int(len(out_ids)), "distinct_post": int(len(in_ids)), "distinct_any": int(len(all_ids)),
        "sum_out_strength": int(out_s.sum()), "sum_in_strength": int(in_s.sum()), "sum_out_degree": int(out_d.sum()), "sum_in_degree": int(in_d.sum()),
        "consistency_ok": bool(int(out_s.sum()) == int(in_s.sum()) == S["wsum"] and int(out_d.sum()) == int(in_d.sum()) == S["rows"]),
        "unmatched_ids_distinct": int(len(unmatched)), "unmatched_ids_examples": unmatched[:10].tolist(),
        "rows_with_unmatched_pre": S["unm_pre"], "rows_with_unmatched_post": S["unm_post"], "rows_with_any_unmatched": S["unm_any"],
        "annotated_bodies": int(len(ids_all)), "annotated_not_in_connections": int((~member(ids_all, all_ids)).sum()),
        "weight_ge": {str(t_): int(wc[wc[:, 0] >= t_, 1].sum()) for t_ in [1, 2, 3, 5, 10, 20, 50, 100]},
        "weight_sum_ge": {str(t_): int((wc[wc[:, 0] >= t_, 0] * wc[wc[:, 0] >= t_, 1]).sum()) for t_ in [1, 5, 10, 20]},
        "candidate_neuron_sets": cn,
    }
    os.makedirs(a.out, exist_ok=True)
    tag = "" if a.max_batches is None else f"_first{ck['nb']}batches"
    json.dump(res, open(f"{a.out}/validation_summary{tag}.json", "w"), indent=1)
    pd.DataFrame(wc, columns=["weight", "n_records"]).to_csv(f"{a.out}/weight_histogram{tag}.csv", index=False)
    log(f"final: wrote {a.out}/validation_summary{tag}.json")
    print(json.dumps({k: v for k, v in res.items() if k != "candidate_neuron_sets"}, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("conn"); ap.add_argument("annot")
    ap.add_argument("--max-batches", type=int, default=None)
    ap.add_argument("--out", default="outputs/tables")
    ap.add_argument("--tmp", default="data/interim/validation")
    ap.add_argument("--stages", default="scan,dup,final")
    a = ap.parse_args()
    for st in a.stages.split(","):
        {"scan": stage_scan, "dup": stage_dup, "final": stage_final}[st](a)
